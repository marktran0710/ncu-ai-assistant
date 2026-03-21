import argparse
import json
import os
import sys
import pickle
from pathlib import Path
import numpy as np

# -- Dependencies --
try:
    from openai import OpenAI  
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Missing libraries! Run: pip install openai sentence_transformers numpy")
    sys.exit(1)

# -- Config --
COURSES_FILE = "ncu_courses_en.jsonl"
INDEX_FILE   = "ncu_index.pkl"
MODEL_NAME   = "qwen2.5:14b"
API_KEY      = "ollama"
BASE_URL     = "http://localhost:11434/v1" 

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- NEW: Local Embedding & Indexing Logic ---

class LocalEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def embed_batch(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, text):
        return self.model.encode([text])[0]

class VectorIndex:
    def __init__(self, embedder):
        self.embedder = embedder
        self.vectors = []
        self.documents = []

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({"vectors": self.vectors, "documents": self.documents}, f)

    @classmethod
    def load(cls, path, embedder):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        instance = cls(embedder)
        instance.vectors = data["vectors"]
        instance.documents = data["documents"]
        return instance

    def search(self, query, top_k=3):
        query_vec = self.embedder.embed_query(query)
        # Calculate Cosine Similarity
        norms = np.linalg.norm(self.vectors, axis=1)
        q_norm = np.linalg.norm(query_vec)
        scores = np.dot(self.vectors, query_vec) / (norms * q_norm)
        
        best_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in best_indices if scores[i] > 0.3]

# --- Agent Logic ---

SYSTEM_PROMPT = """You are an NCU Course Assistant. 
You have access to a tool called 'search_ncu_courses'. 

Rules:
1. If a user asks about courses or professors, ALWAYS use the 'search_ncu_courses' tool.
2. Provide your final answer based ONLY on the tool's results.
3. Use bullet points for lists.
4. Respond in English.
"""

class NCUReActAgent:
    def __init__(self, index):
        self.index = index
        self.history = []
        self.tools = [{
            "type": "function",
            "function": {
                "name": "search_ncu_courses",
                "description": "Search for NCU course information by name, code, or professor.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search keyword"},
                    },
                    "required": ["query"],
                },
            }
        }]

    def search_ncu_courses(self, query: str) -> str:
        print(f"  [Action] Querying index for: {query}...")
        results = self.index.search(query)
        if not results: return "No matching courses found."
        return "\n\n".join([r["text"] for r in results])

    def chat(self, user_message: str):
        self.history.append({"role": "user", "content": user_message})
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.history,
            tools=self.tools,
            tool_choice="auto"
        )

        message = response.choices[0].message
        
        if message.tool_calls:
            self.history.append(message) 
            for tool_call in message.tool_calls:
                if tool_call.function.name == "search_ncu_courses":
                    args = json.loads(tool_call.function.arguments)
                    observation = self.search_ncu_courses(args.get("query", ""))
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "search_ncu_courses",
                        "content": f"Observation: {observation}"
                    })

            final_res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.history,
            )
            final_message = final_res.choices[0].message
            self.history.append(final_message)
            return final_message.content

        self.history.append(message)
        return message.content

# --- Main Execution ---

def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    embedder = LocalEmbedder()

    if args.build:
        print("--- Building Index... ---")
        docs = []
        if not os.path.exists(COURSES_FILE):
            print(f"Error: {COURSES_FILE} not found!")
            return
            
        with open(COURSES_FILE, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                # Use _text_en for the searchable content
                text_content = c.get("_text_en", str(c))
                docs.append({"text": text_content, "metadata": c})
        
        idx = VectorIndex(embedder)
        idx.vectors = embedder.embed_batch([d["text"] for d in docs])
        idx.documents = docs
        idx.save(INDEX_FILE)
        print("Done building index!")
    else:
        if not Path(INDEX_FILE).exists():
            print("Error: Please run with --build first.")
            return
        
        index = VectorIndex.load(INDEX_FILE, embedder)
        agent = NCUReActAgent(index)
        
        print(f"\nAssistant ({MODEL_NAME}) ready! Type 'exit' to quit.")
        while True:
            u_input = input("\nYou: ").strip()
            if u_input.lower() in ['exit', 'quit']: break
            print(f"\nAssistant: {agent.chat(u_input)}")

if __name__ == "__main__":
    run_main()