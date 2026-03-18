import argparse
import json
import os
import pickle
import sys
import re
from pathlib import Path

# -- Dependencies --
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("Missing: pip install google-genai")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Missing: pip install numpy")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Missing: pip install sentence-transformers")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

# -- Config --
_key = os.environ.get("GEMINI_API_KEY", "")
if not _key:
    print("[ERROR] GEMINI_API_KEY not found.")
    sys.exit(1)

_client = genai.Client(api_key=_key)

COURSES_FILE   = "ncu_courses_en.jsonl"
INDEX_FILE     = "ncu_index.pkl"
GEMINI_MODEL   = "gemini-2.0-flash"
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# -- System Prompt for ReAct --
SYSTEM_PROMPT = """You are an NCU Course Assistant. 
You have access to a tool called 'search_ncu_courses'. 

Rules:
1. If a user asks about courses, ALWAYS use the 'search_ncu_courses' tool.
2. If the tool returns no results, state: "I don't know — that information is not in the NCU course database."
3. Do not hallucinate course details.
4. Use bullet points for lists. Include period codes and times (e.g. Period 5, 13:00–13:50).
"""

# -- Embedding & Index Classes (Same as before) --
class LocalEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
    def embed_batch(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)
    def embed(self, text):
        return self.model.encode(text, normalize_embeddings=True)

class VectorIndex:
    def __init__(self, embedder):
        self.embedder = embedder
        self.vectors = None
        self.documents = []

    def search(self, query, top_k=8):
        if self.vectors is None: return []
        q_vec = self.embedder.embed(query)
        scores = self.vectors @ q_vec
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [{**self.documents[i], "score": float(scores[i])} for i in top_idx if scores[i] > 0.15]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"vectors": self.vectors, "documents": self.documents, "embedder": self.embedder}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls(data["embedder"])
        idx.vectors, idx.documents = data["vectors"], data["documents"]
        return idx

# -- The ReAct Agent Logic --
class NCUReActAgent:
    def __init__(self, index):
        self.index = index
        self.history = []
        
        # Define the tool for Gemini
        self.tools = [self.search_ncu_courses]

    def search_ncu_courses(self, query: str) -> str:
        """Searches the NCU database for course names, instructors, times, and classrooms."""
        print(f"  [Tool Action] Searching index for: {query}...")
        results = self.index.search(query)
        if not results:
            return "No courses found for this query."
        return "\n\n".join([r["text"] for r in results])

    def chat(self, user_message: str):
        # Add user message to history
        self.history.append(genai_types.Content(role="user", parts=[genai_types.Part(text=user_message)]))

        # ReAct Loop: Gemini can call tools and we provide the "Observation"
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=self.history,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=self.tools,
            ),
        )

        # Handle tool calls if they exist
        for part in response.candidates[0].content.parts:
            if part.call:
                # Execute the tool
                observation = self.search_ncu_courses(**part.call.args)
                
                # Feed the observation back to the model
                self.history.append(response.candidates[0].content) # Model's thought/call
                self.history.append(genai_types.Content(
                    role="user", # In many SDKs, tool results are sent back as a user/tool role
                    parts=[genai_types.Part(text=f"Observation: {observation}")]
                ))
                
                # Get final answer after observation
                final_response = _client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=self.history,
                    config=genai_types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
                )
                self.history.append(final_response.candidates[0].content)
                return final_response.text

        # If no tool call was needed (unlikely for course queries)
        self.history.append(response.candidates[0].content)
        return response.text

# -- Entry & CLI (Minimal updates to handle the new class) --
def run_agent():
    if not Path(INDEX_FILE).exists():
        print("Run --build first.")
        return
    index = VectorIndex.load(INDEX_FILE)
    agent = NCUReActAgent(index)
    
    print("NCU ReAct Agent Ready. (Type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit"): break
        print("\nAssistant: ", end="")
        print(agent.chat(user_input))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    
    if args.build:
        # Building logic (Load JSONL -> Embed -> Save)
        from sentence_transformers import SentenceTransformer
        docs = []
        with open(COURSES_FILE, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                docs.append({"text": c.get("_text_en", ""), "metadata": c})
        embedder = LocalEmbedder()
        idx = VectorIndex(embedder)
        idx.vectors = embedder.embed_batch([d["text"] for d in docs])
        idx.documents = docs
        idx.save(INDEX_FILE)
    else:
        run_agent()