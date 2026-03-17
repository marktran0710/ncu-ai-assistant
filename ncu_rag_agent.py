"""
NCU Course RAG Agent
=====================
Embeds NCU course data from ncu_courses_en.jsonl into a local vector store
and answers questions strictly from that data. If the answer is not in the
data, it replies "I don't know."

Requirements:
    pip install google-genai numpy

Environment variable:
    GEMINI_API_KEY=your_key_here

Files needed in the same directory:
    ncu_courses_en.jsonl      (from ncu_course_scraper.py)
    ncu_timeslot_lookup.json
    ncu_building_lookup.json

Usage:
    # Step 1 — build the index (run once, saves ncu_index.pkl)
    python ncu_rag_agent.py --build

    # Step 2 — start the chat agent
    python ncu_rag_agent.py
"""

import argparse
import json
import os
import pickle
import sys
import re
from pathlib import Path

# ── Check dependencies ─────────────────────────────────────────────────────────

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

# ── Gemini client setup (chat only, no embedding) ──────────────────────────────

_key = os.environ.get("GEMINI_API_KEY", "")
if not _key:
    print("[ERROR] GEMINI_API_KEY not found. Add it to your .env file:")
    print("        GEMINI_API_KEY=your_key_here")
    sys.exit(1)

_client = genai.Client(api_key=_key)

# ── Config ─────────────────────────────────────────────────────────────────────

COURSES_FILE   = "ncu_courses_en.jsonl"
TIMESLOT_FILE  = "ncu_timeslot_lookup.json"
BUILDING_FILE  = "ncu_building_lookup.json"
INDEX_FILE     = "ncu_index.pkl"

GEMINI_CHAT_MODEL = "gemini-2.0-flash"

# Local embedding model — free, no API key, supports Chinese + English
# Downloads ~90MB on first run, cached locally after that
EMBED_MODEL_NAME  = "paraphrase-multilingual-MiniLM-L12-v2"

TOP_K                = 8
SIMILARITY_THRESHOLD = 0.15

SYSTEM_PROMPT = """You are an NCU (National Central University, Taiwan) course information assistant.

Your ONLY knowledge source is the course data provided to you in the <context> block.
Rules:
- Answer ONLY using information from the context.
- If the answer is not in the context, say exactly: "I don't know — that information is not in the NCU course database."
- Never make up course names, instructors, schedules, or room numbers.
- Be concise and precise. Use bullet points for lists of courses.
- When mentioning schedules, include both the period code and the actual time (e.g. "Period 5, 13:00–13:50").
- When mentioning classrooms, include the building name if available."""

# ── Local sentence-transformers embedder (free, no API key) ───────────────────

class LocalEmbedder:
    """
    Uses sentence-transformers running locally on CPU.
    Model: paraphrase-multilingual-MiniLM-L12-v2
      - Free, ~90MB download (cached after first run)
      - Supports 50+ languages including Chinese and English
      - 384-dim embeddings
    """
    def __init__(self):
        print(f"Loading local embedding model '{EMBED_MODEL_NAME}' ...")
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        print("  Model loaded.")

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vecs = self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.extend(vecs)
            print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        return np.array(all_embeddings, dtype=np.float32)

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)


# ── Vector index ───────────────────────────────────────────────────────────────

class VectorIndex:
    def __init__(self, embedder):
        self.embedder  = embedder
        self.vectors:  np.ndarray | None = None
        self.documents: list[dict] = []

    def build(self, documents: list[dict]) -> None:
        self.documents = documents
        texts = [d["text"] for d in documents]
        print(f"Building embeddings for {len(texts)} documents...")
        self.vectors = self.embedder.embed_batch(texts)
        print("Index built.")

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        if self.vectors is None:
            return []
        q_vec = self.embedder.embed(query)
        # cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        normed = self.vectors / np.where(norms > 0, norms, 1)
        q_norm = np.linalg.norm(q_vec)
        q_normed = q_vec / q_norm if q_norm > 0 else q_vec
        scores = normed @ q_normed
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score >= SIMILARITY_THRESHOLD:
                results.append({**self.documents[idx], "score": score})
        return results

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"vectors": self.vectors, "documents": self.documents,
                         "embedder": self.embedder}, f)
        print(f"Index saved to {path}")

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls(data["embedder"])
        idx.vectors   = data["vectors"]
        idx.documents = data["documents"]
        return idx


# ── Document loader ────────────────────────────────────────────────────────────

def load_documents() -> list[dict]:
    """
    Load courses from JSONL. Each document = one course with its _text_en field
    as the embedding text, plus a rich metadata block for the context window.
    """
    if not Path(COURSES_FILE).exists():
        print(f"[ERROR] {COURSES_FILE} not found. Run ncu_course_scraper.py first.")
        sys.exit(1)

    docs = []
    with open(COURSES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            text = c.get("_text_en", "")
            if not text:
                continue
            docs.append({
                "text":      text,
                "course_id": c.get("course_id", ""),
                "metadata":  c,
            })

    print(f"Loaded {len(docs)} course documents.")
    return docs


def load_lookups() -> tuple[dict, dict]:
    timeslot = {}
    building = {}
    if Path(TIMESLOT_FILE).exists():
        with open(TIMESLOT_FILE, encoding="utf-8") as f:
            timeslot = json.load(f)
    if Path(BUILDING_FILE).exists():
        with open(BUILDING_FILE, encoding="utf-8") as f:
            building = json.load(f)
    return timeslot, building


# ── RAG agent ──────────────────────────────────────────────────────────────────

class NCUAgent:
    def __init__(self, index: VectorIndex, timeslot: dict, building: dict):
        self.index    = index
        self.timeslot = timeslot
        self.building = building
        self.history: list[genai_types.Content] = []

        # Build prefix → dept_name_en map from the scraped data
        self.prefix_map: dict[str, str] = {}
        for doc in index.documents:
            m = doc.get("metadata", {})
            prefix = m.get("dept_code_prefix", "")
            dept   = m.get("dept_name_en", "")
            if prefix and dept and prefix not in self.prefix_map:
                self.prefix_map[prefix] = dept

        print(f"  Loaded {len(self.prefix_map)} dept code prefixes for query expansion.")

    def _build_context(self, results: list[dict]) -> str:
        if not results:
            return "(No relevant course data found in the database.)"
        return "\n\n---\n\n".join(r["text"] for r in results)

    def _augment_query(self, query: str) -> str:
        """
        Expand dept code prefixes found in the query into full dept names.
        e.g. "CS courses on Monday" → "CS Computer Science and Engineering courses on Monday"
        """
        expanded = query
        for prefix, dept_name in self.prefix_map.items():
            pattern = rf"\b{re.escape(prefix)}\b"
            if re.search(pattern, expanded, flags=re.IGNORECASE):
                expanded = re.sub(
                    pattern,
                    f"{prefix} {dept_name}",
                    expanded,
                    flags=re.IGNORECASE,
                )
        return expanded

    def chat(self, user_message: str) -> str:
        augmented = self._augment_query(user_message)
        results   = self.index.search(augmented, top_k=TOP_K)
        context   = self._build_context(results)

        user_turn = f"<context>\n{context}\n</context>\n\nQuestion: {user_message}"

        # Build full contents list: history + current turn
        contents = self.history + [
            genai_types.Content(role="user", parts=[genai_types.Part(text=user_turn)])
        ]

        response = _client.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=1024,
            ),
        )
        answer = response.text

        # Store clean version in history (without context block)
        self.history.append(genai_types.Content(role="user",  parts=[genai_types.Part(text=user_message)]))
        self.history.append(genai_types.Content(role="model", parts=[genai_types.Part(text=answer)]))

        return answer

    def reset(self) -> None:
        self.history = []
        print("Conversation reset.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_index() -> None:
    docs = load_documents()
    print(f"Using local sentence-transformers model '{EMBED_MODEL_NAME}' for embeddings.")
    embedder = LocalEmbedder()
    index = VectorIndex(embedder)
    index.build(docs)
    index.save(INDEX_FILE)
    print(f"\nDone. Run `python ncu_rag_agent.py` to start chatting.")


def run_agent() -> None:
    if not Path(INDEX_FILE).exists():
        print(f"[ERROR] Index not found. Run first: python ncu_rag_agent.py --build")
        sys.exit(1)

    print("Loading index...")
    index = VectorIndex.load(INDEX_FILE)
    timeslot, building = load_lookups()

    print(f"Index loaded. {len(index.documents)} courses.")
    print(f"Embedder: local sentence-transformers ({EMBED_MODEL_NAME})")
    print(f"Chat model: Gemini {GEMINI_CHAT_MODEL}")
    print("-" * 55)
    print("NCU Course Assistant — ask anything about NCU courses.")
    print("Commands: 'reset' = clear history | 'quit' = exit")
    print("-" * 55)

    agent = NCUAgent(index, timeslot, building)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Goodbye.")
            break
        if user_input.lower() == "reset":
            agent.reset()
            continue

        print("\nAssistant: ", end="", flush=True)
        answer = agent.chat(user_input)
        print(answer)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCU Course RAG Agent")
    parser.add_argument(
        "--build", action="store_true",
        help="Build the vector index from ncu_courses_en.jsonl (run once)"
    )
    args = parser.parse_args()

    if args.build:
        build_index()
    else:
        run_agent()