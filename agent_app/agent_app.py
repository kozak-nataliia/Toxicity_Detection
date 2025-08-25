# agent_app.py
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from rich import print
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv() # This loads variables from .env into os.environ
# Optional LLM agent (uses OPENAI_API_KEY if present)
USE_AGENT = bool(os.getenv("OPENAI_API_KEY", "").strip())
if USE_AGENT:
    from langchain_openai import ChatOpenAI
    from langchain.tools import Tool
    from langchain.agents import initialize_agent, AgentType

# ---- Paths ----
MODEL_DIR = "toxic_roberta_model"  # change if yours is elsewhere
DB_DIR = "rag_store"



# ---------------------------
# 1) Toxicity classifier tool
# ---------------------------
@dataclass
class ToxicClassifier:
    model_dir: str
    threshold: float = 0.5

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        cfg = AutoConfig.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, config=cfg)
        self.model.eval()
        # Device choice
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.inference_mode()
    def classify(self, text: str) -> Dict[str, Any]:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        logits = self.model(**enc).logits
        # binary single-label model -> sigmoid
        prob = torch.sigmoid(logits).squeeze().item()
        label = "toxic" if prob >= self.threshold else "non-toxic"
        return {"label": label, "probability": round(prob, 4)}


# ---------------------------
# 2) RAG tool
# ---------------------------
import re, math
from pathlib import Path

class StudentRAG:
    def __init__(self, db_dir: str, profile_path: str = "student_profile.md"):
        # Imports from the split packages
        from langchain_huggingface import HuggingFaceEmbeddings  # or: langchain_community
        from langchain_chroma import Chroma

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vs = Chroma(persist_directory=db_dir, embedding_function=self.embeddings)
        self.retriever = self.vs.as_retriever(search_kwargs={"k": 3})
        self.profile_path = Path(profile_path)

    # ---------- helpers ----------
    @staticmethod
    def _normalize_lines(text: str):
        raw = [l for l in text.splitlines()]
        # strip markdown bullets/headers and whitespace
        return [re.sub(r'^[#*\-\s]+', '', l).strip() for l in raw if l.strip()]

    @staticmethod
    def _extract_kv(lines):
        """Extract ('key', 'value') from any 'key: value' line."""
        kv = []
        for l in lines:
            if ":" in l:
                k, v = l.split(":", 1)
                k = k.strip()
                v = v.strip().strip(" -–—.:;!?")
                if k and v:
                    kv.append((k, v))
        return kv

    @staticmethod
    def _cosine(u, v):
        dot = sum(a*b for a, b in zip(u, v))
        nu = math.sqrt(sum(a*a for a in u))
        nv = math.sqrt(sum(b*b for b in v))
        return 0.0 if (nu == 0 or nv == 0) else dot / (nu * nv)

    def _best_match_value(self, question: str, kv_pairs):
        """Pick the value whose key is semantically closest to the question."""
        if not kv_pairs:
            return None
        q_emb = self.embeddings.embed_query(question)
        best = None
        best_score = -1.0
        for key, value in kv_pairs:
            k_emb = self.embeddings.embed_query(key)
            s = self._cosine(q_emb, k_emb)
            if s > best_score:
                best_score, best = s, value
        return best

    # ---------- main ----------
    def ask(self, question: str):
        # Step 1: retrieve relevant chunks (new API, no deprecation)
        docs = self.retriever.invoke(question)
        retrieved_text = "\n".join(d.page_content for d in docs)
        lines = self._normalize_lines(retrieved_text)
        kv = self._extract_kv(lines)

        # Step 2: if retrieval had no KV, fall back to parsing the whole profile file
        if not kv and self.profile_path.exists():
            all_text = self.profile_path.read_text(encoding="utf-8")
            kv = self._extract_kv(self._normalize_lines(all_text))

        # Step 3: semantic match question -> key, return value
        value = self._best_match_value(question, kv)
        if value:
            return {
                "answer": value,
                "context_snippets": [d.page_content[:200] for d in docs] if docs else []
            }

        # Step 4: graceful fallback
        if lines:
            # show the most informative line (not a header)
            first_info = next((l for l in lines if ":" in l), lines[0])
            return {"answer": first_info, "context_snippets": [d.page_content[:200] for d in docs] if docs else []}

        return {"answer": "I couldn't find that in the profile.", "context_snippets": []}

# ---------------------------
# 3) Wire up as an agent (if OPENAI_API_KEY set)
# ---------------------------
def build_agent(classifier: ToxicClassifier, rag: StudentRAG):
    """LangChain ReAct agent that can choose tools."""
    def classify_tool_fn(q: str) -> str:
        res = classifier.classify(q)
        return f"Label: {res['label']}, probability={res['probability']}"

    def rag_tool_fn(q: str) -> str:
        res = rag.ask(q)
        return f"Answer: {res['answer']}"

    tools = [
        Tool(
            name="toxic_classifier",
            description="Classify a message as toxic or non-toxic. Input is the raw text.",
            func=classify_tool_fn
        ),
        Tool(
            name="student_rag",
            description="Answer questions about the student using the knowledge base (e.g., 'When was the student born?'). Input is the question.",
            func=rag_tool_fn
        ),
    ]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent


# ---------------------------
# 4) CLI
# ---------------------------
def main():
    print("[bold]Loading tools...[/bold]")
    classifier = ToxicClassifier(MODEL_DIR)
    rag = StudentRAG(DB_DIR)

    if USE_AGENT:
        print("[green]Agent mode (LangChain) — using OPENAI_API_KEY[/green]")
        agent = build_agent(classifier, rag)
        print("Type your request (e.g., 'Is \"go away\" toxic?' or 'When was the student born?'). Type 'exit' to quit.")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"exit", "quit"}:
                break
            try:
                result = agent.invoke({"input": q})   # new API
                # AgentExecutor returns a dict like {"input": "...", "output": "...", ...}
                out = result.get("output", result)
                print(f"[cyan]{out}[/cyan]")

            except Exception as e:
                print(f"Error: {e}")
    else:
        print("[yellow]Fallback mode (no API key): use explicit commands[/yellow]")
        print("Commands:\n  classify: <text>\n  ask: <question>\n  exit")
        while True:
            try:
                line = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line.lower() in {"exit", "quit"}:
                break
            if line.startswith("classify:"):
                text = line[len("classify:"):].strip()
                res = classifier.classify(text)
                print(f"Label: {res['label']}  prob: {res['probability']}")
            elif line.startswith("ask:"):
                q = line[len("ask:"):].strip()
                res = rag.ask(q)
                print(f"Answer: {res['answer']}")
            else:
                print("Unknown command. Use 'classify:' or 'ask:' or 'exit'.")

if __name__ == "__main__":
    main()
