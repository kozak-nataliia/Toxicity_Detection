# toxicity-agent-gradio/app/rag.py
from pathlib import Path
from typing import Optional

# NEW imports (no deprecation warnings)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DEFAULT_STORE_DIR = Path(__file__).resolve().parents[1] / "rag_store"
DEFAULT_COLLECTION = "student_profile"

def load_retriever(store_dir: Optional[Path] = None, k: int = 1):
    store_path = Path(store_dir or DEFAULT_STORE_DIR)
    if not store_path.exists():
        raise FileNotFoundError(
            f"Chroma store not found at {store_path}. "
            f"Run agent_app/build_index.py first."
        )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
            persist_directory=str(store_path),
            collection_name=DEFAULT_COLLECTION,          # keep consistent in build & read
            embedding_function=embeddings,
        )   
    return vectordb.as_retriever(search_kwargs={"k": k})
