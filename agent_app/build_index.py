# build_index.py
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

DOC_PATH = Path("student_profile.md")
DB_DIR = "rag_store"

def main():
    if not DOC_PATH.exists():
        raise FileNotFoundError("student_profile.md not found. Please create it first.")

    loader = TextLoader(str(DOC_PATH), encoding="utf-8")
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=DB_DIR)
    print(f"Built vector store at ./{DB_DIR}")

if __name__ == "__main__":
    main()
