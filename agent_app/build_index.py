# build_index.py
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader  # OK in 0.3.x


BASE = Path(__file__).resolve().parent
STORE_DIR = BASE / "rag_store"
COLLECTION = "student_profile"
DOC_PATH = BASE / "student_profile.md"

def main():
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    loader = TextLoader(str(DOC_PATH), encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma(
        persist_directory=str(STORE_DIR),
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )
    # Upsert chunks (ids optional)
    vectordb.add_documents(splits)
    print(f"Built Chroma store at {STORE_DIR} (collection='{COLLECTION}', chunks={len(splits)})")


if __name__ == "__main__":
    main()
