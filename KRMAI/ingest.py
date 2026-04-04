import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Configuration ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300


def _collect_supported_files(source_dirs, supported):
    """Recursively collects supported files from a list of source directories."""
    collected = []
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            continue

        for root, _, files in os.walk(source_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in supported:
                    continue
                file_path = os.path.join(root, filename)
                collected.append((source_dir, file_path, ext))

    collected.sort(key=lambda item: item[1])
    return collected


def load_documents(source_dirs):
    """Loads PDFs, DOCX, and TXT files from configured source directories."""
    documents = []

    supported = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader}
    file_items = _collect_supported_files(source_dirs, supported)

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created '{DATA_DIR}' directory.")

    if not file_items:
        print("No supported documents found in configured source directories.")
        return []

    for source_dir, file_path, ext in file_items:
        loader_cls = supported.get(ext)
        rel_source = os.path.relpath(file_path, source_dir)
        if loader_cls:
            try:
                loader = loader_cls(file_path)
                docs = loader.load()
                # Tag every doc with the original filename for citations
                for doc in docs:
                    # Ensure metadata dict exists before assigning
                    if not hasattr(doc, "metadata") or doc.metadata is None:
                        doc.metadata = {"source": rel_source}
                    else:
                        doc.metadata["source"] = rel_source
                documents.extend(docs)
                print(f"  Loaded: {rel_source} ({len(docs)} page(s))")
            except Exception as e:
                print(f"  Error loading {rel_source}: {e}")

    return documents


def chunk_documents(documents):
    """Splits documents into smaller overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n\n", "\n\n", "\n---", "\n===", "\n~~~", "\n", " ", ""],
    )
    return text_splitter.split_documents(documents)


def create_vector_store(chunks):
    """Creates the ChromaDB vector store (wipes old DB first for a clean rebuild)."""
    if not chunks:
        print("No chunks to ingest.")
        return

    # Wipe stale DB so we always have a clean, consistent store
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Cleared old vector store at {CHROMA_PATH}")

    print("Initializing embedding model (first run downloads ~80 MB)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Creating vector store...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    print(f"Successfully ingested {len(chunks)} chunks into ChromaDB at {CHROMA_PATH}")


def main():
    source_dirs = [DATA_DIR]

    print(f"{'=' * 50}")
    print(f"  Document Ingestion Pipeline")
    print(f"{'=' * 50}")
    print("\nLoading documents from sources:")
    for source_dir in source_dirs:
        print(f"  - {source_dir}")

    documents = load_documents(source_dirs)

    if documents:
        print(f"\nTotal: {len(documents)} document pages loaded.")
        chunks = chunk_documents(documents)
        print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).\n")
        create_vector_store(chunks)
    else:
        print("\nNo documents found. Add PDF, DOCX, or TXT files to the 'data/' folder.")

    print(f"\n{'=' * 50}")


if __name__ == "__main__":
    main()
