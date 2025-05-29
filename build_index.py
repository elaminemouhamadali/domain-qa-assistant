# build_index.py

# handle paths
from pathlib import Path

# split long text into chunks (important for token limits)
from langchain.text_splitter import RecursiveCharacterTextSplitter
# local vector store (storing document embedding here)
from langchain.vectorstores import FAISS
# convert text into dense vectors
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load .env for configs or keys
load_dotenv()

# Paths for raw data txt files and output directory to save FAISS index
DATA_DIR = Path("data/raw")
INDEX_DIR = Path("index")

# Load files
def load_documents():
    docs = []
    # Iterate over files in raw dir
    for file_path in DATA_DIR.glob("*.txt"):
        # Read file as plain text
        with file_path.open('r', encoding='utf-8') as f:
            # Collect them in docs list
            docs.append(f.read())
    return docs

# Split docs into chunks
'''
LLMs have context length limits (usually 4000-8000 tokens)
We break large docs into 1000-token chunks with 200-token overlap
Improves retrieval accuracy and avoids cutting off important context
Recursive splitter tries to split along logical boundaries (paragraphs, sentences)
befre falling back to raw characters
'''
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents(docs)
    return chunks

# Build FAISS index
def build_faiss_index(chunks):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = FAISS.from_documents(chunks, embedder)
    index.save_local(str(INDEX_DIR))
    print(f"Saved FAISS index to {INDEX_DIR.resolve()}")

if __name__ == "__main__":
    if not DATA_DIR.exists():
        print(f"Data directory {DATA_DIR} does not exist!")
    else:
        docs = load_documents()
        if not docs:
            print(f"No .txt files found in {DATA_DIR}")
        else:
            chunks = split_documents(docs)
            build_faiss_index(chunks)
