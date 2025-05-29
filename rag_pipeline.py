from pathlib import Path
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import pipeline

# Paths
INDEX_DIR = Path("index")

# Step 1: Load FAISS index
def load_faiss_index():
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(str(INDEX_DIR), embedder, allow_dangerous_deserialization=True)
    print(f"✅ Loaded FAISS index from {INDEX_DIR.resolve()}")
    return db

# Step 2: Set up local Hugging Face LLM
def load_local_llm():
    local_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=0  # Use GPU 0; set -1 for CPU
    )
    llm = HuggingFacePipeline(pipeline=local_pipeline)
    print("✅ Loaded local Hugging Face model (flan-t5-base)")
    return llm

# Step 3: Create RAG pipeline
def create_qa_pipeline():
    db = load_faiss_index()
    llm = load_local_llm()
    qa_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        chain_type="stuff"
    )
    print("✅ RAG pipeline is ready")
    return qa_pipeline

# Step 4: Run queries
def run_query(question):
    qa_pipeline = create_qa_pipeline()
    result = qa_pipeline.invoke({"query": question})
    return result

if __name__ == "__main__":
    print("🔹 Welcome to the Local RAG Assistant! Type 'exit' to quit.")
    while True:
        question = input("\n❓ Ask a question: ")
        if question.lower() == 'exit':
            print("👋 Exiting. Goodbye!")
            break
        try:
            answer = run_query(question)
            print(f"\n💬 Answer:\n{answer}")
        except Exception as e:
            import traceback
            print(f"❌ Error: {e}")
            traceback.print_exc()
