import streamlit as st
from rag_pipeline import create_qa_pipeline

st.set_page_config(page_title="Local RAG Assistant", page_icon="🛰️")

st.title("🛰️ Local RAG Assistant")
st.write("Ask questions based on your local document knowledge base.")

# Initialize pipeline (only once, cache to avoid reloading each time)
@st.cache_resource
def load_pipeline():
    return create_qa_pipeline()

qa_pipeline = load_pipeline()

# User input
question = st.text_input("❓ Enter your question:")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                result = qa_pipeline.invoke({"query": question})
                st.success("💬 Answer:")
                st.write(result)
            except Exception as e:
                st.error(f"❌ Error: {e}")
