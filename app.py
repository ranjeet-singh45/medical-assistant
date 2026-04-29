import streamlit as st
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------
# Fix import paths
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "src"))

from src.config import DATA_PATH, VECTOR_STORE_PATH, OPENAI_API_KEY
from src.data_loader import DataLoader
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.rag import RAGPipeline

# ---------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🧬",
    layout="wide"
)

# ---------------------------------------------------------------
# 🎨 UI Styling (Main + Sidebar Matching)
# ---------------------------------------------------------------
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

/* Sidebar matches main */
section[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Sidebar inner */
section[data-testid="stSidebar"] > div {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    border: none;
    background: rgba(255,255,255,0.15);
    color: white;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.08);
    padding: 12px;
    border-radius: 12px;
}

/* Input box */
textarea {
    background: rgba(255,255,255,0.1) !important;
    color: white !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Initialize RAG
# ---------------------------------------------------------------
@st.cache_resource
def initialize_rag_system():
    vector_store = VectorStore()

    if not vector_store.load_vector_store():
        with st.spinner("🔄 Building vector database (first run)..."):
            data_loader = DataLoader(DATA_PATH)
            documents = data_loader.get_documents()
            vector_store.create_embeddings(documents)
            vector_store.save_vector_store()

    retriever = Retriever(vector_store)
    rag_pipeline = RAGPipeline(retriever)
    return rag_pipeline

# ---------------------------------------------------------------
# Main App
# ---------------------------------------------------------------
def main():

    st.title("🏥 Medical AI Assistant")
    st.caption("AI-powered medical Q&A using RAG + OpenAI")

    # ⚠️ Disclaimer
    st.warning("⚠️ This is NOT medical advice. Always consult a doctor.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")

        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")

        st.subheader("💡 Try these questions")

        example_questions = [
            "What are dengue symptoms?",
            "How to prevent diabetes?",
            "What causes high blood pressure?",
            "Symptoms of COVID-19?"
        ]

        for q in example_questions:
            if st.button(q):
                st.session_state.messages.append({"role": "user", "content": q})

    # API key check
    if not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY missing in .env")
        st.stop()

    # Initialize system
    try:
        rag_pipeline = initialize_rag_system()
    except Exception as e:
        st.error(f"🚨 Initialization Failed: {e}")
        st.stop()

    # Chat memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask your medical question..."):

        # Store user
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                try:
                    result = rag_pipeline.generate_answer(prompt)
                    answer = result["answer"]

                    st.markdown(answer)

                    # Sources
                    with st.expander("📚 Sources"):
                        for i, src in enumerate(result["sources"], 1):
                            st.markdown(f"**Source {i}:** {src.get('qtype','General')}")
                            st.write(src.get("question", ""))
                            st.write(src.get("answer", "")[:200] + "...")

                except Exception as e:
                    answer = f"❌ Error: {str(e)}"
                    st.error(answer)

        # Save response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


if __name__ == "__main__":
    main()