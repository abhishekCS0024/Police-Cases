import streamlit as st
from retrieval.retriever import retrieve
from generation.generator import generate_answer

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Indian Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Indian Legal RAG Assistant")
st.caption("Powered by Pinecone + Hugging Face Embeddings + Groq LLM")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Top-K Retrieved Chunks", 1, 10, 3)
show_context = st.sidebar.checkbox("Show retrieved context", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("📚 **Dataset**: IPC, CrPC, Evidence Act, Constitution, Procedures")

# ---------------------------
# User input
# ---------------------------
query = st.text_area(
    "Ask a legal question:",
    placeholder="e.g. Explain IPC Section 34 in simple terms",
    height=100
)

ask_btn = st.button("🔍 Ask")

# ---------------------------
# RAG Execution
# ---------------------------
if ask_btn and query.strip():

    with st.spinner("🔎 Retrieving relevant legal sections..."):
        context_chunks = retrieve(query, top_k=top_k)

    if not context_chunks:
        st.warning("No relevant context found in the database.")
    else:
        with st.spinner("🧠 Generating answer using Groq LLM..."):
            answer = generate_answer(query, context_chunks)

        # ---------------------------
        # Answer
        # ---------------------------
        st.subheader("🧠 Answer")
        st.write(answer)

        # ---------------------------
        # Context (optional)
        # ---------------------------
        if show_context:
            st.subheader("📄 Retrieved Context")
            for i, chunk in enumerate(context_chunks, 1):
                with st.expander(f"Context {i}"):
                    st.write(chunk)

elif ask_btn:
    st.warning("Please enter a question.")
