import streamlit as st

from rag_engine import (
    extract_text_from_pdf,
    split_text_into_chunks,
    create_vector_store,
    get_answer
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Document Q&A (Groq RAG)",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Q&A System")
st.markdown("Upload a PDF and ask questions using Groq-powered RAG")

# -----------------------------
# Session state init (IMPORTANT FIX)
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])


# -----------------------------
# Process PDF
# -----------------------------
if uploaded_file is not None:

    with st.spinner("Reading and processing your PDF..."):

        # Extract text
        raw_text = extract_text_from_pdf(uploaded_file)

        if not raw_text.strip():
            st.error("❌ Could not extract text from PDF")
        else:
            # Split into chunks
            chunks = split_text_into_chunks(raw_text)

            # Create vector store
            st.session_state.vector_store = create_vector_store(chunks)

    st.success("✅ PDF processed successfully!")


# -----------------------------
# Ask question
# -----------------------------
question = st.text_input("Ask a question about your document:")

if question:

    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload a PDF first")
    else:
        with st.spinner("Finding the answer..."):
            answer = get_answer(
                st.session_state.vector_store,
                question
            )

        st.markdown("### Answer")
        st.write(answer)

else:
    st.info("Upload a PDF to start asking questions")