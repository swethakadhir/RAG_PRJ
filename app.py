import streamlit as st

from rag_engine import extract_text_from_pdf
from rag_engine import split_text_into_chunks
from rag_engine import create_vector_store
from rag_engine import get_answer

# Configure the page
st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="centered"
)

# Title and description
st.title("📄 Document Q&A System")
st.markdown("Upload any PDF and ask questions about it.")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    with st.spinner("Reading and processing your PDF..."):

        # Step 1: Extract text
        raw_text = extract_text_from_pdf(uploaded_file)

        # Step 2: Split into chunks
        chunks = split_text_into_chunks(raw_text)

        # Step 3: Create vector database
        st.session_state.vector_store = create_vector_store(chunks)

    st.success("PDF processed successfully!")

    # Ask a question
    question = st.text_input(
        "Ask a question about your document:"
    )

    if question:

        with st.spinner("Finding the answer..."):

           answer = get_answer(
             st.session_state.vector_store,
             question
           )
        st.markdown("### Answer")
        st.write(answer)

else:
    st.info("Please upload a PDF file to get started.")