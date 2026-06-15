import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# ---------------------------
# 1. PDF TEXT EXTRACTION
# ---------------------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# ---------------------------
# 2. TEXT CHUNKING
# ---------------------------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# ---------------------------
# 3. VECTOR DB CREATION
# ---------------------------
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    return vectorstore


# ---------------------------
# 4. GROQ LLM SETUP
# ---------------------------
def get_llm():
    return ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="llama3-8b-8192"
    )


# ---------------------------
# 5. QA CHAIN (RAG PIPELINE)
# ---------------------------
def get_qa_chain(vectorstore):
    llm = get_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    return qa_chain