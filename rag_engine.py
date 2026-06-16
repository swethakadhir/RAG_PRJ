import os

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# -----------------------------------
# Extract text from PDF
# -----------------------------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text


# -----------------------------------
# Split text into chunks
# -----------------------------------
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    return chunks


# -----------------------------------
# Create vector database
# -----------------------------------
def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
    texts=chunks,
    embedding=embeddings
    )
    return vector_store


# -----------------------------------
# Answer question
# -----------------------------------
from langchain_groq import ChatGroq

def get_answer(vector_store, question):

    docs = vector_store.similarity_search(question, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    llm = ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="llama3-8b-8192"
    )

    response = llm.invoke(prompt)

    return response.content