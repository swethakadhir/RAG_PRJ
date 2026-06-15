import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text


def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vector_store

import os
from google import genai

import os
from openai import OpenAI

def get_answer(vector_store, question):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return "ERROR: OPENAI_API_KEY not found in Streamlit secrets."

    client = OpenAI(api_key=api_key)

    # Retrieve relevant documents from vector DB
    docs = vector_store.similarity_search(question, k=4)

    context = "\n\n".join(
        doc.page_content for doc in docs if doc.page_content
    )

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}

If the answer is not in the context, say:
"I could not find this in the document."
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content