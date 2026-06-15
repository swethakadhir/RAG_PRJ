import os

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings



def extract_text_from_pdf(pdf_file):
    """Read all text from a PDF file"""

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        text += page.extract_text()

    return text

def split_text_into_chunks(text):
    """Split long text into smaller overlapping chunks"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = splitter.split_text(text)

    return chunks

def create_vector_store(chunks):
    """Convert text chunks into embeddings and store them"""

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings
    )

    return vector_store

def get_answer(vector_store, question):
    """Find relevant chunks and ask Gemini to answer"""

    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash"
    google_api_key=os.getenv("GOOGLE_API_KEY")
    temperature=0.3
    )

    docs = vector_store.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I could not find that information in the document."
"""

    response = llm.invoke(prompt)

    return response.content