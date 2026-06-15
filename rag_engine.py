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

def get_answer(vector_store, question):
    import os
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return "ERROR: GOOGLE_API_KEY not found in Streamlit secrets."

    # Stable Gemini model for LangChain
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    # Retrieve relevant documents
    docs = vector_store.similarity_search(question, k=4)

    context = "\n\n".join(
        doc.page_content for doc in docs if doc.page_content
    )

    prompt = f"""
Answer ONLY using the context below.

Context:
{context}

Question:
{question}

If the answer is not in the context, say:
"I could not find that information in the document."
"""

    try:
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"Gemini API Error: {str(e)}"