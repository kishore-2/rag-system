import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import gradio as gr
from groq import Groq

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError("GROQ_API_KEY is missing! Please add it as a secret in the Hugging Face Space settings.")

client = Groq(api_key=api_key)

# Global variable to store the processed PDF vector store
global_vector_store = None

def process_pdf(pdf_file):
    """
    Process the uploaded PDF:
      - Load the PDF using PyPDFLoader.
      - Split the text into chunks.
      - Generate embeddings and build a vector store.
    Returns the vector store (or an error message as a string).
    """
    try:
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    except Exception as e:
        return f"Error: {str(e)}"

def process_query(vector_store, query):
    """
    Use the stored vector store to retrieve relevant document chunks and generate an answer.
    """
    try:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        retrieved_docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"You are a helpful assistant. Based on the following document:\n{context}\nAnswer the question: {query}"
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192"
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def rag_interface(pdf, query):
    """
    If a PDF is uploaded, process it and update the global vector store.
    If no PDF is uploaded and a global vector store exists, use it.
    Then answer the query using the stored vector store.
    """
    global global_vector_store
    if pdf:
        result = process_pdf(pdf)
        if isinstance(result, str):
            # An error occurred during PDF processing
            return result
        global_vector_store = result
    else:
        if global_vector_store is None:
            return "Please upload a PDF file."
    if not query:
        return "Please enter a query."
    return process_query(global_vector_store, query)

interface = gr.Interface(
    fn=rag_interface,
    inputs=["file", "text"],
    outputs="text",
    title="RAG System with Gradio",
    description="Upload a PDF and ask a query to retrieve answers.",
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", share=True)
