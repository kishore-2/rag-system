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
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError("GROQ_API_KEY is missing! Please add it as a secret in the Hugging Face Space settings.")

client = Groq(api_key=api_key)

def process_pdf_and_query(pdf_file, query):
    try:
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        vector_store = FAISS.from_documents(chunks, embeddings)
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
    if not pdf:
        return "Please upload a PDF file."
    if not query:
        return "Please enter a query."
    return process_pdf_and_query(pdf, query)

interface = gr.Interface(
    fn=rag_interface,
    inputs=["file", "text"],
    outputs="text",
    title="RAG System with Gradio",
    description="Upload a PDF and ask a query to retrieve answers.",
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", share=True)
