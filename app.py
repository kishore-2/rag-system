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

# Load API key for Groq
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError("GROQ_API_KEY is missing! Please add it as a secret in the Hugging Face Space settings.")

client = Groq(api_key=api_key)

EXAMPLE_QUERIES = [
    "Provide a brief outline of this document.",
    "What is the summary points of this document?",
    "List the main topics discussed in this document.",
    "What are the key takeaways from this document?",
    "Explain the purpose of this document.",
    "What are the key findings of this document?",
    "What are the main conclusions of this document?",
    "What are the limitations of this document?",
    "Identify the challenges or problems discussed in this document.",
    "What methodology or approach is discussed in this document?",
]

# Persistent storage for PDF vector store
vector_store = None

def process_pdf(pdf_file):
    """Loads and processes the uploaded PDF, storing embeddings globally."""
    global vector_store  # Keep the vector store persistent across function calls
    
    try:
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()
        
        # Split text into chunks for processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Create embeddings and store them in FAISS
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        vector_store = FAISS.from_documents(chunks, embeddings)

        return "PDF uploaded and processed successfully. You can now ask questions."
    
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def query_pdf(query):
    """Handles user queries using the stored PDF vector store."""
    global vector_store  # Retrieve the stored embeddings

    if vector_store is None:
        return "Please upload a PDF first before asking questions."
    
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
        return f"Error answering question: {str(e)}"

# Gradio Interface with Stateful Storage
with gr.Blocks() as interface:
    gr.Markdown("## RAG System with Gradio - Chat with Your PDF 📄🤖")

    pdf_input = gr.File(label="Upload a PDF")
    query_input = gr.Textbox(label="Ask a question about the PDF", placeholder="Type your query here...")
    submit_button = gr.Button("Submit")

    output_text = gr.Textbox(label="Answer")

    # Upload Button & Store PDF Vector Store
    pdf_upload_button = gr.Button("Process PDF")
    pdf_upload_status = gr.Textbox(label="Status", interactive=False)

    # Maintain state across interactions
    pdf_memory = gr.State(None)

    pdf_upload_button.click(process_pdf, inputs=[pdf_input], outputs=[pdf_upload_status], state=pdf_memory)
    submit_button.click(query_pdf, inputs=[query_input], outputs=[output_text], state=pdf_memory)

interface.launch()
