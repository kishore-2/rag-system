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
from verifytoken import verify_groq_key
from groq import Groq

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise EnvironmentError("GROQ_API_KEY is missing! Please add it as a secret or in a .env file.")

verification_message = verify_groq_key()
print(verification_message)

# Initialize Groq client
client = Groq(api_key=api_key)

def process_pdf_and_query(pdf_file, query):
    try:
        start_time = time.time()

        # Step 1: Load and process the PDF
        print("Loading PDF...")
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()
        if not documents:
            return "No text was extracted from the PDF. Ensure the file is not empty or corrupted."
        print(f"PDF loaded in {time.time() - start_time:.2f} seconds.")

        # Step 2: Split text into manageable chunks
        start_split_time = time.time()
        print("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            return "Failed to split the document into smaller chunks."
        print(f"Text split into chunks in {time.time() - start_split_time:.2f} seconds.")

        # Step 3: Embed the text
        start_embed_time = time.time()
        print("Embedding text...")
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        vector_store = FAISS.from_documents(chunks, embeddings)
        print(f"Text embedded in {time.time() - start_embed_time:.2f} seconds.")

        # Step 4: Retrieve relevant documents
        start_retrieve_time = time.time()
        print("Retrieving relevant documents...")
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        retrieved_docs = retriever.get_relevant_documents(query)
        if not retrieved_docs:
            return "No relevant documents found for the query."
        print(f"Documents retrieved in {time.time() - start_retrieve_time:.2f} seconds.")

        # Step 5: Prompt engineering and query
        start_prompt_time = time.time()
        print("Generating response from Groq API...")
        prompt_template = """You are a helpful assistant. Based on the following document: {context}
        Answer the question: {question}"""
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = prompt_template.format(context=context, question=query)

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            timeout=120  # Timeout for Groq API call
        )
        print(f"Groq API response generated in {time.time() - start_prompt_time:.2f} seconds.")

        return response.choices[0].message.content

    except Exception as e:
        return f"An error occurred: {str(e)}"

def rag_interface(pdf, query):
    if not pdf:
        return "Please upload a PDF file."
    if not query:
        return "Please enter a query."
    return process_pdf_and_query(pdf, query)

# Set up Gradio UI
interface = gr.Interface(
    fn=rag_interface,
    inputs=["file", "text"],
    outputs="text",
    title="RAG System with Gradio",
    description="Upload a PDF and ask a query to retrieve answers.",
)

if __name__ == "__main__":
    interface.launch(share=True)
