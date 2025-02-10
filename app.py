import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
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

# Global cache for example PDFs: key is a name, value is the vector store
example_pdf_cache = {}

# Mapping of example PDF names to their file paths (use raw strings for Windows paths)
EXAMPLE_PDF_FILES = {
    "AI in Health Care REPORT": r"example_pdfs\AI in Health Care  REPORT.pdf",
    "Using ai to address medical needs": r"example_pdfs\Using ai to address medical needs.pdf",
    "The DevOps Handbook": r"example_pdfs\The DevOps Handbook.pdf",
}

def process_pdf_from_file(file_path):
    """
    Processes a PDF from a file path:
      - Loads and splits text.
      - Generates embeddings and builds a vector store.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        vector_store = FAISS.from_documents(chunks, embedder)
        return vector_store
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_example_pdf(example_name):
    """
    Loads an example PDF by its name, using a cache to avoid reprocessing.
    """
    global example_pdf_cache
    if example_name in example_pdf_cache:
        return example_pdf_cache[example_name]
    file_path = EXAMPLE_PDF_FILES.get(example_name)
    if file_path is None:
        return None
    vector_store = process_pdf_from_file(file_path)
    if vector_store:
        example_pdf_cache[example_name] = vector_store
    return vector_store

def query_pdf(query, vector_store):
    """
    Uses the provided vector store to retrieve a response for the query.
    """
    if vector_store is None:
        return "No processed PDF available."
    try:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"You are a helpful assistant. Based on the following document:\n{context}\nAnswer the question: {query}"
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192"
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error answering query: {str(e)}"

def process_pdf_interface(pdf_file, example_pdf):
    """
    Processes the PDF either from an uploaded file or from a selected example.
    Stores the resulting vector store in the global variable.
    """
    global global_vector_store
    if pdf_file:
        try:
            pdf_bytes = pdf_file.read()
            temp_path = "uploaded_temp.pdf"
            with open(temp_path, "wb") as f:
                f.write(pdf_bytes)
            vector_store = process_pdf_from_file(temp_path)
            global_vector_store = vector_store
            return "✅ PDF processed from upload."
        except Exception as e:
            return f"❌ Error processing uploaded PDF: {str(e)}"
    elif example_pdf:
        vector_store = load_example_pdf(example_pdf)
        if vector_store is None:
            return "❌ Error processing example PDF."
        global_vector_store = vector_store
        return "✅ PDF processed from example selection."
    else:
        return "Please upload a PDF or select an example PDF."

def ask_query_interface(query):
    """
    Answers the query using the globally stored vector store.
    """
    if global_vector_store is None:
        return "❌ Please process a PDF first."
    if not query:
        return "Please enter a query."
    return query_pdf(query, global_vector_store)

# Gradio Interface
with gr.Blocks() as interface:
    gr.Markdown("## RAG System with Gradio - Chat with Your PDF")
    
    with gr.Tab("Upload/Select PDF"):
        with gr.Column():
            pdf_input = gr.File(label="Upload a PDF", file_types=[".pdf"])
            example_pdf_dropdown = gr.Dropdown(
                label="Or select an Example PDF",
                choices=list(EXAMPLE_PDF_FILES.keys())
            )
            process_button = gr.Button("Process PDF")
            process_status = gr.Textbox(label="Status", interactive=False)
    
    with gr.Tab("Ask Query"):
        with gr.Column():
            query_input = gr.Textbox(label="Ask a question", placeholder="Type your query here...")
            submit_button = gr.Button("Submit")
            answer_output = gr.Textbox(label="Answer")
            gr.Examples(
                examples=[[q] for q in [
                    "Provide a brief outline of this document.",
                    "What is the summary points of this document?",
                    "List the main topics discussed in this document.",
                    "What are the key takeaways from this document?",
                    "Explain the purpose of this document?",
                ]],
                inputs=query_input,
                label="Example Queries"
            )
    
    # Wire up the buttons:
    process_button.click(process_pdf_interface, inputs=[pdf_input, example_pdf_dropdown], outputs=[process_status])
    submit_button.click(ask_query_interface, inputs=[query_input], outputs=[answer_output])

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", share=True)
