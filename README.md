# RAG System with Gradio

### Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system using Python, LangChain, and Gradio. It enables users to upload a PDF or use an example PDF and interact with its content by asking queries. The system uses advanced AI to retrieve relevant content and provide meaningful answers.

---

### Features
- Upload your own PDF or use the included example PDF (you can find in this repository).
- Automatically split documents into manageable chunks for efficient processing.
- Retrieve relevant content using a FAISS vector store.
- Generate context-aware answers with the Groq API.

---

### Live Demo
🎉 Try the live application on Hugging Face Spaces:  
[**RAG System with Gradio**](https://huggingface.co/spaces/huggyy/rag-project)

---

### Example Queries
You can use the example PDF provided in the application, **"Using AI to Address Medical Needs"**, and try these sample queries:
1. **What are the benefits of AI in healthcare?**
2. **How does AI assist in patient monitoring?**
3. **What are the ethical considerations of using AI in healthcare?**

---

### Setup Instructions

#### 1. Clone the Repository
```bash
git clone https://github.com/kishore-2/rag-system.git
cd rag-system
```
#### 2. Create a Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```
#### 3. Install Dependencies
Install the required Python packages using requirements.txt:

```bash
pip install -r requirements.txt
```
#### 4. Add Your Groq API Key
Create a .env file in the root directory and add your API key:

```plaintext
GROQ_API_KEY=your_api_key_here
```
#### 5. Run the Application Locally
Start the Gradio application:

```bash
python app.py
```
Access the application in your browser via the local link (e.g., http://127.0.0.1:7860).

### Using the Application:
#### 1. Upload a PDF
#### 2. Drag and drop your PDF or click "Upload Your PDF" to choose a file.
#### 3. Enter a query in the text box, such as:
#### 4. "What is the summary of this document?"
#### 5. "Explain the key points in section 3."
#### 6. Use the Example PDF
#### 7. Check the "Use Example PDF" checkbox.
#### 8. Click any of the predefined queries or type your own.
#### 9. Get Answers
#### 10. The system retrieves relevant content from the PDF and generates an answer using the Groq API.

### Project Structure
```bash
rag-system/
├── app.py                 # Main Gradio application
├── verifytoken.py         # Token validation logic
├── requirements.txt       # Python dependencies
├── Using AI to Address Medical Needs.pdf  # Example PDF
├── .env                   # Environment variables (ignored in Git)
├── .gitignore             # Ignored files configuration
└── README.md              # Project documentation
```
Deploying to Hugging Face Spaces

### Clone your Hugging Face Space repository:
```bash
git clone https://huggingface.co/spaces/huggyy/rag-project
cd rag-project
```
### Copy the required files:
app.py
requirements.txt
Any additional files like example PDFs.

### Add and push the files:
```bash
git add .
git commit -m "Deploy RAG system"
git push
```
Add your API key as a secret in the Hugging Face Space settings.

### Dependencies
```bash
gradio
langchain
pypdf
faiss-cpu
groq
python-dotenv
langchain-community
sentence-transformers
pymupdf
huggingface-h
```

### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request. For major changes, open an issue to discuss what you would like to contribute.
