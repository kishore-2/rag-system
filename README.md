# RAG System with Gradio

## Overview
This project, **"Chat with Your Documents"**, was developed as part of the **Infosys Springboard Internship 5.0** program. The application implements a **Retrieval-Augmented Generation (RAG)**  system using Python, LangChain, and Gradio. It enables users to upload a PDF and interact with its content by asking queries. The system uses advanced AI to retrieve relevant content and provide meaningful answers. 

---

## Features
- Upload your own PDF or use the example PDF here --> **"[AI in Health Care  REPORT.pdf](https://github.com/user-attachments/files/18326815/AI.in.Health.Care.REPORT.pdf)"**
).
- Automatically split documents into manageable chunks for efficient processing.
- Retrieve relevant content using a FAISS vector store.
- Generate context-aware answers with the Groq API.

---

## Live Demo
🎉 Try the live application on Hugging Face Spaces:  [**RAG System with Gradio**](https://huggingface.co/spaces/huggyy/rag-project)

---

## Output

<img src="https://github.com/user-attachments/assets/c9ddf363-b1ac-4a7e-acd2-eabc3ff48f56" alt="Output pic" width="700" />

---

## Usefulness

#### This project helps researchers, professionals, and students find answers quickly and accurately from specific documents. Here's why it’s valuable:

- **Focused Search**: Unlike regular search engines, this tool looks only within a single document, giving precise and relevant answers.
- **Time-Saving**: Users can avoid unnecessary information and directly get what they need.
- **Easy to Use**: The interface is simple, allowing anyone to interact with their documents using natural language.
- **Real-Life Applications**:
  - Researchers analyzing specific studies or papers.
  - Lawyers reviewing contracts or case studies.
  - Business professionals exploring reports or financial data.

With advanced AI tools and an intuitive design, this project makes document analysis fast, easy, and efficient.

---

## Example Queries

You can download and use the example PDF provided here -->  **"[AI in Health Care  REPORT.pdf](https://github.com/user-attachments/files/18326815/AI.in.Health.Care.REPORT.pdf)"**, and try these sample queries:
1. **List two challenges of AI in healthcare.**
2. **What is discussed in the "Future Directions" section?**
3. **How can AI improve healthcare access in low-resource settings?**
4. **What are the limitations of AI in critical medical decision-making?**

---

## Using the Application

#### 1. Upload a PDF
- Drag and drop your PDF or click **"Upload Your PDF"** to choose a file.

#### 2. Ask Your Questions
- Enter your custom query in the text box, such as:
  - "What is the summary of this document?"
  - "Explain the key points in section 3."

#### 3. Use the Example PDF
- You can also use the example PDF and queries **"mentioned above ↑"**.
  
#### 4. Get Answers
- Click **"Ask the AI"** to retrieve context-aware answers generated using the Groq API.

---

## Architecture Diagram

<img src="https://github.com/user-attachments/assets/cd2fcaaf-19ce-4037-ad42-5fe5c5718853" alt="RAG System Diagram" width="700" />

---

## Setup Instructions

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
#### 5. Verify the Token
Run the verifytoken.py script to ensure your API key is valid:
```bash
python verifytoken.py
```
If the token is valid, you should see:
```csharp
The token is validated and working good!
```
#### 6. Run the Application Locally
Start the Gradio application:
```bash
python app.py
```
Access the application in your browser via the local link (e.g., http://127.0.0.1:7860).

---

### License
This project is licensed under the MIT License. See the LICENSE file for details.

---

### Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request. For major changes, open an issue to discuss what you would like to contribute.
