# RAG System with Gradio

### Overview
This project, **"Chat with Your Documents"**, was developed as part of the **Infosys Springboard Internship 5.0** program. The application implements a **Retrieval-Augmented Generation (RAG)**  system using Python, LangChain, and Gradio. It enables users to upload a PDF and interact with its content by asking queries. The system uses advanced AI to retrieve relevant content and provide meaningful answers. In addition to the RAG system, this project includes a **Pandas AI Agent** which leverages advanced AI to analyze tabular data in a Pandas DataFrame. It enables you to query datasets in natural language, extract insights, and draw conclusions effortlessly. This implementation was built using **LangChain**, **Groq**, and **Pandas** to process data and respond to queries.

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

## Example Queries


You can download and use the example PDF provided here -->  **"[AI in Health Care  REPORT.pdf](https://github.com/user-attachments/files/18326815/AI.in.Health.Care.REPORT.pdf)"**, and try these sample queries:
1. **List two challenges of AI in healthcare.**
2. **What is discussed in the "Future Directions" section?**
3. **How can AI improve healthcare access in low-resource settings?**
4. **What are the limitations of AI in critical medical decision-making?**

---

## Using the Application:

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

## Architecture Diagram


<img src="https://github.com/user-attachments/assets/cd2fcaaf-19ce-4037-ad42-5fe5c5718853" alt="RAG System Diagram" width="700" />

---

# Pandas AI Agent

## Features
- Analyze datasets interactively using natural language queries.
- Retrieve statistical summaries and correlations from the data.
- Gain insights and conclusions about data relationships.

---

## Setup Instructions

#### 1. Clone the Repository
Ensure you have the repository set up locally:
```bash
git clone https://github.com/kishore-2/rag-system.git
cd rag-system
```
#### 2. Add the Dataset
Ensure the dataset file is present, or download and use this xlsx --> [Simple_linear_regression_prediction.xlsx](https://github.com/user-attachments/files/18259771/Simple_linear_regression_prediction.xlsx). 
This dataset represents a regression example with the following columns:
```plaintext
X (Feature): Input feature values.
Y (Target): Target values for prediction.
Prediction (Y'): Predicted values for the target.
Residual (Y - Y'): Difference between actual and predicted target values.
```
#### 3. Install Dependencies
Make sure all required Python libraries are installed:
```bash
pip install -r requirements-1.txt
```
#### 4. Add Your Groq API Key
Set up the GROQ_API_KEY in your .env file:
```plaintext
GROQ_API_KEY=your_api_key_here
```
#### # 5. Run the Script
Launch the pandas_ai_agent.py script:
```bash
python pandas_ai_agent.py
```

---

### License
This project is licensed under the MIT License. See the LICENSE file for details.

---

### Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request. For major changes, open an issue to discuss what you would like to contribute.
