# RAGENT CHATBOT
## Agentic-RAG-Chatbot

An agent-based Retrieval-Augmented Generation (RAG) chatbot using Streamlit, FAISS, and Google Gemini API for answering questions from uploaded documents.

## 🚀 Features

- **Multi-Document Support:** Upload and process PDF, DOCX, CSV, PPTX, TXT, and Markdown files.
- **Agentic Architecture:** Separates concerns into specialized agents (Ingestion, Retrieval, LLM Response).
- **Retrieval-Augmented Generation (RAG):** Enhances LLM responses with relevant context from your documents.
- **FAISS Vector Store:** Efficiently stores and searches document embeddings.
- **Google Gemini API:** Powers the LLM interactions for generating answers and embeddings.
- **Streamlit UI:** Provides an intuitive web interface for document upload and chatbot interaction.
- **Dark Mode:** Default dark theme for a comfortable user experience.

## 🗂️ Project Structure

```
AgenticRAGChatbot/
├── .streamlit/
│   └── config.toml           # Streamlit configuration (e.g., default theme)
├── agents/
│   ├── __init__.py           # Makes 'agents' a Python package
│   ├── ingestion_agent.py    # Handles document parsing and chunking
│   ├── retrieval_agent.py    # Manages vector store search for context
│   └── llm_response_agent.py # Interfaces with the LLM for answers
├── mcp/
│   ├── __init__.py           # Makes 'mcp' a Python package
│   └── mcp_message_protocol.py # Defines the communication protocol (MCPMessage, MCPMessageType)
├── utils/
│   ├── __init__.py           # Makes 'utils' a Python package
│   ├── utils_document_processor.py # Utilities for parsing and chunking documents
│   └── utils_vector_store.py # Utilities for FAISS vector store and embeddings
├── .gitignore                # Specifies intentionally untracked files to ignore
├── main.py                   # Main Streamlit application entry point
└── requirements.txt          # List of Python dependencies
```

## 📆 Setup and Running Instructions

### Prerequisites

- Git
- Python 3.9+
- pip
- Google Gemini API Key

### 1. Clone the Repository

```bash
cd C:/Users/SHEIK\ MOHAMMED\ SUHEL/Documents/
git clone https://github.com/suhelsheik10/agentic-rag-chatbot.git
cd agentic-rag-chatbot
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv RAG

# Activate virtual environment:
# Git Bash / WSL
source RAG/Scripts/activate
# Command Prompt / PowerShell
# .\RAG\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your Gemini API Key

```bash
# Bash / WSL
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
# PowerShell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

### 5. Run the Application

```bash
streamlit run main.py
```

## 🌐 Usage

- **Upload Documents:** Use the sidebar to upload your documents.
- **Process Uploads:** Click "Process Uploads" to store chunks in vector DB.
- **Ask Questions:** Type questions in the chat input. The agents work together to answer.
- **Clear Data:** Use "Clear All Data" to reset the session.

## ✅ Making Changes & Pushing Updates

```bash
git add .
git commit -m "Descriptive message"
git push origin main
```

## ⚠️ Troubleshooting

- **ModuleNotFoundError:** Ensure virtual environment is activated.
- **Missing `__init__.py`:** Add these in each package folder.
- **"No current event loop" error:** Use sync API methods.
- **LLM not responding:** Verify `GEMINI_API_KEY` is set.

---

> 🚧 **Note:** Contributions, PRs, and feedback are welcome to improve this project.
