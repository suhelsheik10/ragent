# main.py
import streamlit as st
import os
import tempfile
import logging
import sys
from typing import List, Dict, Any, Tuple

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CORRECTED IMPORTS ---
# This ensures the current directory (AgenticRAGChatbot) is in Python's search path.
# This is crucial for finding 'agents', 'mcp', 'utils' as packages directly.
current_project_root = os.path.dirname(os.path.abspath(__file__))
if current_project_root not in sys.path:
    sys.path.insert(0, current_project_root)
    logging.info(f"Added {current_project_root} to sys.path") # Uncomment if you want to see this log

try:
    from mcp.mcp_message_protocol import MCPMessage, MCPMessageType
    from utils.utils_vector_store import VectorStore
    from utils.utils_document_processor import DocumentProcessor
    from agents.ingestion_agent import IngestionAgent
    from agents.retrieval_agent import RetrievalAgent
    from agents.llm_response_agent import LLMResponseAgent

except ImportError as e:
    st.error(f"Error loading core modules. Please ensure your project structure is correct and dependencies are installed. Missing: {e}")
    st.stop()

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")

# --- Session State Initialization ---
def initialize_session_state():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
        logging.info("VectorStore initialized in session state.")
    if "ingestion_agent" not in st.session_state:
        st.session_state.ingestion_agent = IngestionAgent(st.session_state.vector_store)
        logging.info("IngestionAgent initialized in session state.")
    if "retrieval_agent" not in st.session_state:
        st.session_state.retrieval_agent = RetrievalAgent(st.session_state.vector_store)
        logging.info("RetrievalAgent initialized in session state.")
    if "llm_response_agent" not in st.session_state:
        # Pass API key from Streamlit secrets or environment variable
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            st.warning("GEMINI_API_KEY environment variable not set. LLM functionality may be limited or fail.")
        st.session_state.llm_response_agent = LLMResponseAgent(api_key=gemini_api_key)
        logging.info("LLMResponseAgent initialized in session state.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        logging.info("Chat messages initialized in session state.")
    if "uploaded_files_info" not in st.session_state:
        st.session_state.uploaded_files_info = [] # Stores name and original name
        logging.info("Uploaded files info initialized.")

initialize_session_state()

# --- Agent Communication Utility ---
def send_message_to_agent(message: MCPMessage) -> MCPMessage:
    """Simulates message passing between agents based on receiver."""
    if message.receiver == "IngestionAgent":
        return st.session_state.ingestion_agent.process_message(message)
    elif message.receiver == "RetrievalAgent":
        return st.session_state.retrieval_agent.process_message(message)
    elif message.receiver == "LLMResponseAgent":
        return st.session_state.llm_response_agent.process_message(message)
    else:
        logging.error(f"Unknown receiver for message: {message.receiver}")
        return MCPMessage(
            sender="System",
            receiver=message.sender,
            message_type=MCPMessageType.ERROR,
            payload={"error": f"Unknown agent receiver: {message.receiver}"},
            trace_id=message.trace_id
        )

# --- UI Components ---
st.title("📄💬 Agentic RAG Chatbot")

# Sidebar for document upload and controls
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, CSV, PPTX, TXT, or MD files",
        type=["pdf", "docx", "csv", "pptx", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Uploads"):
            if st.session_state.vector_store.index.ntotal > 0:
                st.warning("New uploads will be added to the existing document base. Clear data first if you want to start fresh.")

            files_to_process = []
            current_uploaded_file_names = {f['original_name'] for f in st.session_state.uploaded_files_info}

            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    # Check if file was already processed in current session
                    if uploaded_file.name in current_uploaded_file_names:
                        st.info(f"Skipping '{uploaded_file.name}' as it was already processed in this session.")
                        continue

                    # Save uploaded file to a temporary location
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    files_to_process.append({
                        "name": uploaded_file.name,
                        "path": file_path,
                        "original_name": uploaded_file.name # To display original name in UI
                    })

                if files_to_process:
                    with st.spinner("Processing documents... This may take a moment."):
                        ingestion_message = MCPMessage(
                            sender="UI",
                            receiver="IngestionAgent",
                            message_type=MCPMessageType.DOCUMENT_UPLOAD,
                            payload={"files": files_to_process}
                        )
                        ingestion_response = send_message_to_agent(ingestion_message)

                        if ingestion_response.type == MCPMessageType.LLM_RESPONSE: # IngestionAgent sends LLM_RESPONSE status
                            st.success(f"Document processing complete: {ingestion_response.payload.get('message')}")
                            # Update the list of successfully uploaded files in session state
                            for doc_meta in ingestion_response.payload.get("processed_documents_metadata", []):
                                if doc_meta["status"] == "success":
                                    st.session_state.uploaded_files_info.append({
                                        "name": doc_meta["name"],
                                        "original_name": doc_meta["name"], # Use name for original_name as it's the actual filename
                                        "chunks_count": doc_meta["chunks_count"]
                                    })
                            st.session_state.messages.append({"role": "assistant", "content": ingestion_response.payload.get('message')})
                        elif ingestion_response.type == MCPMessageType.ERROR:
                            st.error(f"Error during document processing: {ingestion_response.payload.get('error', 'Unknown error')}")
                            st.session_state.messages.append({"role": "assistant", "content": f"Failed to process documents: {ingestion_response.payload.get('error', 'Unknown error')}"})
                        else:
                            st.error("Unexpected response from Ingestion Agent.")
                            st.session_state.messages.append({"role": "assistant", "content": "An unexpected error occurred during document processing."})
                else:
                    st.info("No new files to process.")

    st.subheader("Currently Processed Documents:")
    if st.session_state.uploaded_files_info:
        for i, file_info in enumerate(st.session_state.uploaded_files_info):
            st.write(f"- {file_info['original_name']} ({file_info['chunks_count']} chunks)")
    else:
        st.info("No documents processed yet.")

    if st.button("Clear All Data"):
        st.session_state.vector_store.clear_store()
        st.session_state.messages = []
        st.session_state.uploaded_files_info = []
        st.success("All processed data and chat history cleared!")
        logging.info("All data cleared by user request.")
        st.rerun()

# --- Main Chat Interface ---
st.header("Chat with your documents")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_content = ""
            response_sources = []

            # Step 1: Send query to RetrievalAgent
            retrieval_message = MCPMessage(
                sender="UI",
                receiver="RetrievalAgent",
                message_type=MCPMessageType.QUERY_REQUEST,
                payload={"query": prompt, "k_chunks": 5} # Request top 5 chunks
            )
            retrieval_response = send_message_to_agent(retrieval_message)

            if retrieval_response.type == MCPMessageType.CONTEXT_RESPONSE:
                # Step 2: Send context and query to LLMResponseAgent
                llm_query_payload = {
                    "query": prompt,
                    "top_chunks": retrieval_response.payload.get("top_chunks", []),
                    "sources": retrieval_response.payload.get("sources", [])
                }
                llm_message = MCPMessage(
                    sender="RetrievalAgent", # Sender is RetrievalAgent as it forwards context
                    receiver="LLMResponseAgent",
                    message_type=MCPMessageType.CONTEXT_RESPONSE, # Using CONTEXT_RESPONSE type
                    payload=llm_query_payload,
                    trace_id=retrieval_message.trace_id # Maintain trace ID
                )
                llm_response = send_message_to_agent(llm_message)

                if llm_response.type == MCPMessageType.LLM_RESPONSE:
                    response_content = llm_response.payload.get("answer", "No answer generated.")
                    response_sources = llm_response.payload.get("sources", [])
                elif llm_response.type == MCPMessageType.ERROR:
                    response_content = f"Error from LLM Agent: {llm_response.payload.get('error', 'Unknown error')}"
                else:
                    response_content = "Unexpected response type from LLM Agent."

            elif retrieval_response.type == MCPMessageType.ERROR:
                response_content = f"Error from Retrieval Agent: {retrieval_response.payload.get('error', 'Unknown error')}"
                # If retrieval fails, try to answer with LLM directly (without context)
                if st.session_state.vector_store.index.ntotal == 0:
                    response_content += "\n\nNo documents have been processed. Please upload and process documents first."
                else:
                    st.warning("Retrieval failed, attempting to answer without specific context from documents.")
                    llm_message_direct = MCPMessage(
                        sender="UI",
                        receiver="LLMResponseAgent",
                        message_type=MCPMessageType.QUERY_REQUEST, # Use QUERY_REQUEST for direct LLM call
                        payload={"query": prompt},
                        trace_id=retrieval_message.trace_id
                    )
                    llm_response_direct = send_message_to_agent(llm_message_direct)
                    if llm_response_direct.type == MCPMessageType.LLM_RESPONSE:
                        response_content = llm_response_direct.payload.get("answer", "No answer generated.")
                        response_sources = llm_response_direct.payload.get("sources", [])
                    else:
                        response_content = f"Further error when attempting direct LLM call: {llm_response_direct.payload.get('error', 'Unknown error')}"

            else:
                response_content = "An unexpected error occurred during the process."

            st.markdown(response_content)
            if response_sources:
                st.caption(f"Sources: {', '.join(response_sources)}")

        st.session_state.messages.append({"role": "assistant", "content": response_content})