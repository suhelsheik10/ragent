import json
import logging
# Removed asyncio import as it's no longer needed for synchronous calls
import sys
import os

# Removed sys.path manipulation here as main.py handles adding project root to path
# current_dir_llma = os.path.dirname(os.path.abspath(__file__))
# project_root_llma = os.path.abspath(os.path.join(current_dir_llma, '..', '..'))
# if project_root_llma not in sys.path:
#     sys.path.insert(0, project_root_llma)

from mcp.mcp_message_protocol import MCPMessage, MCPMessageType, BaseMCPMessageHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMResponseAgent(BaseMCPMessageHandler):
    """
    The LLMResponseAgent is responsible for:
    - Receiving context (retrieved chunks) and the original query.
    - Formatting a suitable prompt for the Large Language Model (LLM).
    - Interacting with the LLM API to get an answer.
    - Sending the LLM's response back to the UI.
    """
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash"):
        super().__init__("LLMResponseAgent")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") # Prioritize explicit key, then env var
        self.model_name = model_name
        self.client = None # Initialize client later to avoid issues if API key isn't immediately available
        logging.info(f"LLMResponseAgent initialized for model: {self.model_name}.")

    def _initialize_llm_client(self):
        """Initializes the Google Generative AI client."""
        if not self.api_key:
            logging.error("LLM API key not provided or found in environment variables.")
            return False
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
            logging.info(f"Initialized Google Generative AI client for model: {self.model_name}")
            return True
        except ImportError:
            logging.error("google-generativeai library not found. Please install it (`pip install google-generativeai`).")
            return False
        except Exception as e:
            logging.error(f"Error initializing LLM client: {e}", exc_info=True)
            return False

    def process_message(self, message: MCPMessage) -> MCPMessage:
        """
        Processes incoming MCP messages, specifically handling CONTEXT_RESPONSE messages.

        Args:
            message (MCPMessage): The incoming message from the RetrievalAgent.

        Returns:
            MCPMessage: An LLM_RESPONSE message containing the generated answer
                        and sources, or an ERROR message.
        """
        if message.type == MCPMessageType.CONTEXT_RESPONSE:
            return self._handle_context_response(message)
        elif message.type == MCPMessageType.QUERY_REQUEST:
            # If a query comes directly without context, act as a standalone LLM call
            # This might happen if no documents are uploaded, or if retrieval fails.
            return self._handle_direct_query_request(message)
        else:
            logging.warning(f"LLMResponseAgent received unhandled message type: {message.type.name}")
            return self._create_response_message(
                receiver=message.sender,
                message_type=MCPMessageType.ERROR,
                payload={"error": f"Unhandled message type: {message.type.name}"},
                trace_id=message.trace_id
            )

    def _handle_direct_query_request(self, message: MCPMessage) -> MCPMessage:
        """
        Handles a direct QUERY_REQUEST when no context is provided (e.g., no documents uploaded).
        """
        query = message.payload.get("query")
        if not query:
            logging.error("No query provided in direct QUERY_REQUEST payload.")
            return self._create_response_message(
                receiver=message.sender,
                message_type=MCPMessageType.ERROR,
                payload={"error": "Query is missing from payload for direct LLM call."},
                trace_id=message.trace_id
            )

        logging.info(f"Handling direct query without context: '{query}'")
        full_prompt = f"Answer the following question. If you don't know, state that you cannot answer based on provided information:\nQuestion: {query}\nAnswer:"

        llm_answer = self._call_llm(full_prompt)

        return self._create_response_message(
            receiver="UI",
            message_type=MCPMessageType.LLM_RESPONSE,
            payload={"answer": llm_answer, "sources": []}, # No sources for direct query
            trace_id=message.trace_id
        )

    def _handle_context_response(self, message: MCPMessage) -> MCPMessage:
        """
        Handles the CONTEXT_RESPONSE message by generating an LLM answer.
        """
        query = message.payload.get("query")
        top_chunks = message.payload.get("top_chunks", [])
        sources = message.payload.get("sources", [])

        if not query:
            logging.error("No query found in CONTEXT_RESPONSE payload.")
            return self._create_response_message(
                receiver=message.sender,
                message_type=MCPMessageType.ERROR,
                payload={"error": "Query is missing from context response."},
                trace_id=message.trace_id
            )

        if not top_chunks:
            logging.info(f"No relevant context found for query: '{query}'. Responding accordingly.")
            answer = "I couldn't find relevant information in the uploaded documents to answer your question. Please try rephrasing or uploading more relevant documents."
            return self._create_response_message(
                receiver="UI",
                message_type=MCPMessageType.LLM_RESPONSE,
                payload={"answer": answer, "sources": []},
                trace_id=message.trace_id
            )

        # Construct the prompt for the LLM
        context_str = "\n".join([f"Source: {sources[i%len(sources)] if sources else 'Unknown'}\nContent: {chunk}" for i, chunk in enumerate(top_chunks)])

        # Add instructions to the LLM to use the provided context and cite sources
        prompt = f"""You are a helpful assistant. Use the following retrieved context to answer the question.
If the answer is not available in the provided context, politely state that you don't have enough information.
Cite the source document(s) by name (e.g., 'From: document_name.pdf') if you use information from them.
Do not make up information.

Context:
{context_str}

Question: {query}

Answer:"""

        logging.info(f"Sending prompt to LLM (truncated): {prompt[:500]}...")
        llm_answer = self._call_llm(prompt)

        return self._create_response_message(
            receiver="UI",
            message_type=MCPMessageType.LLM_RESPONSE,
            payload={"answer": llm_answer, "sources": sources},
            trace_id=message.trace_id
        )

    def _call_llm(self, prompt: str) -> str:
        """
        Makes an API call to the Generative AI model.
        Uses synchronous API call for compatibility with Streamlit's execution model.
        """
        if not self.client and not self._initialize_llm_client():
            return "Error: LLM client could not be initialized due to missing API key or library."

        try:
            # Use the synchronous generate_content method
            response = self.client.generate_content(prompt)

            # Access the text from the response
            answer = response.text
            logging.info("Successfully received response from LLM.")
            return answer
        except Exception as e:
            logging.error(f"Error calling LLM API: {e}", exc_info=True)
            return f"An error occurred while generating the answer: {e}"

# Example Usage (for testing)
if __name__ == "__main__":
    # For local testing, ensure GEMINI_API_KEY is set in your environment
    # or pass it directly to the constructor.
    # E.g., export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    api_key_for_testing = os.environ.get("GEMINI_API_KEY", "YOUR_ACTUAL_GEMINI_API_KEY_HERE")

    if api_key_for_testing == "YOUR_ACTUAL_GEMINI_API_KEY_HERE":
        print("\nWARNING: Please set the GEMINI_API_KEY environment variable or replace 'YOUR_ACTUAL_GEMINI_API_KEY_HERE' with your actual key for the LLMResponseAgent to function correctly.\n")

    llm_agent = LLMResponseAgent(api_key=api_key_for_testing)

    # Simulate a CONTEXT_RESPONSE message
    context_message = MCPMessage(
        sender="RetrievalAgent",
        receiver="LLMResponseAgent",
        message_type=MCPMessageType.CONTEXT_RESPONSE,
        payload={
            "query": "What is the capital of France?",
            "top_chunks": [
                "Paris is the capital and most populous city of France.",
                "The Eiffel Tower is a landmark in Paris."
            ],
            "sources": ["Wikipedia.pdf", "TravelGuide.docx"]
        }
    )

    print(f"Simulating context response message: {context_message.to_json()}")
    response_from_llm_agent = llm_agent.process_message(context_message)
    print(f"\nLLMResponseAgent Response (with context): {response_from_llm_agent.to_json()}")

    # Simulate a QUERY_REQUEST message (no context)
    direct_query_message = MCPMessage(
        sender="UI",
        receiver="LLMResponseAgent",
        message_type=MCPMessageType.QUERY_REQUEST,
        payload={"query": "What is the primary function of a CPU?"}
    )

    print(f"\nSimulating direct query request message (no context): {direct_query_message.to_json()}")
    response_from_llm_agent_direct = llm_agent.process_message(direct_query_message)
    print(f"\nLLMResponseAgent Response (direct query): {response_from_llm_agent_direct.to_json()}")
