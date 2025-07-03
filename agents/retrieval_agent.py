import logging
from typing import List, Dict, Any, Optional
import sys
import os

# Add the project root to the Python path for standalone execution
current_dir_ra = os.path.dirname(os.path.abspath(__file__))
project_root_ra = os.path.abspath(os.path.join(current_dir_ra, '..', '..'))
if project_root_ra not in sys.path:
    sys.path.insert(0, project_root_ra)

from mcp.mcp_message_protocol import MCPMessage, MCPMessageType, BaseMCPMessageHandler
from utils.utils_vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RetrievalAgent(BaseMCPMessageHandler):
    """
    The RetrievalAgent is responsible for:
    - Receiving user queries.
    - Generating embeddings for the queries.
    - Performing semantic search against the vector store.
    - Returning the most relevant chunks along with their source information.
    """
    def __init__(self, vector_store: VectorStore):
        super().__init__("RetrievalAgent")
        self.vector_store = vector_store
        logging.info("RetrievalAgent initialized.")

    def process_message(self, message: MCPMessage) -> MCPMessage:
        """
        Processes incoming MCP messages, specifically handling QUERY_REQUEST messages.

        Args:
            message (MCPMessage): The incoming message.

        Returns:
            MCPMessage: A CONTEXT_RESPONSE message containing retrieved context
                        or an ERROR message.
        """
        if message.type == MCPMessageType.QUERY_REQUEST:
            return self._handle_query_request(message)
        else:
            logging.warning(f"RetrievalAgent received unhandled message type: {message.type.name}")
            return self._create_response_message(
                receiver=message.sender,
                message_type=MCPMessageType.ERROR,
                payload={"error": f"Unhandled message type: {message.type.name}"},
                trace_id=message.trace_id
            )

    def _handle_query_request(self, message: MCPMessage) -> MCPMessage:
        """
        Handles the QUERY_REQUEST message by performing a semantic search.
        """
        query = message.payload.get("query")
        k_chunks = message.payload.get("k_chunks", 5) # Default to 5 chunks
        if not query:
            logging.error("No query provided in QUERY_REQUEST payload.")
            return self._create_response_message(
                receiver=message.sender,
                message_type=MCPMessageType.ERROR,
                payload={"error": "Query is missing from payload."},
                trace_id=message.trace_id
            )

        logging.info(f"Retrieving top {k_chunks} chunks for query: '{query}'")
        try:
            retrieved_chunks_info = self.vector_store.similarity_search(query, k=k_chunks)

            top_chunks_text = [chunk["text"] for chunk in retrieved_chunks_info]
            # Collect unique sources
            sources = sorted(list(set(chunk["source"] for chunk in retrieved_chunks_info if "source" in chunk)))

            response_payload = {
                "top_chunks": top_chunks_text,
                "query": query,
                "sources": sources,
                "retrieved_chunks_details": retrieved_chunks_info # Include full details for debugging/advanced use
            }

            logging.info(f"Retrieved {len(top_chunks_text)} chunks. Sources: {sources}")

            return self._create_response_message(
                receiver="LLMResponseAgent", # Send to LLMResponseAgent as per workflow
                message_type=MCPMessageType.CONTEXT_RESPONSE,
                payload=response_payload,
                trace_id=message.trace_id
            )
        except Exception as e:
            logging.error(f"Error during retrieval for query '{query}': {e}", exc_info=True)
            return self._create_response_message(
                receiver=message.sender, # Send error back to sender (UI)
                message_type=MCPMessageType.ERROR,
                payload={"error": f"Failed to retrieve context: {e}"},
                trace_id=message.trace_id
            )

# Example Usage (for testing)
if __name__ == "__main__":
    # Dummy VectorStore for testing RetrievalAgent
    class DummyVectorStore:
        def __init__(self):
            # This is a very simplified similarity search for testing
            # In a real scenario, this would be actual embeddings and FAISS index.
            self.data = [
                {"text": "Key Performance Indicators (KPIs) are crucial for evaluating business success.", "source": "report.pdf"},
                {"text": "Revenue increased by 15% in Q1, driven by strong sales in the tech sector.", "source": "sales_data.csv"},
                {"text": "Customer satisfaction scores reached 92%, exceeding targets.", "source": "metrics.xlsx"},
                {"text": "The project timeline was adjusted due to unforeseen delays in procurement.", "source": "project_plan.docx"},
                {"text": "Agile methodologies promote iterative development and collaboration.", "source": "agile_guide.md"},
            ]
            self.ntotal = len(self.data) # For testing purposes

        def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
            logging.info(f"Dummy search for query: '{query}'")
            results = []
            # Simple keyword matching for dummy search
            for item in self.data:
                # Prioritize direct matches, then split query for broader matches
                if query.lower() in item["text"].lower():
                    results.append({"text": item["text"], "source": item["source"], "similarity_score": 1.0})
                elif any(keyword in item["text"].lower() for keyword in query.lower().split()):
                    results.append({"text": item["text"], "source": item["source"], "similarity_score": 0.8})

            # Ensure unique results (by text content, for simplicity in dummy) and then sort
            unique_results = {}
            for res in results:
                if res["text"] not in unique_results or res["similarity_score"] > unique_results[res["text"]].get("similarity_score", 0):
                    unique_results[res["text"]] = res
            results = list(unique_results.values())

            # If no direct match, return some dummy results to avoid empty
            if not results and self.data:
                # Return the most generic sounding one, or just the first
                results = [{"text": self.data[0]["text"], "source": self.data[0]["source"], "similarity_score": 0.1}]


            return sorted(results, key=lambda x: x["similarity_score"], reverse=True)[:k]

    dummy_vector_store = DummyVectorStore()
    retrieval_agent = RetrievalAgent(dummy_vector_store)

    # Simulate a QUERY_REQUEST message
    query_message = MCPMessage(
        sender="UI",
        receiver="RetrievalAgent",
        message_type=MCPMessageType.QUERY_REQUEST,
        payload={"query": "What are the main KPIs?", "k_chunks": 2}
    )

    print(f"Simulating query request message: {query_message.to_json()}")
    response_message = retrieval_agent.process_message(query_message)
    print(f"\nRetrievalAgent Response: {response_message.to_json()}")

    query_message_no_match = MCPMessage(
        sender="UI",
        receiver="RetrievalAgent",
        message_type=MCPMessageType.QUERY_REQUEST,
        payload={"query": "Tell me about space travel.", "k_chunks": 1}
    )
    print(f"\nSimulating query request message (no match): {query_message_no_match.to_json()}")
    response_message_no_match = retrieval_agent.process_message(query_message_no_match)
    print(f"\nRetrievalAgent Response (no match): {response_message_no_match.to_json()}")