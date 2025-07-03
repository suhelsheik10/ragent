import json
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
import sys
import os

# Add the project root to the Python path for standalone execution
current_dir_mcp = os.path.dirname(os.path.abspath(__file__))
project_root_mcp = os.path.abspath(os.path.join(current_dir_mcp, '..', '..'))
if project_root_mcp not in sys.path:
    sys.path.insert(0, project_root_mcp)

class MCPMessageType(Enum):
    """
    Defines the types of messages that can be exchanged between agents
    within the Model Context Protocol (MCP).
    """
    DOCUMENT_UPLOAD = "DOCUMENT_UPLOAD"
    QUERY_REQUEST = "QUERY_REQUEST"
    CONTEXT_RESPONSE = "CONTEXT_RESPONSE"
    LLM_RESPONSE = "LLM_RESPONSE"
    ERROR = "ERROR"
    # Add more types as needed, e.g., AGENT_STATUS, CONFIG_UPDATE

class MCPMessage:
    """
    Represents a structured message in the Model Context Protocol (MCP).
    Used for inter-agent communication.
    """
    def __init__(self,
                 sender: str,
                 receiver: str,
                 message_type: MCPMessageType,
                 payload: Dict[str, Any],
                 trace_id: Optional[str] = None):
        """
        Initializes an MCPMessage.

        Args:
            sender (str): The name or ID of the sending agent.
            receiver (str): The name or ID of the intended receiving agent.
            message_type (MCPMessageType): The type of the message.
            payload (Dict[str, Any]): A dictionary containing the actual data/content
                                      of the message.
            trace_id (Optional[str]): An optional unique ID to trace a series of
                                       related messages across agents. If not provided,
                                       a new one will be generated.
        """
        self.sender = sender
        self.receiver = receiver
        self.type = message_type
        self.trace_id = trace_id if trace_id is not None else str(uuid.uuid4())
        self.payload = payload

    def to_dict(self) -> Dict[str, Any]:
        """Converts the MCPMessage object to a dictionary."""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type.value,  # Use .value for enum
            "trace_id": self.trace_id,
            "payload": self.payload
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """Creates an MCPMessage object from a dictionary."""
        return cls(
            sender=data["sender"],
            receiver=data["receiver"],
            message_type=MCPMessageType(data["type"]), # Convert string back to enum
            payload=data["payload"],
            trace_id=data.get("trace_id")
        )

    def to_json(self) -> str:
        """Converts the MCPMessage object to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'MCPMessage':
        """Creates an MCPMessage object from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        """Returns a string representation of the MCPMessage."""
        return (f"MCPMessage(sender='{self.sender}', receiver='{self.receiver}', "
                f"type='{self.type.name}', trace_id='{self.trace_id}', payload={self.payload})")


class BaseMCPMessageHandler:
    """
    Base class for all agents that need to process MCP messages.
    Provides a common interface for message handling.
    """
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def process_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """
        Abstract method to process an incoming MCP message.
        Each agent must implement its specific message handling logic here.

        Args:
            message (MCPMessage): The incoming message to process.

        Returns:
            Optional[MCPMessage]: An MCPMessage response, or None if no direct
                                  response is needed immediately (e.g., for async ops).
        """
        raise NotImplementedError("Each agent must implement process_message.")

    def _create_response_message(self,
                                 receiver: str,
                                 message_type: MCPMessageType,
                                 payload: Dict[str, Any],
                                 trace_id: str) -> MCPMessage:
        """Helper to create a response MCPMessage."""
        return MCPMessage(
            sender=self.agent_name,
            receiver=receiver,
            message_type=message_type,
            payload=payload,
            trace_id=trace_id
        )

# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    # Create a document upload message
    upload_msg = MCPMessage(
        sender="UI",
        receiver="IngestionAgent",
        message_type=MCPMessageType.DOCUMENT_UPLOAD,
        payload={
            "files": [
                {"name": "document1.pdf", "path": "/tmp/document1.pdf"},
                {"name": "sales_report.docx", "path": "/tmp/sales_report.docx"}
            ],
            "timestamp": "2024-07-03T10:00:00Z"
        }
    )
    print("Document Upload Message:")
    print(upload_msg.to_json())

    # Create a query request message
    query_msg = MCPMessage(
        sender="UI",
        receiver="RetrievalAgent",
        message_type=MCPMessageType.QUERY_REQUEST,
        payload={"query": "What are the key performance indicators?"}
    )
    print("\nQuery Request Message:")
    print(query_msg.to_json())

    # Simulate a context response message
    context_response_msg = MCPMessage(
        sender="RetrievalAgent",
        receiver="LLMResponseAgent",
        message_type=MCPMessageType.CONTEXT_RESPONSE,
        payload={
            "top_chunks": ["Relevant chunk 1 about KPIs.", "Relevant chunk 2 with metrics."],
            "query": "What are the key performance indicators?",
            "sources": ["document1.pdf", "sales_report.docx"]
        },
        trace_id=query_msg.trace_id # Link to original query
    )
    print("\nContext Response Message:")
    print(context_response_msg.to_json())

    # Simulate an LLM response message
    llm_response_msg = MCPMessage(
        sender="LLMResponseAgent",
        receiver="UI",
        message_type=MCPMessageType.LLM_RESPONSE,
        payload={
            "answer": "The key performance indicators tracked include...",
            "sources": ["document1.pdf", "sales_report.docx"]
        },
        trace_id=query_msg.trace_id
    )
    print("\nLLM Response Message:")
    print(llm_response_msg.to_json())

    # Test serialization/deserialization
    json_str = llm_response_msg.to_json()
    reconstructed_msg = MCPMessage.from_json(json_str)
    print("\nReconstructed Message from JSON:")
    print(reconstructed_msg.to_json())
    assert reconstructed_msg.sender == llm_response_msg.sender
    assert reconstructed_msg.type == llm_response_msg.type
    print("Serialization/Deserialization successful!")