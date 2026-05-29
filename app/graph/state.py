from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    The master state schema tracking data elements across execution steps.
    """
    user_id: str
    incoming_message: str
    agent_response: str

    chat_history: List[Dict[str, Any]]
    conversation_summary: str 
    retrieved_content: str
    human_takeover: bool
    
