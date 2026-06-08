from typing import TypedDict, List, Dict, Any 

class AgentState(TypedDict, total=False):
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
    skip_response: bool
    skip_llm: bool
    owner_message: bool
    resumed_after_takeover: bool
    
