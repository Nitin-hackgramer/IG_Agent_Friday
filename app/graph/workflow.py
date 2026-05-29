from langgraph.graph import StateGraph, END 
from app.graph.state import AgentState
from app.graph.nodes import ai_responder_node, summarize_history_node 

# initializing state-driven tracking graph
builder = StateGraph(AgentState)

# Registering our text generation node as an execution worker
builder.add_node("memory_manager", summarize_history_node)
builder.add_node("llm_brain", ai_responder_node)

# Configuring the logical execution route map
builder.set_entry_point("memory_manager")
builder.add_edge("memory_manager", "llm_brain" )
builder.add_edge("llm_brain", END)
 