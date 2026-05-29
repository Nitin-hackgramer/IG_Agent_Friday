import os
import re
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from app.graph.state import AgentState

load_dotenv()

# Initiate the Groq inference engine
llm = ChatGroq(
    temperature=0.85,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
)


def clean_reasoning_tokens(raw_text: str) -> str:
    """
    Parses out model reasoning block to prevent internal thoughts from leaking to public DMs.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    return cleaned.strip()


def polish_dm_response(raw_text: str) -> str:
    """
    Keeps model output shaped like a real Instagram DM instead of a drafted answer.
    """
    text = clean_reasoning_tokens(raw_text)
    text = re.sub(r"^(Nitin|Assistant|AI)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text.strip())
    text = re.sub(r"\s+", " ", text)

    # Remove list formatting if the model slips into assistant mode.
    text = re.sub(r"^\s*[-*]\s*", "", text)

    # Instagram chats feel odd when one reply becomes a whole paragraph.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 2:
        text = " ".join(sentences[:2])

    # Keep runaway generations from sounding like essays.
    words = text.split()
    if len(words) > 28:
        text = " ".join(words[:28]).rstrip(".,;:") + "..."

    return text.strip()


async def summarize_history_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Detects context size inflation. If history exceeds 6 statements (3 turns),
    it condenses the oldest turn into a unified summary string.
    """
    # Allow callers to opt-out of any LLM calls (useful during human takeover)
    if state.get("skip_llm", False):
        return {}
    history = state.get("chat_history", [])
    current_summary = state.get("conversation_summary", "")

    # If the window is small or history is missing, do nothing and pass state forward
    if len(history) <= 6:
        return {}

    # Isolate the oldest turn to condense, leaving the last 2 messages intact
    messages_to_condense = history[:-2]
    retained_history = history[-2:]

    # Format a prompt specifically for summarizing the history segment
    summary_prompt = f"""
Progressive summary so far:
{current_summary}

New older chat turns to integrate:
{messages_to_condense}

Rewrite the memory as a short private note for the next reply.
Keep only stable facts the user actually confirmed, their preferences, and important open threads.
Do not preserve the assistant's guesses, defensive corrections, filler, or wording style.
If something is uncertain, mark it as uncertain instead of making it sound proven.
"""

    response = await llm.ainvoke([("user", summary_prompt)])

    # Return updates to compress the active window size
    return {
        "conversation_summary": response.content.strip(),
        "chat_history": retained_history,
    }


async def ai_responder_node(state: AgentState) -> Dict[str, Any]:
    """
    Node: Generate the response using the compressed background summary and the remaining short-term dialogue window
    """
    user_query = state.get("incoming_message", "")
    history = state.get("chat_history", [])
    # If requested, only persist incoming messages without calling the LLM
    if state.get("skip_response", False) or state.get("skip_llm", False):
        updated_history = list(history)
        # Support owner (human) messages stored as assistant role when flagged
        if state.get("owner_message", False):
            if user_query:
                updated_history.append({"role": "assistant", "content": user_query})
        else:
            if user_query:
                updated_history.append({"role": "user", "content": user_query})

        # Persist only the updated history; do not generate or send replies
        return {"agent_response": "", "chat_history": updated_history}
    summary = state.get("conversation_summary", "")

    # Natural DM persona prompt
    system_instructions = """
You are replying from Nitin Sharma's Instagram DM.
Nitin is a 20-year-old B.Tech student at IPEC Delhi, just finished 4th semester.
He likes coding, tech, and anime, but he does not force those topics.

Write like a normal Indian college friend texting on Instagram:
- Casual Hinglish, mostly simple Hindi/English words.
- 1 short DM only, usually 4-14 words. Max 2 short sentences.
- Match the user's energy. If they are teasing, tease lightly. If they are serious, be normal and direct.
- Sound a little imperfect and human: small fillers like "haan", "arre", "bhai", "acha", "matlab" are okay, but do not overuse them.
- Do not give lectures, summaries, lists, explanations, or customer-support style replies.
- Do not repeat the user's exact words back unless it feels natural.
- Do not mention "history", "memory", "summary", "previous conversation", or how you know something.
- Use emojis rarely. Most replies should have no emoji.
- If you are unsure, be casual: "yaad nahi aa raha bhai", "shayad", "pata nahi".
- If the user catches a mistake, accept it casually instead of defending: "haan my bad, mix ho gaya".
- Never invent personal facts, schedules, college details, relationships, or promises.
- If asked whether this is an AI/bot/automation, be honest in a simple way.

Use memory only as soft background context. Recent user messages matter more than old memory.
If memory conflicts with the current message, follow the current message.
"""

    if summary:
        system_instructions += f"\n\nPrivate memory note:\n{summary}"

    # Standard Conversational array for LLM Invocation
    messages = [("system", system_instructions)]
    for turn in history:
        role = "human" if turn["role"] == "user" else turn["role"]
        messages.append((role, turn["content"]))
    messages.append(("human", user_query))

    # Getting the raw response from the model with thinking tags intact
    raw_completion = await llm.ainvoke(messages)

    # Cleaning the response by removing any internal reasoning
    polished_output = polish_dm_response(raw_completion.content)

    # Update short-term history tracking with the latest exchange
    updated_history = list(history)
    updated_history.append({"role": "user", "content": user_query})
    updated_history.append({"role": "assistant", "content": polished_output})

    return {"agent_response": polished_output, "chat_history": updated_history}
