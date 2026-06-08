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

RECENT_MESSAGE_LIMIT = 24


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


def neutralize_gendered_address(text: str) -> str:
    """
    Avoid guessing the user's gender or comfort level from an Instagram PSID.
    """
    replacements = {
        r"\bbhai\b": "",
        r"\bbro\b": "",
        r"\bdude\b": "",
        r"\bdidi\b": "",
        r"\bsis\b": "",
        r"\bbehen\b": "",
    }
    neutral = text
    for pattern, replacement in replacements.items():
        neutral = re.sub(pattern, replacement, neutral, flags=re.IGNORECASE)
    neutral = re.sub(r"\s+", " ", neutral)
    return neutral.strip()


def normalize_for_repeat_check(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def get_recent_assistant_replies(
    history: list[Dict[str, Any]], limit: int = 4
) -> list[str]:
    replies = [
        turn.get("content", "")
        for turn in history
        if turn.get("role") == "assistant" and turn.get("content")
    ]
    return replies[-limit:]


def is_repetitive_reply(candidate: str, history: list[Dict[str, Any]]) -> bool:
    """
    Catch short-loop replies like "kya hua?", "kya haal hai?", etc.
    """
    normalized_candidate = normalize_for_repeat_check(candidate)
    if not normalized_candidate:
        return True

    candidate_words = normalized_candidate.split()
    for previous in get_recent_assistant_replies(history):
        normalized_previous = normalize_for_repeat_check(previous)
        if normalized_candidate == normalized_previous:
            return True

        previous_words = normalized_previous.split()
        if len(candidate_words) <= 5 and len(previous_words) <= 5:
            if normalized_candidate in normalized_previous:
                return True
            if normalized_previous in normalized_candidate:
                return True
            if candidate_words[:2] == previous_words[:2]:
                return True

    return False


async def summarize_history_node(state: AgentState) -> Dict[str, Any]:
    """
    Middleware-style node: keep recent DMs raw and summarize older turns.
    """
    # Allow callers to opt-out of any LLM calls (useful during human takeover)
    if state.get("skip_llm", False):
        return {}
    history = state.get("chat_history", [])
    current_summary = state.get("conversation_summary", "")

    # Keep the short-term window intact for natural replies.
    if len(history) <= RECENT_MESSAGE_LIMIT:
        return {}

    messages_to_condense = history[:-RECENT_MESSAGE_LIMIT]
    retained_history = history[-RECENT_MESSAGE_LIMIT:]

    summary_prompt = f"""
Progressive summary so far:
{current_summary}

Older chat turns to integrate:
{messages_to_condense}

Rewrite the memory as a concise private summary for the next reply.
Keep only stable facts the user actually confirmed, their preferences, and important open threads.
Keep useful relationship/context details, but avoid stale small talk and repeated greetings.
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
- Sound a little imperfect and human: small fillers like "haan", "arre", "acha", "matlab", "yaar" are okay, but do not overuse them.
- Do not use gendered address words like "bhai", "bro", "didi", "sis", "behen", or "ladki/ladka" unless the user clearly confirmed that exact preference.
- If you do not know the user's gender, keep the wording neutral. Prefer no address word, or use "yaar" sparingly.
- Do not reuse the same opening or filler from recent replies. Avoid loops like "kya hua?", "kya haal hai?", or the same greeting again.
- Do not give lectures, summaries, lists, explanations, or customer-support style replies.
- Do not repeat the user's exact words back unless it feels natural.
- Do not mention "history", "memory", "summary", "previous conversation", or how you know something.
- Use emojis rarely. Most replies should have no emoji.
- If you are unsure, be casual: "yaad nahi aa raha", "shayad", "pata nahi".
- If the user catches a mistake, accept it casually instead of defending: "haan my bad, mix ho gaya".
- Never invent personal facts, schedules, college details, relationships, or promises.
- If asked whether this is an AI/bot/automation, be honest in a simple way.

Use memory only as soft background context. Recent user messages matter more than old memory.
If memory conflicts with the current message, follow the current message.
"""

    # If this message resumes after a human takeover, prepend a clear resume note so the model
    # prioritizes previous context and avoids generic first-time greetings.
    if state.get("resumed_after_takeover", False) and summary:
        resume_note = (
            "Resume chat with this user. Private memory note: "
            + summary
            + "\nDo not open with a first-time greeting; continue the prior conversation naturally.\n\n"
        )
        system_instructions = resume_note + system_instructions

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
    polished_output = neutralize_gendered_address(
        polish_dm_response(raw_completion.content)
    )

    # Give the model one chance to fix low-effort repeated replies.
    if is_repetitive_reply(polished_output, history):
        retry_messages = (
            messages[:-1]
            + [
                (
                    "system",
                    "That reply repeats a recent short phrase. Write a fresh neutral DM, "
                    "no gendered address words, no repeated greeting/opening.",
                )
            ]
            + messages[-1:]
        )
        retry_completion = await llm.ainvoke(retry_messages)
        retry_output = neutralize_gendered_address(
            polish_dm_response(retry_completion.content)
        )
        if retry_output and not is_repetitive_reply(retry_output, history):
            polished_output = retry_output
        else:
            for fallback in (
                "haan bata, kya scene hai?",
                "acha, bol kya chal raha?",
                "samjha, bata phir?",
            ):
                if not is_repetitive_reply(fallback, history):
                    polished_output = fallback
                    break

    # Update short-term history tracking with the latest exchange
    updated_history = list(history)
    updated_history.append({"role": "user", "content": user_query})
    updated_history.append({"role": "assistant", "content": polished_output})

    return {"agent_response": polished_output, "chat_history": updated_history}
