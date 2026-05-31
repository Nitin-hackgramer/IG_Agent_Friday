import logging
from datetime import datetime
from typing import Any, List, Tuple
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

logger = logging.getLogger("uvicorn")


async def _summarize_messages_llm(msgs: List[Tuple[str, str, datetime]]) -> str:
    """Use Groq LLM to produce a short private memory summary from buffered messages.

    msgs: list of tuples (sender_id, text, timestamp)
    Returns a short summary suitable for `conversation_summary`.
    """
    if not msgs:
        return ""

    # Build a compact conversation transcript for summarization
    turns = []
    for sender_id, text, ts in msgs:
        role = "owner" if str(sender_id) == str(os.getenv("OWNER_ID", "")) else "user"
        turns.append(f"{role}: {text}")

    transcript = "\n".join(turns[-20:])

    prompt = f"""
Summarize the following short Instagram DM conversation into a single short private memory note (1-2 short sentences).
Include: relationship hints (friend/family/customer), clear user preferences, and the most recent topic.
Keep it neutral and concise (no apologies, no assistant reasoning). Return only the summary.

Conversation:
{transcript}
"""

    try:
        llm = ChatGroq(
            temperature=0.25,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
        )
        resp = await llm.ainvoke([("user", prompt)])
        summary = resp.content.strip()
        # keep it compact
        if len(summary) > 200:
            summary = summary[:197] + "..."
        return summary
    except Exception as e:
        logger.warning(f"LLM summarization failed: {e}")
        # fallback heuristic: use last two messages
        fallback = " ".join([t[1] for t in msgs[-2:]])
        return (fallback[:197] + "...") if len(fallback) > 200 else fallback


async def start_takeover(app: Any, owner_id: str) -> None:
    """Enable human takeover mode."""
    app.state.human_takeover = True
    app.state.takeover_owner_id = owner_id
    app.state.takeover_started = datetime.utcnow()
    logger.info("Takeover mode enabled.")


async def save_message(
    app: Any, user_id: str, sender_id: str, message_text: str
) -> None:
    """Persist an incoming message into the agent's persistent memory without invoking the LLM.

    This function prefers using the compiled agent's checkpointing so the stored
    conversation becomes part of the same thread state used at runtime.
    """
    compiled_agent = getattr(app.state, "compiled_agent", None)
    owner_id = getattr(app.state, "owner_id", None)

    if not compiled_agent:
        # fallback: keep in memory until stop_takeover runs
        if not hasattr(app.state, "takeover_buffer"):
            app.state.takeover_buffer = []
        app.state.takeover_buffer.append(
            (user_id, sender_id, message_text, datetime.utcnow())
        )
        return

    # Use the compiled agent to persist the incoming message to the thread checkpoint.
    initial_state = {
        "user_id": user_id,
        "incoming_message": message_text,
        "agent_response": "",
        "chat_history": [],
        "conversation_summary": "",
        "retrieved_content": "",
        "human_takeover": True,
        # Instruct nodes to NOT call any LLM or emit replies while saving
        "skip_response": True,
        "skip_llm": True,
        "owner_message": (
            True if owner_id and str(sender_id) == str(owner_id) else False
        ),
    }

    try:
        await compiled_agent.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": f"instagram_{user_id}"}},
        )
    except Exception as e:
        logger.error(f"Error saving takeover message for {user_id}: {e}")
        # on any error, keep in-memory as fallback
        if not hasattr(app.state, "takeover_buffer"):
            app.state.takeover_buffer = []
        app.state.takeover_buffer.append(
            (user_id, sender_id, message_text, datetime.utcnow())
        )


async def stop_takeover(app: Any) -> None:
    """Disable human takeover mode, persist buffered messages, and notify owner."""
    app.state.human_takeover = False
    owner_id = getattr(app.state, "takeover_owner_id", None)
    compiled_agent = getattr(app.state, "compiled_agent", None)

    # Flush in-memory buffer if present
    if hasattr(app.state, "takeover_buffer") and compiled_agent:
        buffer = app.state.takeover_buffer
        by_user = {}
        for user_id, sender_id, text, ts in buffer:
            by_user.setdefault(user_id, []).append((sender_id, text, ts))

        for user_id, msgs in by_user.items():
            chat_history = []
            for sender_id, text, ts in msgs:
                role = (
                    "assistant"
                    if owner_id and str(sender_id) == str(owner_id)
                    else "user"
                )
                chat_history.append({"role": role, "content": text})

            # Generate a short private summary for the buffered conversation
            try:
                conversation_summary = await _summarize_messages_llm(msgs)
            except Exception:
                conversation_summary = ""

            initial_state = {
                "user_id": user_id,
                "incoming_message": "",
                "agent_response": "",
                "chat_history": chat_history,
                "conversation_summary": conversation_summary,
                "retrieved_content": "",
                "human_takeover": False,
                "skip_response": True,
                "skip_llm": True,
            }

            try:
                await compiled_agent.ainvoke(
                    initial_state,
                    config={"configurable": {"thread_id": f"instagram_{user_id}"}},
                )
                # mark that next incoming message for this user should include a resume note
                try:
                    if not hasattr(app.state, "resume_pending"):
                        app.state.resume_pending = set()
                    app.state.resume_pending.add(user_id)
                except Exception:
                    pass
            except Exception as e:
                logger.error(
                    f"Error persisting buffered takeover messages for {user_id}: {e}"
                )

        # clear buffer
        app.state.takeover_buffer = []

    if owner_id:
        logger.info("Botback completed and takeover mode disabled.")
