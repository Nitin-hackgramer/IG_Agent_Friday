import logging
from datetime import datetime
from typing import Any

from app.services.meta_api import send_instagram_dm

logger = logging.getLogger("uvicorn")


async def start_takeover(app: Any, owner_id: str) -> None:
    """Enable human takeover mode and notify owner."""
    app.state.human_takeover = True
    app.state.takeover_owner_id = owner_id
    app.state.takeover_started = datetime.utcnow()

    try:
        await send_instagram_dm(
            owner_id,
            "Takeover ON — bot paused for all users. Conversations will be recorded.",
        )
    except Exception as e:
        logger.error(f"Failed to notify owner about takeover: {e}")


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

            initial_state = {
                "user_id": user_id,
                "incoming_message": "",
                "agent_response": "",
                "chat_history": chat_history,
                "conversation_summary": "",
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
            except Exception as e:
                logger.error(
                    f"Error persisting buffered takeover messages for {user_id}: {e}"
                )

        # clear buffer
        app.state.takeover_buffer = []

    # Notify owner that bot is back
    if owner_id:
        try:
            await send_instagram_dm(
                owner_id,
                "Botback — takeover messages saved to memory. Bot resumed and will respond to new messages.",
            )
        except Exception as e:
            logger.error(f"Failed to notify owner about botback: {e}")
