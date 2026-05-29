from fastapi import FastAPI, Request, Response, status, BackgroundTasks
from contextlib import asynccontextmanager
import os
from psycopg_pool import AsyncConnectionPool, pool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.services.meta_api import send_instagram_dm
from app.services.takeover_service import (
    start_takeover,
    save_message as save_takeover_message,
    stop_takeover,
)
from app.graph.workflow import builder
import logging

logger = logging.getLogger("uvicorn")

VERIFY_TOKEN = "testtest"


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """
    Manages the application start and stop lifecycle events securely.
    guarantees network sockets only open inside active async event loops.
    """
    db_url = os.getenv("SUPABASE_DB_URL")
    logger.info("Connecting to Supabase checkpoint system...")

    async with AsyncConnectionPool(
        conninfo=db_url,
        max_size=10,
        min_size=1,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": None},
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        app.state.compiled_agent = builder.compile(checkpointer=checkpointer)
        # expose the DB pool and takeover defaults on the app state for takeover handling
        app.state.db_pool = pool
        app.state.human_takeover = False
        app.state.owner_id = os.getenv("OWNER_ID")
        app.state.takeover_started = None
        logger.info(
            "Agent successfully compiled with persistent Supabase checkpointer."
        )

        yield

    logger.info("Supabase database connection pool shut down cleanly.")


# Bind the lifespan context manager explicitly into the core FastAPI container
app = FastAPI(lifespan=app_lifespan)


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/webhook")
async def verify_webhook(request: Request):
    """Handles Meta's initial handshake verification request."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("Webhook successfully verified.")

        # Meta strictly expects the exact challenge string back as a plain text response
        return Response(
            content=challenge, media_type="text/plain", status_code=status.HTTP_200_OK
        )

    return Response(
        content="Verification token mismatch", status_code=status.HTTP_400_BAD_REQUEST
    )


async def handle_message_processing(psid: str, incoming_text: str, agent_engine):
    """
    Background worker that executes the LangGraph state machine and fires the DM responses with persistent thread mapping configurations.
    """

    config = {"configurable": {"thread_id": f"instagram_{psid}"}}

    # Package current webhook criteria into our master state dictionary layout
    initial_state = {
        "user_id": psid,
        "incoming_message": incoming_text,
        "agent_response": "",
        "chat_history": [],
        "conversation_summary": "",
        "retrieved_content": "",
        "human_takeover": False,
    }

    # Run the graph asynchronously across our node engine
    final_state = await agent_engine.ainvoke(initial_state, config=config)

    # Extract the cleaned, thought-stripped string output
    final_reply = final_state["agent_response"]

    # Fire the outbound API call back to the user
    if final_reply:
        await send_instagram_dm(recipient_id=psid, message_text=final_reply)


@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handles live incoming Instagram direct messages."""
    payload = await request.json()
    logger.info(f"Incoming Event Payload: {payload}")

    # Simple chect to extract text data
    try:
        if payload.get("object") == "instagram":
            for entry in payload.get("entry", []):
                for messaging_event in entry.get("messaging", []):
                    if (
                        "message" in messaging_event
                        and "text" in messaging_event["message"]
                    ):
                        message = messaging_event["message"]
                        psid = messaging_event["sender"]["id"]
                        message_text = message["text"]
                        is_echo = message.get("is_echo", False)
                        is_self = message.get("is_self", False)

                        live_agent = request.app.state.compiled_agent
                        owner_id = getattr(request.app.state, "owner_id", None)
                        cmd = (
                            message_text.strip().lower()
                            if owner_id and str(psid) == str(owner_id)
                            else ""
                        )
                        is_owner_command = cmd in {"takeover", "botback"}

                        # Ignore all normal echo/self events from our own outgoing page messages.
                        if (is_echo or is_self) and not is_owner_command:
                            logger.info(
                                f"Ignored outgoing/self event from {psid}: {message_text}"
                            )
                            continue

                        # Owner commands to toggle takeover mode
                        if is_owner_command:
                            if cmd == "takeover":
                                background_tasks.add_task(
                                    start_takeover, request.app, owner_id
                                )
                                logger.info(
                                    "Owner requested takeover. Switched to human takeover mode."
                                )
                                return Response(
                                    content="EVENT_RECEIVED", status_code=200
                                )
                            if cmd == "botback":
                                background_tasks.add_task(stop_takeover, request.app)
                                logger.info(
                                    "Owner requested botback. Processing takeover buffer."
                                )
                                return Response(
                                    content="EVENT_RECEIVED", status_code=200
                                )

                        # If a human takeover is active, persist incoming messages only and do not respond
                        if getattr(request.app.state, "human_takeover", False):
                            background_tasks.add_task(
                                save_takeover_message,
                                request.app,
                                psid,
                                psid,
                                message_text,
                            )
                            logger.info(
                                f"Human takeover active — saved message from {psid} only."
                            )
                            return Response(content="EVENT_RECEIVED", status_code=200)

                        # Hand the processing off to a background thread instantly so we can immediately return 200 OK back to Meta
                        background_tasks.add_task(
                            handle_message_processing, psid, message_text, live_agent
                        )

                        logger.info(f"User {psid} sent: {message_text}")

    except Exception as e:
        logger.error(f"Error parsing webhook content: {e}")

    # Meta requires a rapid 200 OK acknowledgment to prevent retries
    return Response(content="EVENT_RECEIVED", status_code=status.HTTP_200_OK)


# INFO:     2a03:2880:2ff:52:::0 - "GET /webhook?hub.mode=subscribe&hub.challenge=509448174&hub.verify_token=testtest&hub_mode=subscribe&hub_challenge=509448174&hub_verify_token=testtest HTTP/1.1" 200 OK
