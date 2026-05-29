**Ig-Agent (Instagram) — Interactive DM Agent**

A lightweight Instagram DM agent that uses Groq LLM for on-the-fly DM generation and Supabase/Postgres for persistent memory. Built with FastAPI and LangGraph, this project demonstrates a production-capable conversational automation with a human takeover flow (takeover / botback).

**Live Demo:** https://ig-agent-friday.onrender.com/

**Why this project matters**

- **Recruiters / Founders:** Shows full-stack AI integration — model, persistence, webhooks, and deployment.
- **Engineers / Students:** Practical example of applying LangChain/Groq, background workers, and durable checkpoints.
- **Non-technical stakeholders:** Demonstrates how an owner can pause automation, review chat history, and resume the bot with memory preserved.

**Current Features**

- **Instagram webhook listener:** inbound events are handled at `/webhook` and acknowledged quickly for Meta compliance.
- **Groq LLM replies:** Groq-powered conversational generation in [app/graph/nodes.py](app/graph/nodes.py) producing natural DM-style replies.
- **Persistent memory (Supabase/Postgres):** LangGraph checkpointing via `langgraph.checkpoint.postgres.aio.AsyncPostgresSaver` stores agent state.
- **Human takeover flow:** Owner can send `takeover` to pause automated replies and `botback` to resume; messages during takeover are persisted and not replied to.
- **Graceful background processing:** Incoming messages are processed in FastAPI `BackgroundTasks` and mapped to thread-aware checkpoints.

**Tech Stack**

- **Backend:** FastAPI
- **Async HTTP client:** httpx
- **LLM:** Groq via `langchain-groq` (ChatGroq)
- **State machine:** LangGraph
- **Checkpoint storage:** Supabase/Postgres via `langgraph-checkpoint-postgres`
- **Deployment:** Render (example deployment linked above)

**Repository Structure (brief)**

- **app/main.py**: FastAPI entrypoint, webhook endpoints, lifecycle setup, takeover command handling.
- **app/services/meta_api.py**: Meta/IG Graph API send helper.
- **app/services/takeover_service.py**: takeover/botback lifecycle and persistence helpers.
- **app/graph/nodes.py**: LLM prompting, response polishing, and memory summarization.
- **app/graph/workflow.py**: LangGraph state graph definition.
- **app/graph/state.py**: Typed state schema for LangGraph.
- **app/db/**: Database helper placeholders (can be extended with application-specific storage logic).
- **.env**: local environment variables (DO NOT COMMIT; included here for dev convenience — rotate secrets immediately).

**Quick Start (development)**

- Install dependencies: `pip install -r app/requirements.txt`
- Set environment variables (example):
    - `PAGE_ACCESS_TOKEN` (Meta page token)
    - `GROQ_API_KEY` (Groq key)
    - `SUPABASE_DB_URL` (Postgres connection string)
    - `OWNER_ID` (owner numeric id used to send takeover/botback)
- Run locally: `python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`

**Production / Render**

- Build command: `pip install -r app/requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT` (or `gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:$PORT`)
- Health check path: `/` or `/health` (the project includes a root health endpoint)

**Security & Ops notes**

- Rotate all keys in `.env` immediately if they were ever committed. Use Render/CI environment variables or a secrets vault in production.
- Verify your Meta webhook `VERIFY_TOKEN` matches the `VERIFY_TOKEN` in `app/main.py` (or move it to an env var for safety).
- Replace `PAGE_ACCESS_TOKEN` with a page-level token that has messaging permissions. Check recipient ids before sending to avoid IG API errors.

**How takeover works (brief)**

- Owner sends `takeover` to the bot account (owner id must match `OWNER_ID`).
- The app enters human takeover mode and saves incoming messages to checkpoint (no replies are sent while takeover is active).
- Owner sends `botback` to flush stored messages into memory and resume automated responses. Messages that arrived during takeover are not replied to automatically.

**Planned / Future features**

- Owner-managed knowledge base: allow the owner to upload PDFs/docs/markdown to populate the agent’s private memory (useful for brand/product/company policies).
- CRM integration: import owner spreadsheets/CRM data so the agent can track leads, map prospects, and maintain contextual chat metadata for sales workflows.
- Admin UI: dashboard for takeover control, message review, and memory editing.

**Contributing**

- Open issues and PRs are welcome. For a quick start:
    - Fork the repo
    - Create a feature branch: `git checkout -b feat/your-feature`
    - Open a PR describing the change

**Contact / Links**

- Live demo: https://ig-agent-friday.onrender.com/
- Code: this repository

**License**

- MIT (or choose your preferred license)

Enjoy — if you want, I can also: add a CONTRIBUTING.md, CI checks, and a small admin UI prototype for takeover control.
