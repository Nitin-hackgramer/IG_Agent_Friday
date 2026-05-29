***Ig-Agent (Instagram) — Interactive DM Agent***

A lightweight Instagram DM agent that uses Groq LLM for on-the-fly DM generation and Supabase/Postgres for persistent memory. Built with FastAPI and LangGraph, this project demonstrates a production-capable conversational automation with a human takeover flow (takeover / botback).

**Project Live Link:** https://ig-agent-friday.onrender.com/

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

**Quick Start (development)**

- Install dependencies: `pip install -r app/requirements.txt`
- Set environment variables (example):
    - `PAGE_ACCESS_TOKEN` (Meta page token)
    - `GROQ_API_KEY` (Groq key)
    - `SUPABASE_DB_URL` (Postgres connection string)
    - `OWNER_ID` (owner Instagram's numeric id used to send takeover/botback)
- Run locally: `python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`

**Planned / Future features**

- Owner-managed knowledge base: allow the owner to upload PDFs/docs/markdown to populate the agent’s private memory (useful for brand/product/company policies).
- CRM integration: import owner spreadsheets/CRM data so the agent can track leads, map prospects, and maintain contextual chat metadata for sales workflows.
- Admin UI: dashboard for takeover control, message review, and memory editing.

**Contributing**

- Open issues and PRs are welcome. For a quick start:
    - Fork the repo
    - Create a feature branch: `git checkout -b feat/your-feature`
    - Open a PR describing the change
