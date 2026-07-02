# ✈️ AI Travel Planning System

A multi-agent AI travel planner built with **LangGraph**, **Groq (Llama 3.3 70B)**, and **Streamlit**. Give it a plain-English trip request — *"Plan a 7-day Japan trip under ₹2 lakhs"* — and four specialized agents work together to research flights and hotels, build a day-by-day itinerary, and hand back a polished final report, with conversation memory persisted in PostgreSQL.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Agent Architecture](#agent-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Running Locally](#running-locally)
- [Deploying on Streamlit Cloud](#deploying-on-streamlit-cloud)
- [Rate Limits & Error Handling](#rate-limits--error-handling)
- [Roadmap](#roadmap)
- [License](#license)

---

## How It Works

The system is a **LangGraph state graph**: a fixed pipeline of four agent nodes that pass a shared state object down the chain, each one enriching it before handing off to the next.

```
User query
    │
    ▼
┌─────────────────┐
│  Flight Agent    │  → queries AviationStack API
└────────┬─────────┘
         ▼
┌─────────────────┐
│  Hotel Agent     │  → queries Tavily web search
└────────┬─────────┘
         ▼
┌─────────────────┐
│ Itinerary Agent  │  → Groq LLM synthesizes flights + hotels into a day-by-day plan
└────────┬─────────┘
         ▼
┌─────────────────┐
│  Final Agent     │  → Groq LLM writes the polished Markdown trip report
└────────┬─────────┘
         ▼
   Streamlit UI (live streamed) + PostgreSQL checkpoint
```

Each node's output is streamed to the Streamlit frontend in real time via `app.stream(..., stream_mode="updates")`, so the UI shows each agent completing its step live rather than waiting for the whole pipeline to finish.

## Agent Architecture

| # | Agent | File / Function | Responsibility | Data Source |
|---|-------|------------------|-----------------|--------------|
| 1 | **Flight Agent** | `main.py → flight_agent()` | Searches for flight options matching the trip request | [AviationStack API](https://aviationstack.com/) via `tools/flight_tool.py` |
| 2 | **Hotel Agent** | `main.py → hotel_agent()` | Searches the web for hotel recommendations | [Tavily Search API](https://tavily.com/) via `tools/tavily_tool.py` |
| 3 | **Itinerary Agent** | `main.py → itinerary_agent()` | Uses the LLM to turn flight + hotel results into a structured day-by-day itinerary | Groq (`llama-3.3-70b-versatile`) |
| 4 | **Final Agent** | `main.py → final_agent()` | Uses the LLM to compose the final, formatted Markdown trip report | Groq (`llama-3.3-70b-versatile`) |

**Shared state (`TravelState`)** flows through every node:

```python
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]  # accumulated chat history
    user_query: str        # the original trip request
    flight_results: str    # set by flight_agent
    hotel_results: str     # set by hotel_agent
    itinerary: str         # set by itinerary_agent
    llm_calls: int         # running count of LLM invocations
```

**Persistence:** the graph is compiled with a `PostgresSaver` checkpointer, keyed by a `thread_id` (the "User ID" field in the sidebar). This means each user's conversation state is durable across sessions — the same `thread_id` picks up prior context.

## Tech Stack

- **[LangGraph](https://www.langchain.com/langgraph)** — orchestrates the multi-agent pipeline as a state graph
- **[LangChain](https://www.langchain.com/)** — LLM/message abstractions
- **[Groq](https://groq.com/)** — fast LLM inference (`llama-3.3-70b-versatile`)
- **[Streamlit](https://streamlit.io/)** — web UI, with live-streamed agent progress
- **[PostgreSQL](https://www.postgresql.org/)** — durable conversation/checkpoint memory (via [Neon](https://neon.tech/) or any Postgres instance)
- **[Tavily](https://tavily.com/)** — real-time web search for hotel research
- **[AviationStack](https://aviationstack.com/)** — flight data API
- **[Tenacity](https://tenacity.readthedocs.io/)** — retry/backoff for transient API rate limits

## Project Structure

```
AI-Travel-Planning-System/
├── main.py                # LangGraph pipeline: state, agent nodes, graph, checkpointer
├── frontend.py             # Streamlit UI: sidebar, hero, live agent stream, final report
├── requirements.txt
├── tools/
│   ├── flight_tool.py      # AviationStack API wrapper
│   └── tavily_tool.py      # Tavily search API wrapper
├── travel_plans/           # Auto-saved Markdown reports (created at runtime)
└── .env                    # API keys & DB connection string (not committed)
```

## Getting Started

## Environment Variables

Create a `.env` file in the project root:

```dotenv
AVIATIONSTACK_API_KEY="your_aviationstack_key"
GROQ_API_KEY="your_groq_key"
TAVILY_API_KEY="your_tavily_key"
DATABASE_URL="postgresql://user:password@host/dbname?sslmode=require"
```

> ⚠️ **Never commit your `.env` file.** Make sure it's listed in `.gitignore`. If a key is ever exposed (e.g. pasted into a chat, a public repo, or a screenshot), rotate it immediately from the provider's dashboard.

## Deploying on Streamlit Cloud

1. Push this repo to GitHub (with `.env` excluded).
2. In [Streamlit Cloud](https://streamlit.io/cloud), create a new app pointing at `frontend.py`.
3. Under **Settings → Secrets**, add the same four variables from your `.env` (`AVIATIONSTACK_API_KEY`, `GROQ_API_KEY`, `TAVILY_API_KEY`, `DATABASE_URL`).
4. Deploy. Logs are available via **Manage app** in the app's lower-right corner if something goes wrong.

## Rate Limits & Error Handling

Groq's free tier enforces requests-per-minute, tokens-per-minute, and requests-per-day limits on `llama-3.3-70b-versatile`. To keep the app resilient:

- LLM calls in `itinerary_agent` and `final_agent` go through `safe_llm_invoke()`, which retries on `RateLimitError` with exponential backoff (up to 5 attempts).
- If the limit still hasn't cleared after retries, the affected step returns a clear inline note instead of crashing the pipeline.
- The Streamlit frontend wraps the agent stream in a try/except so any remaining failure surfaces as a readable message (with a collapsible technical-details panel), not a raw traceback.

If you hit this often, consider a paid Groq tier or a smaller/cheaper model for the itinerary/final steps.

## Roadmap

- [ ] Real hotel pricing/availability API (currently web-search based)
- [ ] Multi-city itinerary support
- [ ] Budget-aware flight/hotel filtering
- [ ] User authentication instead of free-text thread IDs
- [ ] Export itinerary to PDF / calendar (.ics)

## License

MIT — see [LICENSE](LICENSE) for details (add one if not already present).
