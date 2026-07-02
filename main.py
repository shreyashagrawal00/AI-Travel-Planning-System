"""
pip install langgraph langchain langchain-openai langchain-groq
langchain-community langchain-tavily
psycopg[binary] psycopg_pool
python-dotenv tavily-python
requests streamlit

Create PostgreSQL database:

CREATE DATABASE langgraph_memory;
"""

import os
import operator
from typing import TypedDict, Annotated

import psycopg
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq
from groq import RateLimitError

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging

from tools.flight_tool import search_flights
from tools.tavily_tool import tavily_search

# -------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")

logger = logging.getLogger("travel_planner")

# -------------------------------------------------------
# LLM
# -------------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
)


class LLMUnavailableError(Exception):
    """Raised when the LLM still fails after all retries (e.g. rate limit
    quota fully exhausted, not just a transient burst)."""


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _invoke_with_retry(messages):
    """Call the Groq LLM, retrying with exponential backoff on 429s.

    Groq's free tier enforces RPM/TPM/RPD limits. Most 429s are transient
    (a short burst), so a few retries with backoff usually clears them.
    If Groq returns a `retry_after` hint in the error body, honor it.
    """
    return llm.invoke(messages)


def safe_llm_invoke(messages, fallback_text: str):
    """Wraps _invoke_with_retry so a persistent rate limit doesn't crash
    the whole graph run — the pipeline still completes with a clear note
    about what happened, instead of an unhandled 500."""
    try:
        return _invoke_with_retry(messages)
    except RateLimitError as e:
        logger.error(f"Groq rate limit exhausted after retries: {e}")
        return AIMessage(
            content=(
                f"{fallback_text}\n\n"
                "_⚠️ This step couldn't complete because the Groq API rate "
                "limit was hit and didn't clear after several retries. "
                "This is a quota limit on the API key (free tier), not an "
                "app bug — wait a minute and try again, or use an API key "
                "with a higher limit._"
            )
        )

# -------------------------------------------------------
# State
# -------------------------------------------------------

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


# -------------------------------------------------------
# Flight Agent
# -------------------------------------------------------

def flight_agent(state: TravelState):

    query = state["user_query"]

    try:
        flights = search_flights(query)
    except Exception as e:
        flights = f"Unable to fetch flights.\n\n{e}"

    return {
        "flight_results": flights,
        "messages": [
            AIMessage(content="Flight results fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# -------------------------------------------------------
# Hotel Agent
# -------------------------------------------------------

def hotel_agent(state: TravelState):

    query = f"Best hotels for {state['user_query']}"

    try:
        hotels = tavily_search(query)
    except Exception as e:
        hotels = f"Unable to fetch hotels.\n\n{e}"

    return {
        "hotel_results": hotels,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# -------------------------------------------------------
# Itinerary Agent
# -------------------------------------------------------

def itinerary_agent(state: TravelState):

    prompt = f"""
You are an expert travel planner.

Create a complete itinerary.

User Request:
{state["user_query"]}

Flight Information:
{state["flight_results"]}

Hotel Information:
{state["hotel_results"]}

Return a well-structured itinerary.
"""

    response = safe_llm_invoke(
        [
            SystemMessage(
                content="You are an expert travel planner."
            ),
            HumanMessage(content=prompt),
        ],
        fallback_text="Itinerary could not be generated.",
    )

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# -------------------------------------------------------
# Final Agent
# -------------------------------------------------------

def final_agent(state: TravelState):

    prompt = f"""
Generate the final travel report.

Flights
--------
{state["flight_results"]}

Hotels
--------
{state["hotel_results"]}

Itinerary
--------
{state["itinerary"]}

Produce a beautiful final answer in Markdown.
"""

    response = safe_llm_invoke(
        [
            HumanMessage(content=prompt)
        ],
        fallback_text="Final report could not be generated.",
    )

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# -------------------------------------------------------
# Build Graph
# -------------------------------------------------------

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# -------------------------------------------------------
# PostgreSQL Checkpointer
# -------------------------------------------------------
# Uses a connection pool instead of a single raw connection.
# Managed/serverless Postgres (e.g. Neon) silently closes idle
# connections after a period of inactivity. A single long-lived
# connection then fails on the next query with
# "the connection is closed" (psycopg.OperationalError). A pool
# detects dead connections and transparently reconnects instead.

from psycopg_pool import ConnectionPool

_connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

try:

    pool = ConnectionPool(
        conninfo=DATABASE_URL,
        min_size=1,
        max_size=10,
        kwargs=_connection_kwargs,
        # Verify a connection is alive before handing it out; if the
        # server has silently dropped it, open a fresh one instead.
        check=ConnectionPool.check_connection,
    )

    checkpointer = PostgresSaver(pool)

    try:
        checkpointer.setup()
    except Exception:
        # Database already initialized
        pass

except Exception as e:
    raise RuntimeError(
        f"Unable to connect to PostgreSQL.\n\n{e}"
    )


# -------------------------------------------------------
# Compile Graph
# -------------------------------------------------------

app = graph.compile(
    checkpointer=checkpointer
)


# -------------------------------------------------------
# CLI
# -------------------------------------------------------

if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "user_aarohi"
        }
    }

    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0,
        },
        config=config,
    )

    print("\n" + "=" * 60)
    print("FINAL RESPONSE")
    print("=" * 60)

    for msg in result["messages"]:
        print(msg.content)