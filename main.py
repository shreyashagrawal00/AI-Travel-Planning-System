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

from tools.flight_tool import search_flights
from tools.tavily_tool import tavily_search

# -------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")

# -------------------------------------------------------
# LLM
# -------------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
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

    response = llm.invoke(
        [
            SystemMessage(
                content="You are an expert travel planner."
            ),
            HumanMessage(content=prompt),
        ]
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

    response = llm.invoke(
        [
            HumanMessage(content=prompt)
        ]
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

try:

    conn = psycopg.connect(
        DATABASE_URL,
        autocommit=True,
    )

    checkpointer = PostgresSaver(conn)

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