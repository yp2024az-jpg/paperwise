from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

from backend.config import get_tavily_api_key
from backend.llm import llm
from backend.models import BtwRouteDecision

load_dotenv(Path(__file__).with_name(".env"))


def handle_btw(query: str) -> Generator[str, None, None]:
    """Off-topic side channel — never touches the vector store or checkpointer."""
    route_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Decide if answering this question requires a real-time web search (recent events, "
         "current prices, breaking news) or if your general knowledge is sufficient."),
        ("human", "{query}"),
    ])
    decision = (route_prompt | llm.with_structured_output(BtwRouteDecision)).invoke({"query": query})

    if decision.needs_web_search:
        client = TavilyClient(api_key=get_tavily_api_key())
        results = client.search(query, max_results=3)
        context = "\n\n".join(r["content"] for r in results["results"])
        sources = "\n".join(f"- {r['url']}" for r in results["results"])

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Answer the question using the web search results below. Be concise.\n\n"
             f"Results:\n{context}\n\nSources:\n{sources}"),
            ("human", "{query}"),
        ])
    else:
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question concisely from your general knowledge."),
            ("human", "{query}"),
        ])

    for chunk in (answer_prompt | llm).stream({"query": query}):
        if chunk.content:
            yield chunk.content
