# day54_web_search_agent/tools.py
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


# Tavily web search tool
web_search = TavilySearchResults(
    max_results=3,
    api_key=os.getenv("TAVILY_API_KEY")
)

web_search.name        = "web_search"
web_search.description = "Search the web for current information. Use for recent events, news, prices, or any information that may have changed recently."


@tool
def summarize_search_results(results: str) -> str:
    """
    Summarize and format web search results into a clean output.
    Pass the raw search results as a string.
    """
    if not results:
        return "No results to summarize."
    lines = results.split('\n')[:10]
    return "Key findings:\n" + "\n".join(f"-> {line}" for line in lines if line.strip())