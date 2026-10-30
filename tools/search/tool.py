"""
Web search tool using Tavily Search API.
"""

from langchain_tavily import TavilySearch
from app.config import settings

search_tool = TavilySearch(
    name="tavily_search",
    max_results=5,
    topic="general",
    search_depth="advanced",
    tavily_api_key=settings.tavily_api_key if settings.tavily_api_key else None,
)

