from typing import Dict, Any
from tavily import TavilyClient
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

tavily_client = TavilyClient()

class WebSearchInput(BaseModel):
    query: str = Field(description="Search query using only plain text, no accented characters.")

class LimitedWebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web for information. "
        "Queries must use only plain text characters, no accented or special characters."
    )
    args_schema: type[BaseModel] = WebSearchInput
    max_searches: int = 7
    search_count: int = 0

    def reset(self):
        """Resetea el contador para una nueva invocación del agente."""
        self.search_count = 0

    def _run(self, query: str) -> Dict[str, Any]:
        if self.search_count >= self.max_searches:
            return {
                "message": f"Search limit of {self.max_searches} reached. Stop searching and summarize your findings now."
            }
        self.search_count += 1
        print(f"[WebSearch] Query #{self.search_count}/{self.max_searches}: {query}")
        try:
            return tavily_client.search(query)
        except Exception as e:
            return {"error": str(e)}