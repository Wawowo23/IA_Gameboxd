from langchain.agents import AgentState
from typing import List, Optional

class GameState(AgentState):
    full_game_title: Optional[str] = None
    platform: Optional[str] = None
    genres: List[str] = []
    tags: List[str] = []