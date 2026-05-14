from pydantic import BaseModel, Field
from typing import Any, List, Optional

class Store(BaseModel):
    """Information about the digital storefront where the game is sold."""
    store_name: str = Field(
        description="The formal name of the store (e.g., Steam, Epic Games Store, GOG)."
    )
    store_url: str = Field(
        description="The direct link to the game's product page on this specific store."
    )

class Game(BaseModel):
    """Detailed metadata and categorization for a videogame."""
    title: str = Field(description="The official full name of the videogame.")
    description: str = Field(description="A concise summary of the game's plot, setting, or core mechanics.")
    genres: List[str] = Field(description="A list of primary genres (e.g., RPG, Shooter, Platformer).")
    tags: List[str] = Field(description="A list of descriptive tags (e.g., Souls-like, Open World, Cyberpunk, 2D).")

class Rating(BaseModel):
    """Detailed data of the scores and reviews of a videogame."""
    metacritic_score: Optional[int] = Field(description="The Metacritic aggregate score (0-100).")
    ign_score: Optional[float] = Field(description="The score given by IGN (0-10).")
    top_reviews: List[str] = Field(description="A list of 2-3 highlighted quotes from critics.")


class DiscountInfo(BaseModel):
    """Detailed data of the best discounts available for a specific game."""
    game: Game = Field(description="Detailed data of the discounted game")
    discounted_price: float = Field(description="The discounted price available in the store")
    store: Store = Field(description="Detailed data of the store in which the discount was found")


class RatingInfo(BaseModel):
    """Consolidated review scores and player feedback."""
    game: Game = Field(description="Detailed data of the game being rated.")
    rating: Rating = Field(description="Detailed data of the scores and reviews of the game.")


class Trophy(BaseModel):
    """Individual trophy details."""
    name: str = Field(description="The name of the trophy.")
    description: str = Field(description="The official requirement to unlock it.")
    rarity: Optional[str] = Field(description="Rarity or type (Bronze, Silver, Gold, Platinum).")
    guide: Optional[str] = Field(description="How-to instructions. Only filled if a specific trophy is requested.")

class TrophyStats(BaseModel):
    """Summary of the achievement system."""
    total_count: int = Field(description="Total number of trophies.")
    difficulty_rating: str = Field(description="Difficulty (e.g., '4/10' or 'Moderate'). Source: PSNProfiles.")
    trophy_list: List[Trophy] = Field(default_factory=list, description="The list of trophies or the specific guide requested.")

class CompletionistInfo(BaseModel):
    """Main output class for the agent."""
    game: Game  # Usamos la clase Game que ya definimos antes
    main_hours: Optional[float] = Field(description="Hours for main story. Source: HowLongToBeat.")
    completionist_hours: Optional[float] = Field(description="Hours for 100%. Source: HowLongToBeat.")
    trophy_data: TrophyStats
    status_message: Optional[str] = Field(description="Important notices (e.g., Nintendo warning or 'Game not found').")

class RecommendationInfo(BaseModel):
    """Suggestions of similar games based on a specific title."""
    original_game: Game = Field(description="The game used as the basis for the recommendations.")
    recommendation_reason: str = Field(description="Short explanation of why these games are being suggested.")
    suggested_games: List[Game] = Field(description="A list of games that share similar genres, tags, or mechanics.")


class FinalOrchestratorOutput(BaseModel):
    """Final response containing the sub-agent data and the Spanish analysis."""
    subagent_json: Any = Field(description="The original JSON object from the specialized agent.")
    analisis_final: str = Field(description="Análisis detallado en español de los resultados y la intención del usuario.")
