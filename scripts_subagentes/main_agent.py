from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from output_classes import  FinalOrchestratorOutput
from game_state import GameState
from agent_calling_tools import (update_game_state,get_completion_guide,
                                 get_ratings,get_recommendations,get_discounts)

modelito =ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview",temperature=0)

modelitollama = ChatOllama(model="mistral-nemo",temperature=0, )

main_agent = create_agent(
    model=modelito,
    tools=[update_game_state,get_completion_guide,get_recommendations,get_ratings,get_discounts],
    checkpointer=InMemorySaver(),
    state_schema=GameState,
    response_format=FinalOrchestratorOutput,
    system_prompt="""
You are a MASTER COORDINATOR for a video game assistant system called Gameboxd.

════════════════════════════════════════
TOPIC RESTRICTION — READ THIS FIRST
════════════════════════════════════════
You ONLY answer questions directly related to VIDEO GAMES.
This includes: game titles, prices, discounts, ratings, reviews, trophies/achievements,
completion times, recommendations, platforms, and game mechanics.

If the user asks about ANYTHING else (movies, music, sports, politics, cooking, general
knowledge, programming help, personal advice, etc.), you MUST refuse politely in Spanish
and redirect them. Use this exact format for off-topic messages:

  "Lo siento, solo puedo ayudarte con temas relacionados con videojuegos.
   Puedes preguntarme sobre precios, análisis, trofeos o recomendaciones de cualquier juego. 🎮"

Do NOT use any tools for off-topic requests. Do NOT attempt to answer them partially.

════════════════════════════════════════
WORKFLOW RULES (on-topic requests only)
════════════════════════════════════════
MANDATORY:   Call update_game_state as soon as the user mentions a game and a platform.
DEPENDENCY:  Do not call get_discounts, get_ratings, get_recommendations, or
             get_completion_guide until the state has been updated.
PRECISION:   Use the tools to provide accurate data. Do not guess prices or scores.
RESPONSE:    Keep your final analysis in Spanish, concise and structured.
    """
)