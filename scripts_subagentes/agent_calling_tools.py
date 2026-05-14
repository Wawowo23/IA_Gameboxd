from langchain.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from langchain.tools import tool, ToolRuntime
from sub_agents import rating_agent,discount_agent,completion_agent,recommendation_agent,discount_search,rating_search,reco_search,completion_search

@tool
def get_discounts(runtime: ToolRuntime) -> dict:
    """Busca el mejor precio y oferta para el juego actual en tiendas digitales."""
    title = runtime.state.get("full_game_title")
    platform = runtime.state.get("platform")

    if not title:
        # Devolvemos un mensaje que "obligue" al modelo a usar la otra herramienta
        return "ERROR: No conozco el nombre del juego. DETÉN la búsqueda y usa primero 'update_game_state' para registrar el título y la plataforma."

    print(f"DEBUG: 🔍 Buscando ofertas para {title} en {platform}...")
    discount_search.reset()
    response = discount_agent.invoke({"messages": [HumanMessage(content=f"Busca la mejor oferta para {title} en {platform}")]})
    print("DEBUG: ✅ Sub-agente respondió correctamente.")
    return response['messages'][-1].content


@tool
def get_ratings(runtime: ToolRuntime) -> dict:
    """Obtiene las notas de Metacritic e IGN junto con reseñas destacadas."""

    # 1. Usamos .get() en lugar de [] para evitar que Python lance un KeyError
    title = runtime.state.get("full_game_title")

    # 2. Si el título está vacío (None), le mandamos un error a la IA para que lo arregle
    if not title:
        return "ERROR INTERNO: No has definido el juego. Debes llamar a la herramienta 'update_game_state' indicando el título del juego antes de poder usar esta herramienta."

    # 3. Si todo está bien, continuamos normal
    rating_search.reset()
    response = rating_agent.invoke({"messages": [HumanMessage(content=f"Reseñas y notas para {title}")]})
    return response['messages'][-1].content


@tool
def get_recommendations(runtime: ToolRuntime) -> dict:
    """Sugiere juegos similares basados en géneros y mecánicas del juego base."""

    # 1. Usamos .get() en lugar de [] para evitar que Python lance un KeyError
    title = runtime.state.get("full_game_title")

    # 2. Si el título está vacío (None), le mandamos un error a la IA para que lo arregle
    if not title:
        return "ERROR INTERNO: No has definido el juego. Debes llamar a la herramienta 'update_game_state' indicando el título del juego antes de poder usar esta herramienta."

    # 3. Si todo está bien, continuamos normal
    reco_search.reset()
    response = recommendation_agent.invoke(
        {"messages": [HumanMessage(content=f"Recomienda juegos similares a {title}")]})
    return response['messages'][-1].content


@tool
def get_completion_guide(runtime: ToolRuntime) -> dict:
    """Obtiene tiempos de duración (HLTB) y guías de trofeos/logros."""

    # 1. Usamos .get() en lugar de [] para evitar que Python lance un KeyError
    title = runtime.state.get("full_game_title")

    # 2. Si el título está vacío (None), le mandamos un error a la IA para que lo arregle
    if not title:
        return "ERROR INTERNO: No has definido el juego. Debes llamar a la herramienta 'update_game_state' indicando el título del juego antes de poder usar esta herramienta."

    # 3. Si todo está bien, continuamos normal
    platform = runtime.state.get("platform", "todas las plataformas")
    completion_search.reset()
    response = completion_agent.invoke(
        {"messages": [HumanMessage(content=f"Guía de trofeos y duración para {title} en {platform}")]})
    return response['messages'][-1].content


@tool
def update_game_state(full_game_title: str, platform: str, runtime: ToolRuntime) -> Command:
    """
    Updates the global game state.
    Args:
        full_game_title: The EXACT and COMPLETE name of the game (e.g. 'Elden Ring').
        platform: The specific platform (e.g. 'PC', 'PS5').
    """
    # Si el modelo intenta mandar "No especificado", lanzamos un error que el LLM pueda leer
    if "especificado" in full_game_title.lower() or "especificado" in platform.lower():
        return "ERROR: Debes extraer el nombre real del juego y la plataforma de la pregunta del usuario."

    return Command(
        update={
            "full_game_title": full_game_title,
            "platform": platform,
            "messages": [ToolMessage(content=f"STATE_UPDATED: {full_game_title} ({platform})",
                                     tool_call_id=runtime.tool_call_id)]
        }
    )