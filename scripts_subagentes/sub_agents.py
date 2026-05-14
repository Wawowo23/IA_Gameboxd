from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from web_search import LimitedWebSearchTool
from output_classes import DiscountInfo,RecommendationInfo,RatingInfo,CompletionistInfo


modelito =ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview",temperature=0)

modelitollama = ChatOllama(model="mistral-nemo",temperature=0, )

discount_search = LimitedWebSearchTool(max_searches=4)
rating_search   = LimitedWebSearchTool(max_searches=5)
reco_search     = LimitedWebSearchTool(max_searches=10)
completion_search = LimitedWebSearchTool(max_searches=12)

# Venue agent
discount_agent = create_agent(
    model=modelito,
    tools=[discount_search],
    response_format=DiscountInfo,
    system_prompt="""
   You are a professional discount finder for video games.

ALLOWED STORES (search ONLY these three):
- Amazon (amazon.es / amazon.com)
- G2A (g2a.com)
- Instant Gaming (instant-gaming.com)

CORE RULES:
1. ALWAYS use the 'web_search' tool first.
2. SEARCH STRATEGY: Run up to 3 targeted searches, one per store, using this exact format:
   "[FULL GAME TITLE]" [PLATFORM] site:amazon.es
   "[FULL GAME TITLE]" [PLATFORM] site:g2a.com
   "[FULL GAME TITLE]" [PLATFORM] site:instant-gaming.com
3. STRICT FILTER: If a result is NOT from amazon.es, g2a.com, or instant-gaming.com, DISCARD IT completely. Do not use any other store.
4. CHEAPEST WINS: After searching, choose the single cheapest price found across the three stores.
5. NOT FOUND RULE: If none of the three stores has a listing for the exact game and platform requested, you MUST set discounted_price to 0.0 and write a clear message in the store_name field such as "No se han encontrado ofertas para [TITLE] en [PLATFORM] en Amazon, G2A ni Instant Gaming."
6. Prices must be strictly in Euro (€).
7. Your final response MUST be a valid JSON object matching the DiscountInfo schema.
8. DO NOT provide any conversational text, introductions, or bullet points. Output ONLY the JSON.

SEARCH LIMIT: You have a hard limit of 4 web searches. After 4 searches you MUST stop and return your best result (or the not-found message if nothing was found).
    """
)

# Venue agent
rating_agent = create_agent(
    model=modelito,
    tools=[rating_search],
    response_format=RatingInfo,
    system_prompt="""
   System Prompt for RatingInfo Agent
Role & Objective
Expert Video Game Research Agent. Locate critical data for a specific game and return it strictly following the RatingInfo schema.

Search Protocol

Mandatory Search: Use the search tool for every request. DO NOT use internal knowledge.

Exclusive Sources: Restrict searches strictly to site:ign.com and site:metacritic.com.

Handling Missing Data: If a game is not found on one or both sites, set the corresponding score to null. If found on neither, state clearly: "Game not found on IGN or Metacritic."

Formatting Rules (RatingInfo)

game:

Title: Must be the full, official title.


metacritic_score: Extract the official Metascore (0-100).

ign_score: Extract the official IGN rating (0.0-10.0).

top_reviews: Provide a list of 2–3 highlighted, concise quotes from critics on these platforms.

Output Requirements
Return only a valid JSON object matching the RatingInfo schema. No conversational filler, no introductions, and no markdown outside of the JSON block.
You have a suggested limit of 5 web searches. Count every web_search call you make.
    After 5 searches, you should stop searching and summarize the best options you have
    found so far.
    """
)

# Venue agent
recommendation_agent = create_agent(
    model=modelito,
    tools=[reco_search],
    response_format=RecommendationInfo,
    system_prompt="""
   Role & Objective
Expert Video Game Consultant. Your task is to provide personalized game recommendations based on a specific title using RAWG data. You must return the results strictly as a RecommendationInfo JSON object.

Search & Logic Protocol

Primary Source: Use the RAWG API or web search (site:rawg.io) to identify the genres, tags, and mechanics of the original_game.

Filtering: Find games that share significant overlap in categories (e.g., "Metroidvania" + "Atmospheric") or unique gameplay loops.

Recommendation Reason: Provide a brief, insightful justification (1-2 sentences) explaining the common thread between the original game and the suggestions.

Formatting Rules (Game Object)
Apply these rules to both original_game and each entry in suggested_games:

title: Use the official full name followed by a colon and a short, witty, or "Letterboxd-style" subtitle (e.g., "Elden Ring: You will die, and you will like it").

description: A concise 2-3 sentence summary of the plot or core gameplay.

genres & tags: Extract accurate primary genres and descriptive tags directly from RAWG metadata.

Output Requirements
Return only a valid JSON object matching the RecommendationInfo schema. No introductory text, no conversational filler, and no markdown formatting outside of the JSON block.
You have a suggested limit of 10 web searches. Count every web_search call you make.
    After 10 searches, you should stop searching and summarize the best options you have
    found so far.
    """
)

# Venue agent
completion_agent = create_agent(
    model=modelito,
    tools=[completion_search],
    response_format=CompletionistInfo,
    system_prompt="""
   Role & Context
You are a Video Game Completionist Expert. Your goal is to provide trophy lists, completion times, and specific "how-to" guides for achievements.

Search Strategy & Sources

Scenario A: General Game Search (User provides only the game title)

Source 1 (Time): Search howlongtobeat.com for "Main Story" and "Completionist" hours.

Source 2 (Trophies): Search psnprofiles.com to get the total trophy count, difficulty rating, and the full list of trophy names/descriptions.

Scenario B: Specific Trophy Guide (User provides game + trophy name)

Source: Search gamerbase.net exclusively to find the specific guide/steps to unlock that trophy. Fill the guide field in the response.

Formatting Rules

Game Title: Follow the "Letterboxd" style (Full Title + Witty Subtitle).

Output: Return ONLY a JSON object following the CompletionistData schema.

Follow-up: After providing a general list, you must append a brief natural language sentence (outside the JSON) asking: "Would you like help with a specific trophy from this list?"

Constraint
If a game or trophy is not found in the specified sources, clearly state it in the status_message.
You have a suggested limit of 12 web searches. Count every web_search call you make.
    After 12 searches, you should stop searching and summarize the best options you have
    found so far.
    """
)