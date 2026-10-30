
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import math
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langgraph.types import interrupt, Command


# Load environment variables
load_dotenv()


# ---------------------------------------------------------
# LLM: vLLM OpenAI-compatible endpoint
# ---------------------------------------------------------
llm = ChatOpenAI(
    model=os.getenv("VLLM_MODEL", "qwen3.8-27b"),
    base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
    temperature=0.7,
    timeout=90,
    max_retries=1,
)


# ---------------------------------------------------------
# Embeddings: local Ollama
# ---------------------------------------------------------
embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)


# ---------------------------------------------------------
# PDF RAG
# ---------------------------------------------------------
def _rag_db_path(thread_id: str) -> str:
    """Return an isolated FAISS directory for one conversation."""
    safe_thread_id = "".join(
        char for char in str(thread_id) if char.isalnum() or char in ("-", "_")
    )
    if not safe_thread_id:
        raise ValueError("A valid conversation thread ID is required for PDF RAG.")
    return os.path.join("faiss_db", safe_thread_id)


def has_rag_index(thread_id: str) -> bool:
    """Check whether this conversation has uploaded/indexed a PDF."""
    db_path = _rag_db_path(thread_id)
    return (
        os.path.isfile(os.path.join(db_path, "index.faiss"))
        and os.path.isfile(os.path.join(db_path, "index.pkl"))
    )


def ingest_rag_document(file_path, thread_id: str):
    """Create embeddings and a FAISS index isolated to the current chat."""
    DB_PATH = _rag_db_path(thread_id)
    os.makedirs(DB_PATH, exist_ok=True)

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    if not docs:
        raise ValueError("The uploaded PDF did not contain readable pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No readable text was extracted from the uploaded PDF.")

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)
    return len(chunks)


def get_retriever(thread_id: str):
    """Load the current conversation's PDF index only."""
    DB_PATH = _rag_db_path(thread_id)

    if not has_rag_index(thread_id):
        return None

    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )


# ---------------------------------------------------------
# RAG tool
# ---------------------------------------------------------
@tool
def rag_tool(query: str, config: RunnableConfig) -> str:
    """
    Retrieve information only from PDFs uploaded in the current conversation.

    Use this tool for questions about an uploaded PDF/document. Each chat has
    its own FAISS index, so documents from other conversations are not used.

    Args:
        query: The question or search query used to retrieve PDF content.
        config: LangGraph runtime configuration used to identify the chat.
    """
    thread_id = (config or {}).get("configurable", {}).get("thread_id")

    if not thread_id:
        return (
            "I can't search a PDF because this conversation has no valid "
            "thread ID. Please start a new chat and upload the PDF again."
        )

    if not has_rag_index(thread_id):
        return (
            "I can't answer this from a PDF yet because no PDF has been "
            "uploaded and indexed in this chat. Please attach the relevant "
            "PDF using the + button in the chat input, wait for the upload "
            "confirmation, and ask your question again. You can still ask "
            "general questions, request web searches, or ask about weather "
            "without uploading a PDF."
        )

    retriever = get_retriever(thread_id)
    if retriever is None:
        return (
            "No PDF index is available for this chat. Please upload the "
            "relevant PDF again and wait for processing to finish."
        )

    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF uploaded to this chat."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)


# ---------------------------------------------------------
# Tools
# ---------------------------------------------------------

# Tavily search integration
search_tool = TavilySearch(
    name="tavily_search",
    max_results=5,
    topic="general",
    search_depth="advanced",
)


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """
    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given symbol using Alpha Vantage.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        return {"error": "ALPHA_VANTAGE_API_KEY is missing from .env"}

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Stock API request failed: {str(e)}"}


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, interrupt and wait for a
    human decision ("yes" / anything else).
    """
    decision = interrupt(
        f"Approve buying {quantity} shares of {symbol}? (yes/no)"
    )

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }

    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }



@tool
def get_current_weather(location: str) -> str:
    """
    Get the current real-time weather for a given city or location.

    Args:
        location: City or location name, for example:
                  "Ahmedabad", "London, UK", or "New York, US".
    """
    try:
        # Step 1: Resolve city name to coordinates using Open-Meteo
        geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"

        geocoding_params = {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json",
        }

        geo_response = requests.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10,
        )
        geo_response.raise_for_status()

        locations = geo_response.json().get("results", [])

        if not locations:
            return f"Could not find the location: {location}"

        place = locations[0]
        latitude = place["latitude"]
        longitude = place["longitude"]
        resolved_name = place.get("name", location)
        country = place.get("country", "")
        admin1 = place.get("admin1", "")

        # Step 2: Get current weather without an API key
        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "apparent_temperature,pressure_msl,"
                "weather_code,wind_speed_10m,visibility"
            ),
            "timezone": "auto",
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10,
        )
        weather_response.raise_for_status()

        weather_data = weather_response.json()
        current = weather_data["current"]

        temperature = current["temperature_2m"]
        feels_like = current["apparent_temperature"]
        humidity = current["relative_humidity_2m"]
        pressure = current.get("pressure_msl", "N/A")
        wind_speed = current.get("wind_speed_10m", "N/A")
        visibility_meters = current.get("visibility")

        visibility_km = (
            round(visibility_meters / 1000, 1)
            if visibility_meters is not None
            else "N/A"
        )

        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snowfall",
            73: "Moderate snowfall",
            75: "Heavy snowfall",
            80: "Rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
            96: "Thunderstorm with hail",
            99: "Thunderstorm with heavy hail",
        }

        weather_code = current.get("weather_code")
        description = weather_codes.get(
            weather_code, "Unknown conditions"
        )

        location_parts = [resolved_name]

        if admin1:
            location_parts.append(admin1)

        if country:
            location_parts.append(country)

        display_location = ", ".join(location_parts)

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {description}\n"
            f"- Temperature: {temperature}°C\n"
            f"- Feels like: {feels_like}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- Wind speed: {wind_speed} km/h\n"
            f"- Visibility: {visibility_km} km"
        )

    except requests.Timeout:
        return "The weather service request timed out. Please try again."

    except requests.HTTPError as error:
        status_code = (
            error.response.status_code if error.response else "unknown"
        )
        return f"Weather API returned an HTTP error: {status_code}"

    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"

    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"



# ---------------------------------------------------------
# Tool list — original tool workflow retained
# ---------------------------------------------------------
tools = [
    search_tool,
    calculator,
    get_stock_price,
    get_current_weather,
    rag_tool,
    purchase_stock,
]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------
# State
# ---------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------
# Node 1: Chat node
# ---------------------------------------------------------

def _get_deterministic_tool_call(user_text: str):
    """Return a tool call for intents that should not depend on model tool selection.

    This is a small routing layer in front of the LLM. It keeps the existing
    LangGraph ToolNode workflow, but ensures that obvious tool requests (for
    example current prices, latest news, arithmetic, weather, and PDF questions)
    actually execute the matching tool even when the model does not emit a
    native tool call.
    """
    import re
    import uuid

    text = user_text.strip()
    lower = text.lower()
    if not text:
        return None

    def call(name: str, args: dict):
        return {
            "name": name,
            "args": args,
            "id": str(uuid.uuid4()),
            "type": "tool_call",
        }

    # Current stock quotes are different from retail/product prices.
    stock_terms = ("stock price", "share price", "stock quote", "market price of", "ticker price")
    if any(term in lower for term in stock_terms):
        match = re.search(r"\b(?:for|of|ticker|symbol)\s+([A-Z]{1,6})\b", text)
        if not match:
            match = re.search(r"\b([A-Z]{1,6})\b", text)
        if match:
            return call("get_stock_price", {"symbol": match.group(1).upper()})

    # Current weather must be retrieved, not answered from model memory.
    weather_terms = ("weather", "temperature in", "is it raining", "rain in", "forecast in")
    if any(term in lower for term in weather_terms):
        match = re.search(r"\b(?:in|at|for|near)\s+([A-Za-z][A-Za-z .,'-]{1,60}?)(?:[?.!,]|$)", text, re.IGNORECASE)
        location = match.group(1).strip() if match else "Ahmedabad"
        location = re.sub(r"\b(today|now|currently|right now|please|weather)\b", "", location, flags=re.IGNORECASE).strip(" ,.-")
        if location:
            return call("get_current_weather", {"location": location})

    # Web search for recent/current or changing information, including retail prices.
    search_terms = (
        "latest", "current price", "today's price", "today price", "price of",
        "latest price", "latest news", "recent news", "today's news", "current news",
        "search the web", "search online", "official documentation", "recent reviews",
        "release date", "available now", "current specifications", "latest specifications",
        "compare prices", "source urls", "verify this", "what happened today",
    )
    product_or_news = any(term in lower for term in search_terms)
    if product_or_news:
        return call("tavily_search", {"query": text})

    # PDF/document questions should use the indexed document retriever.
    document_terms = (
        "uploaded pdf", "uploaded document", "in the pdf", "in this pdf",
        "from the pdf", "according to the document", "in the document",
        "uploaded file", "in the uploaded", "csm ppt", "this document",
    )
    if any(term in lower for term in document_terms):
        return call("rag_tool", {"query": text})

    # Stock purchase must go through the existing interrupt/HITL workflow.
    if any(term in lower for term in ("buy shares", "purchase shares", "buy stock", "purchase stock", "buy stocks")):
        symbol_match = re.search(r"\b(?:of|in|for)\s+([A-Z]{1,6})\b", text)
        quantity_match = re.search(r"\b(\d+)\s+(?:shares|stocks)\b", lower)
        if symbol_match and quantity_match:
            return call("purchase_stock", {"symbol": symbol_match.group(1).upper(), "quantity": int(quantity_match.group(1))})

    # Arithmetic expressions / explicit calculation requests use the calculator.
    math_request = any(term in lower for term in (
        "calculate", "calculator", "divided by", "multiply", "multiplied by",
        "percentage of", "percent of", "square root", "what is ", "what's ",
    ))
    has_math_symbols = bool(re.search(r"\d\s*[+*/^%]\s*\d", text))
    if math_request or has_math_symbols:
        # Handle natural-language arithmetic first so trailing instructions
        # such as "verify using the calculator tool" are not passed to eval.
        division = re.search(r"(?i)([\d,.]+)\s+divided by\s+([\d,.]+)", text)
        if division:
            left = division.group(1).replace(",", "")
            right = division.group(2).replace(",", "")
            return call("calculator", {"expression": f"{left} / {right}"})

        multiplication = re.search(r"(?i)([\d,.]+)\s+(?:multiplied by|times)\s+([\d,.]+)", text)
        if multiplication:
            left = multiplication.group(1).replace(",", "")
            right = multiplication.group(2).replace(",", "")
            operator = "*"
            return call("calculator", {"expression": f"{left} {operator} {right}"})

        expression = text
        expression = re.sub(r"(?i)^.*?\b(?:calculate|calculator|what is|what's|compute)\b\s*", "", expression).strip(" ?")
        expression = re.sub(r"(?i)\s+times\s+", " * ", expression)
        # Extract only a math expression, ignoring explanatory text after it.
        match = re.search(r"[0-9][0-9,.]*(?:\s*[+*/%^-]\s*[0-9][0-9,.]*)+", expression)
        if match:
            safe_expression = match.group(0).replace(",", "")
            return call("calculator", {"expression": safe_expression})

    return None


def chat_node(state: ChatState):
    """Use deterministic routing for clear tool intents, then let the LLM answer."""

    # Only inspect a fresh user message. After a tool runs, the latest message
    # is a ToolMessage and the LLM must interpret the tool result instead.
    latest_message = state["messages"][-1] if state.get("messages") else None
    if isinstance(latest_message, HumanMessage):
        forced_tool_call = _get_deterministic_tool_call(str(latest_message.content))
        if forced_tool_call is not None:
            return {"messages": [AIMessage(content="", tool_calls=[forced_tool_call])]}

    system_message = SystemMessage(
        content="""
You are a helpful Agentic Chatbot with access to tools.

Your primary responsibility is to understand the user's intent and
automatically use the appropriate tool whenever it is needed.

TOOL SELECTION RULES — FOLLOW THESE STRICTLY:

1. WEB SEARCH (search_tool / Tavily):
   You MUST use search_tool before answering questions involving:
   - Latest or current prices of products, including phones, laptops,
     cars, electronics, and other goods.
   - Latest news, current events, recent announcements, and rumors.
   - Current product availability, release dates, specifications,
     comparisons, offers, and market information.
   - Latest official documentation, recent reviews, or information
     that may have changed.
   - Requests asking to search the web, find sources, or verify claims.

   Examples:
   - "What is the latest iPhone 18 Pro Max price?"
     -> Use search_tool.
   - "What are today's NVIDIA DGX Spark specifications?"
     -> Use search_tool.
   - "What is the latest technology news?"
     -> Use search_tool.
   - "Compare current prices of two laptops."
     -> Use search_tool.

   Do not answer these questions from memory alone.
   Do not ask the user to explicitly name the tool.
   Do not claim information is current without checking it.

2. CALCULATOR:
   Use calculator for arithmetic, division, percentages, financial
   calculations, and other calculations that can be verified numerically.
   Do not rely on mental arithmetic when the calculator is available.

3. STOCK PRICE:
   Use get_stock_price for current stock prices and stock-market quotes.
   A product's retail price is NOT a stock price.
   For example, an iPhone price requires search_tool, whereas Apple's
   AAPL share price requires get_stock_price.

4. WEATHER:
   Use get_current_weather for current weather conditions at a location.

5. PDF / DOCUMENT QUESTIONS:
   Use rag_tool whenever the user asks about an uploaded PDF/document or asks a question
   that clearly depends on a document uploaded in this conversation.
   Retrieve relevant content before answering and do not invent document content.
   If no PDF has been uploaded and indexed in this chat, tell the user immediately
   that a PDF is required and ask them to upload it. Never retry PDF retrieval repeatedly.

6. STOCK PURCHASE:
   Use purchase_stock only when the user requests a stock purchase.
   Follow the existing human-in-the-loop approval process.
   Never claim a real trade was executed if the tool only simulates it.

7. GENERAL QUESTIONS:
   Answer directly when no external tool or document retrieval is needed.
   Do not call tools unnecessarily for simple greetings, writing,
   explanations, or stable general knowledge.

AFTER TOOL EXECUTION:
- Read the tool result carefully.
- Give one clear, complete answer.
- Summarize useful findings instead of merely repeating raw tool output.
- Include source URLs when web search results provide them.
- Clearly state when current information could not be verified.
- Never invent search results, prices, citations, or tool outputs.
- Do not tell the user to request a tool manually when a tool is appropriate.

IMPORTANT:
Choose tools based on the meaning and intent of the question, not merely
whether the user explicitly mentions a tool.
"""
    )

    messages = [
        system_message,
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}



# ---------------------------------------------------------
# Node 2: Tool node
# ---------------------------------------------------------
tool_node = ToolNode(tools)


# ---------------------------------------------------------
# Checkpointer — same database file
# ---------------------------------------------------------
conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False,
)
checkpoint = SqliteSaver(conn)


# ---------------------------------------------------------
# Graph — same nodes and edges
# ---------------------------------------------------------
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)


# ---------------------------------------------------------
# Helper functions for Streamlit frontend
# ---------------------------------------------------------
def get_all_threads():
    all_threads = set()

    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config["configurable"]["thread_id"])

    return list(all_threads)
