import logging
import os
import re
from openai import OpenAI
import chromadb

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    DISTANCE_THRESHOLD,
    DISTANCE_THRESHOLD_SCOPED,
    FALLBACK_MESSAGE,
    MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPINION_COMPETITOR_MESSAGE,
    TOP_K,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cached Chroma collection reference
_chroma_client = None
_chroma_collection = None

COMPETITORS = {
    "emaar", "damac", "sobha", "aldar", "nakheel", "meraas", "omniyat",
    "binghatti", "danube", "azizi", "roshn", "bayut", "property finder",
    "propertyfinder", "aqarmap", "savills", "knight frank", "cbre", "jll",
}

OPINION_PATTERNS = [
    r"\bbetter\s+than\b",
    r"\balternatives?\b",
    r"\bcompetitors?\b",
    r"\b(?:which\s+(?:developer|real\s+estate)\s+is\s+best|best\s+developer|best\s+real\s+estate)\b",
    r"\bwhich\s+is\s+better\b",
    r"\bshould\s+(?:i|we|anyone|one)\s+buy\b",
    r"\b(?:good|bad|wise|worthwhile)\s+investment\b",
    r"\bworth\s+investing\b",
    r"\brecommend\s+an?\b",
]


def detect_source(query: str) -> str:
    """
    Detects if query targets 'darglobal' or 'wasalt'.
    Case-insensitive, ignoring spaces and hyphens.
    Returns 'darglobal', 'wasalt', or 'all'.
    """
    clean = re.sub(r"[\s\-_]+", "", query.lower())
    has_darglobal = (
        "darglobal" in clean
        or "daralarkan" in clean
        or bool(re.search(r"\bdar[\s\-_]*(?:global|al[\s\-_]*arkan)\b", query, re.I))
    )
    has_wasalt = "wasalt" in clean

    if has_darglobal and has_wasalt:
        return "all"
    elif has_darglobal:
        return "darglobal"
    elif has_wasalt:
        return "wasalt"
    return "all"


def expand_query(query: str) -> str:
    """
    Expands vague or short terms with relevant domain keywords
    to improve embedding retrieval accuracy before querying Chroma.
    """
    q_lower = query.lower()
    words = set(re.findall(r"\b\w+\b", q_lower))
    expansions = []

    rank_terms = {"top", "best", "featured", "popular", "flagship", "main"}
    prop_terms = {
        "properties", "property", "projects", "project",
        "developments", "development", "listings", "listing",
    }
    luxury_terms = {"luxury", "high-end", "highend", "premium", "exclusive"}
    residential_terms = {
        "villas", "villa", "apartments", "apartment", "flats", "flat",
        "houses", "house", "residences", "residence",
    }

    if words & rank_terms:
        expansions.append("featured projects flagship developments")
    if words & prop_terms:
        expansions.append("projects developments residences")
    if words & luxury_terms:
        expansions.append("luxury residential premium")
    if words & residential_terms:
        expansions.append("residential units homes")

    if expansions:
        return f"{query} {' '.join(expansions)}"
    return query


def is_opinion_or_competitor(query: str) -> bool:
    """
    Detects if query is asking for competitor comparisons, opinions, recommendations,
    or financial/investment advice that fall outside verified factual scope.
    Returns True to trigger guardrail response, False otherwise.
    """
    q_lower = query.lower()

    # Direct opinion patterns
    for pattern in OPINION_PATTERNS:
        if re.search(pattern, q_lower):
            return True

    # "vs" or "versus" check
    if re.search(r"\b(?:vs\.?|versus)\b", q_lower):
        if any(c in q_lower for c in COMPETITORS):
            return True
        is_dar_vs_wasalt = bool(re.search(r"\b(?:dar[\s\-_]*(?:global|al[\s\-_]*arkan))\s+(?:vs\.?|versus)\s+wasalt\b", q_lower)) or \
                           bool(re.search(r"\bwasalt\s+(?:vs\.?|versus)\s+(?:dar[\s\-_]*(?:global|al[\s\-_]*arkan))\b", q_lower))
        if not is_dar_vs_wasalt:
            return True

    # Competitor mentions in comparative context
    if any(c in q_lower for c in COMPETITORS):
        if re.search(r"\b(?:compare|compared|or|like)\b", q_lower):
            return True

    return False


def get_collection():
    global _chroma_client, _chroma_collection
    if _chroma_collection is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        try:
            _chroma_collection = _chroma_client.get_collection(CHROMA_COLLECTION_NAME)
        except Exception as e:
            logger.warning("Collection '%s' not found. Creating empty collection. (%s)", CHROMA_COLLECTION_NAME, e)
            _chroma_collection = _chroma_client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
    return _chroma_collection


def retrieve(query: str, source_filter: str = "all") -> tuple[list[str], list[dict], list[float]]:
    """
    Retrieves the top-k most relevant chunks from Chroma collection.
    Filters by source ("wasalt" or "darglobal") unless source_filter == "all".
    Returns (chunks, metadatas, distances).
    """
    collection = get_collection()
    where_filter = None
    if source_filter and source_filter.lower() != "all":
        where_filter = {"source": source_filter.lower()}

    # Check collection size
    count = collection.count()
    if count == 0:
        print(f"[RAG Retrieve] Collection is empty. Returning 0 results.")
        return [], [], []

    n_results = min(TOP_K, count)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
    )

    chunks = results["documents"][0] if results.get("documents") else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else []
    distances = results["distances"][0] if results.get("distances") else []

    best_dist = round(distances[0], 4) if distances else None
    all_dists = [round(d, 4) for d in distances]
    print(
        f"[RAG Retrieve] Query: \"{query}\" | Filter: {source_filter} | "
        f"Found: {len(chunks)} | Best Distance: {best_dist} | Distances: {all_dists}"
    )

    return chunks, metadatas, distances


def generate(
    query: str,
    history: list[dict] | None = None,
    source_filter: str = "all",
) -> dict:
    """
    Answers query using RAG.
    Checks for out-of-scope opinion/competitor queries before retrieval.
    Applies automatic source detection and query expansion.
    Uses DISTANCE_THRESHOLD_SCOPED (0.65) when scoped, or DISTANCE_THRESHOLD (0.50) when unscoped.
    Returns: {"answer": str, "sources": list[str]}
    """
    # 1. Guardrail for out-of-scope opinion, competitor, or financial advice queries
    if is_opinion_or_competitor(query):
        print(f"[RAG Guardrail] Query flagged as opinion/competitor inquiry: \"{query}\". Returning fixed guardrail response.")
        return {
            "answer": OPINION_COMPETITOR_MESSAGE,
            "sources": [],
        }

    # 2. Check if this is a follow-up asking about sources/origin of prior response
    effective_query = query
    META_SOURCE_PHRASES = (
        "where did you get this",
        "where did you find this",
        "where is this from",
        "what are your sources",
        "what is your source",
        "cite your source",
        "where did you get that",
        "source of this information",
    )
    if history and any(p in query.lower() for p in META_SOURCE_PHRASES):
        for turn in reversed(history):
            if turn.get("role") == "user" and turn.get("content"):
                effective_query = turn.get("content")
                logger.info("Using previous user turn as retrieval context for source follow-up: '%s'", effective_query)
                break

    # 3. Automatic source detection (sidebar filter, when set explicitly, still wins)
    if not source_filter or source_filter.lower() == "all":
        effective_source = detect_source(query)
    else:
        effective_source = source_filter.lower()

    # 4. Scoped vs unscoped distance threshold
    is_scoped = effective_source in ("darglobal", "wasalt")
    threshold = DISTANCE_THRESHOLD_SCOPED if is_scoped else DISTANCE_THRESHOLD
    print(
        f"[RAG] Query: \"{query}\" | Source Filter: {effective_source} (scoped={is_scoped}) | "
        f"Using distance threshold: {threshold}"
    )

    # 5. Query expansion for retrieval
    retrieval_query = expand_query(effective_query)
    chunks, metadatas, distances = retrieve(retrieval_query, source_filter=effective_source)

    # 6. Rejection gate: no results or distance above threshold
    if not distances or distances[0] > threshold:
        best_d = round(distances[0], 4) if distances else "None"
        print(
            f"[RAG Fallback] Best distance ({best_d}) exceeds threshold ({threshold}) "
            f"or no results. Returning fallback without calling LLM."
        )
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": [],
        }

    # Extract unique source URLs
    unique_sources = []
    for meta in metadatas:
        url = meta.get("url")
        if url and url not in unique_sources:
            unique_sources.append(url)

    # Build context string
    context_blocks = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas), 1):
        title = meta.get("title", "Property Listing")
        url = meta.get("url", "")
        src = meta.get("source", "").capitalize()
        context_blocks.append(f"[{src} Document {i}] Title: {title}\nURL: {url}\nContent:\n{chunk}")

    context_str = "\n\n".join(context_blocks)

    # System prompt enforcing strict grounding
    system_prompt = (
        "You are the DarGlobal & Wasalt AI Property Assistant.\n"
        "Your task is to answer user property inquiries using ONLY the provided verified context.\n\n"
        "STRICT INSTRUCTIONS:\n"
        "1. Answer ONLY from the facts provided in the context below.\n"
        "2. NEVER invent, hallucinate, or assume prices, locations, square footage, amenities, or property details. "
        "If specific pricing is not provided in the context, explicitly state that verified price information is not available in the sources.\n"
        "3. Cite specific sources and URLs from the context when discussing properties.\n"
        f"4. If the context does not contain sufficient verified information to answer the question, output EXACTLY:\n"
        f"   \"{FALLBACK_MESSAGE}\"\n"
        "5. The sources do not rank properties. If the user asks for top or best properties, present the main projects described in the context as 'key projects' and say the sources do not provide an official ranking. Do not invent rankings, prices, or sales figures.\n"
        "6. Never give financial or investment advice, never criticize or rank companies, and never state facts about other companies.\n"
        "7. Keep your tone professional, concise, and structured."
    )

    # Format conversation messages
    messages = [{"role": "system", "content": system_prompt}]

    # Include up to the last 6 conversation turns
    if history:
        for turn in history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # Add current user turn with context
    current_prompt = (
        f"CONTEXT INFORMATION:\n{context_str}\n\n"
        f"USER QUESTION: {query}\n\n"
        "Please provide a factual answer based exclusively on the context above. Cite the relevant source URL(s)."
    )
    messages.append({"role": "user", "content": current_prompt})

    # Call OpenRouter LLM via OpenAI SDK
    api_key = OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        msg = (
            "OpenRouter API key is not configured. Please set OPENROUTER_API_KEY in your .env file to enable AI responses.\n\n"
            f"**Retrieved Context ({len(chunks)} chunks):**\n" + "\n\n".join(f"- {c[:250]}..." for c in chunks[:3])
        )
        return {
            "answer": msg,
            "sources": unique_sources,
        }

    import time
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        timeout=30.0,
    )

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2,
            )
            answer = response.choices[0].message.content.strip()
            return {
                "answer": answer,
                "sources": unique_sources,
            }
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            if attempt == 0:
                logger.warning("OpenRouter %s encountered (%s). Retrying once after 3 seconds...", type(e).__name__, e)
                time.sleep(3)
                continue
            else:
                logger.error("OpenRouter request failed after retry: %s", e)
                friendly_msg = (
                    "The AI service is currently experiencing high demand or a temporary connection timeout. "
                    "Please try your request again in a few moments."
                )
                return {
                    "answer": friendly_msg,
                    "sources": unique_sources,
                }
        except Exception as e:
            logger.error("Error calling OpenRouter API: %s", e)
            return {
                "answer": f"Unable to generate response due to AI service error: {e}",
                "sources": unique_sources,
            }


if __name__ == "__main__":
    print("\n--- Testing RAG Retrieval & Relevance Filtering ---")
    test_queries = [
        ("What villas are available for sale in Riyadh?", "all"),
        ("Are there apartments for rent in Jeddah?", "wasalt"),
        ("Tell me about quantum physics in black holes", "all"),
    ]

    for q, f in test_queries:
        print(f"\nQuery: '{q}' (filter: {f})")
        chunks, metas, dists = retrieve(q, source_filter=f)
        if dists and dists[0] <= DISTANCE_THRESHOLD:
            print(f"  -> ACCEPTED (Best distance {dists[0]:.4f} <= {DISTANCE_THRESHOLD})")
            print(f"  -> Top result title: {metas[0].get('title')}")
            print(f"  -> Top result url:   {metas[0].get('url')}")
        else:
            best_d = dists[0] if dists else "No results"
            print(f"  -> REJECTED (Best distance {best_d} > {DISTANCE_THRESHOLD}) -> Fallback would be returned.")
