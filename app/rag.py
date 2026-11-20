import logging
import os
from openai import OpenAI
import chromadb

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    DISTANCE_THRESHOLD,
    FALLBACK_MESSAGE,
    MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    TOP_K,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Cached Chroma collection reference
_chroma_client = None
_chroma_collection = None


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
    If no result or best distance > DISTANCE_THRESHOLD, returns fallback without calling LLM.
    Otherwise builds prompt from chunks and last 6 history turns and calls OpenRouter via OpenAI SDK.
    Returns: {"answer": str, "sources": list[str]}
    """
    # Check if this is a follow-up asking about sources/origin of prior response
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

    chunks, metadatas, distances = retrieve(effective_query, source_filter=source_filter)

    # Rejection gate: no results or distance above threshold
    if not distances or distances[0] > DISTANCE_THRESHOLD:
        best_d = round(distances[0], 4) if distances else "None"
        print(
            f"[RAG Fallback] Best distance ({best_d}) exceeds threshold ({DISTANCE_THRESHOLD}) "
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
        "2. NEVER invent, hallucinate, or assume prices, locations, square footage, amenities, or property details.\n"
        "3. Cite specific sources and URLs from the context when discussing properties.\n"
        f"4. If the context does not contain sufficient verified information to answer the question, output EXACTLY:\n"
        f"   \"{FALLBACK_MESSAGE}\"\n"
        "5. Keep your tone professional, concise, and structured."
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
