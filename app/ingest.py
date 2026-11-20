import json
import logging
import re
from pathlib import Path

import chromadb

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DARGLOBAL_RAW_JSON,
    MANUAL_DARGLOBAL_DIR,
    WASALT_RAW_JSON,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def sync_manual_darglobal_files() -> list[dict]:
    """
    Reads data/manual/darglobal/*.txt.
    Format:
      Line 1: "URL: ..."
      Line 2: "TITLE: ..."
      Line 3+: page text
    Skips files under 200 characters.
    Writes/updates data/raw/darglobal.json and returns the loaded items.
    """
    MANUAL_DARGLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    txt_files = list(MANUAL_DARGLOBAL_DIR.glob("*.txt"))
    logger.info("Found %d manual DarGlobal .txt file(s) in %s", len(txt_files), MANUAL_DARGLOBAL_DIR)

    darglobal_docs = []
    for filepath in txt_files:
        if filepath.name.lower() in ("readme.txt", "readme.md"):
            continue

        try:
            content = filepath.read_text(encoding="utf-8").strip()
            if len(content) < 200:
                logger.warning("Skipping %s: total character count (%d) is under 200", filepath.name, len(content))
                continue

            lines = content.splitlines()
            if len(lines) < 3:
                logger.warning("Skipping %s: file has fewer than 3 lines", filepath.name)
                continue

            url_line = lines[0].strip()
            title_line = lines[1].strip()

            if not url_line.upper().startswith("URL:") or not title_line.upper().startswith("TITLE:"):
                logger.warning("Skipping %s: must start with 'URL: ...' on line 1 and 'TITLE: ...' on line 2", filepath.name)
                continue

            url = url_line.split(":", 1)[1].strip()
            title = title_line.split(":", 1)[1].strip()
            body_text = "\n".join(lines[2:]).strip()

            if len(body_text) < 200:
                logger.warning("Skipping %s: body text is under 200 characters", filepath.name)
                continue

            darglobal_docs.append({
                "source": "darglobal",
                "url": url,
                "title": title or url,
                "text": body_text,
            })
            logger.info("Loaded manual page: %s (%s) - %d chars", title, filepath.name, len(body_text))
        except Exception as e:
            logger.error("Error reading manual file %s: %s", filepath, e)

    # Save to data/raw/darglobal.json if manual files are provided
    if darglobal_docs:
        DARGLOBAL_RAW_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(DARGLOBAL_RAW_JSON, "w", encoding="utf-8") as f:
            json.dump(darglobal_docs, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d DarGlobal manual page(s) to %s", len(darglobal_docs), DARGLOBAL_RAW_JSON)
    else:
        # If no valid manual files exist yet, ensure darglobal.json is written as empty list
        with open(DARGLOBAL_RAW_JSON, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        darglobal_docs = []

    return darglobal_docs


def load_raw_json(json_path: Path, source_name: str) -> list[dict]:
    """Loads raw json file if it exists, returning a list of dicts."""
    if not json_path.exists():
        logger.warning("Raw data file not found: %s", json_path)
        return []

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                logger.info("Loaded %d raw pages for source '%s' from %s", len(data), source_name, json_path)
                return data
            else:
                logger.warning("Expected list in %s, got %s", json_path, type(data))
                return []
    except Exception as e:
        logger.error("Error loading %s: %s", json_path, e)
        return []


def clean_text(text: str) -> str:
    """Normalizes whitespace and cleans up formatting."""
    # Replace multiple spaces/tabs with single space
    cleaned = re.sub(r"[ \t]+", " ", text)
    # Replace 3 or more newlines with double newline
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def chunk_document(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Chunks text by paragraph first, then to about 800 chars with 100 overlap.
    """
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    current_chunk = ""

    for p in paragraphs:
        # If single paragraph exceeds chunk_size, split by sliding window
        if len(p) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            start = 0
            while start < len(p):
                end = start + chunk_size
                sub = p[start:end].strip()
                if sub:
                    chunks.append(sub)
                start += (chunk_size - chunk_overlap)
            continue

        # Check if appending this paragraph exceeds chunk_size
        if current_chunk and (len(current_chunk) + len(p) + 2 > chunk_size):
            chunks.append(current_chunk.strip())
            overlap = current_chunk[-chunk_overlap:].strip() if len(current_chunk) > chunk_overlap else ""
            current_chunk = (overlap + "\n\n" + p).strip() if overlap else p
        else:
            current_chunk = (current_chunk + "\n\n" + p).strip() if current_chunk else p

    if current_chunk and current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [c for c in chunks if len(c) >= 50]


def ingest_data():
    print("\n" + "=" * 60)
    print("STARTING INGESTION PIPELINE")
    print("=" * 60)

    # 1. Sync manual DarGlobal files first
    darglobal_docs = sync_manual_darglobal_files()

    # 2. Load Wasalt data
    wasalt_docs = load_raw_json(WASALT_RAW_JSON, "wasalt")

    all_raw_docs = darglobal_docs + wasalt_docs
    logger.info("Total raw documents collected: %d", len(all_raw_docs))

    # 3. Clean whitespace and drop documents under 200 characters
    cleaned_docs = []
    for doc in all_raw_docs:
        raw_text = doc.get("text", "")
        cleaned = clean_text(raw_text)
        if len(cleaned) < 200:
            logger.info("Dropping page from %s (length %d < 200): %s", doc.get("source"), len(cleaned), doc.get("url"))
            continue
        cleaned_docs.append({
            "source": doc.get("source", "unknown"),
            "url": doc.get("url", ""),
            "title": doc.get("title", ""),
            "text": cleaned,
        })

    logger.info("Documents after cleaning and length filtering (>= 200 chars): %d", len(cleaned_docs))

    # 4. Chunk documents
    chunk_counts_by_source = {"darglobal": 0, "wasalt": 0}
    all_chunks = []
    all_metadatas = []
    all_ids = []

    for doc_idx, doc in enumerate(cleaned_docs):
        source = doc["source"]
        chunks = chunk_document(doc["text"], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{source}_{doc_idx}_{chunk_idx}"
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": source,
                "url": doc["url"],
                "title": doc["title"],
            })
            all_ids.append(chunk_id)
            chunk_counts_by_source[source] = chunk_counts_by_source.get(source, 0) + 1

    # 5. Add to Chroma collection (idempotent: recreate collection)
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
        logger.info("Deleted existing Chroma collection '%s'", CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Created clean Chroma collection '%s' with cosine similarity", CHROMA_COLLECTION_NAME)

    # Batch add chunks to Chroma
    BATCH_SIZE = 100
    total_added = len(all_chunks)
    for i in range(0, total_added, BATCH_SIZE):
        batch_docs = all_chunks[i : i + BATCH_SIZE]
        batch_meta = all_metadatas[i : i + BATCH_SIZE]
        batch_ids = all_ids[i : i + BATCH_SIZE]
        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids,
        )

    # Print summary statistics
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Collection Name: {CHROMA_COLLECTION_NAME}")
    print(f"Storage Path:    {CHROMA_DB_DIR}")
    print(f"Total Chunks:    {total_added}")
    print("\nChunk Counts per Source:")
    for src, count in sorted(chunk_counts_by_source.items()):
        print(f"  - {src}: {count} chunks")
    print("=" * 60 + "\n")

    return chunk_counts_by_source


if __name__ == "__main__":
    ingest_data()
