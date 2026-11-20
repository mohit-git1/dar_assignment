# DarGlobal & Wasalt AI Property Assistant

A Retrieval-Augmented Generation (RAG) assistant over public property data from DarGlobal and Wasalt.

## Architecture & Stack
- **Python 3.11**
- **Web Scraping:** `requests`, `beautifulsoup4`, `lxml`
- **Vector Store & Embeddings:** `chromadb.PersistentClient` using default built-in ONNX runtime MiniLM-L6-v2 embedder (zero PyTorch or sentence-transformers dependencies)
- **RAG & LLM:** OpenAI Python SDK configured for OpenRouter (`https://openrouter.ai/api/v1`, model `openrouter/free`)
- **User Interface:** Streamlit interactive chat interface with source citations and platform filters

---

## Data Collection
- **Wasalt (`https://wasalt.sa/en`)**: Public listing and property category pages were scraped directly within `robots.txt` guidelines. The scraper respects rate limits (1.5-second sleep interval), extracts page titles, headings, listing specifications, and paragraphs, and handles Next.js hydration data when needed. Sensitive, login-protected, or administrative endpoints are strictly excluded.
- **DarGlobal (`https://www.darglobal.co.uk`)**: DarGlobal's website is protected by Imperva Incapsula bot protection (WAF) which returns an anti-bot JavaScript challenge to automated HTTP clients. To respect site protections without attempting intrusive bypasses, public pages from DarGlobal are collected manually and stored under `data/manual/darglobal/`.

### Adding Manual DarGlobal Pages
Place plain `.txt` files in `data/manual/darglobal/` using this format:
- **Line 1:** `URL: <page_url>`
- **Line 2:** `TITLE: <page_title>`
- **Line 3+:** Readable public text content from the page (descriptions, amenities, unit types, etc.)

Files under 200 characters are automatically skipped. Ingestion (`python -m app.ingest`) is idempotent and parses all `.txt` files in `data/manual/darglobal/`, updates `data/raw/darglobal.json`, and rebuilds the vector index.

---

## Setup & Running

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env` and set your OpenRouter API key:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 3. Scraping & Ingestion
```bash
# Run web scraper (respects robots.txt and logs stats)
python -m app.scraper

# Ingest scraped data and manual DarGlobal files into ChromaDB
python -m app.ingest
```

### 4. Testing RAG Retrieval
```bash
python -m app.rag
```

### 5. Launch the Streamlit Chat UI
```bash
streamlit run app/chat.py
```
