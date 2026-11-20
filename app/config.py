import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

DARGLOBAL_RAW_JSON = RAW_DATA_DIR / "darglobal.json"
WASALT_RAW_JSON = RAW_DATA_DIR / "wasalt.json"
MANUAL_DARGLOBAL_DIR = DATA_DIR / "manual" / "darglobal"

# Load environment variables from .env
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Retrieval & Ingestion constants
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
TOP_K = int(os.getenv("TOP_K", 5))

# Distance threshold for relevance (cosine distance: relevant <= 0.42, unrelated >= 0.85)
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", 0.50))
DISTANCE_THRESHOLD_SCOPED = float(os.getenv("DISTANCE_THRESHOLD_SCOPED", 0.65))

# Guardrail response for out-of-scope opinion and comparison queries
OPINION_COMPETITOR_MESSAGE = (
    "I can only answer questions about DarGlobal and Wasalt using the public information I've collected, "
    "so I can't compare them with other companies or recommend alternatives. "
    "I can tell you about DarGlobal's projects, Wasalt's property listings, locations, amenities, "
    "or how the two companies differ. What would you like to know?"
)

# Model & API configuration
MODEL = os.getenv("MODEL", "openrouter/free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Fallback message
FALLBACK_MESSAGE = "I couldn't find verified information about that in the collected sources."

# Chroma collection name
CHROMA_COLLECTION_NAME = "properties"
