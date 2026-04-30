from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    APP_NAME: str = "Disability Rights & Accessibility Guide"
    APP_VERSION: str = "1.0.0"
    SECRET_KEY: str = "disability-rights-india-gov-secret-2024"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    DATABASE_URL: str = "sqlite:///./disability_rights.db"

    FAISS_INDEX_PATH: str = str(BACKEND_ROOT / "faiss_index")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"
    OLLAMA_TIMEOUT_SECONDS: int = 2

    # Keep initial server startup instant; warmup can be enabled explicitly via env.
    RAG_WARMUP_ON_STARTUP: bool = False

    # Keep chat responsive if heavyweight document parsing would slow down fallback path.
    ENABLE_PDF_FALLBACK: bool = False

    # Translation calls can be slow/unreliable on poor networks; make it configurable.
    # Enable translations by default so regional language responses are returned.
    TRANSLATION_ENABLED: bool = True

    OPENAI_API_KEY: Optional[str] = None
    GROK_API_KEY: Optional[str] = None

    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 3

    class Config:
        env_file = ".env"

settings = Settings()
