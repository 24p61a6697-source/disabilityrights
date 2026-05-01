from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional
from pathlib import Path
import os

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # ---------------- BASIC ---------------- #
    APP_NAME: str = "Disability Rights & Accessibility Guide"
    APP_VERSION: str = "1.0.0"

    # 🔥 MUST come from env
    SECRET_KEY: str = "your-secret-key-change-in-production"

    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    # ---------------- DATABASE ---------------- #
    DATABASE_URL: str = "sqlite:///./disability_rights.db"

    # ---------------- VECTOR DB ---------------- #
    FAISS_INDEX_PATH: str = str(BACKEND_ROOT / "faiss_index")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # ---------------- LLM ---------------- #
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"
    OLLAMA_TIMEOUT_SECONDS: int = 10  # ✅ realistic

    # ---------------- FEATURES ---------------- #
    RAG_WARMUP_ON_STARTUP: bool = True
    ENABLE_PDF_FALLBACK: bool = False

    TRANSLATION_ENABLED: bool = True

    # ---------------- API KEYS ---------------- #
    OPENAI_API_KEY: Optional[str] = None
    GROK_API_KEY: Optional[str] = None

    # ---------------- RAG CONFIG ---------------- #
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 3

    # ---------------- VALIDATION ---------------- #

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret(cls, v):
        if not v or len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v

    @field_validator("FAISS_INDEX_PATH")
    @classmethod
    def ensure_faiss_path(cls, v):
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @field_validator("TRANSLATION_ENABLED")
    @classmethod
    def validate_translation(cls, v, values):
        # Allow translation via local translation libraries (deep-translator / googletrans)
        # even when LLM API keys are not present. Keep the user's setting as-is.
        return bool(v)

    class Config:
        env_file = ".env"
        extra = "ignore"


# ---------------- LOAD SETTINGS ---------------- #

settings = Settings()