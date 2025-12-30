from pydantic_settings import BaseSettings
from pydantic import field_validator
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "FutureX Finance API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key-for-dev")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./futurex.db")
    MONGODB_URL: str = os.getenv("MONGODB_URL", "")
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    
    
    # CORS - using different field name to avoid Pydantic env parsing
    # Will be populated from CORS_ORIGINS env var after instantiation
    cors_allowed_origins: List[str] = []
    
    # Storage
    STORAGE_BUCKET: str = os.getenv("STORAGE_BUCKET", "futurex-assets")
    
    class Config:
        case_sensitive = True

# Create settings instance
settings = Settings()

# Manually parse CORS_ORIGINS from environment and set to cors_allowed_origins
_cors_raw = os.getenv("CORS_ORIGINS", "")
if _cors_raw and _cors_raw.strip():
    if _cors_raw.startswith('['):
        try:
            import json
            settings.cors_allowed_origins = json.loads(_cors_raw)
        except:
            settings.cors_allowed_origins = [origin.strip() for origin in _cors_raw.split(',') if origin.strip()]
    else:
        settings.cors_allowed_origins = [origin.strip() for origin in _cors_raw.split(',') if origin.strip()]
else:
    settings.cors_allowed_origins = ["http://localhost:3000", "http://localhost:3001"]

