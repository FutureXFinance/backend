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
    
    # CORS - will be parsed from comma-separated string
    CORS_ORIGINS: List[str] = []
    
    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        # If None or empty, return defaults
        if v is None or v == [] or v == '':
            return ["http://localhost:3000", "http://localhost:3001"]
        # If string, parse comma-separated values
        if isinstance(v, str):
            origins = [origin.strip() for origin in v.split(',') if origin.strip()]
            return origins if origins else ["http://localhost:3000", "http://localhost:3001"]
        # If already a list, return it
        if isinstance(v, list):
            return v if v else ["http://localhost:3000", "http://localhost:3001"]
        # Fallback to defaults
        return ["http://localhost:3000", "http://localhost:3001"]
    
    # Storage
    STORAGE_BUCKET: str = os.getenv("STORAGE_BUCKET", "futurex-assets")
    
    class Config:
        case_sensitive = True

settings = Settings()
