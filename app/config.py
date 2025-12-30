from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

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
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:3000", "https://futurexfinance.vercel.app"]
    
    # Storage
    STORAGE_BUCKET: str = os.getenv("STORAGE_BUCKET", "futurex-assets")
    
    class Config:
        case_sensitive = True

settings = Settings()
