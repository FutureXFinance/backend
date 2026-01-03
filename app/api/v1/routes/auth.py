"""
Authentication API Routes
Handles user sign in, sign up, and token management
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, timedelta
import jwt
import bcrypt
from sqlalchemy.orm import Session

from app.database import get_db
from app.config import settings

router = APIRouter()

# JWT Configuration
SECRET_KEY = settings.SECRET_KEY if hasattr(settings, 'SECRET_KEY') else "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


class SignUpRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    company: Optional[str] = None


class TokenResponse(BaseModel):
    token: str
    user: dict


def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


@router.post("/signin", response_model=TokenResponse)
async def sign_in(request: SignInRequest, db: Session = Depends(get_db)):
    """
    Sign in user with email and password
    
    Returns JWT token and user information
    """
    try:
        from app.models.user import User
        
        # Query user from database
        user = db.query(User).filter(User.email == request.email).first()
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Verify password
        if not verify_password(request.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(status_code=403, detail="Account is disabled")
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        # Prepare user data
        user_data = {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "company": user.company,
            "role": user.role
        }
        
        # Create access token
        token = create_access_token(data={"sub": user.email, "user_id": user.id})
        
        return {
            "token": token,
            "user": user_data
        }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")


@router.post("/signup", response_model=TokenResponse)
async def sign_up(request: SignUpRequest, db: Session = Depends(get_db)):
    """
    Register new user
    
    Returns JWT token and user information
    """
    try:
        from app.models.user import User
        
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == request.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Hash password
        hashed_password = hash_password(request.password)
        
        # Create new user
        new_user = User(
            email=request.email,
            password_hash=hashed_password,
            full_name=request.full_name,
            company=request.company,
            role="user",
            is_active=True,
            is_verified=False
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Prepare user data
        user_data = {
            "id": new_user.id,
            "email": new_user.email,
            "full_name": new_user.full_name,
            "company": new_user.company,
            "role": new_user.role
        }
        
        # Create access token
        token = create_access_token(data={"sub": new_user.email, "user_id": new_user.id})
        
        return {
            "token": token,
            "user": user_data
        }
            
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Sign up failed: {str(e)}")


@router.post("/verify-token")
async def verify_token(token: str):
    """
    Verify JWT token
    
    Returns user information if token is valid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        user_id = payload.get("user_id")
        
        if email is None or user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # TODO: Fetch user from database
        return {
            "valid": True,
            "user_id": user_id,
            "email": email
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
