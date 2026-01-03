"""
Seed demo user for authentication testing
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models.user import User
import bcrypt


def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def seed_demo_user():
    """Create demo user for testing"""
    db = SessionLocal()
    
    try:
        # Check if demo user already exists
        existing_user = db.query(User).filter(User.email == "demo@futurexfinance.com").first()
        
        if existing_user:
            print("✓ Demo user already exists")
            return
        
        # Create demo user
        demo_user = User(
            email="demo@futurexfinance.com",
            password_hash=hash_password("demo123"),
            full_name="Demo User",
            company="FutureX Finance",
            role="admin",
            is_active=True,
            is_verified=True
        )
        
        db.add(demo_user)
        db.commit()
        
        print("✓ Demo user created successfully!")
        print(f"  Email: demo@futurexfinance.com")
        print(f"  Password: demo123")
        
    except Exception as e:
        print(f"✗ Error creating demo user: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    print("Seeding demo user...")
    seed_demo_user()
