"""
Seed script for fraud detection database
Generates sample transactions and user profiles for testing
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import random
import uuid
from decimal import Decimal

from app.database import SessionLocal, engine, Base
from app.models.fraud import Transaction, UserProfile, FraudAnalysisResult
from app.services.fraud.detector import fraud_detector

# Sample data
TRANSACTION_TYPES = ['purchase', 'withdrawal', 'transfer', 'payment', 'deposit']
MERCHANT_CATEGORIES = ['grocery', 'restaurant', 'online', 'travel', 'entertainment', 'gas', 'healthcare', 'general']
MERCHANT_NAMES = [
    'Amazon', 'Walmart', 'Target', 'Starbucks', 'McDonald\'s',
    'Shell', 'Chevron', 'CVS', 'Walgreens', 'Best Buy',
    'Home Depot', 'Costco', 'Whole Foods', 'Uber', 'Lyft'
]

def generate_sample_users(db, count=10):
    """Generate sample user profiles"""
    print(f"Generating {count} user profiles...")
    
    users = []
    for i in range(count):
        user = UserProfile(
            user_id=uuid.uuid4(),
            account_created_at=datetime.utcnow() - timedelta(days=random.randint(30, 730)),
            typical_transaction_amount=Decimal(str(round(random.uniform(20, 200), 2))),
            typical_location_lat=Decimal(str(round(random.uniform(25, 48), 6))),
            typical_location_lon=Decimal(str(round(random.uniform(-120, -70), 6))),
            avg_daily_transactions=random.randint(1, 10),
            avg_monthly_spend=Decimal(str(round(random.uniform(500, 5000), 2))),
            risk_profile=random.choice(['LOW', 'LOW', 'LOW', 'MEDIUM', 'HIGH'])
        )
        users.append(user)
        db.add(user)
    
    db.commit()
    print(f"✓ Created {count} user profiles")
    return users

def generate_sample_transactions(db, users, count=100):
    """Generate sample transactions"""
    print(f"Generating {count} transactions...")
    
    transactions = []
    for i in range(count):
        user = random.choice(users)
        
        # Generate transaction
        amount = round(random.uniform(5, 1000), 2)
        
        # Occasionally generate suspicious transactions
        is_suspicious = random.random() < 0.1  # 10% suspicious
        
        if is_suspicious:
            # Make it more suspicious
            amount = round(random.uniform(5000, 15000), 2)
            location_distance = round(random.uniform(500, 2000), 2)
            count_last_hour = random.randint(10, 20)
        else:
            location_distance = round(random.uniform(0, 50), 2)
            count_last_hour = random.randint(0, 3)
        
        transaction = Transaction(
            transaction_id=f"TXN{uuid.uuid4().hex[:12].upper()}",
            user_id=user.user_id,
            amount=Decimal(str(amount)),
            currency='USD',
            transaction_type=random.choice(TRANSACTION_TYPES),
            merchant_category=random.choice(MERCHANT_CATEGORIES),
            merchant_name=random.choice(MERCHANT_NAMES),
            location_lat=user.typical_location_lat + Decimal(str(round(random.uniform(-0.5, 0.5), 6))),
            location_lon=user.typical_location_lon + Decimal(str(round(random.uniform(-0.5, 0.5), 6))),
            location_distance_km=Decimal(str(location_distance)),
            device_id=f"device_{random.randint(1, 5)}",
            timestamp=datetime.utcnow() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
        )
        
        transactions.append(transaction)
        db.add(transaction)
    
    db.commit()
    print(f"✓ Created {count} transactions")
    return transactions

def analyze_transactions(db, transactions):
    """Run fraud analysis on all transactions"""
    print(f"Analyzing {len(transactions)} transactions...")
    
    for i, transaction in enumerate(transactions):
        # Prepare transaction data for analysis
        tx_data = {
            'transaction_id': transaction.transaction_id,
            'amount': float(transaction.amount),
            'timestamp': transaction.timestamp.isoformat(),
            'type': transaction.transaction_type,
            'merchant_category': transaction.merchant_category,
            'location_distance_km': float(transaction.location_distance_km or 0),
            'account_age_days': (datetime.utcnow() - transaction.created_at).days,
            'recent_transaction_count': random.randint(1, 10),
            'amount_last_hour': float(transaction.amount) * random.uniform(0.5, 2),
            'count_last_hour': random.randint(0, 5)
        }
        
        # Analyze
        result = fraud_detector.detect(tx_data)
        
        # Save result
        fraud_result = FraudAnalysisResult(
            transaction_id=transaction.transaction_id,
            is_fraudulent=result['is_fraudulent'],
            risk_score=Decimal(str(result['risk_score'])),
            risk_level=result['risk_level'],
            ml_score=Decimal(str(result['ml_score'])) if result['ml_score'] else None,
            rule_score=Decimal(str(result['rule_score'])),
            anomalies=result['anomalies'],
            recommendation=result['recommendation'],
            details=result['details'],
            model_version='1.0.0'
        )
        
        db.add(fraud_result)
        
        if (i + 1) % 20 == 0:
            print(f"  Analyzed {i + 1}/{len(transactions)} transactions...")
    
    db.commit()
    print(f"✓ Completed fraud analysis for all transactions")

def main():
    """Main seeding function"""
    print("=" * 60)
    print("FutureX Finance - Fraud Detection Database Seeding")
    print("=" * 60)
    
    # Create tables
    print("\nCreating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Tables created")
    
    # Create session
    db = SessionLocal()
    
    try:
        # Check if data already exists
        existing_users = db.query(UserProfile).count()
        if existing_users > 0:
            print(f"\n⚠ Database already has {existing_users} users")
            response = input("Do you want to clear and reseed? (yes/no): ")
            if response.lower() != 'yes':
                print("Seeding cancelled")
                return
            
            # Clear existing data
            print("\nClearing existing data...")
            db.query(FraudAnalysisResult).delete()
            db.query(Transaction).delete()
            db.query(UserProfile).delete()
            db.commit()
            print("✓ Cleared existing data")
        
        # Generate data
        print("\n" + "=" * 60)
        users = generate_sample_users(db, count=10)
        transactions = generate_sample_transactions(db, users, count=100)
        analyze_transactions(db, transactions)
        
        # Summary
        print("\n" + "=" * 60)
        print("SEEDING SUMMARY")
        print("=" * 60)
        print(f"Users created: {len(users)}")
        print(f"Transactions created: {len(transactions)}")
        print(f"Fraud analyses completed: {len(transactions)}")
        
        # Statistics
        fraudulent_count = db.query(FraudAnalysisResult).filter(
            FraudAnalysisResult.is_fraudulent == True
        ).count()
        print(f"\nFraud Statistics:")
        print(f"  Fraudulent: {fraudulent_count}")
        print(f"  Legitimate: {len(transactions) - fraudulent_count}")
        print(f"  Fraud Rate: {(fraudulent_count / len(transactions) * 100):.1f}%")
        
        print("\n✅ Database seeding completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during seeding: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    main()
