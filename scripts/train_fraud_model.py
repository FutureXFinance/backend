"""
Model training script for fraud detection
Trains the ensemble models on historical data
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from datetime import datetime, timedelta
import random
from app.services.fraud.detector_enhanced import EnhancedFraudDetector

def generate_training_data(count=1000):
    """Generate synthetic training data"""
    print(f"Generating {count} training transactions...")
    
    transactions = []
    
    for i in range(count):
        # Generate mostly normal transactions (90%)
        is_normal = random.random() < 0.9
        
        if is_normal:
            # Normal transaction
            amount = round(random.uniform(10, 500), 2)
            hour = random.randint(8, 20)  # Business hours
            location_distance = round(random.uniform(0, 50), 2)
            count_last_hour = random.randint(0, 3)
            account_age = random.randint(30, 730)
        else:
            # Anomalous transaction
            amount = round(random.uniform(1000, 10000), 2)
            hour = random.choice([2, 3, 4, 23, 0, 1])  # Unusual hours
            location_distance = round(random.uniform(500, 2000), 2)
            count_last_hour = random.randint(5, 15)
            account_age = random.randint(1, 10)
        
        transaction = {
            "amount": amount,
            "timestamp": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}T{hour:02d}:{random.randint(0, 59):02d}:00Z",
            "type": random.choice(['purchase', 'payment', 'withdrawal', 'transfer']),
            "merchant_category": random.choice(['grocery', 'restaurant', 'gas', 'general', 'online', 'travel']),
            "location_distance_km": location_distance,
            "account_age_days": account_age,
            "recent_transaction_count": random.randint(1, 10),
            "amount_last_hour": amount * random.uniform(0.5, 2),
            "count_last_hour": count_last_hour
        }
        
        transactions.append(transaction)
    
    print(f"✓ Generated {count} transactions")
    return transactions

def train_model():
    """Train the fraud detection model"""
    print("=" * 60)
    print("Fraud Detection Model Training")
    print("=" * 60)
    
    # Initialize detector
    detector = EnhancedFraudDetector()
    
    # Generate training data
    training_data = generate_training_data(count=1000)
    
    # Train model
    print("\nTraining ensemble models...")
    result = detector.train(training_data)
    
    print("\n" + "=" * 60)
    print("Training Results:")
    print("=" * 60)
    print(f"Status: {result.get('status')}")
    print(f"Transactions: {result.get('transactions_count')}")
    print(f"Features: {result.get('features_count')}")
    print(f"Models: {', '.join(result.get('models', []))}")
    print(f"Version: {result.get('model_version')}")
    
    # Test on a few samples
    print("\n" + "=" * 60)
    print("Testing on Sample Transactions:")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Normal Transaction",
            "data": {
                "amount": 50.00,
                "timestamp": "2024-12-31T14:30:00Z",
                "type": "purchase",
                "merchant_category": "grocery",
                "location_distance_km": 5,
                "account_age_days": 365,
                "recent_transaction_count": 3,
                "amount_last_hour": 50,
                "count_last_hour": 1
            }
        },
        {
            "name": "High Amount",
            "data": {
                "amount": 15000.00,
                "timestamp": "2024-12-31T14:30:00Z",
                "type": "purchase",
                "merchant_category": "online",
                "location_distance_km": 10,
                "account_age_days": 365,
                "recent_transaction_count": 5,
                "amount_last_hour": 15000,
                "count_last_hour": 1
            }
        },
        {
            "name": "Suspicious Pattern",
            "data": {
                "amount": 5000.00,
                "timestamp": "2024-12-31T02:30:00Z",
                "type": "withdrawal",
                "merchant_category": "online",
                "location_distance_km": 1500,
                "account_age_days": 3,
                "recent_transaction_count": 15,
                "amount_last_hour": 8000,
                "count_last_hour": 12
            }
        }
    ]
    
    for test in test_cases:
        result = detector.detect(test["data"])
        print(f"\n{test['name']}:")
        print(f"  Risk Score: {result['risk_score']}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Fraudulent: {result['is_fraudulent']}")
        print(f"  Recommendation: {result['recommendation']}")
    
    print("\n✅ Model training completed successfully!")
    print(f"Models saved to: {detector.model_dir}")

if __name__ == "__main__":
    train_model()
