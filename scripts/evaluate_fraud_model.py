"""
Model evaluation script for fraud detection
Evaluates model performance on test data
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from app.services.fraud.detector_enhanced import EnhancedFraudDetector
import random

def generate_labeled_test_data(count=200):
    """Generate labeled test data with known fraud/legitimate labels"""
    print(f"Generating {count} labeled test transactions...")
    
    transactions = []
    labels = []
    
    # Generate 50% fraudulent, 50% legitimate
    for i in range(count):
        is_fraud = i < (count // 2)
        
        if is_fraud:
            # Fraudulent transaction characteristics
            amount = round(random.uniform(5000, 20000), 2)
            hour = random.choice([1, 2, 3, 23, 0])
            location_distance = round(random.uniform(1000, 3000), 2)
            count_last_hour = random.randint(8, 20)
            account_age = random.randint(1, 7)
            amount_last_hour = amount * random.uniform(1.5, 3)
        else:
            # Legitimate transaction characteristics
            amount = round(random.uniform(10, 500), 2)
            hour = random.randint(8, 20)
            location_distance = round(random.uniform(0, 50), 2)
            count_last_hour = random.randint(0, 3)
            account_age = random.randint(30, 730)
            amount_last_hour = amount * random.uniform(0.5, 1.5)
        
        transaction = {
            "amount": amount,
            "timestamp": f"2024-12-31T{hour:02d}:{random.randint(0, 59):02d}:00Z",
            "type": random.choice(['purchase', 'payment', 'withdrawal']),
            "merchant_category": random.choice(['grocery', 'restaurant', 'online', 'travel']),
            "location_distance_km": location_distance,
            "account_age_days": account_age,
            "recent_transaction_count": random.randint(1, 15),
            "amount_last_hour": amount_last_hour,
            "count_last_hour": count_last_hour
        }
        
        transactions.append(transaction)
        labels.append(1 if is_fraud else 0)  # 1 = fraud, 0 = legitimate
    
    print(f"✓ Generated {count} labeled transactions")
    print(f"  Fraudulent: {sum(labels)}")
    print(f"  Legitimate: {len(labels) - sum(labels)}")
    
    return transactions, labels

def evaluate_model():
    """Evaluate the fraud detection model"""
    print("=" * 60)
    print("Fraud Detection Model Evaluation")
    print("=" * 60)
    
    # Initialize detector
    detector = EnhancedFraudDetector()
    
    if not detector.is_trained:
        print("\n⚠ Model is not trained!")
        print("Please run train_fraud_model.py first")
        return
    
    # Generate test data
    test_transactions, true_labels = generate_labeled_test_data(count=200)
    
    # Run predictions
    print("\nRunning predictions...")
    predictions = []
    risk_scores = []
    
    for tx in test_transactions:
        result = detector.detect(tx)
        predictions.append(1 if result['is_fraudulent'] else 0)
        risk_scores.append(result['risk_score'] / 100)  # Normalize to 0-1
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("Evaluation Metrics:")
    print("=" * 60)
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    print(f"\nAccuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-Score:  {f1:.2%}")
    
    # Try to calculate AUC-ROC
    try:
        auc = roc_auc_score(true_labels, risk_scores)
        print(f"AUC-ROC:   {auc:.2%}")
    except:
        print("AUC-ROC:   N/A")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("                Legit  Fraud")
    print(f"Actual Legit    {cm[0][0]:4d}   {cm[0][1]:4d}")
    print(f"       Fraud    {cm[1][0]:4d}   {cm[1][1]:4d}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(
        true_labels, predictions,
        target_names=['Legitimate', 'Fraudulent'],
        digits=3
    ))
    
    # Performance interpretation
    print("\n" + "=" * 60)
    print("Performance Interpretation:")
    print("=" * 60)
    
    if accuracy >= 0.90:
        print("✅ Excellent accuracy (≥90%)")
    elif accuracy >= 0.80:
        print("✓ Good accuracy (80-90%)")
    else:
        print("⚠ Needs improvement (<80%)")
    
    if precision >= 0.85:
        print("✅ Excellent precision - Low false positives")
    elif precision >= 0.70:
        print("✓ Good precision")
    else:
        print("⚠ High false positive rate")
    
    if recall >= 0.85:
        print("✅ Excellent recall - Catches most fraud")
    elif recall >= 0.70:
        print("✓ Good recall")
    else:
        print("⚠ Missing too many fraudulent transactions")
    
    print("\n✅ Evaluation completed!")

if __name__ == "__main__":
    evaluate_model()
