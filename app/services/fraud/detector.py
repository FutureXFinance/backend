"""
Production-Grade Fraud Detection Service
Implements ML-based anomaly detection using PyOD and custom risk scoring
"""

import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import uuid

try:
    from pyod.models.iforest import IForest
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    logging.warning("PyOD not available - using fallback fraud detection")

from sklearn.preprocessing import StandardScaler
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionFeatureExtractor:
    """Extract features from transaction data for ML models"""
    
    @staticmethod
    def extract_features(transaction: Dict[str, Any]) -> np.ndarray:
        """
        Extract numerical features from transaction
        
        Features:
        - Amount
        - Hour of day
        - Day of week
        - Transaction type (encoded)
        - Merchant category (encoded)
        - Location distance (if available)
        - Account age (days)
        - Recent transaction count
        """
        features = []
        
        # Amount
        amount = float(transaction.get('amount', 0))
        features.append(amount)
        
        # Time-based features
        timestamp = transaction.get('timestamp', datetime.utcnow().isoformat())
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            features.append(dt.hour)  # Hour of day (0-23)
            features.append(dt.weekday())  # Day of week (0-6)
        except:
            features.extend([12, 3])  # Default: noon, Wednesday
        
        # Transaction type encoding
        tx_type = transaction.get('type', 'purchase')
        type_encoding = {
            'purchase': 1,
            'withdrawal': 2,
            'transfer': 3,
            'deposit': 4,
            'payment': 5
        }
        features.append(type_encoding.get(tx_type, 1))
        
        # Merchant category
        category = transaction.get('merchant_category', 'general')
        category_encoding = {
            'general': 1,
            'online': 2,
            'travel': 3,
            'entertainment': 4,
            'grocery': 5,
            'gas': 6,
            'restaurant': 7,
            'healthcare': 8
        }
        features.append(category_encoding.get(category, 1))
        
        # Location-based (if available)
        location_distance = transaction.get('location_distance_km', 0)
        features.append(float(location_distance))
        
        # Account age (days)
        account_age = transaction.get('account_age_days', 30)
        features.append(float(account_age))
        
        # Recent transaction count
        recent_tx_count = transaction.get('recent_transaction_count', 5)
        features.append(float(recent_tx_count))
        
        # Velocity features
        amount_last_hour = transaction.get('amount_last_hour', 0)
        features.append(float(amount_last_hour))
        
        count_last_hour = transaction.get('count_last_hour', 0)
        features.append(float(count_last_hour))
        
        return np.array(features)


class RiskScorer:
    """Calculate risk scores based on various factors"""
    
    @staticmethod
    def calculate_rule_based_score(transaction: Dict[str, Any]) -> float:
        """
        Calculate risk score using rule-based approach
        Returns score between 0-100
        """
        risk_score = 0.0
        
        amount = float(transaction.get('amount', 0))
        
        # High amount transactions
        if amount > 10000:
            risk_score += 30
        elif amount > 5000:
            risk_score += 20
        elif amount > 1000:
            risk_score += 10
        
        # Unusual time
        timestamp = transaction.get('timestamp', datetime.utcnow().isoformat())
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hour = dt.hour
            # Late night/early morning transactions
            if hour < 6 or hour > 22:
                risk_score += 15
        except:
            pass
        
        # High velocity
        count_last_hour = transaction.get('count_last_hour', 0)
        if count_last_hour > 10:
            risk_score += 25
        elif count_last_hour > 5:
            risk_score += 15
        
        amount_last_hour = transaction.get('amount_last_hour', 0)
        if amount_last_hour > 20000:
            risk_score += 20
        
        # Foreign/distant location
        location_distance = transaction.get('location_distance_km', 0)
        if location_distance > 1000:
            risk_score += 20
        elif location_distance > 500:
            risk_score += 10
        
        # New account
        account_age = transaction.get('account_age_days', 30)
        if account_age < 7:
            risk_score += 15
        elif account_age < 30:
            risk_score += 5
        
        # High-risk merchant categories
        category = transaction.get('merchant_category', 'general')
        high_risk_categories = ['online', 'travel', 'entertainment']
        if category in high_risk_categories:
            risk_score += 10
        
        # Cap at 100
        return min(risk_score, 100.0)
    
    @staticmethod
    def combine_scores(ml_score: float, rule_score: float) -> float:
        """Combine ML and rule-based scores"""
        # Weighted average: 60% ML, 40% rules
        if ml_score >= 0:
            return (ml_score * 0.6) + (rule_score * 0.4)
        else:
            return rule_score


class FraudDetector:
    """Main fraud detection orchestrator"""
    
    def __init__(self):
        self.pyod_available = PYOD_AVAILABLE
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        
        # Initialize model if PyOD is available
        if self.pyod_available:
            # Use Isolation Forest for anomaly detection
            self.model = IForest(
                n_estimators=100,
                contamination=0.1,  # Expected fraud rate
                random_state=42
            )
            logger.info("Fraud detector initialized with Isolation Forest")
        else:
            logger.warning("PyOD not available - using rule-based detection only")
    
    def train(self, historical_transactions: List[Dict[str, Any]]) -> None:
        """
        Train the model on historical transaction data
        
        Args:
            historical_transactions: List of past transactions (normal behavior)
        """
        if not self.pyod_available or len(historical_transactions) < 10:
            logger.warning("Insufficient data or PyOD unavailable - skipping training")
            return
        
        # Extract features
        features_list = []
        for tx in historical_transactions:
            features = TransactionFeatureExtractor.extract_features(tx)
            features_list.append(features)
        
        X = np.array(features_list)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        self.is_trained = True
        
        logger.info(f"Model trained on {len(historical_transactions)} transactions")
    
    def detect(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect if a transaction is fraudulent
        
        Args:
            transaction: Transaction data dictionary
            
        Returns:
            Detection result with risk score and details
        """
        transaction_id = transaction.get('transaction_id', str(uuid.uuid4()))
        timestamp = datetime.utcnow().isoformat()
        
        # Extract features
        features = TransactionFeatureExtractor.extract_features(transaction)
        
        # ML-based score
        ml_score = -1
        anomaly_score = 0.0
        
        if self.pyod_available and self.is_trained:
            try:
                # Normalize features
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                
                # Predict anomaly score (0 = normal, 1 = anomaly)
                prediction = self.model.predict(features_scaled)[0]
                
                # Get anomaly score (higher = more anomalous)
                anomaly_score = self.model.decision_function(features_scaled)[0]
                
                # Convert to 0-100 scale
                # Normalize anomaly score to 0-100 range
                ml_score = min(max((anomaly_score + 1) * 50, 0), 100)
                
            except Exception as e:
                logger.error(f"ML detection error: {e}")
                ml_score = -1
        
        # Rule-based score
        rule_score = RiskScorer.calculate_rule_based_score(transaction)
        
        # Combined score
        final_score = RiskScorer.combine_scores(ml_score, rule_score)
        
        # Determine if fraudulent
        is_fraudulent = final_score > 70
        
        # Risk level
        if final_score > 80:
            risk_level = "CRITICAL"
        elif final_score > 60:
            risk_level = "HIGH"
        elif final_score > 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Identify anomalies
        anomalies = self._identify_anomalies(transaction, rule_score)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            is_fraudulent, risk_level, anomalies
        )
        
        return {
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "is_fraudulent": is_fraudulent,
            "risk_score": round(final_score, 2),
            "risk_level": risk_level,
            "ml_score": round(ml_score, 2) if ml_score >= 0 else None,
            "rule_score": round(rule_score, 2),
            "anomalies": anomalies,
            "recommendation": recommendation,
            "details": {
                "amount": transaction.get('amount'),
                "type": transaction.get('type'),
                "merchant_category": transaction.get('merchant_category'),
                "model_used": "Isolation Forest" if self.is_trained else "Rule-based only"
            }
        }
    
    def _identify_anomalies(
        self, transaction: Dict[str, Any], rule_score: float
    ) -> List[str]:
        """Identify specific anomalies in the transaction"""
        anomalies = []
        
        amount = float(transaction.get('amount', 0))
        if amount > 5000:
            anomalies.append(f"High transaction amount: ${amount:,.2f}")
        
        count_last_hour = transaction.get('count_last_hour', 0)
        if count_last_hour > 5:
            anomalies.append(f"High transaction velocity: {count_last_hour} in last hour")
        
        location_distance = transaction.get('location_distance_km', 0)
        if location_distance > 500:
            anomalies.append(f"Unusual location: {location_distance}km from normal")
        
        account_age = transaction.get('account_age_days', 30)
        if account_age < 7:
            anomalies.append(f"New account: {account_age} days old")
        
        timestamp = transaction.get('timestamp', datetime.utcnow().isoformat())
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hour = dt.hour
            if hour < 6 or hour > 22:
                anomalies.append(f"Unusual time: {hour:02d}:00")
        except:
            pass
        
        return anomalies
    
    def _generate_recommendation(
        self, is_fraudulent: bool, risk_level: str, anomalies: List[str]
    ) -> str:
        """Generate action recommendation"""
        if is_fraudulent:
            return "BLOCK - High fraud probability. Require additional verification."
        elif risk_level == "HIGH":
            return "REVIEW - Manual review recommended before approval."
        elif risk_level == "MEDIUM":
            return "MONITOR - Approve but flag for monitoring."
        else:
            return "APPROVE - Low risk transaction."


# Global instance
fraud_detector = FraudDetector()

# Auto-train with sample data if available
def initialize_with_sample_data():
    """Initialize detector with sample normal transactions"""
    sample_transactions = [
        {
            "amount": 50.00,
            "timestamp": "2024-01-01T14:30:00Z",
            "type": "purchase",
            "merchant_category": "grocery",
            "location_distance_km": 5,
            "account_age_days": 365,
            "recent_transaction_count": 3,
            "amount_last_hour": 50,
            "count_last_hour": 1
        },
        {
            "amount": 120.00,
            "timestamp": "2024-01-02T18:45:00Z",
            "type": "purchase",
            "merchant_category": "restaurant",
            "location_distance_km": 10,
            "account_age_days": 365,
            "recent_transaction_count": 5,
            "amount_last_hour": 120,
            "count_last_hour": 1
        },
        # Add more sample transactions...
    ]
    
    # Generate more synthetic normal transactions
    for i in range(50):
        sample_transactions.append({
            "amount": np.random.uniform(10, 500),
            "timestamp": f"2024-01-{(i % 30) + 1:02d}T{np.random.randint(8, 20):02d}:00:00Z",
            "type": np.random.choice(['purchase', 'payment']),
            "merchant_category": np.random.choice(['grocery', 'restaurant', 'gas', 'general']),
            "location_distance_km": np.random.uniform(0, 50),
            "account_age_days": 365,
            "recent_transaction_count": np.random.randint(1, 10),
            "amount_last_hour": np.random.uniform(0, 500),
            "count_last_hour": np.random.randint(0, 3)
        })
    
    fraud_detector.train(sample_transactions)

# Initialize on import
initialize_with_sample_data()
