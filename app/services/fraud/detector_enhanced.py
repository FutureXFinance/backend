"""
Enhanced Production-Grade Fraud Detection Service
Implements advanced ML-based anomaly detection with ensemble models
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import uuid
import pickle
from pathlib import Path

try:
    from pyod.models.iforest import IForest
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


class AdvancedFeatureExtractor:
    """
    Extract advanced features from transaction data for ML models
    Implements 15 features for comprehensive fraud detection
    """
    
    @staticmethod
    def extract_features(transaction: Dict[str, Any], user_profile: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Extract 15 numerical features from transaction
        
        Features:
        1. amount (normalized log scale)
        2. hour_of_day (0-23)
        3. day_of_week (0-6)
        4. is_weekend (binary)
        5. transaction_type_encoded (1-5)
        6. merchant_category_encoded (1-8)
        7. location_distance_km
        8. account_age_days
        9. recent_transaction_count (last 24h)
        10. amount_last_hour
        11. count_last_hour
        12. amount_deviation_from_typical (if user profile available)
        13. velocity_score (calculated)
        14. is_unusual_time (binary: late night/early morning)
        15. is_high_risk_category (binary)
        """
        features = []
        
        # 1. Amount (log scale for better distribution)
        amount = float(transaction.get('amount', 0))
        features.append(np.log1p(amount))  # log(1 + amount) to handle 0
        
        # 2-4. Time-based features
        timestamp = transaction.get('timestamp', datetime.utcnow().isoformat())
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hour = dt.hour
            weekday = dt.weekday()
            is_weekend = 1 if weekday >= 5 else 0
            
            features.append(hour)  # 2. Hour of day (0-23)
            features.append(weekday)  # 3. Day of week (0-6)
            features.append(is_weekend)  # 4. Is weekend
        except:
            features.extend([12, 3, 0])  # Default: noon, Wednesday, not weekend
        
        # 5. Transaction type encoding
        tx_type = transaction.get('type', 'purchase')
        type_encoding = {
            'purchase': 1,
            'withdrawal': 2,
            'transfer': 3,
            'deposit': 4,
            'payment': 5
        }
        features.append(type_encoding.get(tx_type, 1))
        
        # 6. Merchant category
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
        
        # 7. Location distance
        location_distance = transaction.get('location_distance_km', 0)
        features.append(float(location_distance))
        
        # 8. Account age (days)
        account_age = transaction.get('account_age_days', 30)
        features.append(float(account_age))
        
        # 9. Recent transaction count
        recent_tx_count = transaction.get('recent_transaction_count', 5)
        features.append(float(recent_tx_count))
        
        # 10. Amount last hour
        amount_last_hour = transaction.get('amount_last_hour', 0)
        features.append(float(amount_last_hour))
        
        # 11. Count last hour
        count_last_hour = transaction.get('count_last_hour', 0)
        features.append(float(count_last_hour))
        
        # 12. Amount deviation from typical (if user profile available)
        if user_profile and user_profile.get('typical_transaction_amount'):
            typical_amount = float(user_profile['typical_transaction_amount'])
            deviation = abs(amount - typical_amount) / (typical_amount + 1)  # Normalized deviation
            features.append(deviation)
        else:
            features.append(0.0)  # No profile, assume no deviation
        
        # 13. Velocity score (composite of amount and count velocity)
        velocity_score = 0.0
        if count_last_hour > 0:
            avg_amount_per_tx = amount_last_hour / count_last_hour
            velocity_score = (count_last_hour * 0.5) + (avg_amount_per_tx / 1000 * 0.5)
        features.append(velocity_score)
        
        # 14. Is unusual time (late night/early morning)
        try:
            is_unusual_time = 1 if (hour < 6 or hour > 22) else 0
        except:
            is_unusual_time = 0
        features.append(is_unusual_time)
        
        # 15. Is high-risk category
        high_risk_categories = ['online', 'travel', 'entertainment']
        is_high_risk = 1 if category in high_risk_categories else 0
        features.append(is_high_risk)
        
        return np.array(features)


class EnhancedRiskScorer:
    """Enhanced risk scoring with more sophisticated rules"""
    
    @staticmethod
    def calculate_rule_based_score(transaction: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Calculate risk score using enhanced rule-based approach
        Returns (score, triggered_rules) tuple
        """
        risk_score = 0.0
        triggered_rules = []
        
        amount = float(transaction.get('amount', 0))
        
        # Amount-based rules (weighted by severity)
        if amount > 10000:
            risk_score += 35
            triggered_rules.append("Very high amount (>$10,000)")
        elif amount > 5000:
            risk_score += 25
            triggered_rules.append("High amount (>$5,000)")
        elif amount > 1000:
            risk_score += 10
            triggered_rules.append("Elevated amount (>$1,000)")
        
        # Time-based rules
        timestamp = transaction.get('timestamp', datetime.utcnow().isoformat())
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hour = dt.hour
            
            # Late night/early morning (higher risk)
            if hour < 6 or hour > 22:
                risk_score += 15
                triggered_rules.append(f"Unusual time ({hour:02d}:00)")
            
            # Weekend transactions (slightly elevated risk)
            if dt.weekday() >= 5:
                risk_score += 5
                triggered_rules.append("Weekend transaction")
        except:
            pass
        
        # Velocity rules (transaction frequency)
        count_last_hour = transaction.get('count_last_hour', 0)
        if count_last_hour > 10:
            risk_score += 30
            triggered_rules.append(f"Very high velocity ({count_last_hour} txns/hour)")
        elif count_last_hour > 5:
            risk_score += 20
            triggered_rules.append(f"High velocity ({count_last_hour} txns/hour)")
        elif count_last_hour > 3:
            risk_score += 10
            triggered_rules.append(f"Elevated velocity ({count_last_hour} txns/hour)")
        
        # Amount velocity
        amount_last_hour = transaction.get('amount_last_hour', 0)
        if amount_last_hour > 20000:
            risk_score += 25
            triggered_rules.append(f"High spending velocity (${amount_last_hour:,.0f}/hour)")
        elif amount_last_hour > 10000:
            risk_score += 15
            triggered_rules.append(f"Elevated spending velocity (${amount_last_hour:,.0f}/hour)")
        
        # Location-based rules
        location_distance = transaction.get('location_distance_km', 0)
        if location_distance > 1000:
            risk_score += 25
            triggered_rules.append(f"Foreign location ({location_distance:.0f}km away)")
        elif location_distance > 500:
            risk_score += 15
            triggered_rules.append(f"Distant location ({location_distance:.0f}km away)")
        elif location_distance > 100:
            risk_score += 5
            triggered_rules.append(f"Unusual location ({location_distance:.0f}km away)")
        
        # Account age rules
        account_age = transaction.get('account_age_days', 30)
        if account_age < 7:
            risk_score += 20
            triggered_rules.append(f"Very new account ({account_age} days)")
        elif account_age < 30:
            risk_score += 10
            triggered_rules.append(f"New account ({account_age} days)")
        
        # Merchant category rules
        category = transaction.get('merchant_category', 'general')
        high_risk_categories = {
            'online': 15,
            'travel': 12,
            'entertainment': 10
        }
        if category in high_risk_categories:
            category_risk = high_risk_categories[category]
            risk_score += category_risk
            triggered_rules.append(f"High-risk category ({category})")
        
        # Cap at 100
        return min(risk_score, 100.0), triggered_rules
    
    @staticmethod
    def combine_scores(ml_score: float, rule_score: float, ml_weight: float = 0.6) -> float:
        """
        Combine ML and rule-based scores with configurable weighting
        Default: 60% ML, 40% rules
        """
        if ml_score >= 0:
            return (ml_score * ml_weight) + (rule_score * (1 - ml_weight))
        else:
            return rule_score


class EnhancedFraudDetector:
    """
    Enhanced fraud detection orchestrator with ensemble models
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        self.pyod_available = PYOD_AVAILABLE
        self.scaler = StandardScaler()
        self.iforest_model = None
        self.lof_model = None
        self.is_trained = False
        self.model_version = "2.0.0"
        
        # Model directory
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = Path(__file__).parent.parent.parent.parent / "ai_models" / "fraud_detection" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models if PyOD is available
        if self.pyod_available:
            # Isolation Forest - good for global anomalies
            self.iforest_model = IForest(
                n_estimators=100,
                contamination=0.1,  # Expected fraud rate
                random_state=42,
                behaviour='new'
            )
            
            # LOF - good for local anomalies
            self.lof_model = LOF(
                n_neighbors=20,
                contamination=0.1,
                novelty=True  # For prediction on new data
            )
            
            logger.info("Enhanced fraud detector initialized with Isolation Forest + LOF")
            
            # Try to load existing models
            self._load_models()
        else:
            logger.warning("PyOD not available - using rule-based detection only")
    
    def _load_models(self) -> bool:
        """Load pre-trained models if they exist"""
        try:
            iforest_path = self.model_dir / "iforest_model.pkl"
            lof_path = self.model_dir / "lof_model.pkl"
            scaler_path = self.model_dir / "scaler.pkl"
            
            if iforest_path.exists() and lof_path.exists() and scaler_path.exists():
                with open(iforest_path, 'rb') as f:
                    self.iforest_model = pickle.load(f)
                with open(lof_path, 'rb') as f:
                    self.lof_model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                self.is_trained = True
                logger.info("Loaded pre-trained models successfully")
                return True
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
        
        return False
    
    def _save_models(self) -> None:
        """Save trained models to disk"""
        try:
            with open(self.model_dir / "iforest_model.pkl", 'wb') as f:
                pickle.dump(self.iforest_model, f)
            with open(self.model_dir / "lof_model.pkl", 'wb') as f:
                pickle.dump(self.lof_model, f)
            with open(self.model_dir / "scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info(f"Models saved to {self.model_dir}")
        except Exception as e:
            logger.error(f"Could not save models: {e}")
    
    def train(self, historical_transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the ensemble models on historical transaction data
        
        Args:
            historical_transactions: List of past transactions (normal behavior)
            
        Returns:
            Training metrics and statistics
        """
        if not self.pyod_available or len(historical_transactions) < 10:
            logger.warning("Insufficient data or PyOD unavailable - skipping training")
            return {"status": "skipped", "reason": "insufficient_data_or_no_pyod"}
        
        # Extract features
        features_list = []
        for tx in historical_transactions:
            features = AdvancedFeatureExtractor.extract_features(tx)
            features_list.append(features)
        
        X = np.array(features_list)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.iforest_model.fit(X_scaled)
        
        # Train LOF
        self.lof_model.fit(X_scaled)
        
        self.is_trained = True
        
        # Save models
        self._save_models()
        
        logger.info(f"Ensemble models trained on {len(historical_transactions)} transactions")
        
        return {
            "status": "success",
            "transactions_count": len(historical_transactions),
            "features_count": X.shape[1],
            "model_version": self.model_version,
            "models": ["IsolationForest", "LOF"]
        }
    
    def detect(self, transaction: Dict[str, Any], user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect if a transaction is fraudulent using ensemble approach
        
        Args:
            transaction: Transaction data dictionary
            user_profile: Optional user profile for enhanced features
            
        Returns:
            Detection result with risk score and details
        """
        transaction_id = transaction.get('transaction_id') or str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Extract features
        features = AdvancedFeatureExtractor.extract_features(transaction, user_profile)
        
        # ML-based scores
        iforest_score = -1
        lof_score = -1
        ml_score = -1
        
        if self.pyod_available and self.is_trained:
            try:
                # Normalize features
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                
                # Get Isolation Forest score
                iforest_anomaly = self.iforest_model.decision_function(features_scaled)[0]
                iforest_score = min(max((iforest_anomaly + 1) * 50, 0), 100)
                
                # Get LOF score
                lof_anomaly = -self.lof_model.decision_function(features_scaled)[0]  # LOF returns negative
                lof_score = min(max(lof_anomaly * 50, 0), 100)
                
                # Ensemble: average of both models
                ml_score = (iforest_score + lof_score) / 2
                
            except Exception as e:
                logger.error(f"ML detection error: {e}")
                ml_score = -1
        
        # Rule-based score
        rule_score, triggered_rules = EnhancedRiskScorer.calculate_rule_based_score(transaction)
        
        # Combined score
        final_score = EnhancedRiskScorer.combine_scores(ml_score, rule_score)
        
        # Determine if fraudulent (threshold: 70)
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
        
        # Generate recommendation
        recommendation = self._generate_recommendation(is_fraudulent, risk_level, triggered_rules)
        
        return {
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "is_fraudulent": is_fraudulent,
            "risk_score": round(final_score, 2),
            "risk_level": risk_level,
            "ml_score": round(ml_score, 2) if ml_score >= 0 else None,
            "rule_score": round(rule_score, 2),
            "anomalies": triggered_rules,
            "recommendation": recommendation,
            "details": {
                "amount": transaction.get('amount'),
                "type": transaction.get('type'),
                "merchant_category": transaction.get('merchant_category'),
                "model_used": "Ensemble (IForest+LOF)" if self.is_trained else "Rule-based only",
                "model_version": self.model_version,
                "iforest_score": round(iforest_score, 2) if iforest_score >= 0 else None,
                "lof_score": round(lof_score, 2) if lof_score >= 0 else None,
                "features_count": len(features)
            }
        }
    
    def _generate_recommendation(
        self, is_fraudulent: bool, risk_level: str, anomalies: List[str]
    ) -> str:
        """Generate action recommendation based on risk assessment"""
        if is_fraudulent:
            return "BLOCK - High fraud probability. Require additional verification before processing."
        elif risk_level == "HIGH":
            return "REVIEW - Manual review recommended before approval. Consider additional authentication."
        elif risk_level == "MEDIUM":
            return "MONITOR - Approve but flag for monitoring. Watch for similar patterns."
        else:
            return "APPROVE - Low risk transaction. Process normally."


# Global instance
fraud_detector = EnhancedFraudDetector()

# Auto-train with sample data if available
def initialize_with_sample_data():
    """Initialize detector with sample normal transactions"""
    sample_transactions = []
    
    # Generate synthetic normal transactions
    for i in range(100):
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
    
    result = fraud_detector.train(sample_transactions)
    logger.info(f"Initialization complete: {result}")

# Initialize on import
initialize_with_sample_data()
