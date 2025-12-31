"""
Fraud Detection API Routes
Handles transaction analysis and fraud detection
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from sqlalchemy.orm import Session

from app.services.fraud.detector_enhanced import fraud_detector
from app.services.fraud.service import FraudDetectionService
from app.database import get_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


class TransactionInput(BaseModel):
    """Transaction data for fraud analysis"""
    transaction_id: Optional[str] = None
    amount: float = Field(..., gt=0, description="Transaction amount")
    timestamp: Optional[str] = Field(None, description="ISO format timestamp")
    type: str = Field(default="purchase", description="Transaction type")
    merchant_category: str = Field(default="general", description="Merchant category")
    location_distance_km: float = Field(default=0, description="Distance from usual location")
    account_age_days: int = Field(default=30, description="Account age in days")
    recent_transaction_count: int = Field(default=0, description="Recent transaction count")
    amount_last_hour: float = Field(default=0, description="Total amount in last hour")
    count_last_hour: int = Field(default=0, description="Transaction count in last hour")


class FraudAnalysisResult(BaseModel):
    """Fraud analysis result"""
    transaction_id: str
    timestamp: str
    is_fraudulent: bool
    risk_score: float
    risk_level: str
    ml_score: Optional[float]
    rule_score: float
    anomalies: List[str]
    recommendation: str
    details: Dict[str, Any]


class TrainingData(BaseModel):
    """Historical transactions for model training"""
    transactions: List[Dict[str, Any]]


@router.post("/analyze", response_model=FraudAnalysisResult)
async def analyze_transaction(
    transaction: TransactionInput,
    user_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Analyze a transaction for fraud and save to database
    
    - **amount**: Transaction amount (required)
    - **type**: Transaction type (purchase, withdrawal, transfer, etc.)
    - **merchant_category**: Category of merchant
    - **location_distance_km**: Distance from user's normal location
    - **account_age_days**: Age of the account
    - **recent_transaction_count**: Number of recent transactions
    - **amount_last_hour**: Total amount transacted in last hour
    - **count_last_hour**: Number of transactions in last hour
    - **user_id**: Optional user ID for profile lookup
    
    Returns detailed fraud analysis with risk score and recommendations
    """
    try:
        # Convert to dict
        transaction_dict = transaction.dict()
        
        # Add timestamp if not provided
        if not transaction_dict.get('timestamp'):
            transaction_dict['timestamp'] = datetime.utcnow().isoformat()
        
        # Perform fraud detection with database integration
        result = FraudDetectionService.analyze_transaction_with_db(
            transaction_dict, db, user_id
        )
        
        logger.info(
            f"Analyzed transaction {result['transaction_id']}: "
            f"Risk={result['risk_score']}, Fraudulent={result['is_fraudulent']}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Fraud analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/batch-analyze")
async def batch_analyze_transactions(transactions: List[TransactionInput]):
    """
    Analyze multiple transactions in batch
    
    Useful for processing historical data or bulk analysis
    """
    try:
        results = []
        
        for transaction in transactions:
            transaction_dict = transaction.dict()
            if not transaction_dict.get('timestamp'):
                transaction_dict['timestamp'] = datetime.utcnow().isoformat()
            
            result = fraud_detector.detect(transaction_dict)
            results.append(result)
        
        # Calculate summary statistics
        total = len(results)
        fraudulent_count = sum(1 for r in results if r['is_fraudulent'])
        avg_risk_score = sum(r['risk_score'] for r in results) / total if total > 0 else 0
        
        return {
            "total_analyzed": total,
            "fraudulent_count": fraudulent_count,
            "fraud_rate": (fraudulent_count / total * 100) if total > 0 else 0,
            "average_risk_score": round(avg_risk_score, 2),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )


@router.post("/train")
async def train_model(training_data: TrainingData):
    """
    Train the fraud detection model with historical data
    
    Provide a list of normal (non-fraudulent) transactions to train the model
    """
    try:
        transactions = training_data.transactions
        
        if len(transactions) < 10:
            raise HTTPException(
                status_code=400,
                detail="At least 10 transactions required for training"
            )
        
        fraud_detector.train(transactions)
        
        return {
            "status": "success",
            "message": f"Model trained on {len(transactions)} transactions",
            "model_type": "Isolation Forest" if fraud_detector.is_trained else "Rule-based only"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


@router.get("/history")
async def get_fraud_history(limit: int = 50):
    """
    Get fraud detection history
    
    In production, this would query a database
    Currently returns mock data
    """
    # TODO: Implement database query
    return {
        "message": "History endpoint - implement database integration",
        "limit": limit
    }


@router.get("/statistics")
async def get_statistics():
    """
    Get fraud detection statistics
    
    Returns overall statistics about fraud detection performance
    """
    return {
        "model_status": "trained" if fraud_detector.is_trained else "untrained",
        "model_type": "Isolation Forest + Rules" if fraud_detector.pyod_available else "Rules only",
        "pyod_available": fraud_detector.pyod_available,
        "features_count": 10,
        "risk_thresholds": {
            "low": "0-40",
            "medium": "40-60",
            "high": "60-80",
            "critical": "80-100"
        }
    }


@router.get("/health")
async def health_check():
    """Check fraud detection service health"""
    return {
        "status": "healthy",
        "pyod_available": fraud_detector.pyod_available,
        "model_trained": fraud_detector.is_trained,
        "model_type": "Isolation Forest" if fraud_detector.is_trained else "Rule-based"
    }


@router.post("/simulate")
async def simulate_fraud_scenarios():
    """
    Simulate various fraud scenarios for testing
    
    Returns analysis results for different risk levels
    """
    scenarios = [
        {
            "name": "Normal Transaction",
            "transaction": {
                "amount": 50.00,
                "type": "purchase",
                "merchant_category": "grocery",
                "location_distance_km": 5,
                "account_age_days": 365,
                "count_last_hour": 1,
                "amount_last_hour": 50
            }
        },
        {
            "name": "High Amount",
            "transaction": {
                "amount": 15000.00,
                "type": "purchase",
                "merchant_category": "online",
                "location_distance_km": 10,
                "account_age_days": 365,
                "count_last_hour": 1,
                "amount_last_hour": 15000
            }
        },
        {
            "name": "High Velocity",
            "transaction": {
                "amount": 200.00,
                "type": "purchase",
                "merchant_category": "online",
                "location_distance_km": 5,
                "account_age_days": 365,
                "count_last_hour": 15,
                "amount_last_hour": 3000
            }
        },
        {
            "name": "Suspicious Location",
            "transaction": {
                "amount": 500.00,
                "type": "purchase",
                "merchant_category": "travel",
                "location_distance_km": 2000,
                "account_age_days": 365,
                "count_last_hour": 1,
                "amount_last_hour": 500
            }
        },
        {
            "name": "New Account + High Amount",
            "transaction": {
                "amount": 5000.00,
                "type": "purchase",
                "merchant_category": "online",
                "location_distance_km": 100,
                "account_age_days": 3,
                "count_last_hour": 5,
                "amount_last_hour": 8000
            }
        }
    ]
    
    results = []
    for scenario in scenarios:
        tx = scenario["transaction"]
        tx["timestamp"] = datetime.utcnow().isoformat()
        
        analysis = fraud_detector.detect(tx)
        results.append({
            "scenario": scenario["name"],
            "analysis": analysis
        })
    
    return {
        "scenarios": results,
        "note": "These are simulated scenarios for testing purposes"
    }


# New Database-Integrated Endpoints

@router.get("/history")
async def get_transaction_history(
    db: Session = Depends(get_db),
    user_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    risk_level: Optional[str] = None,
    is_fraudulent: Optional[bool] = None
):
    try:
        result = FraudDetectionService.get_transaction_history(
            db, user_id, limit, offset, risk_level, is_fraudulent
        )
        return result
    except Exception as e:
        logger.error(f"Error fetching transaction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Database-Integrated Endpoints

@router.get("/history")
async def get_transaction_history(
    db: Session = Depends(get_db),
    user_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    risk_level: Optional[str] = None,
    is_fraudulent: Optional[bool] = None
):
    """Get transaction history with fraud analysis results"""
    try:
        result = FraudDetectionService.get_transaction_history(
            db, user_id, limit, offset, risk_level, is_fraudulent
        )
        return result
    except Exception as e:
        logger.error(f"Error fetching transaction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts(
    db: Session = Depends(get_db),
    status: str = "PENDING",
    severity: Optional[str] = None,
    limit: int = 50
):
    """Get fraud alerts"""
    try:
        result = FraudDetectionService.get_active_alerts(db, status, severity, limit)
        return result
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/alerts/{alert_id}")
async def update_alert(
    alert_id: str,
    db: Session = Depends(get_db),
    status: Optional[str] = None,
    assigned_to: Optional[str] = None,
    resolution_notes: Optional[str] = None
):
    """Update fraud alert status"""
    try:
        if not status:
            raise HTTPException(status_code=400, detail="Status is required")
        
        result = FraudDetectionService.update_alert_status(
            db, alert_id, status, assigned_to, resolution_notes
        )
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/profile")
async def get_user_profile(
    user_id: str,
    db: Session = Depends(get_db)
):
    """Get user risk profile"""
    try:
        profile = FraudDetectionService.get_user_profile(db, user_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail=f"User profile not found for {user_id}")
        
        return profile
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/dashboard")
async def get_dashboard_analytics(
    db: Session = Depends(get_db),
    days: int = 30
):
    """Get fraud detection analytics for dashboard"""
    try:
        analytics = FraudDetectionService.get_dashboard_analytics(db, days)
        return analytics
    except Exception as e:
        logger.error(f"Error fetching dashboard analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain")
async def retrain_model(training_data: Optional[TrainingData] = None):
    """Retrain the fraud detection model"""
    try:
        if training_data and training_data.transactions:
            metrics = fraud_detector.train(training_data.transactions)
        else:
            if not fraud_detector.is_trained:
                raise HTTPException(
                    status_code=400,
                    detail="No training data available"
                )
            metrics = {"message": "Model already trained", "version": fraud_detector.model_version}
        
        return {
            "status": "success",
            "model_version": fraud_detector.model_version,
            "metrics": metrics
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/metrics")
async def get_model_metrics():
    """Get fraud detection model information"""
    try:
        return {
            "model_version": fraud_detector.model_version,
            "is_trained": fraud_detector.is_trained,
            "feature_count": len(fraud_detector.feature_extractor.get_feature_names()),
            "risk_thresholds": {
                "CRITICAL": "> 80",
                "HIGH": "60-80",
                "MEDIUM": "40-60",
                "LOW": "< 40"
            },
            "ensemble_weights": {
                "ml_score": 0.6,
                "rule_score": 0.4
            }
        }
    except Exception as e:
        logger.error(f"Error fetching model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
