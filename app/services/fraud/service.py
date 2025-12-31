"""
Fraud Detection Service with Database Integration
Handles fraud detection with database persistence
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from app.models.fraud import Transaction, FraudAnalysisResult, UserProfile, FraudAlert
from app.services.fraud.detector_enhanced import fraud_detector
import logging

logger = logging.getLogger(__name__)


class FraudDetectionService:
    """
    Service layer for fraud detection with database integration
    """
    
    @staticmethod
    def analyze_transaction_with_db(
        transaction_data: Dict[str, Any],
        db: Session,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a transaction and save results to database
        
        Args:
            transaction_data: Transaction data dictionary
            db: Database session
            user_id: Optional user ID for profile lookup
            
        Returns:
            Fraud analysis result
        """
        # Get user profile if user_id provided
        user_profile = None
        if user_id:
            try:
                profile = db.query(UserProfile).filter(
                    UserProfile.user_id == user_id
                ).first()
                
                if profile:
                    user_profile = {
                        'typical_transaction_amount': float(profile.typical_transaction_amount or 0),
                        'typical_location_lat': float(profile.typical_location_lat or 0),
                        'typical_location_lon': float(profile.typical_location_lon or 0),
                        'avg_daily_transactions': profile.avg_daily_transactions,
                        'avg_monthly_spend': float(profile.avg_monthly_spend or 0)
                    }
            except Exception as e:
                logger.warning(f"Could not fetch user profile: {e}")
        
        # Run fraud detection (this always works)
        result = fraud_detector.detect(transaction_data, user_profile)
        
        # Try to save to database (gracefully handle if tables don't exist)
        try:
            # Save transaction to database
            transaction = Transaction(
                transaction_id=result['transaction_id'],
                user_id=uuid.UUID(user_id) if user_id else None,
                amount=Decimal(str(transaction_data.get('amount', 0))),
                currency=transaction_data.get('currency', 'USD'),
                transaction_type=transaction_data.get('type', 'purchase'),
                merchant_category=transaction_data.get('merchant_category'),
                merchant_name=transaction_data.get('merchant_name'),
                location_distance_km=Decimal(str(transaction_data.get('location_distance_km', 0))),
                timestamp=datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))
            )
            db.add(transaction)
            
            # Save fraud analysis result
            fraud_result = FraudAnalysisResult(
                transaction_id=result['transaction_id'],
                is_fraudulent=result['is_fraudulent'],
                risk_score=Decimal(str(result['risk_score'])),
                risk_level=result['risk_level'],
                ml_score=Decimal(str(result['ml_score'])) if result['ml_score'] is not None else None,
                rule_score=Decimal(str(result['rule_score'])),
                anomalies=result['anomalies'],
                recommendation=result['recommendation'],
                details=result['details'],
                model_version=result['details'].get('model_version')
            )
            db.add(fraud_result)
            
            # Create alert if high risk
            if result['is_fraudulent'] or result['risk_level'] in ['CRITICAL', 'HIGH']:
                alert = FraudAlert(
                    transaction_id=result['transaction_id'],
                    alert_type='FRAUD_DETECTION',
                    severity=result['risk_level'],
                    message=f"Fraudulent transaction detected: {result['recommendation']}",
                    status='PENDING'
                )
                db.add(alert)
            
            db.commit()
            logger.info(f"Transaction {result['transaction_id']} analyzed and saved to database")
            
        except Exception as e:
            # Database save failed (tables might not exist yet), but fraud detection still worked
            logger.warning(f"Could not save to database (tables may not exist): {e}")
            try:
                db.rollback()
            except:
                pass
        
        # Always return the fraud detection result
        return result
    
    @staticmethod
    def get_transaction_history(
        db: Session,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        risk_level: Optional[str] = None,
        is_fraudulent: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Get transaction history with fraud analysis results
        
        Args:
            db: Database session
            user_id: Optional filter by user ID
            limit: Maximum number of results
            offset: Offset for pagination
            risk_level: Optional filter by risk level
            is_fraudulent: Optional filter by fraud status
            
        Returns:
            Dictionary with transactions list and metadata
        """
        try:
            query = db.query(Transaction, FraudAnalysisResult).join(
                FraudAnalysisResult,
                Transaction.transaction_id == FraudAnalysisResult.transaction_id
            )
            
            # Apply filters
            if user_id:
                query = query.filter(Transaction.user_id == uuid.UUID(user_id))
            
            if risk_level:
                query = query.filter(FraudAnalysisResult.risk_level == risk_level)
            
            if is_fraudulent is not None:
                query = query.filter(FraudAnalysisResult.is_fraudulent == is_fraudulent)
            
            # Order by timestamp descending
            query = query.order_by(desc(Transaction.timestamp))
            
            # Get total count
            total = query.count()
            
            # Pagination
            results = query.limit(limit).offset(offset).all()
            
            # Format results
            transactions = []
            for transaction, fraud_result in results:
                transactions.append({
                    'transaction_id': transaction.transaction_id,
                    'user_id': str(transaction.user_id) if transaction.user_id else None,
                    'amount': float(transaction.amount),
                    'currency': transaction.currency,
                    'type': transaction.transaction_type,
                    'merchant_category': transaction.merchant_category,
                    'merchant_name': transaction.merchant_name,
                    'timestamp': transaction.timestamp.isoformat(),
                    'is_fraudulent': fraud_result.is_fraudulent,
                    'risk_score': float(fraud_result.risk_score),
                    'risk_level': fraud_result.risk_level,
                    'recommendation': fraud_result.recommendation,
                    'anomalies': fraud_result.anomalies
                })
            
            return {
                'transactions': transactions,
                'total': total,
                'limit': limit,
                'offset': offset
            }
        except Exception as e:
            logger.error(f"Error fetching transaction history: {e}")
            return {
                'transactions': [],
                'total': 0,
                'limit': limit,
                'offset': offset,
                'error': 'Database tables may not exist yet'
            }
    
    @staticmethod
    def get_active_alerts(
        db: Session,
        status: str = 'PENDING',
        severity: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get active fraud alerts
        
        Args:
            db: Database session
            status: Alert status (PENDING, REVIEWING, RESOLVED)
            severity: Optional filter by severity
            limit: Maximum number of results
            
        Returns:
            Dictionary with alerts list
        """
        try:
            query = db.query(FraudAlert).filter(FraudAlert.status == status)
            
            if severity:
                query = query.filter(FraudAlert.severity == severity)
            
            query = query.order_by(desc(FraudAlert.created_at))
            alerts = query.limit(limit).all()
            
            return {
                'alerts': [
                    {
                        'id': str(alert.id),
                        'transaction_id': alert.transaction_id,
                        'alert_type': alert.alert_type,
                        'severity': alert.severity,
                        'message': alert.message,
                        'status': alert.status,
                        'created_at': alert.created_at.isoformat(),
                        'assigned_to': str(alert.assigned_to) if alert.assigned_to else None
                    }
                    for alert in alerts
                ]
            }
        except Exception as e:
            logger.error(f"Error fetching alerts: {e}")
            return {
                'alerts': [],
                'error': 'Database tables may not exist yet'
            }
    
    @staticmethod
    def update_alert_status(
        db: Session,
        alert_id: str,
        status: str,
        assigned_to: Optional[str] = None,
        resolution_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update fraud alert status
        
        Args:
            db: Database session
            alert_id: Alert ID
            status: New status
            assigned_to: Optional user ID to assign to
            resolution_notes: Optional resolution notes
            
        Returns:
            Updated alert
        """
        try:
            alert = db.query(FraudAlert).filter(FraudAlert.id == uuid.UUID(alert_id)).first()
            
            if not alert:
                return {'error': f'Alert {alert_id} not found'}
            
            alert.status = status
            
            if assigned_to:
                alert.assigned_to = uuid.UUID(assigned_to)
            
            if resolution_notes:
                alert.resolution_notes = resolution_notes
            
            if status == 'RESOLVED':
                alert.resolved_at = datetime.utcnow()
            
            db.commit()
            
            return {
                'id': str(alert.id),
                'transaction_id': alert.transaction_id,
                'status': alert.status,
                'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None
            }
        except Exception as e:
            logger.error(f"Error updating alert: {e}")
            db.rollback()
            return {'error': str(e)}
    
    @staticmethod
    def get_user_profile(db: Session, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user risk profile
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            User profile or None
        """
        try:
            profile = db.query(UserProfile).filter(
                UserProfile.user_id == uuid.UUID(user_id)
            ).first()
            
            if not profile:
                return None
            
            return {
                'user_id': str(profile.user_id),
                'account_created_at': profile.account_created_at.isoformat() if profile.account_created_at else None,
                'typical_transaction_amount': float(profile.typical_transaction_amount or 0),
                'avg_daily_transactions': profile.avg_daily_transactions,
                'avg_monthly_spend': float(profile.avg_monthly_spend or 0),
                'risk_profile': profile.risk_profile,
                'last_updated': profile.last_updated.isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching user profile: {e}")
            return None
    
    @staticmethod
    def get_dashboard_analytics(db: Session, days: int = 30) -> Dict[str, Any]:
        """
        Get fraud detection analytics for dashboard
        
        Args:
            db: Database session
            days: Number of days to analyze
            
        Returns:
            Dashboard analytics
        """
        try:
            since = datetime.utcnow() - timedelta(days=days)
            
            # Total transactions
            total_transactions = db.query(Transaction).filter(
                Transaction.created_at >= since
            ).count()
            
            # Fraudulent transactions
            fraudulent_count = db.query(FraudAnalysisResult).filter(
                and_(
                    FraudAnalysisResult.analyzed_at >= since,
                    FraudAnalysisResult.is_fraudulent == True
                )
            ).count()
            
            # Fraud rate
            fraud_rate = (fraudulent_count / total_transactions * 100) if total_transactions > 0 else 0
            
            # Risk level distribution
            risk_distribution = {}
            for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
                count = db.query(FraudAnalysisResult).filter(
                    and_(
                        FraudAnalysisResult.analyzed_at >= since,
                        FraudAnalysisResult.risk_level == level
                    )
                ).count()
                risk_distribution[level] = count
            
            # Average risk score
            avg_risk_score = db.query(FraudAnalysisResult).filter(
                FraudAnalysisResult.analyzed_at >= since
            ).with_entities(FraudAnalysisResult.risk_score).all()
            
            avg_score = sum(float(score[0]) for score in avg_risk_score) / len(avg_risk_score) if avg_risk_score else 0
            
            # Active alerts
            active_alerts = db.query(FraudAlert).filter(
                FraudAlert.status == 'PENDING'
            ).count()
            
            return {
                'period_days': days,
                'total_transactions': total_transactions,
                'fraudulent_transactions': fraudulent_count,
                'fraud_rate': round(fraud_rate, 2),
                'average_risk_score': round(avg_score, 2),
                'risk_distribution': risk_distribution,
                'active_alerts': active_alerts,
                'model_version': fraud_detector.model_version,
                'model_trained': fraud_detector.is_trained
            }
        except Exception as e:
            logger.error(f"Error fetching dashboard analytics: {e}")
            return {
                'period_days': days,
                'total_transactions': 0,
                'fraudulent_transactions': 0,
                'fraud_rate': 0,
                'average_risk_score': 0,
                'risk_distribution': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0},
                'active_alerts': 0,
                'model_version': fraud_detector.model_version,
                'model_trained': fraud_detector.is_trained,
                'error': 'Database tables may not exist yet'
            }
