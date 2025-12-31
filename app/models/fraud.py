"""
Fraud Detection Database Models
SQLAlchemy models for fraud detection system
"""
from sqlalchemy import Column, String, Float, Boolean, DateTime, JSON, DECIMAL, Integer, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.database import Base


class Transaction(Base):
    """
    Transaction model - stores all financial transactions
    """
    __tablename__ = "transactions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), index=True)
    amount = Column(DECIMAL(15, 2), nullable=False)
    currency = Column(String(3), default='USD')
    transaction_type = Column(String(50), nullable=False)
    merchant_category = Column(String(100))
    merchant_name = Column(String(255))
    location_lat = Column(DECIMAL(10, 8))
    location_lon = Column(DECIMAL(11, 8))
    location_distance_km = Column(DECIMAL(10, 2))
    device_id = Column(String(255))
    ip_address = Column(INET)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationship
    fraud_results = relationship("FraudAnalysisResult", back_populates="transaction")
    
    def __repr__(self):
        return f"<Transaction {self.transaction_id}: ${self.amount}>"


class FraudAnalysisResult(Base):
    """
    Fraud Analysis Result model - stores fraud detection results
    """
    __tablename__ = "fraud_analysis_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id = Column(String(100), ForeignKey('transactions.transaction_id'), index=True)
    is_fraudulent = Column(Boolean, nullable=False)
    risk_score = Column(DECIMAL(5, 2), nullable=False, index=True)
    risk_level = Column(String(20), nullable=False)
    ml_score = Column(DECIMAL(5, 2))
    rule_score = Column(DECIMAL(5, 2))
    anomalies = Column(JSONB)
    recommendation = Column(Text)
    details = Column(JSONB)
    analyzed_at = Column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    model_version = Column(String(50))
    
    # Relationship
    transaction = relationship("Transaction", back_populates="fraud_results")
    
    def __repr__(self):
        return f"<FraudResult {self.transaction_id}: {self.risk_level}>"


class UserProfile(Base):
    """
    User Profile model - stores user behavioral patterns
    """
    __tablename__ = "user_profiles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), unique=True, nullable=False, index=True)
    account_created_at = Column(DateTime(timezone=True))
    typical_transaction_amount = Column(DECIMAL(15, 2))
    typical_location_lat = Column(DECIMAL(10, 8))
    typical_location_lon = Column(DECIMAL(11, 8))
    avg_daily_transactions = Column(Integer)
    avg_monthly_spend = Column(DECIMAL(15, 2))
    risk_profile = Column(String(20), default='LOW')
    last_updated = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserProfile {self.user_id}: {self.risk_profile}>"


class FraudAlert(Base):
    """
    Fraud Alert model - stores fraud alerts for review
    """
    __tablename__ = "fraud_alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id = Column(String(100))
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    status = Column(String(20), default='PENDING', index=True)
    assigned_to = Column(UUID(as_uuid=True))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime(timezone=True))
    resolution_notes = Column(Text)
    
    def __repr__(self):
        return f"<FraudAlert {self.id}: {self.severity} - {self.status}>"
