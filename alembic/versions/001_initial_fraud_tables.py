"""Add fraud detection tables

Revision ID: 001_initial
Revises: 
Create Date: 2025-12-31 13:30:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create transactions table
    op.create_table(
        'transactions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('transaction_id', sa.String(100), nullable=False, unique=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True)),
        sa.Column('amount', sa.DECIMAL(15, 2), nullable=False),
        sa.Column('currency', sa.String(3), server_default='USD'),
        sa.Column('transaction_type', sa.String(50), nullable=False),
        sa.Column('merchant_category', sa.String(100)),
        sa.Column('merchant_name', sa.String(255)),
        sa.Column('location_lat', sa.DECIMAL(10, 8)),
        sa.Column('location_lon', sa.DECIMAL(11, 8)),
        sa.Column('location_distance_km', sa.DECIMAL(10, 2)),
        sa.Column('device_id', sa.String(255)),
        sa.Column('ip_address', postgresql.INET()),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )
    
    # Create indexes for transactions
    op.create_index('idx_transactions_transaction_id', 'transactions', ['transaction_id'])
    op.create_index('idx_transactions_user_id', 'transactions', ['user_id'])
    op.create_index('idx_transactions_timestamp', 'transactions', ['timestamp'])
    op.create_index('idx_transactions_amount', 'transactions', ['amount'])
    
    # Create fraud_analysis_results table
    op.create_table(
        'fraud_analysis_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('transaction_id', sa.String(100), sa.ForeignKey('transactions.transaction_id')),
        sa.Column('is_fraudulent', sa.Boolean(), nullable=False),
        sa.Column('risk_score', sa.DECIMAL(5, 2), nullable=False),
        sa.Column('risk_level', sa.String(20), nullable=False),
        sa.Column('ml_score', sa.DECIMAL(5, 2)),
        sa.Column('rule_score', sa.DECIMAL(5, 2)),
        sa.Column('anomalies', postgresql.JSONB()),
        sa.Column('recommendation', sa.Text()),
        sa.Column('details', postgresql.JSONB()),
        sa.Column('analyzed_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('model_version', sa.String(50)),
    )
    
    # Create indexes for fraud_analysis_results
    op.create_index('idx_fraud_results_transaction', 'fraud_analysis_results', ['transaction_id'])
    op.create_index('idx_fraud_results_risk_score', 'fraud_analysis_results', ['risk_score'])
    op.create_index('idx_fraud_results_analyzed_at', 'fraud_analysis_results', ['analyzed_at'])
    
    # Create user_profiles table
    op.create_table(
        'user_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column('account_created_at', sa.DateTime(timezone=True)),
        sa.Column('typical_transaction_amount', sa.DECIMAL(15, 2)),
        sa.Column('typical_location_lat', sa.DECIMAL(10, 8)),
        sa.Column('typical_location_lon', sa.DECIMAL(11, 8)),
        sa.Column('avg_daily_transactions', sa.Integer()),
        sa.Column('avg_monthly_spend', sa.DECIMAL(15, 2)),
        sa.Column('risk_profile', sa.String(20), server_default='LOW'),
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )
    
    # Create index for user_profiles
    op.create_index('idx_user_profiles_user_id', 'user_profiles', ['user_id'])
    
    # Create fraud_alerts table
    op.create_table(
        'fraud_alerts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('transaction_id', sa.String(100)),
        sa.Column('alert_type', sa.String(50), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('status', sa.String(20), server_default='PENDING'),
        sa.Column('assigned_to', postgresql.UUID(as_uuid=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('resolved_at', sa.DateTime(timezone=True)),
        sa.Column('resolution_notes', sa.Text()),
    )
    
    # Create indexes for fraud_alerts
    op.create_index('idx_fraud_alerts_status', 'fraud_alerts', ['status'])
    op.create_index('idx_fraud_alerts_created_at', 'fraud_alerts', ['created_at'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('fraud_alerts')
    op.drop_table('user_profiles')
    op.drop_table('fraud_analysis_results')
    op.drop_table('transactions')
