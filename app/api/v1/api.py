from fastapi import APIRouter
from app.api.v1.routes import chatbot, fraud, kyc, documents, insights, compliance, auth

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(chatbot.router, prefix="/chatbot", tags=["Chatbot"])
api_router.include_router(fraud.router, prefix="/fraud", tags=["Fraud Detection"])
api_router.include_router(kyc.router, prefix="/kyc", tags=["KYC Verification"])
api_router.include_router(documents.router, prefix="/documents", tags=["Document Processing"])
api_router.include_router(insights.router, prefix="/insights", tags=["Customer Insights"])
api_router.include_router(compliance.router, prefix="/compliance", tags=["Compliance Monitoring"])

