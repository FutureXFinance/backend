from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.services.insights.recommender import insights_engine

router = APIRouter()

class UserInsightRequest(BaseModel):
    user_id: str
    data_points: Optional[List[Dict[str, Any]]] = None

class Recommendation(BaseModel):
    product_name: str
    product_type: str
    relevance_score: float
    reasoning: str

class UserInsightsResponse(BaseModel):
    user_id: str
    top_recommendations: List[Recommendation]
    spending_analysis: Dict[str, Any]

@router.post("/analyze", response_model=UserInsightsResponse)
async def get_user_insights(request: UserInsightRequest):
    try:
        results = await insights_engine.get_recommendations(request.user_id)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
