from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.services.compliance.monitor import compliance_monitor

router = APIRouter()

class ComplianceUpdate(BaseModel):
    title: str
    source: str
    summary: str
    risk_level: str
    date: str

class ComplianceAlertResponse(BaseModel):
    new_updates: List[ComplianceUpdate]
    total_active_alerts: int

@router.get("/alerts", response_model=ComplianceAlertResponse)
async def get_compliance_alerts():
    try:
        results = await compliance_monitor.check_updates()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
