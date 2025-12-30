from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time

router = APIRouter()

class ChatMessage(BaseModel):
    text: str
    sender_id: str

class ChatResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    timestamp: float

@router.post("/chat", response_model=ChatResponse)
async def chat_message(message: ChatMessage):
    # This is a stub. In a real scenario, this would call Rasa or a Transformer model.
    # For now, it's a rule-based prototype.
    text = message.text.lower()
    
    intent = "unknown"
    response_text = "I'm sorry, I'm not sure how to help with that. Could you try rephrasing?"
    
    if "balance" in text:
        intent = "check_balance"
        response_text = "Your current account balance is $24,560.75 across Savings and Checking."
    elif "transfer" in text:
        intent = "transfer_money"
        response_text = "I can help with that. Please specify the recipient and the amount."
    elif "hello" in text or "hi" in text:
        intent = "greet"
        response_text = "Hello! I'm your FutureX Banking Assistant. How can I help you today?"
        
    return ChatResponse(
        text=response_text,
        intent=intent,
        confidence=0.95 if intent != "unknown" else 0.0,
        timestamp=time.time()
    )
