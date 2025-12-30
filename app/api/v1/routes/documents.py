from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import os
import aiofiles
from app.services.documents.parser import document_parser

router = APIRouter()

class ParsingResult(BaseModel):
    document_id: str
    filename: str
    extracted_data: Dict[str, Any]
    status: str

@router.post("/parse", response_model=ParsingResult)
async def parse_document(file: UploadFile = File(...)):
    upload_dir = "uploads/documents"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = f"{upload_dir}/{uuid.uuid4()}_{file.filename}"
    
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
            
        # Call documentation parsing service
        result = document_parser.parse(file_path)
        
        return ParsingResult(
            document_id=str(uuid.uuid4()),
            filename=file.filename,
            extracted_data=result,
            status="SUCCESS"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing Error: {str(e)}")
