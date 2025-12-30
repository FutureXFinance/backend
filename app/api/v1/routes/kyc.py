"""
KYC Verification API Routes
Handles document upload and identity verification
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
import os
import aiofiles
from pathlib import Path
import logging

from app.services.kyc.verifier import kyc_verifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration
UPLOAD_DIR = Path("uploads/kyc")
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class KYCResult(BaseModel):
    """KYC Verification Result Schema"""
    verification_id: str
    status: str = Field(..., description="VERIFIED, REJECTED, or PENDING")
    timestamp: str
    face_match_confidence: float = Field(..., ge=0, le=100)
    details: Dict[str, Any]


class VerificationStatus(BaseModel):
    """Verification status response"""
    verification_id: str
    status: str
    created_at: str


def validate_file(file: UploadFile) -> tuple[bool, str]:
    """Validate uploaded file"""
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check content type
    allowed_content_types = [
        "image/jpeg", "image/jpg", "image/png", "application/pdf"
    ]
    if file.content_type not in allowed_content_types:
        return False, f"Invalid content type: {file.content_type}"
    
    return True, "Valid"


async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """Save uploaded file to disk"""
    try:
        async with aiofiles.open(destination, 'wb') as f:
            content = await upload_file.read()
            
            # Check file size
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"
                )
            
            await f.write(content)
            
    except Exception as e:
        logger.error(f"File save error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


@router.post("/verify", response_model=KYCResult)
async def verify_identity(
    id_document: UploadFile = File(..., description="ID card or passport image"),
    selfie: UploadFile = File(..., description="Selfie photo for face matching")
):
    """
    Verify identity by matching ID document with selfie
    
    - **id_document**: Clear photo of government-issued ID (passport, driver's license, etc.)
    - **selfie**: Recent selfie photo for face matching
    
    Returns verification result with confidence score and extracted data
    """
    
    # Validate files
    id_valid, id_msg = validate_file(id_document)
    if not id_valid:
        raise HTTPException(status_code=400, detail=f"ID document: {id_msg}")
    
    selfie_valid, selfie_msg = validate_file(selfie)
    if not selfie_valid:
        raise HTTPException(status_code=400, detail=f"Selfie: {selfie_msg}")
    
    # Generate unique filenames
    id_filename = f"{uuid.uuid4()}_{id_document.filename}"
    selfie_filename = f"{uuid.uuid4()}_{selfie.filename}"
    
    id_path = UPLOAD_DIR / id_filename
    selfie_path = UPLOAD_DIR / selfie_filename
    
    try:
        # Save files
        await save_upload_file(id_document, id_path)
        await save_upload_file(selfie, selfie_path)
        
        logger.info(f"Processing KYC verification: {id_filename}")
        
        # Perform verification
        result = await kyc_verifier.verify(str(id_path), str(selfie_path))
        
        logger.info(f"Verification complete: {result['verification_id']} - {result['status']}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"KYC verification error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        )
    finally:
        # Optional: Clean up files after processing
        # In production, you might want to keep them or move to cloud storage
        pass


@router.post("/verify-single", response_model=Dict[str, Any])
async def verify_single_document(
    document: UploadFile = File(..., description="Document to verify")
):
    """
    Verify a single document (OCR and quality check only)
    
    Useful for document validation without face matching
    """
    
    # Validate file
    valid, msg = validate_file(document)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)
    
    # Generate unique filename
    filename = f"{uuid.uuid4()}_{document.filename}"
    file_path = UPLOAD_DIR / filename
    
    try:
        # Save file
        await save_upload_file(document, file_path)
        
        # Perform OCR and quality check
        from app.services.kyc.verifier import DocumentValidator, OCRProcessor
        
        validator = DocumentValidator()
        ocr = OCRProcessor()
        
        # Check quality
        quality_ok, quality_msg = validator.check_image_quality(str(file_path))
        
        # Extract text
        ocr_result = ocr.extract_text(str(file_path))
        
        return {
            "document_id": str(uuid.uuid4()),
            "quality_check": {
                "passed": quality_ok,
                "message": quality_msg
            },
            "ocr_result": ocr_result,
            "filename": document.filename
        }
        
    except Exception as e:
        logger.error(f"Document verification error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        )


@router.get("/status/{verification_id}", response_model=VerificationStatus)
async def get_verification_status(verification_id: str):
    """
    Get status of a verification request
    
    In a production system, this would query a database
    """
    # TODO: Implement database lookup
    # For now, return mock response
    return {
        "verification_id": verification_id,
        "status": "PENDING",
        "created_at": "2024-01-01T00:00:00Z"
    }


@router.get("/health")
async def health_check():
    """Check KYC service health"""
    from app.services.kyc.verifier import TESSERACT_AVAILABLE, FACE_RECOGNITION_AVAILABLE
    
    return {
        "status": "healthy",
        "tesseract_available": TESSERACT_AVAILABLE,
        "face_recognition_available": FACE_RECOGNITION_AVAILABLE,
        "upload_dir": str(UPLOAD_DIR),
        "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024
    }


@router.delete("/cleanup")
async def cleanup_old_files():
    """
    Clean up old uploaded files
    
    In production, implement proper file retention policy
    """
    try:
        import time
        from datetime import datetime, timedelta
        
        # Delete files older than 24 hours
        cutoff_time = time.time() - (24 * 60 * 60)
        deleted_count = 0
        
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                deleted_count += 1
        
        return {
            "status": "success",
            "deleted_files": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
