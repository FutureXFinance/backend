"""
Production-Grade KYC Verification Service
Implements OCR, face detection, and document validation
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
import os
import uuid
import logging
from pathlib import Path
from datetime import datetime

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentValidator:
    """Validates document quality and authenticity"""
    
    @staticmethod
    def check_image_quality(image_path: str) -> Tuple[bool, str]:
        """Check if image meets quality requirements"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "Unable to read image file"
            
            # Check resolution
            height, width = img.shape[:2]
            if width < 600 or height < 400:
                return False, f"Image resolution too low ({width}x{height}). Minimum: 600x400"
            
            # Check if image is too blurry using Laplacian variance
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var < 100:
                return False, f"Image too blurry (score: {laplacian_var:.2f})"
            
            # Check brightness
            mean_brightness = np.mean(gray)
            if mean_brightness < 50:
                return False, "Image too dark"
            if mean_brightness > 200:
                return False, "Image too bright"
            
            return True, "Image quality acceptable"
            
        except Exception as e:
            logger.error(f"Quality check error: {e}")
            return False, f"Quality check failed: {str(e)}"
    
    @staticmethod
    def detect_document_edges(image_path: str) -> bool:
        """Detect if document edges are visible"""
        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Count edge pixels
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.size
            edge_ratio = edge_pixels / total_pixels
            
            # Document should have clear edges
            return 0.05 < edge_ratio < 0.3
            
        except Exception as e:
            logger.error(f"Edge detection error: {e}")
            return False


class OCRProcessor:
    """Handles OCR text extraction from documents"""
    
    def __init__(self):
        self.tesseract_available = TESSERACT_AVAILABLE
        if self.tesseract_available:
            # Configure Tesseract path if needed
            tesseract_cmd = os.getenv("TESSERACT_CMD")
            if tesseract_cmd and os.path.exists(tesseract_cmd):
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for better OCR results"""
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        return denoised
    
    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """Extract text from document image"""
        if not self.tesseract_available:
            return {
                "success": False,
                "text": "",
                "error": "Tesseract OCR not available",
                "confidence": 0.0
            }
        
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Extract text with confidence scores
            data = pytesseract.image_to_data(
                processed_img, 
                output_type=pytesseract.Output.DICT
            )
            
            # Get full text
            text = pytesseract.image_to_string(processed_img)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Extract structured data
            structured_data = self._extract_structured_info(text)
            
            return {
                "success": True,
                "text": text.strip(),
                "confidence": avg_confidence,
                "structured_data": structured_data,
                "word_count": len(text.split())
            }
            
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return {
                "success": False,
                "text": "",
                "error": str(e),
                "confidence": 0.0
            }
    
    def _extract_structured_info(self, text: str) -> Dict[str, Any]:
        """Extract structured information from OCR text"""
        import re
        
        structured = {
            "id_numbers": [],
            "dates": [],
            "names": []
        }
        
        # Extract ID-like patterns (numbers with specific formats)
        id_patterns = re.findall(r'\b[A-Z0-9]{6,15}\b', text)
        structured["id_numbers"] = id_patterns[:3]  # Limit to first 3
        
        # Extract date patterns
        date_patterns = re.findall(
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            text
        )
        structured["dates"] = date_patterns[:3]
        
        # Extract capitalized words (potential names)
        name_patterns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        structured["names"] = name_patterns[:5]
        
        return structured


class FaceVerifier:
    """Handles face detection and matching"""
    
    def __init__(self):
        self.face_recognition_available = FACE_RECOGNITION_AVAILABLE
    
    def detect_face(self, image_path: str) -> Tuple[bool, str, int]:
        """Detect face in image using OpenCV"""
        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Use Haar Cascade for face detection
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            face_count = len(faces)
            
            if face_count == 0:
                return False, "No face detected", 0
            elif face_count > 1:
                return False, f"Multiple faces detected ({face_count})", face_count
            else:
                return True, "Single face detected", 1
                
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return False, f"Detection failed: {str(e)}", 0
    
    def match_faces(self, id_image_path: str, selfie_path: str) -> Dict[str, Any]:
        """Match faces between ID and selfie"""
        
        # First check if faces exist in both images
        id_face_ok, id_msg, id_count = self.detect_face(id_image_path)
        selfie_face_ok, selfie_msg, selfie_count = self.detect_face(selfie_path)
        
        if not id_face_ok:
            return {
                "match": False,
                "confidence": 0.0,
                "reason": f"ID image: {id_msg}"
            }
        
        if not selfie_face_ok:
            return {
                "match": False,
                "confidence": 0.0,
                "reason": f"Selfie: {selfie_msg}"
            }
        
        # If face_recognition library is available, use it for matching
        if self.face_recognition_available:
            try:
                id_img = face_recognition.load_image_file(id_image_path)
                selfie_img = face_recognition.load_image_file(selfie_path)
                
                id_encodings = face_recognition.face_encodings(id_img)
                selfie_encodings = face_recognition.face_encodings(selfie_img)
                
                if len(id_encodings) > 0 and len(selfie_encodings) > 0:
                    # Compare faces
                    matches = face_recognition.compare_faces(
                        [id_encodings[0]], selfie_encodings[0], tolerance=0.6
                    )
                    distance = face_recognition.face_distance(
                        [id_encodings[0]], selfie_encodings[0]
                    )[0]
                    
                    confidence = (1.0 - distance) * 100  # Convert to percentage
                    is_match = matches[0] and confidence > 60
                    
                    return {
                        "match": is_match,
                        "confidence": round(confidence, 2),
                        "reason": "Face recognition match" if is_match else "Faces do not match"
                    }
                    
            except Exception as e:
                logger.error(f"Face matching error: {e}")
        
        # Fallback: Use basic face detection similarity
        return self._basic_face_comparison(id_image_path, selfie_path)
    
    def _basic_face_comparison(self, img1_path: str, img2_path: str) -> Dict[str, Any]:
        """Basic face comparison using histogram similarity"""
        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(selfie_path)
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Resize to same size
            gray2_resized = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
            
            # Calculate histogram correlation
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2_resized], [0], None, [256], [0, 256])
            
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            confidence = correlation * 100
            
            return {
                "match": confidence > 70,
                "confidence": round(confidence, 2),
                "reason": "Basic histogram comparison (face_recognition library not available)"
            }
            
        except Exception as e:
            logger.error(f"Basic comparison error: {e}")
            return {
                "match": False,
                "confidence": 0.0,
                "reason": f"Comparison failed: {str(e)}"
            }


class KYCVerifier:
    """Main KYC verification orchestrator"""
    
    def __init__(self):
        self.validator = DocumentValidator()
        self.ocr = OCRProcessor()
        self.face_verifier = FaceVerifier()
        logger.info("KYC Verifier initialized")
        logger.info(f"Tesseract available: {TESSERACT_AVAILABLE}")
        logger.info(f"Face recognition available: {FACE_RECOGNITION_AVAILABLE}")
    
    async def verify(self, id_path: str, selfie_path: str) -> Dict[str, Any]:
        """
        Complete KYC verification workflow
        
        Args:
            id_path: Path to ID document image
            selfie_path: Path to selfie image
            
        Returns:
            Verification result with status and details
        """
        verification_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Step 1: Validate image quality
        id_quality_ok, id_quality_msg = self.validator.check_image_quality(id_path)
        selfie_quality_ok, selfie_quality_msg = self.validator.check_image_quality(selfie_path)
        
        if not id_quality_ok:
            return self._create_rejection_result(
                verification_id, timestamp, f"ID quality issue: {id_quality_msg}"
            )
        
        if not selfie_quality_ok:
            return self._create_rejection_result(
                verification_id, timestamp, f"Selfie quality issue: {selfie_quality_msg}"
            )
        
        # Step 2: OCR extraction from ID
        ocr_result = self.ocr.extract_text(id_path)
        
        if not ocr_result["success"]:
            logger.warning(f"OCR failed: {ocr_result.get('error')}")
        
        # Step 3: Face verification
        face_match_result = self.face_verifier.match_faces(id_path, selfie_path)
        
        # Step 4: Determine verification status
        is_verified = (
            face_match_result["match"] and 
            face_match_result["confidence"] > 60 and
            ocr_result.get("confidence", 0) > 50
        )
        
        status = "VERIFIED" if is_verified else "REJECTED"
        
        return {
            "verification_id": verification_id,
            "status": status,
            "timestamp": timestamp,
            "face_match_confidence": face_match_result["confidence"],
            "details": {
                "id_quality": id_quality_msg,
                "selfie_quality": selfie_quality_msg,
                "face_match": face_match_result["reason"],
                "face_detected": face_match_result["match"],
                "ocr_confidence": ocr_result.get("confidence", 0),
                "ocr_success": ocr_result["success"],
                "extracted_text_preview": ocr_result.get("text", "")[:200],
                "structured_data": ocr_result.get("structured_data", {}),
                "verification_method": "production" if FACE_RECOGNITION_AVAILABLE else "fallback"
            }
        }
    
    def _create_rejection_result(
        self, verification_id: str, timestamp: str, reason: str
    ) -> Dict[str, Any]:
        """Create a rejection result"""
        return {
            "verification_id": verification_id,
            "status": "REJECTED",
            "timestamp": timestamp,
            "face_match_confidence": 0.0,
            "details": {
                "rejection_reason": reason,
                "face_detected": False,
                "ocr_success": False
            }
        }


# Global instance
kyc_verifier = KYCVerifier()
