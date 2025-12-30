"""
Production-Grade Document Parser Service
Implements OCR, NER, and structured data extraction from documents
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import uuid
import re

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import spacy
    # Try to load English model
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except:
        SPACY_AVAILABLE = False
        nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """Preprocess documents for better OCR results"""
    
    @staticmethod
    def preprocess_image(image_path: str) -> np.ndarray:
        """Preprocess image for OCR"""
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Deskew if needed
        coords = np.column_stack(np.where(denoised > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            if abs(angle) > 0.5:  # Only deskew if angle is significant
                (h, w) = denoised.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                denoised = cv2.warpAffine(
                    denoised, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
        
        return denoised


class OCREngine:
    """OCR text extraction engine"""
    
    def __init__(self):
        self.tesseract_available = TESSERACT_AVAILABLE
        if self.tesseract_available:
            tesseract_cmd = os.getenv("TESSERACT_CMD")
            if tesseract_cmd and os.path.exists(tesseract_cmd):
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    def extract_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract text from image"""
        if not self.tesseract_available:
            return {
                "success": False,
                "text": "",
                "error": "Tesseract OCR not available"
            }
        
        try:
            # Preprocess
            preprocessor = DocumentPreprocessor()
            processed_img = preprocessor.preprocess_image(image_path)
            
            # Extract text
            text = pytesseract.image_to_string(processed_img)
            
            # Get detailed data
            data = pytesseract.image_to_data(
                processed_img,
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate confidence
            confidences = [int(c) for c in data['conf'] if c != '-1']
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                "success": True,
                "text": text.strip(),
                "confidence": avg_confidence,
                "word_count": len(text.split()),
                "char_count": len(text)
            }
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text from PDF"""
        if not PDF_AVAILABLE:
            return {
                "success": False,
                "text": "",
                "error": "PyPDF2 not available"
            }
        
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            return {
                "success": True,
                "text": text.strip(),
                "confidence": 100,  # PDF text extraction is deterministic
                "page_count": len(reader.pages),
                "word_count": len(text.split()),
                "char_count": len(text)
            }
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return {
                "success": False,
                "text": "",
                "error": str(e)
            }


class EntityExtractor:
    """Extract named entities and structured data"""
    
    def __init__(self):
        self.spacy_available = SPACY_AVAILABLE
        self.nlp = nlp
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using spaCy"""
        if not self.spacy_available:
            # Fallback to regex-based extraction
            return self._regex_extraction(text)
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"NER error: {e}")
            return self._regex_extraction(text)
    
    def _regex_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Fallback regex-based entity extraction"""
        entities = []
        
        # Email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        for email in emails:
            entities.append({"text": email, "label": "EMAIL", "confidence": 0.9})
        
        # Phone numbers
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        for phone in phones:
            entities.append({"text": phone, "label": "PHONE", "confidence": 0.8})
        
        # Dates
        dates = re.findall(
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            text
        )
        for date in dates:
            entities.append({"text": date, "label": "DATE", "confidence": 0.85})
        
        # Money amounts
        money = re.findall(r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?', text)
        for amount in money:
            entities.append({"text": amount, "label": "MONEY", "confidence": 0.9})
        
        # Invoice/Order numbers
        invoice_nums = re.findall(r'\b(?:INV|ORDER|PO)[-#]?\s*\d+\b', text, re.IGNORECASE)
        for num in invoice_nums:
            entities.append({"text": num, "label": "INVOICE_NUMBER", "confidence": 0.85})
        
        return entities
    
    def extract_key_value_pairs(self, text: str) -> Dict[str, str]:
        """Extract key-value pairs from text"""
        pairs = {}
        
        # Common patterns like "Name: John Doe"
        pattern = r'([A-Z][a-z\s]+):\s*([^\n]+)'
        matches = re.findall(pattern, text)
        
        for key, value in matches:
            pairs[key.strip()] = value.strip()
        
        return pairs


class DocumentParser:
    """Main document parser orchestrator"""
    
    def __init__(self):
        self.ocr = OCREngine()
        self.ner = EntityExtractor()
        logger.info("Document parser initialized")
        logger.info(f"Tesseract available: {TESSERACT_AVAILABLE}")
        logger.info(f"SpaCy available: {SPACY_AVAILABLE}")
        logger.info(f"PDF support: {PDF_AVAILABLE}")
    
    def parse(self, file_path: str, file_type: str = None) -> Dict[str, Any]:
        """
        Parse document and extract structured data
        
        Args:
            file_path: Path to document file
            file_type: File type (pdf, image, etc.)
            
        Returns:
            Parsed document with text and entities
        """
        document_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Determine file type if not provided
        if not file_type:
            if file_path.lower().endswith('.pdf'):
                file_type = 'pdf'
            else:
                file_type = 'image'
        
        # Extract text
        if file_type == 'pdf':
            ocr_result = self.ocr.extract_from_pdf(file_path)
        else:
            ocr_result = self.ocr.extract_from_image(file_path)
        
        if not ocr_result["success"]:
            return {
                "document_id": document_id,
                "timestamp": timestamp,
                "success": False,
                "error": ocr_result.get("error", "OCR failed")
            }
        
        text = ocr_result["text"]
        
        # Extract entities
        entities = self.ner.extract_entities(text)
        
        # Extract key-value pairs
        key_value_pairs = self.ner.extract_key_value_pairs(text)
        
        # Categorize document
        doc_type = self._categorize_document(text, entities)
        
        return {
            "document_id": document_id,
            "timestamp": timestamp,
            "success": True,
            "document_type": doc_type,
            "extracted_text": text,
            "text_length": len(text),
            "word_count": ocr_result.get("word_count", 0),
            "ocr_confidence": ocr_result.get("confidence", 0),
            "entities": entities,
            "key_value_pairs": key_value_pairs,
            "metadata": {
                "file_type": file_type,
                "page_count": ocr_result.get("page_count"),
                "extraction_method": "Tesseract OCR" if file_type == 'image' else "PDF Text",
                "ner_method": "spaCy" if SPACY_AVAILABLE else "Regex"
            }
        }
    
    def _categorize_document(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> str:
        """Categorize document type based on content"""
        text_lower = text.lower()
        
        # Check for invoice indicators
        invoice_keywords = ['invoice', 'bill', 'payment due', 'total amount']
        if any(keyword in text_lower for keyword in invoice_keywords):
            return "invoice"
        
        # Check for contract indicators
        contract_keywords = ['agreement', 'contract', 'terms and conditions', 'parties']
        if any(keyword in text_lower for keyword in contract_keywords):
            return "contract"
        
        # Check for receipt indicators
        receipt_keywords = ['receipt', 'purchased', 'transaction', 'thank you for your purchase']
        if any(keyword in text_lower for keyword in receipt_keywords):
            return "receipt"
        
        # Check for ID document
        id_keywords = ['passport', 'driver license', 'identification', 'date of birth']
        if any(keyword in text_lower for keyword in id_keywords):
            return "identification"
        
        return "general"


# Global instance
document_parser = DocumentParser()

import os
