"""
Deterministic document classification service.
Uses keyword rules and regex patterns only - no LLM fallback.
Returns "unclassified" if no confident match is found.
"""

import logging
import re
from typing import Literal

from config import ClassificationConfig
from services.ocr import OCRResult, get_ocr_service

logger = logging.getLogger(__name__)

DocumentType = Literal["CNIC", "DOMICILE", "TRANSCRIPT", "unclassified"]


class ClassificationService:
    """Deterministic document classification based on keywords and patterns."""
    
    def __init__(self):
        """Initialize classification service."""
        self.ocr_service = get_ocr_service()
    
    def classify(self, ocr_results: list[OCRResult]) -> DocumentType:
        """
        Classify document based on OCR results using deterministic rules.
        
        Priority order: DOMICILE -> CNIC -> TRANSCRIPT
        If no rule matches confidently, return "unclassified".
        
        Args:
            ocr_results: List of OCRResult objects from OCR
            
        Returns:
            Document type: "CNIC", "DOMICILE", "TRANSCRIPT", or "unclassified"
        """
        # Get full text for pattern matching
        full_text = self.ocr_service.get_full_text(ocr_results)
        text_lower = full_text.lower()
        
        # Check DOMICILE keywords (highest priority)
        if self._check_domicile_keywords(ocr_results, text_lower):
            logger.info("Document classified as DOMICILE")
            return "DOMICILE"
        
        # Check CNIC keywords and pattern
        if self._check_cnic_keywords(ocr_results, text_lower) or self._check_cnic_pattern(full_text):
            logger.info("Document classified as CNIC")
            return "CNIC"
        
        # Check TRANSCRIPT keywords
        if self._check_transcript_keywords(ocr_results, text_lower):
            logger.info("Document classified as TRANSCRIPT")
            return "TRANSCRIPT"
        
        # No confident match found
        logger.warning("Document could not be confidently classified")
        return "unclassified"
    
    def _check_domicile_keywords(self, ocr_results: list[OCRResult], text_lower: str) -> bool:
        """
        Check if document contains DOMICILE keywords.
        
        Args:
            ocr_results: List of OCRResult objects
            text_lower: Full text in lowercase
            
        Returns:
            True if domicile keywords found
        """
        keyword_matches = self.ocr_service.find_text_by_keywords(
            ocr_results, 
            ClassificationConfig.DOMICILE_KEYWORDS, 
            case_sensitive=False
        )
        
        # Check if minimum number of keywords are present
        if len(keyword_matches) >= ClassificationConfig.MIN_KEYWORD_MATCHES:
            logger.debug(f"Domicile keywords found: {len(keyword_matches)}")
            return True
        
        # Also check in full text for additional coverage
        full_text_matches = sum(1 for keyword in ClassificationConfig.DOMICILE_KEYWORDS 
                               if keyword.lower() in text_lower)
        
        return full_text_matches >= ClassificationConfig.MIN_KEYWORD_MATCHES
    
    def _check_cnic_keywords(self, ocr_results: list[OCRResult], text_lower: str) -> bool:
        """
        Check if document contains CNIC keywords.
        
        Args:
            ocr_results: List of OCRResult objects
            text_lower: Full text in lowercase
            
        Returns:
            True if CNIC keywords found
        """
        keyword_matches = self.ocr_service.find_text_by_keywords(
            ocr_results, 
            ClassificationConfig.CNIC_KEYWORDS, 
            case_sensitive=False
        )
        
        if len(keyword_matches) >= ClassificationConfig.MIN_KEYWORD_MATCHES:
            logger.debug(f"CNIC keywords found: {len(keyword_matches)}")
            return True
        
        # Check in full text
        full_text_matches = sum(1 for keyword in ClassificationConfig.CNIC_KEYWORDS 
                               if keyword.lower() in text_lower)
        
        return full_text_matches >= ClassificationConfig.MIN_KEYWORD_MATCHES
    
    def _check_cnic_pattern(self, full_text: str) -> bool:
        """
        Check if document contains CNIC number pattern.
        
        Args:
            full_text: Full text from OCR
            
        Returns:
            True if CNIC pattern found
        """
        pattern = ClassificationConfig.CNIC_PATTERN
        matches = re.findall(pattern, full_text)
        
        if matches:
            logger.debug(f"CNIC pattern found: {len(matches)} matches")
            return True
        
        return False
    
    def _check_transcript_keywords(self, ocr_results: list[OCRResult], text_lower: str) -> bool:
        """
        Check if document contains TRANSCRIPT keywords.
        
        Args:
            ocr_results: List of OCRResult objects
            text_lower: Full text in lowercase
            
        Returns:
            True if transcript keywords found
        """
        keyword_matches = self.ocr_service.find_text_by_keywords(
            ocr_results, 
            ClassificationConfig.TRANSCRIPT_KEYWORDS, 
            case_sensitive=False
        )
        
        if len(keyword_matches) >= ClassificationConfig.MIN_KEYWORD_MATCHES:
            logger.debug(f"Transcript keywords found: {len(keyword_matches)}")
            return True
        
        # Check in full text
        full_text_matches = sum(1 for keyword in ClassificationConfig.TRANSCRIPT_KEYWORDS 
                               if keyword.lower() in text_lower)
        
        return full_text_matches >= ClassificationConfig.MIN_KEYWORD_MATCHES
    
    def get_classification_confidence(self, ocr_results: list[OCRResult], 
                                    doc_type: DocumentType) -> float:
        """
        Get confidence score for classification based on keyword matches.
        
        Args:
            ocr_results: List of OCRResult objects
            doc_type: The determined document type
            
        Returns:
            Confidence score between 0 and 1
        """
        if doc_type == "unclassified":
            return 0.0
        
        full_text = self.ocr_service.get_full_text(ocr_results)
        text_lower = full_text.lower()
        
        # Count keyword matches based on document type
        if doc_type == "DOMICILE":
            keywords = ClassificationConfig.DOMICILE_KEYWORDS
        elif doc_type == "CNIC":
            keywords = ClassificationConfig.CNIC_KEYWORDS
        elif doc_type == "TRANSCRIPT":
            keywords = ClassificationConfig.TRANSCRIPT_KEYWORDS
        else:
            return 0.0
        
        # Count matches in text
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        
        # Simple confidence based on keyword matches
        # More matches = higher confidence
        confidence = min(matches / max(len(keywords) * 0.3, 1), 1.0)
        
        logger.debug(f"Classification confidence for {doc_type}: {confidence:.2f}")
        return confidence


# Singleton instance
_classification_service = None

def get_classification_service() -> ClassificationService:
    """Get the singleton classification service instance."""
    global _classification_service
    if _classification_service is None:
        _classification_service = ClassificationService()
    return _classification_service