"""
Transcript extraction service using label-anchored deterministic extraction with LLM fallback.
First tries anchor-based extraction; falls back to LLM if insufficient fields are found.
"""

import logging
import re

from pydantic import BaseModel, ValidationError

from config import TranscriptExtractionConfig
from services.extract_cnic import ExtractedField
from services.llm_service import get_llm_service
from services.ocr import OCRResult, get_ocr_service

logger = logging.getLogger(__name__)


class TranscriptData(BaseModel):
    """Pydantic model for transcript data validation."""
    roll_number: str
    student_name: str
    father_name: str | None = None
    degree_program: str | None = None
    total_marks: str | None = None
    obtained_marks: str | None = None
    grade: str | None = None
    semester: str | None = None


class TranscriptExtractor:
    """Transcript field extraction with anchor-based approach and LLM fallback."""
    
    def __init__(self):
        """Initialize Transcript extractor."""
        self.ocr_service = get_ocr_service()
        self.llm_service = get_llm_service()
    
    def extract(self, ocr_results: list[OCRResult]) -> dict[str, ExtractedField]:
        """
        Extract Transcript fields using anchor-based approach with LLM fallback.
        
        Args:
            ocr_results: List of OCRResult objects
            
        Returns:
            Dictionary mapping field names to ExtractedField objects
        """
        # Try anchor-based extraction first
        extracted_fields = self._anchor_based_extraction(ocr_results)
        
        # Check if we extracted enough fields
        if self._sufficient_fields_extracted(extracted_fields):
            logger.info("Anchor-based extraction sufficient, skipping LLM")
            return extracted_fields
        
        # Fall back to LLM if insufficient fields
        logger.info("Insufficient fields from anchor-based extraction, trying LLM fallback")
        llm_fields = self._llm_fallback_extraction(ocr_results)
        
        if llm_fields:
            # Merge LLM results with anchor results (LLM takes precedence)
            extracted_fields.update(llm_fields)
            logger.info("LLM fallback extraction successful")
        else:
            logger.warning("LLM fallback failed, using anchor-based results only")
        
        return extracted_fields
    
    def _anchor_based_extraction(self, ocr_results: list[OCRResult]) -> dict[str, ExtractedField]:
        """
        Extract fields using label-anchored approach.
        
        Args:
            ocr_results: List of OCRResult objects
            
        Returns:
            Dictionary mapping field names to ExtractedField objects
        """
        extracted_fields = {}
        
        # Extract each field using its label anchors
        for field_name, label_variations in TranscriptExtractionConfig.LABEL_ANCHORS.items():
            field = self._extract_field_by_anchor(ocr_results, field_name, label_variations)
            if field:
                extracted_fields[field_name] = field
                logger.debug(f"Extracted {field_name}: {field.value} (confidence: {field.extraction_confidence:.2f})")
        
        # Try pattern-based extraction for roll numbers as additional fallback
        if 'roll_number' not in extracted_fields:
            roll_field = self._extract_roll_number_by_pattern(ocr_results)
            if roll_field:
                extracted_fields['roll_number'] = roll_field
        
        return extracted_fields
    
    def _extract_field_by_anchor(self, ocr_results: list[OCRResult], 
                                 field_name: str, 
                                 label_variations: list[str]) -> ExtractedField | None:
        """
        Extract a field by finding its label anchor and nearby value.
        
        Args:
            ocr_results: List of OCRResult objects
            field_name: Name of the field to extract
            label_variations: List of possible label text variations
            
        Returns:
            ExtractedField object or None if not found
        """
        # Find label anchors
        label_results = []
        for variation in label_variations:
            matches = self.ocr_service.find_text_by_keywords(
                ocr_results, 
                {variation}, 
                case_sensitive=False
            )
            label_results.extend(matches)
        
        if not label_results:
            logger.debug(f"No label anchor found for {field_name}")
            return None
        
        # Use the first/most prominent label
        best_label = self._select_best_label(label_results)
        
        # Find nearby text (value)
        nearby_results = self.ocr_service.find_text_nearby(
            ocr_results,
            best_label,
            max_distance=TranscriptExtractionConfig.SEARCH_RADIUS,
            direction='right_or_below'
        )
        
        if not nearby_results:
            logger.debug(f"No nearby text found for {field_name}")
            return None
        
        # Select the best candidate for the value
        best_value = self._select_best_value(nearby_results, field_name)
        
        # Calculate extraction confidence based on OCR confidence
        confidence = self._calculate_extraction_confidence(best_label, best_value)
        
        return ExtractedField(
            field_name=field_name,
            value=best_value.text,
            source_bbox=best_value.bbox,
            extraction_confidence=confidence,
            extraction_method="anchor"
        )
    
    def _extract_roll_number_by_pattern(self, ocr_results: list[OCRResult]) -> ExtractedField | None:
        """
        Extract roll number using pattern matching as fallback.
        
        Args:
            ocr_results: List of OCRResult objects
            
        Returns:
            ExtractedField object or None if not found
        """
        for pattern in TranscriptExtractionConfig.ROLL_NUMBER_PATTERNS:
            for result in ocr_results:
                if re.match(pattern, result.text.strip()):
                    logger.debug(f"Found roll number by pattern: {result.text}")
                    return ExtractedField(
                        field_name='roll_number',
                        value=result.text.strip(),
                        source_bbox=result.bbox,
                        extraction_confidence=result.confidence,
                        extraction_method="pattern"
                    )
        
        return None
    
    def _select_best_label(self, label_results: list[OCRResult]) -> OCRResult:
        """
        Select the best label from multiple candidates.
        
        Args:
            label_results: List of OCRResult objects that match label variations
            
        Returns:
            Best OCRResult to use as label
        """
        # Prefer the label with highest confidence
        return max(label_results, key=lambda x: x.confidence)
    
    def _select_best_value(self, nearby_results: list[OCRResult], 
                         field_name: str) -> OCRResult:
        """
        Select the best value from nearby text candidates.
        
        Args:
            nearby_results: List of nearby OCRResult objects
            field_name: Name of the field being extracted
            
        Returns:
            Best OCRResult to use as value
        """
        # Filter candidates based on field-specific criteria
        candidates = []
        
        for result in nearby_results:
            # Skip if it's just a label or stop word
            if self._is_label_text(result.text):
                continue
            
            # Field-specific filtering
            if field_name == 'roll_number':
                # Prefer text that looks like a roll number
                if self._looks_like_roll_number(result.text):
                    candidates.append(result)
            elif field_name == 'grade':
                # Prefer text that looks like a grade
                if self._looks_like_grade(result.text):
                    candidates.append(result)
            elif field_name in ['total_marks', 'obtained_marks']:
                # Prefer numeric values
                if result.text.replace('.', '').replace('/', '').isdigit():
                    candidates.append(result)
            else:
                # For other fields, take the first reasonable candidate
                if len(result.text) > 1:  # Skip single characters
                    candidates.append(result)
        
        # If no specific candidates, take the first nearby result
        if not candidates:
            candidates = nearby_results
        
        # Select the candidate with highest confidence
        return max(candidates, key=lambda x: x.confidence)
    
    def _is_label_text(self, text: str) -> bool:
        """
        Check if text is likely a label rather than a value.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be a label
        """
        label_indicators = [
            'name', 'number', 'roll', 'marks', 'grade', 'program', 'semester', 'of', 'the',
            'student', 'obtained', 'total'
        ]
        text_lower = text.lower().strip()
        # Use word-boundary matching to avoid false positives in legitimate values
        # Also check for common label patterns ending with colon
        if text_lower.endswith(':'):
            return True
        return any(re.search(rf'\b{indicator}\b', text_lower) for indicator in label_indicators)
    
    def _looks_like_roll_number(self, text: str) -> bool:
        """
        Check if text looks like a roll number.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be a roll number
        """
        for pattern in TranscriptExtractionConfig.ROLL_NUMBER_PATTERNS:
            if re.match(pattern, text.strip()):
                return True
        return False
    
    def _looks_like_grade(self, text: str) -> bool:
        """
        Check if text looks like a grade.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be a grade
        """
        grade_indicators = ['A', 'B', 'C', 'D', 'F', 'first', 'second', 'third', 'division']
        text_upper = text.upper().strip()
        return any(indicator in text_upper for indicator in grade_indicators)
    
    def _calculate_extraction_confidence(self, label: OCRResult, 
                                       value: OCRResult) -> float:
        """
        Calculate extraction confidence based on OCR confidences.
        
        Args:
            label: Label OCRResult
            value: Value OCRResult
            
        Returns:
            Confidence score between 0 and 1
        """
        # Average of label and value confidences
        return (label.confidence + value.confidence) / 2
    
    def _sufficient_fields_extracted(self, extracted_fields: dict[str, ExtractedField]) -> bool:
        """
        Check if sufficient fields were extracted via anchor-based approach.
        
        Args:
            extracted_fields: Dictionary of extracted fields
            
        Returns:
            True if sufficient fields extracted
        """
        return len(extracted_fields) >= TranscriptExtractionConfig.MIN_ANCHOR_FIELDS
    
    def _llm_fallback_extraction(self, ocr_results: list[OCRResult]) -> dict[str, ExtractedField] | None:
        """
        Fall back to LLM extraction for transcript fields.
        
        Args:
            ocr_results: List of OCRResult objects
            
        Returns:
            Dictionary mapping field names to ExtractedField objects or None if failed
        """
        if not self.llm_service or not self.llm_service.is_available():
            logger.warning("LLM service not available for fallback")
            return None
        
        # Get full text for LLM
        raw_text = self.ocr_service.get_full_text(ocr_results)
        
        # Call LLM for extraction
        llm_data = self.llm_service.extract_transcript_fields(raw_text)
        
        if not llm_data:
            logger.warning("LLM extraction returned no data")
            return None
        
        # Validate against Pydantic model
        try:
            validated_data = TranscriptData(**llm_data)
            logger.info("LLM data validated successfully")
        except ValidationError as e:
            logger.error(f"LLM data validation failed: {e!s}")
            return None
        
        # Convert to ExtractedField objects
        extracted_fields = {}
        for field_name, value in validated_data.model_dump().items():
            if value:  # Only include non-empty values
                extracted_fields[field_name] = ExtractedField(
                    field_name=field_name,
                    value=str(value),
                    source_bbox=[0, 0, 0, 0],  # LLM doesn't provide bbox
                    extraction_confidence=0.7,  # Fixed confidence for LLM
                    extraction_method="llm"
                )
        
        return extracted_fields
    
    def get_extraction_summary(self, extracted_fields: dict[str, ExtractedField]) -> dict:
        """
        Get summary of extraction results.
        
        Args:
            extracted_fields: Dictionary of extracted fields
            
        Returns:
            Summary dictionary with statistics
        """
        total_fields = len(TranscriptExtractionConfig.LABEL_ANCHORS)
        extracted_count = len(extracted_fields)
        required_fields = len(TranscriptExtractionConfig.REQUIRED_FIELDS)
        required_extracted = sum(1 for field in TranscriptExtractionConfig.REQUIRED_FIELDS 
                                if field in extracted_fields)
        
        avg_confidence = 0.0
        if extracted_fields:
            avg_confidence = sum(f.extraction_confidence for f in extracted_fields.values()) / extracted_count
        
        # Check if LLM was used
        llm_used = any(f.extraction_method == "llm" for f in extracted_fields.values())
        
        return {
            "total_fields": total_fields,
            "extracted_count": extracted_count,
            "required_fields": required_fields,
            "required_extracted": required_extracted,
            "avg_confidence": avg_confidence,
            "extraction_rate": extracted_count / total_fields if total_fields > 0 else 0,
            "required_satisfied": required_extracted == required_fields,
            "llm_used": llm_used
        }


# Singleton instance
_transcript_extractor = None

def get_transcript_extractor() -> TranscriptExtractor:
    """Get the singleton Transcript extractor instance."""
    global _transcript_extractor
    if _transcript_extractor is None:
        _transcript_extractor = TranscriptExtractor()
    return _transcript_extractor