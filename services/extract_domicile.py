"""
Domicile extraction service using label-anchored deterministic extraction.
Finds field labels in OCR results and extracts nearby values based on spatial relationships.
No LLM used - completely deterministic approach for government documents.
"""

import logging
import re

from config import DomicileExtractionConfig
from services.extract_cnic import ExtractedField
from services.ocr import OCRResult, get_ocr_service

logger = logging.getLogger(__name__)


class DomicileExtractor:
    """Deterministic Domicile field extraction using label anchors."""
    
    def __init__(self):
        """Initialize Domicile extractor."""
        self.ocr_service = get_ocr_service()
    
    def extract(self, ocr_results: list[OCRResult]) -> dict[str, ExtractedField]:
        """
        Extract Domicile fields using label-anchored approach.
        
        Args:
            ocr_results: List of OCRResult objects
            
        Returns:
            Dictionary mapping field names to ExtractedField objects
        """
        extracted_fields = {}
        
        # Extract each field using its label anchors
        for field_name, label_variations in DomicileExtractionConfig.LABEL_ANCHORS.items():
            field = self._extract_field_by_anchor(ocr_results, field_name, label_variations)
            if field:
                extracted_fields[field_name] = field
                logger.debug(f"Extracted {field_name}: {field.value} (confidence: {field.extraction_confidence:.2f})")
        
        # Post-process specific fields
        if 'permanent_province' in extracted_fields:
            extracted_fields['permanent_province'] = self._validate_province(extracted_fields['permanent_province'])
        
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
            max_distance=DomicileExtractionConfig.SEARCH_RADIUS,
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
            if field_name == 'permanent_province':
                # Prefer text that matches a known province
                if self._is_valid_province(result.text):
                    candidates.append(result)
            elif field_name == 'issue_date':
                # Prefer text that looks like a date
                if self._looks_like_date(result.text):
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
            'name', 'number', 'date', 'father', 'permanent', 'of', 'the', 'district',
            'province', 'domicile'
        ]
        text_lower = text.lower().strip()
        # Use word-boundary matching to avoid false positives in legitimate values
        # Also check for common label patterns ending with colon
        if text_lower.endswith(':'):
            return True
        return any(re.search(rf'\b{indicator}\b', text_lower) for indicator in label_indicators)
    
    def _is_valid_province(self, text: str) -> bool:
        """
        Check if text matches a valid Pakistani province.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is a valid province
        """
        text_lower = text.lower().strip()
        return any(province in text_lower for province in DomicileExtractionConfig.VALID_PROVINCES)
    
    def _looks_like_date(self, text: str) -> bool:
        """
        Check if text looks like a date.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be a date
        """
        import re
        date_patterns = [
            r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        ]
        return any(re.match(pattern, text) for pattern in date_patterns)
    
    def _validate_province(self, province_field: ExtractedField) -> ExtractedField:
        """
        Validate and normalize province name.
        
        Args:
            province_field: ExtractedField containing province name
            
        Returns:
            Validated ExtractedField
        """
        text_lower = province_field.value.lower().strip()
        
        # Try to match against valid provinces
        for valid_province in DomicileExtractionConfig.VALID_PROVINCES:
            if valid_province in text_lower or text_lower in valid_province:
                # Return the standardized province name
                return ExtractedField(
                    field_name=province_field.field_name,
                    value=valid_province,
                    source_bbox=province_field.source_bbox,
                    extraction_confidence=province_field.extraction_confidence,
                    extraction_method=province_field.extraction_method
                )
        
        # If no match found, return original but log warning
        logger.warning(f"Province '{province_field.value}' not found in valid provinces list")
        return province_field
    
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
    
    def get_extraction_summary(self, extracted_fields: dict[str, ExtractedField]) -> dict:
        """
        Get summary of extraction results.
        
        Args:
            extracted_fields: Dictionary of extracted fields
            
        Returns:
            Summary dictionary with statistics
        """
        total_fields = len(DomicileExtractionConfig.LABEL_ANCHORS)
        extracted_count = len(extracted_fields)
        required_fields = len(DomicileExtractionConfig.REQUIRED_FIELDS)
        required_extracted = sum(1 for field in DomicileExtractionConfig.REQUIRED_FIELDS 
                                if field in extracted_fields)
        
        avg_confidence = 0.0
        if extracted_fields:
            avg_confidence = sum(f.extraction_confidence for f in extracted_fields.values()) / extracted_count
        
        return {
            "total_fields": total_fields,
            "extracted_count": extracted_count,
            "required_fields": required_fields,
            "required_extracted": required_extracted,
            "avg_confidence": avg_confidence,
            "extraction_rate": extracted_count / total_fields if total_fields > 0 else 0,
            "required_satisfied": required_extracted == required_fields
        }


# Singleton instance
_domicile_extractor = None

def get_domicile_extractor() -> DomicileExtractor:
    """Get the singleton Domicile extractor instance."""
    global _domicile_extractor
    if _domicile_extractor is None:
        _domicile_extractor = DomicileExtractor()
    return _domicile_extractor