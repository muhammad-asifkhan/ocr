"""
Format validation service for extracted data.
Validates CNIC format, dates, required fields before database comparison.
"""

import logging
import re
from datetime import datetime

from config import (
    CNICExtractionConfig,
    DomicileExtractionConfig,
    FormatValidationConfig,
    TranscriptExtractionConfig,
)
from services.extract_cnic import ExtractedField

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when format validation fails."""
    def __init__(self, field_name: str, reason: str, value: str = ""):
        self.field_name = field_name
        self.reason = reason
        self.value = value
        super().__init__(f"{field_name}: {reason} (value: '{value}')")


class FormatValidator:
    """Validates format of extracted data before database comparison."""
    
    def __init__(self):
        """Initialize format validator."""
        self.date_formats = FormatValidationConfig.DATE_FORMATS
    
    def validate_cnic_data(self, extracted_fields: dict[str, ExtractedField]) -> tuple[bool, list[str]]:
        """
        Validate CNIC extracted data format.
        
        Args:
            extracted_fields: Dictionary of extracted fields
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        
        # Check required fields
        for field in CNICExtractionConfig.REQUIRED_FIELDS:
            if field not in extracted_fields or not extracted_fields[field].value.strip():
                errors.append(f"Missing required field: {field}")
        
        # Validate CNIC format
        if 'cnic_number' in extracted_fields:
            cnic_value = extracted_fields['cnic_number'].value.strip()
            if not self._validate_cnic_format(cnic_value):
                errors.append(f"Invalid CNIC format: {cnic_value}")
        
        # Validate date format if present
        if 'date_of_birth' in extracted_fields:
            dob_value = extracted_fields['date_of_birth'].value.strip()
            if dob_value and not self._validate_date_format(dob_value):
                errors.append(f"Invalid date format: {dob_value}")
        
        # Validate name format
        if 'full_name' in extracted_fields:
            name_value = extracted_fields['full_name'].value.strip()
            if name_value and not self._validate_name_format(name_value):
                errors.append(f"Invalid name format: {name_value}")
        
        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"CNIC format validation failed: {errors}")
        
        return is_valid, errors
    
    def validate_domicile_data(self, extracted_fields: dict[str, ExtractedField]) -> tuple[bool, list[str]]:
        """
        Validate Domicile extracted data format.
        
        Args:
            extracted_fields: Dictionary of extracted fields
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        
        # Check required fields
        for field in DomicileExtractionConfig.REQUIRED_FIELDS:
            if field not in extracted_fields or not extracted_fields[field].value.strip():
                errors.append(f"Missing required field: {field}")
        
        # Validate province
        if 'permanent_province' in extracted_fields:
            province_value = extracted_fields['permanent_province'].value.strip()
            if province_value and not self._validate_province_format(province_value):
                errors.append(f"Invalid province: {province_value}")
        
        # Validate date format if present
        if 'issue_date' in extracted_fields:
            date_value = extracted_fields['issue_date'].value.strip()
            if date_value and not self._validate_date_format(date_value):
                errors.append(f"Invalid date format: {date_value}")
        
        # Validate name format
        if 'full_name' in extracted_fields:
            name_value = extracted_fields['full_name'].value.strip()
            if name_value and not self._validate_name_format(name_value):
                errors.append(f"Invalid name format: {name_value}")
        
        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"Domicile format validation failed: {errors}")
        
        return is_valid, errors
    
    def validate_transcript_data(self, extracted_fields: dict[str, ExtractedField]) -> tuple[bool, list[str]]:
        """
        Validate Transcript extracted data format.
        
        Args:
            extracted_fields: Dictionary of extracted fields
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        
        # Check required fields
        for field in TranscriptExtractionConfig.REQUIRED_FIELDS:
            if field not in extracted_fields or not extracted_fields[field].value.strip():
                errors.append(f"Missing required field: {field}")
        
        # Validate roll number format (should be alphanumeric)
        if 'roll_number' in extracted_fields:
            roll_value = extracted_fields['roll_number'].value.strip()
            if roll_value and not self._validate_roll_number_format(roll_value):
                errors.append(f"Invalid roll number format: {roll_value}")
        
        # Validate name format
        if 'student_name' in extracted_fields:
            name_value = extracted_fields['student_name'].value.strip()
            if name_value and not self._validate_name_format(name_value):
                errors.append(f"Invalid name format: {name_value}")
        
        # Validate marks format if present (should be numeric)
        for field in ['total_marks', 'obtained_marks']:
            if field in extracted_fields:
                marks_value = extracted_fields[field].value.strip()
                if marks_value and not self._validate_marks_format(marks_value):
                    errors.append(f"Invalid {field} format: {marks_value}")
        
        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"Transcript format validation failed: {errors}")
        
        return is_valid, errors
    
    def _validate_cnic_format(self, cnic: str) -> bool:
        """
        Validate CNIC format (13 digits).
        
        Args:
            cnic: CNIC string to validate
            
        Returns:
            True if valid CNIC format
        """
        if FormatValidationConfig.STRICT_CNIC_VALIDATION:
            return bool(re.match(CNICExtractionConfig.CNIC_PATTERN, cnic))
        else:
            # Less strict: just check if it's 13 digits
            return bool(re.match(r'^\d{13}$', cnic))
    
    def _validate_date_format(self, date_str: str) -> bool:
        """
        Validate date string against known formats.
        
        Args:
            date_str: Date string to validate
            
        Returns:
            True if valid date format
        """
        for date_format in self.date_formats:
            try:
                datetime.strptime(date_str, date_format)
                return True
            except ValueError:
                continue
        return False
    
    def _validate_name_format(self, name: str) -> bool:
        """
        Validate name format (reasonable length and characters).
        
        Args:
            name: Name string to validate
            
        Returns:
            True if valid name format
        """
        if not name:
            return False
        
        # Check length
        if len(name) < FormatValidationConfig.MIN_NAME_LENGTH:
            return False
        if len(name) > FormatValidationConfig.MAX_NAME_LENGTH:
            return False
        
        # Check for reasonable characters (letters, spaces, hyphens, apostrophes)
        # Allow common name characters across languages
        if not re.match(r'^[a-zA-Z\u0600-\u06FF\s\-\'\.]+$', name):
            return False
        
        return True
    
    def _validate_province_format(self, province: str) -> bool:
        """
        Validate province against known Pakistani provinces.
        
        Args:
            province: Province string to validate
            
        Returns:
            True if valid province
        """
        province_lower = province.lower().strip()
        return any(
            valid_province in province_lower or province_lower in valid_province
            for valid_province in DomicileExtractionConfig.VALID_PROVINCES
        )
    
    def _validate_roll_number_format(self, roll_number: str) -> bool:
        """
        Validate roll number format (alphanumeric).
        
        Args:
            roll_number: Roll number string to validate
            
        Returns:
            True if valid roll number format
        """
        # Allow alphanumeric characters, hyphens, slashes
        return bool(re.match(r'^[A-Za-z0-9\-\/]+$', roll_number))
    
    def _validate_marks_format(self, marks: str) -> bool:
        """
        Validate marks format (numeric or decimal).
        
        Args:
            marks: Marks string to validate
            
        Returns:
            True if valid marks format
        """
        # Allow numeric values with optional decimal point
        return bool(re.match(r'^\d+(\.\d+)?$', marks))
    
    def validate_by_document_type(self, doc_type: str, 
                                 extracted_fields: dict[str, ExtractedField]) -> tuple[bool, list[str]]:
        """
        Route validation based on document type.
        
        Args:
            doc_type: Document type (CNIC, DOMICILE, TRANSCRIPT)
            extracted_fields: Dictionary of extracted fields
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        if doc_type == "CNIC":
            return self.validate_cnic_data(extracted_fields)
        elif doc_type == "DOMICILE":
            return self.validate_domicile_data(extracted_fields)
        elif doc_type == "TRANSCRIPT":
            return self.validate_transcript_data(extracted_fields)
        else:
            logger.error(f"Unknown document type for validation: {doc_type}")
            return False, [f"Unknown document type: {doc_type}"]


# Singleton instance
_format_validator = None

def get_format_validator() -> FormatValidator:
    """Get the singleton format validator instance."""
    global _format_validator
    if _format_validator is None:
        _format_validator = FormatValidator()
    return _format_validator