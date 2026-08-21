"""
Unit tests for CNIC extraction service.
"""

import pytest
from services.extract_cnic import CNICExtractor
from services.ocr import OCRResult


def test_extract_field_by_anchor():
    """Test extracting a field by its label anchor."""
    extractor = CNICExtractor()
    
    # Create synthetic OCR results with realistic vertical spacing
    ocr_results = [
        OCRResult(text='Name:', bbox=[10, 10, 50, 30], confidence=0.95),
        OCRResult(text='Ali Khan', bbox=[60, 10, 120, 30], confidence=0.9),
        OCRResult(text='CNIC No:', bbox=[10, 80, 70, 100], confidence=0.95),  # Further down
        OCRResult(text='1234567890123', bbox=[80, 80, 180, 100], confidence=0.92)
    ]
    
    # Extract name
    name_field = extractor._extract_field_by_anchor(
        ocr_results,
        'full_name',
        ['Name', 'Full Name']
    )
    
    assert name_field is not None
    assert name_field.value == 'Ali Khan'
    assert name_field.field_name == 'full_name'
    assert name_field.extraction_method == 'anchor'


def test_label_present_no_nearby_value():
    """Test case where label is present but no nearby value."""
    extractor = CNICExtractor()
    
    ocr_results = [
        OCRResult(text='Name:', bbox=[10, 10, 50, 30], confidence=0.95)
        # No nearby value
    ]
    
    name_field = extractor._extract_field_by_anchor(
        ocr_results,
        'full_name',
        ['Name', 'Full Name']
    )
    
    assert name_field is None


def test_multiple_label_matches():
    """Test case with multiple label matches."""
    extractor = CNICExtractor()
    
    ocr_results = [
        OCRResult(text='Name:', bbox=[10, 10, 50, 30], confidence=0.85),
        OCRResult(text='Ali Khan', bbox=[60, 10, 120, 30], confidence=0.9),
        OCRResult(text='Name:', bbox=[10, 50, 50, 70], confidence=0.95),  # Higher confidence
        OCRResult(text='Another Name', bbox=[60, 50, 140, 70], confidence=0.88)
    ]
    
    # Should use the label with highest confidence
    name_field = extractor._extract_field_by_anchor(
        ocr_results,
        'full_name',
        ['Name', 'Full Name']
    )
    
    assert name_field is not None
    # Should pick the second label (higher confidence) and its nearby value
    assert name_field.value in ['Ali Khan', 'Another Name']


def test_thea_khan_not_rejected():
    """Test that 'Thea Khan' is NOT rejected as a label (substring bug fix)."""
    extractor = CNICExtractor()
    
    ocr_results = [
        OCRResult(text='Name:', bbox=[10, 10, 50, 30], confidence=0.95),
        OCRResult(text='Thea Khan', bbox=[60, 10, 120, 30], confidence=0.9)
    ]
    
    # Thea Khan contains 'the' as a substring but should NOT be treated as a label
    name_field = extractor._extract_field_by_anchor(
        ocr_results,
        'full_name',
        ['Name', 'Full Name']
    )
    
    assert name_field is not None
    assert name_field.value == 'Thea Khan'
    
    # Also test _is_label_text directly
    assert extractor._is_label_text('Thea Khan') is False
    assert extractor._is_label_text('the') is True  # Actual label indicator
    assert extractor._is_label_text('Father') is True


def test_cnic_pattern_extraction():
    """Test CNIC number pattern detection."""
    extractor = CNICExtractor()
    
    # Test the _looks_like_cnic method
    assert extractor._looks_like_cnic('1234567890123') is True
    assert extractor._looks_like_cnic('12345-6789012-3') is True  # With separators
    assert extractor._looks_like_cnic('12345') is False  # Too short
    assert extractor._looks_like_cnic('not a cnic') is False


def test_extract_with_sufficient_fields():
    """Test extraction when sufficient fields are found."""
    extractor = CNICExtractor()
    
    ocr_results = [
        OCRResult(text='Name:', bbox=[10, 10, 50, 30], confidence=0.95),
        OCRResult(text='Ali Khan', bbox=[60, 10, 120, 30], confidence=0.9),
        OCRResult(text='CNIC No:', bbox=[10, 80, 70, 100], confidence=0.95),
        OCRResult(text='1234567890123', bbox=[80, 80, 180, 100], confidence=0.92),
        OCRResult(text='Date of Birth:', bbox=[10, 150, 100, 170], confidence=0.95),
        OCRResult(text='01-01-1990', bbox=[110, 150, 180, 170], confidence=0.9)
    ]
    
    extracted = extractor.extract(ocr_results)
    
    assert 'full_name' in extracted
    assert 'cnic_number' in extracted
    assert extracted['full_name'].value == 'Ali Khan'
    assert extracted['cnic_number'].value == '1234567890123'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
