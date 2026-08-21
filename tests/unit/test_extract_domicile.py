"""
Unit tests for Domicile extraction service.
"""

import pytest
from services.extract_domicile import DomicileExtractor, ExtractedField
from services.ocr import OCRResult


def test_extract_field_by_anchor():
    """Test extracting a field by its label anchor."""
    extractor = DomicileExtractor()
    
    ocr_results = [
        OCRResult(text='Name:', bbox=[10, 10, 50, 30], confidence=0.95),
        OCRResult(text='Sara Ahmed', bbox=[60, 10, 130, 30], confidence=0.9),
        OCRResult(text='Province:', bbox=[10, 80, 70, 100], confidence=0.95),
        OCRResult(text='Sindh', bbox=[80, 80, 120, 100], confidence=0.92)
    ]
    
    name_field = extractor._extract_field_by_anchor(
        ocr_results,
        'full_name',
        ['Name', 'Full Name']
    )
    
    assert name_field is not None
    assert name_field.value == 'Sara Ahmed'
    assert name_field.field_name == 'full_name'


def test_thea_khan_not_rejected_domicile():
    """Test that 'Thea Khan' is NOT rejected as a label in domicile extraction."""
    extractor = DomicileExtractor()
    
    ocr_results = [
        OCRResult(text='Name:', bbox=[10, 10, 50, 30], confidence=0.95),
        OCRResult(text='Thea Khan', bbox=[60, 10, 120, 30], confidence=0.9)
    ]
    
    name_field = extractor._extract_field_by_anchor(
        ocr_results,
        'full_name',
        ['Name', 'Full Name']
    )
    
    assert name_field is not None
    assert name_field.value == 'Thea Khan'
    
    # Test _is_label_text directly
    assert extractor._is_label_text('Thea Khan') is False
    assert extractor._is_label_text('the') is True


def test_extract_province():
    """Test province extraction."""
    extractor = DomicileExtractor()
    
    ocr_results = [
        OCRResult(text='Province:', bbox=[10, 10, 70, 30], confidence=0.95),
        OCRResult(text='Punjab', bbox=[80, 10, 130, 30], confidence=0.92)
    ]
    
    province_field = extractor._extract_field_by_anchor(
        ocr_results,
        'permanent_province',
        ['Province', 'Permanent Province']
    )
    
    assert province_field is not None
    assert province_field.value == 'Punjab'


def test_extract_with_sufficient_fields():
    """Test extraction when sufficient fields are found."""
    extractor = DomicileExtractor()
    
    ocr_results = [
        OCRResult(text='Name:', bbox=[10, 10, 50, 30], confidence=0.95),
        OCRResult(text='Sara Ahmed', bbox=[60, 10, 130, 30], confidence=0.9),
        OCRResult(text='Province:', bbox=[10, 80, 70, 100], confidence=0.95),
        OCRResult(text='Sindh', bbox=[80, 80, 120, 100], confidence=0.92),
        OCRResult(text='District:', bbox=[10, 150, 60, 170], confidence=0.95),
        OCRResult(text='Karachi', bbox=[70, 150, 120, 170], confidence=0.9)
    ]
    
    extracted = extractor.extract(ocr_results)
    
    assert 'full_name' in extracted
    assert 'permanent_province' in extracted
    assert extracted['full_name'].value == 'Sara Ahmed'
    assert extracted['permanent_province'].value.lower() == 'sindh'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
