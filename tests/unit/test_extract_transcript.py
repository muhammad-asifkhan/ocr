"""
Unit tests for Transcript extraction service.
"""

import pytest
from services.extract_transcript import TranscriptExtractor, ExtractedField
from services.ocr import OCRResult


def test_extract_field_by_anchor():
    """Test extracting a field by its label anchor."""
    extractor = TranscriptExtractor()
    
    ocr_results = [
        OCRResult(text='Roll Number:', bbox=[10, 10, 80, 30], confidence=0.95),
        OCRResult(text='2023-001', bbox=[90, 10, 150, 30], confidence=0.9),
        OCRResult(text='Student Name:', bbox=[10, 50, 100, 70], confidence=0.95),
        OCRResult(text='Usman Ali', bbox=[110, 50, 170, 70], confidence=0.92)
    ]
    
    roll_field = extractor._extract_field_by_anchor(
        ocr_results,
        'roll_number',
        ['Roll Number', 'Roll No']
    )
    
    assert roll_field is not None
    assert roll_field.value == '2023-001'
    assert roll_field.field_name == 'roll_number'


def test_thea_khan_not_rejected_transcript():
    """Test that 'Thea Khan' is NOT rejected as a label in transcript extraction."""
    extractor = TranscriptExtractor()
    
    ocr_results = [
        OCRResult(text='Student Name:', bbox=[10, 10, 100, 30], confidence=0.95),
        OCRResult(text='Thea Khan', bbox=[110, 10, 170, 30], confidence=0.9)
    ]
    
    name_field = extractor._extract_field_by_anchor(
        ocr_results,
        'student_name',
        ['Student Name', 'Name']
    )
    
    assert name_field is not None
    assert name_field.value == 'Thea Khan'
    
    # Test _is_label_text directly
    assert extractor._is_label_text('Thea Khan') is False
    assert extractor._is_label_text('the') is True


def test_extract_roll_number_by_pattern():
    """Test roll number extraction by pattern."""
    extractor = TranscriptExtractor()
    
    ocr_results = [
        OCRResult(text='2023-CS-001', bbox=[10, 10, 100, 30], confidence=0.95),
        OCRResult(text='Some text', bbox=[10, 40, 80, 60], confidence=0.9)
    ]
    
    roll_field = extractor._extract_roll_number_by_pattern(ocr_results)
    
    assert roll_field is not None
    assert roll_field.value == '2023-CS-001'
    assert roll_field.field_name == 'roll_number'


def test_extract_with_sufficient_fields():
    """Test extraction when sufficient fields are found."""
    extractor = TranscriptExtractor()
    
    ocr_results = [
        OCRResult(text='Roll Number:', bbox=[10, 10, 80, 30], confidence=0.95),
        OCRResult(text='2023-001', bbox=[90, 10, 150, 30], confidence=0.9),
        OCRResult(text='Student Name:', bbox=[10, 50, 100, 70], confidence=0.95),
        OCRResult(text='Usman Ali', bbox=[110, 50, 170, 70], confidence=0.92),
        OCRResult(text='Program:', bbox=[10, 90, 70, 110], confidence=0.95),
        OCRResult(text='CS', bbox=[80, 90, 110, 110], confidence=0.9)
    ]
    
    extracted = extractor.extract(ocr_results)
    
    assert 'roll_number' in extracted
    assert 'student_name' in extracted
    assert extracted['roll_number'].value == '2023-001'
    assert extracted['student_name'].value == 'Usman Ali'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
