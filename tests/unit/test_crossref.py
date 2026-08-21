"""
Unit tests for cross-reference service.
"""

import pytest
from services.crossref import CrossReferenceService, ReferenceData
from services.extract_cnic import ExtractedField


def test_exact_cnic_match():
    """Test exact CNIC number matching."""
    service = CrossReferenceService()
    service.reference_db['user001'] = ReferenceData(
        user_id='user001',
        cnic='1234567890123',
        name='Ali Khan',
        province='Punjab'
    )
    
    extracted_fields = {
        'cnic_number': ExtractedField(
            field_name='cnic_number',
            value='1234567890123',
            source_bbox=[0, 0, 100, 20],
            extraction_confidence=0.9,
            extraction_method='anchor'
        ),
        'full_name': ExtractedField(
            field_name='full_name',
            value='Ali Khan',
            source_bbox=[0, 30, 100, 50],
            extraction_confidence=0.85,
            extraction_method='anchor'
        )
    }
    
    match, results = service.match_cnic(extracted_fields, service.reference_db['user001'])
    
    assert match is True
    assert len(results) == 2
    assert results[0].field_name == 'cnic_number'
    assert results[0].matched is True
    assert results[0].similarity_score == 100.0


def test_fuzzy_name_match_above_threshold():
    """Test fuzzy name matching above threshold."""
    service = CrossReferenceService()
    service.reference_db['user001'] = ReferenceData(
        user_id='user001',
        cnic='1234567890123',
        name='Ali Khan',
        province='Punjab'
    )
    
    extracted_fields = {
        'cnic_number': ExtractedField(
            field_name='cnic_number',
            value='1234567890123',
            source_bbox=[0, 0, 100, 20],
            extraction_confidence=0.9,
            extraction_method='anchor'
        ),
        'full_name': ExtractedField(
            field_name='full_name',
            value='Ali Khan',  # Exact match
            source_bbox=[0, 30, 100, 50],
            extraction_confidence=0.85,
            extraction_method='anchor'
        )
    }
    
    match, results = service.match_cnic(extracted_fields, service.reference_db['user001'])
    
    assert match is True
    name_result = [r for r in results if r.field_name == 'full_name'][0]
    assert name_result.matched is True
    assert name_result.similarity_score >= 88  # FUZZY_NAME_THRESHOLD


def test_fuzzy_name_match_below_threshold():
    """Test fuzzy name matching below threshold."""
    service = CrossReferenceService()
    service.reference_db['user001'] = ReferenceData(
        user_id='user001',
        cnic='1234567890123',
        name='Ali Khan',
        province='Punjab'
    )
    
    extracted_fields = {
        'cnic_number': ExtractedField(
            field_name='cnic_number',
            value='1234567890123',
            source_bbox=[0, 0, 100, 20],
            extraction_confidence=0.9,
            extraction_method='anchor'
        ),
        'full_name': ExtractedField(
            field_name='full_name',
            value='Very Different Name',  # Very different from reference
            source_bbox=[0, 30, 100, 50],
            extraction_confidence=0.85,
            extraction_method='anchor'
        )
    }
    
    match, results = service.match_cnic(extracted_fields, service.reference_db['user001'])
    
    # Overall match should be False because name doesn't match
    assert match is False
    name_result = [r for r in results if r.field_name == 'full_name'][0]
    assert name_result.matched is False
    assert name_result.similarity_score < 88


def test_missing_user_id():
    """Test cross-reference with non-existent user_id."""
    service = CrossReferenceService()
    service.reference_db['user001'] = ReferenceData(
        user_id='user001',
        cnic='1234567890123',
        name='Ali Khan',
        province='Punjab'
    )
    
    extracted_fields = {
        'cnic_number': ExtractedField(
            field_name='cnic_number',
            value='1234567890123',
            source_bbox=[0, 0, 100, 20],
            extraction_confidence=0.9,
            extraction_method='anchor'
        )
    }
    
    match, results, error = service.match_by_document_type('CNIC', extracted_fields, 'nonexistent_user')
    
    assert match is False
    assert results == []
    assert error is not None
    assert 'No reference data found' in error


def test_malformed_extracted_fields():
    """Test cross-reference with malformed extracted fields."""
    service = CrossReferenceService()
    service.reference_db['user001'] = ReferenceData(
        user_id='user001',
        cnic='1234567890123',
        name='Ali Khan',
        province='Punjab'
    )
    
    # Empty extracted fields
    extracted_fields = {}
    
    match, results = service.match_cnic(extracted_fields, service.reference_db['user001'])
    
    assert match is False
    # Should return match results with matched=False for missing fields
    assert len(results) == 2  # cnic_number and full_name
    assert all(r.matched is False for r in results)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
