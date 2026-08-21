"""
Unit tests for verdict service.
"""

import pytest
from services.verdict import VerdictService, VerdictEvidence
from services.extract_cnic import ExtractedField
from services.crossref import MatchResult
from config import VerdictConfig


def test_verdict_verified_state():
    """Test that verified verdict state is reachable."""
    service = VerdictService()
    
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
    
    match_results = [
        MatchResult(
            field_name='cnic_number',
            matched=True,
            similarity_score=100.0,
            extracted_value='1234567890123',
            reference_value='1234567890123'
        ),
        MatchResult(
            field_name='full_name',
            matched=True,
            similarity_score=95.0,
            extracted_value='Ali Khan',
            reference_value='Ali Khan'
        )
    ]
    
    verdict, evidence = service.determine_verdict(
        document_type='CNIC',
        user_id='user001',
        extracted_fields=extracted_fields,
        format_valid=True,
        format_errors=[],
        crossref_match=True,
        match_results=match_results,
        classification_confidence=0.9,
        extraction_summary={'avg_confidence': 0.87}
    )
    
    assert verdict == VerdictConfig.VERDICT_VERIFIED
    assert evidence.format_valid is True
    assert evidence.crossref_match is True


def test_verdict_needs_review_state():
    """Test that needs_review verdict state is reachable."""
    service = VerdictService()
    
    extracted_fields = {
        'cnic_number': ExtractedField(
            field_name='cnic_number',
            value='1234567890123',
            source_bbox=[0, 0, 100, 20],
            extraction_confidence=0.9,
            extraction_method='anchor'
        )
    }
    
    match_results = [
        MatchResult(
            field_name='cnic_number',
            matched=True,
            similarity_score=100.0,
            extracted_value='1234567890123',
            reference_value='1234567890123'
        )
    ]
    
    # Format invalid should trigger needs_review
    verdict, evidence = service.determine_verdict(
        document_type='CNIC',
        user_id='user001',
        extracted_fields=extracted_fields,
        format_valid=False,
        format_errors=['Invalid name format'],
        crossref_match=True,
        match_results=match_results,
        classification_confidence=0.9,
        extraction_summary={'avg_confidence': 0.87}
    )
    
    assert verdict == VerdictConfig.VERDICT_NEEDS_REVIEW
    assert evidence.format_valid is False


def test_verdict_rejected_state():
    """Test that rejected verdict state is reachable."""
    service = VerdictService()
    
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
    
    match_results = [
        MatchResult(
            field_name='cnic_number',
            matched=False,
            similarity_score=0.0,
            extracted_value='1234567890123',
            reference_value='9999999999999'
        ),
        MatchResult(
            field_name='full_name',
            matched=False,
            similarity_score=30.0,
            extracted_value='Ali Khan',
            reference_value='Different Name'
        )
    ]
    
    verdict, evidence = service.determine_verdict(
        document_type='CNIC',
        user_id='user001',
        extracted_fields=extracted_fields,
        format_valid=True,
        format_errors=[],
        crossref_match=False,
        match_results=match_results,
        classification_confidence=0.9,
        extraction_summary={'avg_confidence': 0.87}
    )
    
    assert verdict == VerdictConfig.VERDICT_REJECTED
    assert evidence.crossref_match is False


def test_get_overall_verdict_all_verified():
    """Test overall verdict when all documents are verified."""
    service = VerdictService()
    
    individual_verdicts = {
        'cnic': VerdictConfig.VERDICT_VERIFIED,
        'domicile': VerdictConfig.VERDICT_VERIFIED,
        'transcript': VerdictConfig.VERDICT_VERIFIED
    }
    
    overall = service.get_overall_verdict(individual_verdicts)
    assert overall == VerdictConfig.VERDICT_VERIFIED


def test_get_overall_verdict_mixed():
    """Test overall verdict with mixed individual verdicts."""
    service = VerdictService()
    
    individual_verdicts = {
        'cnic': VerdictConfig.VERDICT_VERIFIED,
        'domicile': VerdictConfig.VERDICT_NEEDS_REVIEW,
        'transcript': VerdictConfig.VERDICT_VERIFIED
    }
    
    overall = service.get_overall_verdict(individual_verdicts)
    assert overall == VerdictConfig.VERDICT_NEEDS_REVIEW


def test_get_overall_verdict_any_rejected():
    """Test overall verdict when any document is rejected."""
    service = VerdictService()
    
    individual_verdicts = {
        'cnic': VerdictConfig.VERDICT_VERIFIED,
        'domicile': VerdictConfig.VERDICT_REJECTED,
        'transcript': VerdictConfig.VERDICT_VERIFIED
    }
    
    overall = service.get_overall_verdict(individual_verdicts)
    assert overall == VerdictConfig.VERDICT_REJECTED


def test_get_overall_verdict_all_needs_review():
    """Test overall verdict when all documents need review."""
    service = VerdictService()
    
    individual_verdicts = {
        'cnic': VerdictConfig.VERDICT_NEEDS_REVIEW,
        'domicile': VerdictConfig.VERDICT_NEEDS_REVIEW,
        'transcript': VerdictConfig.VERDICT_NEEDS_REVIEW
    }
    
    overall = service.get_overall_verdict(individual_verdicts)
    assert overall == VerdictConfig.VERDICT_NEEDS_REVIEW


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
