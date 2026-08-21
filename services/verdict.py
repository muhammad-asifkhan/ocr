"""
Verdict service for three-state decision logic: verified / needs_review / rejected.
Determines final verdict based on extraction confidence, format validation, and cross-reference matching.
Fully auditable with evidence preservation.
"""

import logging
from datetime import datetime, timezone

from config import CrossReferenceConfig, VerdictConfig
from services.crossref import MatchResult
from services.extract_cnic import ExtractedField

logger = logging.getLogger(__name__)


def redact_user_id(user_id: str) -> str:
    """Redact user ID for logging."""
    return f"{user_id[:3]}***" if user_id and len(user_id) > 3 else "***"


def redact_user_name(name: str) -> str:
    """Redact user name for logging."""
    return f"{name[0]}***" if name else "***"


class VerdictEvidence:
    """Evidence supporting a verdict decision."""
    def __init__(self, document_type: str, user_id: str):
        self.document_type = document_type
        self.user_id = user_id
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.extraction_confidence: float = 0.0
        self.format_valid: bool = True
        self.format_errors: list[str] = []
        self.crossref_match: bool = False
        self.match_results: list[dict] = []
        self.classification_confidence: float = 0.0
        self.verdict: str = ""
        self.verdict_reasons: list[str] = []
        self.extraction_summary: dict = {}
    
    def to_dict(self) -> dict:
        """Convert to dictionary for audit trail."""
        return {
            "document_type": self.document_type,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "extraction_confidence": self.extraction_confidence,
            "format_valid": self.format_valid,
            "format_errors": self.format_errors,
            "crossref_match": self.crossref_match,
            "match_results": self.match_results,
            "classification_confidence": self.classification_confidence,
            "verdict": self.verdict,
            "verdict_reasons": self.verdict_reasons,
            "extraction_summary": self.extraction_summary
        }


class VerdictService:
    """Service for determining three-state verdicts with evidence preservation."""
    
    def __init__(self):
        """Initialize verdict service."""
        self.config = VerdictConfig
    
    def determine_verdict(self, 
                         document_type: str,
                         user_id: str,
                         extracted_fields: dict[str, ExtractedField],
                         format_valid: bool,
                         format_errors: list[str],
                         crossref_match: bool,
                         match_results: list[MatchResult],
                         classification_confidence: float,
                         extraction_summary: dict) -> tuple[str, VerdictEvidence]:
        """
        Determine the final verdict based on all available evidence.
        
        Args:
            document_type: Type of document (CNIC, DOMICILE, TRANSCRIPT)
            user_id: User identifier
            extracted_fields: Dictionary of extracted fields
            format_valid: Whether format validation passed
            format_errors: List of format validation errors
            crossref_match: Whether cross-reference matching passed
            match_results: List of field match results
            classification_confidence: Confidence in document classification
            extraction_summary: Summary of extraction results
            
        Returns:
            Tuple of (verdict_string, evidence_object)
        """
        evidence = VerdictEvidence(document_type, user_id)
        evidence.format_valid = format_valid
        evidence.format_errors = format_errors
        evidence.crossref_match = crossref_match
        evidence.match_results = [result.to_dict() for result in match_results]
        evidence.classification_confidence = classification_confidence
        evidence.extraction_summary = extraction_summary
        
        # Calculate overall extraction confidence
        evidence.extraction_confidence = self._calculate_overall_confidence(
            extracted_fields, extraction_summary
        )
        
        # Determine verdict
        verdict, reasons = self._apply_verdict_logic(evidence)
        
        evidence.verdict = verdict
        evidence.verdict_reasons = reasons
        
        logger.info(f"Verdict for {document_type} (user: {redact_user_id(user_id)}): {verdict} - {reasons}")
        return verdict, evidence
    
    def _calculate_overall_confidence(self, 
                                     extracted_fields: dict[str, ExtractedField],
                                     extraction_summary: dict) -> float:
        """
        Calculate overall extraction confidence from individual field confidences.
        
        Args:
            extracted_fields: Dictionary of extracted fields
            extraction_summary: Summary of extraction results
            
        Returns:
            Overall confidence score (0-1)
        """
        if not extracted_fields:
            return 0.0
        
        # Use the average confidence from extraction summary if available
        if extraction_summary and 'avg_confidence' in extraction_summary:
            return extraction_summary['avg_confidence']
        
        # Otherwise calculate from individual fields
        confidences = [field.extraction_confidence for field in extracted_fields.values()]
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _apply_verdict_logic(self, evidence: VerdictEvidence) -> tuple[str, list[str]]:
        """
        Apply the three-state verdict logic.
        
        Args:
            evidence: VerdictEvidence object with all available information
            
        Returns:
            Tuple of (verdict, list_of_reasons)
        """
        reasons = []
        
        # Priority 1: Format validation failures -> needs_review
        if not evidence.format_valid:
            reasons.append(f"Format validation failed: {', '.join(evidence.format_errors)}")
            # Any format validation failure should trigger needs_review
            return self.config.VERDICT_NEEDS_REVIEW, reasons
        
        # Priority 2: Classification uncertainty -> needs_review
        if evidence.classification_confidence < self.config.MIN_CLASSIFICATION_CONFIDENCE:
            reasons.append("Low classification confidence")
            return self.config.VERDICT_NEEDS_REVIEW, reasons
        
        # Priority 3: Extraction confidence below threshold -> needs_review
        if evidence.extraction_confidence < self.config.MIN_EXTRACTION_CONFIDENCE:
            reasons.append(f"Low extraction confidence: {evidence.extraction_confidence:.2f}")
            return self.config.VERDICT_NEEDS_REVIEW, reasons
        
        # Priority 4: Analyze cross-reference match results
        if not evidence.crossref_match:
            # Determine if it's a clear mismatch or ambiguous
            match_analysis = self._analyze_match_results(evidence.match_results)
            
            if match_analysis['is_clear_mismatch']:
                reasons.append(f"Clear mismatch: {match_analysis['reason']}")
                return self.config.VERDICT_REJECTED, reasons
            else:
                reasons.append(f"Ambiguous match: {match_analysis['reason']}")
                return self.config.VERDICT_NEEDS_REVIEW, reasons
        
        # Priority 5: Check for ambiguous middle-band similarity scores
        ambiguous_matches = self._find_ambiguous_matches(evidence.match_results)
        if ambiguous_matches:
            reasons.append(f"Ambiguous similarity scores: {ambiguous_matches}")
            return self.config.VERDICT_NEEDS_REVIEW, reasons
        
        # Priority 6: All checks passed -> verified
        reasons.append("All validation checks passed")
        return self.config.VERDICT_VERIFIED, reasons
    
    def _analyze_match_results(self, match_results: list[dict]) -> dict:
        """
        Analyze match results to determine if it's a clear mismatch or ambiguous.
        
        Args:
            match_results: List of match result dictionaries
            
        Returns:
            Dictionary with analysis results
        """
        if not match_results:
            return {
                'is_clear_mismatch': True,
                'reason': 'No match results available'
            }
        
        # Check if any required field has very low similarity
        low_similarity_fields = []
        for result in match_results:
            if result['similarity_score'] < CrossReferenceConfig.LOW_MATCH_THRESHOLD:
                low_similarity_fields.append(result['field_name'])
        
        if low_similarity_fields:
            return {
                'is_clear_mismatch': True,
                'reason': f"Low similarity on required fields: {', '.join(low_similarity_fields)}"
            }
        
        # Check if all required fields have high similarity
        high_similarity_count = sum(
            1 for result in match_results 
            if result['similarity_score'] >= CrossReferenceConfig.HIGH_MATCH_THRESHOLD
        )
        
        if high_similarity_count == len(match_results):
            return {
                'is_clear_mismatch': False,
                'reason': 'All fields have high similarity'
            }
        
        # Otherwise it's ambiguous
        return {
            'is_clear_mismatch': False,
            'reason': 'Mixed similarity scores - some fields match, others do not'
        }
    
    def _find_ambiguous_matches(self, match_results: list[dict]) -> str | None:
        """
        Find matches with similarity scores in the ambiguous middle band.
        
        Args:
            match_results: List of match result dictionaries
            
        Returns:
            Description of ambiguous matches or None
        """
        ambiguous_fields = []
        lower, upper = CrossReferenceConfig.MIDDLE_BAND
        
        for result in match_results:
            score = result['similarity_score']
            if lower <= score < upper:
                ambiguous_fields.append(f"{result['field_name']} ({score:.1f}%)")
        
        if ambiguous_fields:
            return ', '.join(ambiguous_fields)
        
        return None
    
    def get_overall_verdict(self, individual_verdicts: dict[str, str]) -> str:
        """
        Determine overall verdict from individual document verdicts.
        
        Args:
            individual_verdicts: Dictionary mapping document types to their verdicts
            
        Returns:
            Overall verdict string
        """
        # If any document is rejected, overall is rejected
        if any(verdict == self.config.VERDICT_REJECTED for verdict in individual_verdicts.values()):
            return self.config.VERDICT_REJECTED
        
        # If any document needs review, overall is needs_review
        if any(verdict == self.config.VERDICT_NEEDS_REVIEW for verdict in individual_verdicts.values()):
            return self.config.VERDICT_NEEDS_REVIEW
        
        # All documents verified
        return self.config.VERDICT_VERIFIED


# Singleton instance
_verdict_service = None

def get_verdict_service() -> VerdictService:
    """Get the singleton verdict service instance."""
    global _verdict_service
    if _verdict_service is None:
        _verdict_service = VerdictService()
    return _verdict_service