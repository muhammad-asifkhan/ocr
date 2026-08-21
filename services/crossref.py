"""
Cross-reference service for matching extracted data against reference database.
Uses rapidfuzz for fuzzy matching with configurable thresholds.
Consistent matching logic across document types.
"""

import csv
import logging

from rapidfuzz import fuzz

from config import REFERENCE_DB_PATH, CrossReferenceConfig
from services.extract_cnic import ExtractedField

logger = logging.getLogger(__name__)


def redact_cnic(cnic: str) -> str:
    """Redact CNIC number for logging - show first 3 and last 4 digits."""
    if not cnic or len(cnic) < 7:
        return "***"
    return f"{cnic[:3]}...{cnic[-4:]}"


def redact_name(name: str) -> str:
    """Redact name for logging - show only first letter."""
    return f"{name[0]}***" if name else "***"


def redact_match_results(match_results: list) -> list:
    """Redact sensitive data in match results for logging."""
    redacted = []
    for result in match_results:
        redacted_result = result.to_dict().copy()
        field_lower = result.field_name.lower()
        
        # Redact CNIC numbers
        if 'cnic' in field_lower:
            redacted_result['extracted_value'] = redact_cnic(result.extracted_value)
            redacted_result['reference_value'] = redact_cnic(result.reference_value)
        # Redact any name fields (full_name, student_name, father_name)
        elif 'name' in field_lower:
            redacted_result['extracted_value'] = redact_name(result.extracted_value)
            redacted_result['reference_value'] = redact_name(result.reference_value)
        # Province is not PII (public administrative data), so leave as-is
        # Other fields like dates, marks, roll numbers are not sensitive PII
        
        redacted.append(redacted_result)
    return redacted


class ReferenceData:
    """Represents reference data for a user."""
    def __init__(self, user_id: str, cnic: str, name: str, province: str = ""):
        self.user_id = user_id
        self.cnic = cnic
        self.name = name
        self.province = province
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "cnic": self.cnic,
            "name": self.name,
            "province": self.province
        }


class MatchResult:
    """Represents a field match result with evidence."""
    def __init__(self, field_name: str, matched: bool, similarity_score: float, 
                 extracted_value: str, reference_value: str):
        self.field_name = field_name
        self.matched = matched
        self.similarity_score = similarity_score
        self.extracted_value = extracted_value
        self.reference_value = reference_value
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "field_name": self.field_name,
            "matched": self.matched,
            "similarity_score": self.similarity_score,
            "extracted_value": self.extracted_value,
            "reference_value": self.reference_value
        }


class CrossReferenceService:
    """Service for cross-referencing extracted data against reference database."""
    
    def __init__(self):
        """Initialize cross-reference service."""
        self.reference_db: dict[str, ReferenceData] = {}
        self._load_reference_database()
    
    def _load_reference_database(self):
        """Load reference database from CSV file."""
        try:
            if not REFERENCE_DB_PATH.exists():
                logger.warning(f"Reference database not found at {REFERENCE_DB_PATH}")
                return
            
            with open(REFERENCE_DB_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    user_id = row.get('user_id', '').strip()
                    if user_id:
                        self.reference_db[user_id] = ReferenceData(
                            user_id=user_id,
                            cnic=row.get('cnic', '').strip(),
                            name=row.get('name', '').strip(),
                            province=row.get('province', '').strip()
                        )
            
            logger.info(f"Loaded {len(self.reference_db)} reference records")
            
        except Exception as e:
            logger.error(f"Failed to load reference database: {e!s}")
    
    def get_reference_data(self, user_id: str) -> ReferenceData | None:
        """
        Get reference data for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            ReferenceData object or None if not found
        """
        return self.reference_db.get(user_id)
    
    def match_cnic(self, extracted_fields: dict[str, ExtractedField], 
                  reference_data: ReferenceData) -> tuple[bool, list[MatchResult]]:
        """
        Match CNIC extracted data against reference data.
        
        Required matches: cnic_number AND full_name (both must match)
        
        Args:
            extracted_fields: Dictionary of extracted fields
            reference_data: Reference data for the user
            
        Returns:
            Tuple of (overall_match, list_of_field_match_results)
        """
        match_results = []
        
        # CNIC number - exact match required
        cnic_match = self._match_field(
            field_name='cnic_number',
            extracted_value=extracted_fields.get('cnic_number'),
            reference_value=reference_data.cnic,
            match_type='exact'
        )
        match_results.append(cnic_match)
        
        # Full name - fuzzy match required
        name_match = self._match_field(
            field_name='full_name',
            extracted_value=extracted_fields.get('full_name'),
            reference_value=reference_data.name,
            match_type='fuzzy',
            threshold=CrossReferenceConfig.FUZZY_NAME_THRESHOLD
        )
        match_results.append(name_match)
        
        # Overall match: all required fields must match
        required_matches = CrossReferenceConfig.CNIC_REQUIRED_MATCHES
        overall_match = all(
            result.matched for result in match_results 
            if result.field_name in required_matches
        )
        
        logger.info(f"CNIC cross-reference: overall_match={overall_match}, details={redact_match_results(match_results)}")
        return overall_match, match_results
    
    def match_domicile(self, extracted_fields: dict[str, ExtractedField], 
                      reference_data: ReferenceData) -> tuple[bool, list[MatchResult]]:
        """
        Match Domicile extracted data against reference data.
        
        Required matches: full_name AND permanent_province (both must match)
        
        Args:
            extracted_fields: Dictionary of extracted fields
            reference_data: Reference data for the user
            
        Returns:
            Tuple of (overall_match, list_of_field_match_results)
        """
        match_results = []
        
        # Full name - fuzzy match required
        name_match = self._match_field(
            field_name='full_name',
            extracted_value=extracted_fields.get('full_name'),
            reference_value=reference_data.name,
            match_type='fuzzy',
            threshold=CrossReferenceConfig.FUZZY_NAME_THRESHOLD
        )
        match_results.append(name_match)
        
        # Province - fuzzy match required
        province_match = self._match_field(
            field_name='permanent_province',
            extracted_value=extracted_fields.get('permanent_province'),
            reference_value=reference_data.province,
            match_type='fuzzy',
            threshold=CrossReferenceConfig.FUZZY_PROVINCE_THRESHOLD
        )
        match_results.append(province_match)
        
        # Overall match: all required fields must match
        required_matches = CrossReferenceConfig.DOMICILE_REQUIRED_MATCHES
        overall_match = all(
            result.matched for result in match_results 
            if result.field_name in required_matches
        )
        
        logger.info(f"Domicile cross-reference: overall_match={overall_match}, details={redact_match_results(match_results)}")
        return overall_match, match_results
    
    def match_transcript(self, extracted_fields: dict[str, ExtractedField], 
                        reference_data: ReferenceData) -> tuple[bool, list[MatchResult]]:
        """
        Match Transcript extracted data against reference data.
        
        Required matches: student_name (minimum requirement)
        
        Args:
            extracted_fields: Dictionary of extracted fields
            reference_data: Reference data for the user
            
        Returns:
            Tuple of (overall_match, list_of_field_match_results)
        """
        match_results = []
        
        # Student name - fuzzy match required
        name_match = self._match_field(
            field_name='student_name',
            extracted_value=extracted_fields.get('student_name'),
            reference_value=reference_data.name,
            match_type='fuzzy',
            threshold=CrossReferenceConfig.FUZZY_NAME_THRESHOLD
        )
        match_results.append(name_match)
        
        # Overall match: all required fields must match
        required_matches = CrossReferenceConfig.TRANSCRIPT_REQUIRED_MATCHES
        overall_match = all(
            result.matched for result in match_results 
            if result.field_name in required_matches
        )
        
        logger.info(f"Transcript cross-reference: overall_match={overall_match}, details={redact_match_results(match_results)}")
        return overall_match, match_results
    
    def _match_field(self, field_name: str, extracted_value: ExtractedField | None,
                    reference_value: str, match_type: str, 
                    threshold: float = 0.0) -> MatchResult:
        """
        Match a single field against reference value.
        
        Args:
            field_name: Name of the field
            extracted_value: ExtractedField containing the value
            reference_value: Reference value to match against
            match_type: Type of match ('exact' or 'fuzzy')
            threshold: Threshold for fuzzy matching
            
        Returns:
            MatchResult object
        """
        if not extracted_value or not extracted_value.value.strip():
            return MatchResult(
                field_name=field_name,
                matched=False,
                similarity_score=0.0,
                extracted_value="",
                reference_value=reference_value
            )
        
        extracted_text = extracted_value.value.strip()
        
        if match_type == 'exact':
            # Exact match for CNIC numbers
            matched = extracted_text == reference_value
            similarity_score = 100.0 if matched else 0.0
        else:
            # Fuzzy match for names, provinces, etc.
            similarity_score = fuzz.ratio(extracted_text, reference_value)
            matched = similarity_score >= threshold
        
        return MatchResult(
            field_name=field_name,
            matched=matched,
            similarity_score=similarity_score,
            extracted_value=extracted_text,
            reference_value=reference_value
        )
    
    def match_by_document_type(self, doc_type: str, extracted_fields: dict[str, ExtractedField],
                              user_id: str) -> tuple[bool, list[MatchResult], str | None]:
        """
        Route matching based on document type.
        
        Args:
            doc_type: Document type (CNIC, DOMICILE, TRANSCRIPT)
            extracted_fields: Dictionary of extracted fields
            user_id: User identifier
            
        Returns:
            Tuple of (overall_match, match_results, error_message)
        """
        reference_data = self.get_reference_data(user_id)
        
        if not reference_data:
            error_msg = f"No reference data found for user_id: {user_id}"
            logger.warning(error_msg)
            return False, [], error_msg
        
        try:
            if doc_type == "CNIC":
                match, results = self.match_cnic(extracted_fields, reference_data)
                return match, results, None
            elif doc_type == "DOMICILE":
                match, results = self.match_domicile(extracted_fields, reference_data)
                return match, results, None
            elif doc_type == "TRANSCRIPT":
                match, results = self.match_transcript(extracted_fields, reference_data)
                return match, results, None
            else:
                error_msg = f"Unknown document type for matching: {doc_type}"
                logger.error(error_msg)
                return False, [], error_msg
                
        except Exception as e:
            error_msg = f"Error during cross-reference matching: {e!s}"
            logger.error(error_msg)
            return False, [], error_msg
    
    def reload_reference_database(self):
        """Reload reference database from file."""
        self.reference_db.clear()
        self._load_reference_database()
        logger.info("Reference database reloaded")


# Singleton instance
_crossref_service = None

def get_crossref_service() -> CrossReferenceService:
    """Get the singleton cross-reference service instance."""
    global _crossref_service
    if _crossref_service is None:
        _crossref_service = CrossReferenceService()
    return _crossref_service