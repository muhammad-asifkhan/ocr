"""
Configuration for Government Document Verification System
All thresholds, paths, and settings are centralized here for auditability and easy adjustment.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
REVIEW_STORE_PATH = DATA_DIR / "reviews.json"
REFERENCE_DB_PATH = DATA_DIR / "reference_db.csv"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# IMAGE QUALITY GATES
# =============================================================================
class ImageQualityConfig:
    # Blur detection via Laplacian variance
    MIN_BLUR_SCORE = 100.0  # Below this, image is rejected as too blurry
    
    # Resolution requirements
    MIN_WIDTH = 300  # pixels
    MIN_HEIGHT = 300  # pixels
    
    # File size limits
    MAX_FILE_SIZE_MB = 10.0
    MIN_FILE_SIZE_KB = 10
    
    # Allowed MIME types
    ALLOWED_MIME_TYPES = {
        'image/jpeg',
        'image/jpg', 
        'image/png',
        'image/tiff',
        'image/bmp'
    }

# =============================================================================
# OCR CONFIGURATION
# =============================================================================
class OCRConfig:
    ENGINE = "paddleocr"  # Only PaddleOCR used in this system
    CPU_THREADS = 2
    USE_GPU = False
    LANGUAGES = ['en']  # English only for Pakistani documents
    SHOW_LOG = False
    
    # PaddleOCR specific
    USE_ANGLE_CLASSIFIER = False  # Disabled for speed
    
    # Minimum confidence for OCR results to be considered
    MIN_WORD_CONFIDENCE = 0.5

# =============================================================================
# CLASSIFICATION CONFIGURATION
# =============================================================================
class ClassificationConfig:
    # Priority order: DOMICILE -> CNIC -> TRANSCRIPT
    # If no rule matches, return "unclassified"
    
    # DOMICILE keywords (case-insensitive)
    DOMICILE_KEYWORDS = {
        'domicile',
        'permanent address', 
        'permanent residence',
        'province',
        'district of domicile',
        'certificate of domicile'
    }
    
    # CNIC keywords (case-insensitive)
    CNIC_KEYWORDS = {
        'cnic',
        'national identity card',
        'identity card',
        'id card',
        'pakistani identity',
        'computerized national identity card'
    }
    
    # CNIC pattern (13 digits with optional hyphens)
    CNIC_PATTERN = r'\d{5}[-]?\d{7}[-]?\d{1}'
    
    # TRANSCRIPT keywords (case-insensitive)
    TRANSCRIPT_KEYWORDS = {
        'transcript',
        'marksheet',
        'mark sheet',
        'grade',
        'roll no',
        'roll number',
        'registration number',
        'academic record',
        'examination result'
    }
    
    # Keywords must appear in document to trigger classification
    MIN_KEYWORD_MATCHES = 1  # At least 1 keyword required

# =============================================================================
# EXTRACTION CONFIGURATION - CNIC
# =============================================================================
class CNICExtractionConfig:
    # Label anchors for CNIC fields (case-insensitive variations)
    LABEL_ANCHORS = {
        'cnic_number': [
            'identity number',
            'cnic',
            'national identity no',
            'id number',
            'cnic no',
            'identity no'
        ],
        'full_name': [
            'name',
            'full name',
            'applicant name'
        ],
        'father_name': [
            'father name',
            "father's name",
            'father/husband name',
            's/o',
            'd/o'
        ],
        'date_of_birth': [
            'date of birth',
            'dob',
            'birth date'
        ],
        'gender': [
            'gender',
            'sex'
        ]
    }
    
    # Spatial search parameters
    MAX_HORIZONTAL_DISTANCE = 200  # pixels to the right of label
    MAX_VERTICAL_DISTANCE = 50     # pixels below label
    SEARCH_RADIUS = 300            # general search radius around label
    
    # Required fields for format validation
    REQUIRED_FIELDS = ['cnic_number', 'full_name']
    
    # CNIC format validation
    CNIC_LENGTH = 13
    CNIC_PATTERN = r'^\d{13}$'

# =============================================================================
# EXTRACTION CONFIGURATION - DOMICILE
# =============================================================================
class DomicileExtractionConfig:
    # Label anchors for Domicile fields
    LABEL_ANCHORS = {
        'full_name': [
            'name',
            'full name',
            'applicant name'
        ],
        'father_name': [
            'father name',
            "father's name",
            's/o',
            'd/o'
        ],
        'permanent_province': [
            'province',
            'permanent province',
            'province of domicile'
        ],
        'permanent_district': [
            'district',
            'permanent district',
            'district of domicile'
        ],
        'issue_date': [
            'issue date',
            'date of issue',
            'dated'
        ]
    }
    
    # Pakistani provinces for validation
    VALID_PROVINCES = {
        'punjab',
        'sindh', 
        'kpk',
        'khyber pakhtunkhwa',
        'balochistan',
        'gilgit baltistan',
        'gilgit',
        'ajk',
        'azad kashmir',
        'islamabad capital territory'
    }
    
    # Spatial search parameters
    MAX_HORIZONTAL_DISTANCE = 200
    MAX_VERTICAL_DISTANCE = 50
    SEARCH_RADIUS = 300
    
    # Required fields for format validation
    REQUIRED_FIELDS = ['full_name', 'permanent_province']

# =============================================================================
# EXTRACTION CONFIGURATION - TRANSCRIPT
# =============================================================================
class TranscriptExtractionConfig:
    # Label anchors for Transcript fields
    LABEL_ANCHORS = {
        'roll_number': [
            'roll no',
            'roll number',
            'registration no',
            'reg no',
            'reg. no',
            'roll number'
        ],
        'student_name': [
            'name',
            'student name',
            'candidate name',
            'applicant name'
        ],
        'father_name': [
            'father name',
            "father's name",
            's/o',
            'd/o'
        ],
        'degree_program': [
            'degree',
            'program',
            'course',
            'discipline'
        ],
        'total_marks': [
            'total marks',
            'total',
            'maximum marks'
        ],
        'obtained_marks': [
            'obtained marks',
            'obtained',
            'marks obtained',
            'secured'
        ],
        'grade': [
            'grade',
            'division',
            'cgpa'
        ],
        'semester': [
            'semester',
            'term',
            'session'
        ]
    }
    
    # Pattern-based fallback for roll numbers
    ROLL_NUMBER_PATTERNS = [
        r'\b\d{4,10}\b',  # 4-10 digit numbers
        r'\b[A-Z]{2,4}-\d{4,6}\b',  # Pattern like AB-12345
        r'\b\d{4}-\d{4}\b'  # Pattern like 1234-5678
    ]
    
    # Minimum fields required before LLM fallback
    MIN_ANCHOR_FIELDS = 3  # If fewer than 3 fields found via anchors, use LLM
    
    # Spatial search parameters
    MAX_HORIZONTAL_DISTANCE = 250
    MAX_VERTICAL_DISTANCE = 60
    SEARCH_RADIUS = 350
    
    # Required fields for format validation
    REQUIRED_FIELDS = ['roll_number', 'student_name']

# =============================================================================
# LLM CONFIGURATION (TRANSCRIPT FALLBACK ONLY)
# =============================================================================
class LLMConfig:
    # Model path
    MODEL_PATH = MODELS_DIR / "phi-3-mini-4k-instruct-q4.gguf"
    
    # LLM parameters
    N_CTX = 4096  # Context window
    N_THREADS = 4
    TEMPERATURE = 0.1  # Low temperature for consistent outputs
    MAX_TOKENS = 300
    
    # Only used for transcript extraction when anchor-based fails
    ENABLED = True  # Can be disabled to force anchor-only extraction

# =============================================================================
# FORMAT VALIDATION CONFIGURATION
# =============================================================================
class FormatValidationConfig:
    # Date formats accepted (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD)
    DATE_FORMATS = [
        '%d/%m/%Y',
        '%m/%d/%Y', 
        '%Y-%m-%d',
        '%d-%m-%Y',
        '%m-%d-%Y'
    ]
    
    # Minimum length for text fields
    MIN_NAME_LENGTH = 2
    MAX_NAME_LENGTH = 100
    
    # Validation strictness
    STRICT_CNIC_VALIDATION = True  # Must be exactly 13 digits
    ALLOW_EMPTY_OPTIONAL_FIELDS = True

# =============================================================================
# CROSS-REFERENCE CONFIGURATION
# =============================================================================
class CrossReferenceConfig:
    # Fuzzy matching thresholds (rapidfuzz)
    FUZZY_NAME_THRESHOLD = 88  # Minimum similarity score for name matching
    FUZZY_PROVINCE_THRESHOLD = 85  # Minimum similarity for province matching
    FUZZY_INSTITUTION_THRESHOLD = 80  # For transcript institution names
    
    # Required field matches per document type
    CNIC_REQUIRED_MATCHES = ['cnic_number', 'full_name']  # Both must match
    DOMICILE_REQUIRED_MATCHES = ['full_name', 'permanent_province']  # Both must match
    TRANSCRIPT_REQUIRED_MATCHES = ['student_name']  # At minimum
    
    # Match types
    CNIC_MATCH_TYPE = 'exact'  # CNIC must match exactly
    NAME_MATCH_TYPE = 'fuzzy'  # Names use fuzzy matching
    PROVINCE_MATCH_TYPE = 'fuzzy'  # Provinces use fuzzy matching
    
    # Similarity score ranges for verdict logic
    HIGH_MATCH_THRESHOLD = 95  # Above this: clear match
    LOW_MATCH_THRESHOLD = 70   # Below this: clear mismatch
    MIDDLE_BAND = (70, 95)     # Between: needs review

# =============================================================================
# VERDICT CONFIGURATION
# =============================================================================
class VerdictConfig:
    # Three possible verdicts
    VERDICT_VERIFIED = "verified"
    VERDICT_NEEDS_REVIEW = "needs_review"
    VERDICT_REJECTED = "rejected"
    
    # Confidence thresholds
    MIN_EXTRACTION_CONFIDENCE = 0.6  # Below this: needs review
    MIN_OVERALL_CONFIDENCE = 0.7     # Below this: needs review
    MIN_CLASSIFICATION_CONFIDENCE = 0.5  # Below this: needs review
    
    # Conditions for each verdict
    # VERIFIED: All required fields match above threshold AND extraction confidence sufficient
    # NEEDS_REVIEW: Format validation failed, low confidence, ambiguous match scores, or uncertain classification
    # REJECTED: Required fields clearly don't match with high extraction confidence
    
    # Store evidence for audit trail
    STORE_EVIDENCE = True
    EVIDENCE_RETENTION_DAYS = 90

# =============================================================================
# API CONFIGURATION
# =============================================================================
class APIConfig:
    HOST = "0.0.0.0"
    PORT = 8000
    
    # Authentication
    API_KEY_HEADER = "X-API-Key"
    API_KEY = os.getenv("API_KEY", "dev-secret-key-change-in-production")  # Must be set in production
    
    # Rate limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_TIMES = 10  # requests
    RATE_LIMIT_SECONDS = 60  # per 60 seconds
    
    # Request timeout
    REQUEST_TIMEOUT_SECONDS = 120
    
    # Concurrent processing
    MAX_CONCURRENT_DOCUMENTS = 3  # Process all 3 documents concurrently


# =============================================================================
# CORS CONFIGURATION
# =============================================================================
class CORSConfig:
    # Environment-based CORS configuration
    # Set CORS_ALLOWED_ORIGINS env var to comma-separated list (e.g., "https://example.com,https://app.example.com")
    # If not set, defaults to "*" for development (CHANGE IN PRODUCTION)
    ALLOWED_ORIGINS = os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "*"  # Default: wildcard for development - MUST override in production
    ).split(",") if os.getenv("CORS_ALLOWED_ORIGINS") else ["*"]
    
    ALLOW_CREDENTIALS = True
    ALLOW_METHODS = ["*"]  # All HTTP methods
    ALLOW_HEADERS = ["*"]  # All headers

# =============================================================================
# DATA HANDLING CONFIGURATION
# =============================================================================
class DataHandlingConfig:
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = LOGS_DIR / "verification.log"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # PII redaction in logs
    REDACT_CNIC = True  # Show only first 3 and last 4 digits
    REDACT_NAMES = True  # Show only first letter
    LOG_REQUEST_IDS = True  # Always log request IDs for tracing

# =============================================================================
# REVIEW STORAGE CONFIGURATION
# =============================================================================
class ReviewStoreConfig:
    STORAGE_TYPE = "json"  # Options: json, sqlite (future)
    STORAGE_PATH = REVIEW_STORE_PATH
    
    # Auto-cleanup
    AUTO_CLEANUP_OLD_REVIEWS = True
    REVIEW_RETENTION_DAYS = 30

# =============================================================================
# VALIDATION SUMMARY
# =============================================================================
def validate_config():
    """Validate that all required paths and settings are properly configured."""
    issues = []
    
    # Check critical paths
    if not MODELS_DIR.exists():
        issues.append(f"Models directory does not exist: {MODELS_DIR}")
    
    if LLMConfig.ENABLED and not LLMConfig.MODEL_PATH.exists():
        issues.append(f"LLM model not found: {LLMConfig.MODEL_PATH}")
    
    if not REFERENCE_DB_PATH.exists():
        issues.append(f"Reference database not found: {REFERENCE_DB_PATH}")
    
    # Check API key in production
    if APIConfig.API_KEY == "dev-secret-key-change-in-production":
        issues.append("CRITICAL: Using default dev API key. Set API_KEY environment variable in production.")
    
    # Check CORS configuration
    if CORSConfig.ALLOWED_ORIGINS == ["*"]:
        issues.append("CRITICAL: CORS allows all origins ('*'). Set CORS_ALLOWED_ORIGINS environment variable to specific domains in production.")
    
    return issues

if __name__ == "__main__":
    # Test configuration
    issues = validate_config()
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid.")