"""
Main FastAPI application for Government Document Verification System.
Implements authentication, rate limiting, concurrent document processing, and structured error handling.
"""

import asyncio
import logging
import os

# Import services
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import APIConfig, CORSConfig, DataHandlingConfig, VerdictConfig
from services.classify import get_classification_service
from services.crossref import get_crossref_service
from services.extract_cnic import get_cnic_extractor
from services.extract_domicile import get_domicile_extractor
from services.extract_transcript import get_transcript_extractor
from services.ocr import get_ocr_service
from services.preprocess import ImageQualityError, validate_and_preprocess
from services.review_store import get_review_store
from services.validate_format import get_format_validator
from services.verdict import get_verdict_service

# Configure logging
logging.basicConfig(
    level=getattr(logging, DataHandlingConfig.LOG_LEVEL),
    format=DataHandlingConfig.LOG_FORMAT,
    handlers=[
        logging.FileHandler(DataHandlingConfig.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan events
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting Government Document Verification API")
    
    # Validate configuration
    from config import validate_config
    config_issues = validate_config()
    if config_issues:
        logger.warning(f"Configuration issues: {config_issues}")
    
    # Initialize services
    try:
        get_ocr_service()
        get_classification_service()
        get_cnic_extractor()
        get_domicile_extractor()
        get_transcript_extractor()
        get_format_validator()
        get_crossref_service()
        get_verdict_service()
        get_review_store()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Service initialization failed: {e!s}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Government Document Verification API")


# Initialize FastAPI
app = FastAPI(
    title="Government Document Verification API",
    description="Automated document classification, extraction, and verification system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
# Uses environment-driven configuration via CORSConfig
# Set CORS_ALLOWED_ORIGINS env var to comma-separated list for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORSConfig.ALLOWED_ORIGINS,
    allow_credentials=CORSConfig.ALLOW_CREDENTIALS,
    allow_methods=CORSConfig.ALLOW_METHODS,
    allow_headers=CORSConfig.ALLOW_HEADERS,
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =============================================================================
# Pydantic Models
# =============================================================================

class DocumentProcessingResult(BaseModel):
    """Result of processing a single document."""
    document_type: str
    classification_confidence: float
    extracted_fields: dict[str, dict]
    extraction_summary: dict
    format_valid: bool
    format_errors: list
    crossref_match: bool
    match_results: list
    verdict: str
    verdict_reasons: list
    evidence_id: str | None = None


class VerificationRequest(BaseModel):
    """Request model for document verification."""
    # This is for documentation - actual request uses multipart/form-data


class VerificationResponse(BaseModel):
    """Response model for document verification."""
    request_id: str
    overall_verdict: str
    processing_time_seconds: float
    documents: dict[str, DocumentProcessingResult]
    cross_reference_used: bool
    timestamp: str


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    detail: str
    request_id: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    services: dict[str, str]


# =============================================================================
# Authentication
# =============================================================================

async def verify_api_key(api_key: str = Header(..., alias=APIConfig.API_KEY_HEADER)) -> bool:
    """
    Verify API key for authentication.
    
    Args:
        api_key: API key from header
        
    Returns:
        True if authenticated
        
    Raises:
        HTTPException: If authentication fails
    """
    if api_key != APIConfig.API_KEY:
        logger.warning(f"Authentication failed with API key: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return True


# =============================================================================
# Processing Pipeline
# =============================================================================

async def process_single_document(
    image_bytes: bytes,
    file_size: int,
    mime_type: str,
    user_id: str,
    document_type_hint: str,
    request_id: str
) -> DocumentProcessingResult:
    """
    Process a single document through the complete pipeline.
    
    Args:
        image_bytes: Raw image bytes
        file_size: File size in bytes
        mime_type: MIME type of the file
        user_id: User identifier
        document_type_hint: Hint about document type (cnic, domicile, transcript)
        request_id: Request identifier for logging
        
    Returns:
        DocumentProcessingResult object
    """
    logger.info(f"Processing {document_type_hint} document for user {user_id} (request: {request_id})")
    
    try:
        # Step 1: Preprocessing with quality gates
        logger.debug(f"Preprocessing {document_type_hint} image")
        preprocessed_image, _quality_metrics = validate_and_preprocess(image_bytes, file_size, mime_type)
        
        # Step 2: OCR
        logger.debug(f"Running OCR on {document_type_hint}")
        ocr_service = get_ocr_service()
        ocr_results = ocr_service.process_image(preprocessed_image)
        
        if not ocr_results:
            raise ValueError("OCR returned no results - image may be unreadable")
        
        # Step 3: Classification
        logger.debug(f"Classifying {document_type_hint} document")
        classification_service = get_classification_service()
        doc_type = classification_service.classify(ocr_results)
        classification_confidence = classification_service.get_classification_confidence(ocr_results, doc_type)
        
        if doc_type == "unclassified":
            raise ValueError("Document could not be classified with confidence")
        
        # Step 4: Extraction based on document type
        logger.debug(f"Extracting fields from {doc_type}")
        if doc_type == "CNIC":
            extractor = get_cnic_extractor()
            extracted_fields = extractor.extract(ocr_results)
            extraction_summary = extractor.get_extraction_summary(extracted_fields)
        elif doc_type == "DOMICILE":
            extractor = get_domicile_extractor()
            extracted_fields = extractor.extract(ocr_results)
            extraction_summary = extractor.get_extraction_summary(extracted_fields)
        elif doc_type == "TRANSCRIPT":
            extractor = get_transcript_extractor()
            extracted_fields = extractor.extract(ocr_results)
            extraction_summary = extractor.get_extraction_summary(extracted_fields)
        else:
            raise ValueError(f"Unknown document type: {doc_type}")
        
        # Convert ExtractedField objects to dictionaries
        extracted_fields_dict = {
            field_name: field.to_dict() 
            for field_name, field in extracted_fields.items()
        }
        
        # Step 5: Format validation
        logger.debug(f"Validating format for {doc_type}")
        format_validator = get_format_validator()
        format_valid, format_errors = format_validator.validate_by_document_type(
            doc_type, extracted_fields
        )
        
        # Step 6: Cross-reference matching
        logger.debug(f"Cross-referencing {doc_type} against database")
        crossref_service = get_crossref_service()
        crossref_match, match_results, crossref_error = crossref_service.match_by_document_type(
            doc_type, extracted_fields, user_id
        )
        
        if crossref_error:
            logger.warning(f"Cross-reference error: {crossref_error}")
            crossref_match = False
        
        # Step 7: Verdict determination
        logger.debug(f"Determining verdict for {doc_type}")
        verdict_service = get_verdict_service()
        verdict, evidence = verdict_service.determine_verdict(
            document_type=doc_type,
            user_id=user_id,
            extracted_fields=extracted_fields,
            format_valid=format_valid,
            format_errors=format_errors,
            crossref_match=crossref_match,
            match_results=match_results,
            classification_confidence=classification_confidence,
            extraction_summary=extraction_summary
        )
        
        # Step 8: Store review evidence if needs_review
        evidence_id = None
        if verdict == VerdictConfig.VERDICT_NEEDS_REVIEW:
            review_store = get_review_store()
            evidence_id = review_store.store_review(evidence, request_id)
            logger.info(f"Stored review evidence: {evidence_id}")
        
        # Return result
        return DocumentProcessingResult(
            document_type=doc_type,
            classification_confidence=classification_confidence,
            extracted_fields=extracted_fields_dict,
            extraction_summary=extraction_summary,
            format_valid=format_valid,
            format_errors=format_errors,
            crossref_match=crossref_match,
            match_results=[result.to_dict() for result in match_results],
            verdict=verdict,
            verdict_reasons=evidence.verdict_reasons,
            evidence_id=evidence_id
        )
        
    except ImageQualityError as e:
        logger.error(f"Image quality gate failed for {document_type_hint}: {e.reason}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image quality check failed: {e.reason}"
        )
    except Exception as e:
        logger.error(f"Error processing {document_type_hint}: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {e!s}"
        )


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with service status
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        services={
            "ocr": "operational",
            "classification": "operational",
            "extraction": "operational",
            "validation": "operational",
            "crossref": "operational",
            "verdict": "operational"
        }
    )


@app.post("/verify", 
          response_model=VerificationResponse,
          dependencies=[Depends(verify_api_key)])
@limiter.limit(f"{APIConfig.RATE_LIMIT_TIMES}/{APIConfig.RATE_LIMIT_SECONDS} seconds")
async def verify_documents(
    request: Request,
    user_id: str = Form(...),
    cnic_img: UploadFile = File(...),
    domicile_img: UploadFile = File(...),
    transcript_img: UploadFile = File(...)
):
    """
    Main verification endpoint - processes three documents concurrently.
    
    Args:
        user_id: User identifier
        cnic_img: CNIC document image
        domicile_img: Domicile document image
        transcript_img: Transcript document image
        
    Returns:
        VerificationResponse with overall verdict and individual document results
    """
    request_id = str(uuid4())
    start_time = datetime.now(timezone.utc)
    
    logger.info(f"Starting verification request {request_id} for user {user_id}")
    
    try:
        # Read file contents
        cnic_bytes = await cnic_img.read()
        domicile_bytes = await domicile_img.read()
        transcript_bytes = await transcript_img.read()
        
        # Get file metadata
        cnic_size = len(cnic_bytes)
        domicile_size = len(domicile_bytes)
        transcript_size = len(transcript_bytes)
        
        cnic_mime = cnic_img.content_type or "image/jpeg"
        domicile_mime = domicile_img.content_type or "image/jpeg"
        transcript_mime = transcript_img.content_type or "image/jpeg"
        
        # Process documents concurrently
        logger.info(f"Processing documents concurrently for request {request_id}")
        results = await asyncio.gather(
            process_single_document(
                cnic_bytes, cnic_size, cnic_mime, user_id, "cnic", request_id
            ),
            process_single_document(
                domicile_bytes, domicile_size, domicile_mime, user_id, "domicile", request_id
            ),
            process_single_document(
                transcript_bytes, transcript_size, transcript_mime, user_id, "transcript", request_id
            ),
            return_exceptions=True
        )
        
        # Check for processing errors
        documents = {}
        for i, (doc_type, result) in enumerate(zip(["cnic", "domicile", "transcript"], results)):
            if isinstance(result, Exception):
                logger.error(f"Error processing {doc_type}: {result!s}")
                # Create error result
                documents[doc_type] = DocumentProcessingResult(
                    document_type="error",
                    classification_confidence=0.0,
                    extracted_fields={},
                    extraction_summary={},
                    format_valid=False,
                    format_errors=[str(result)],
                    crossref_match=False,
                    match_results=[],
                    verdict=VerdictConfig.VERDICT_NEEDS_REVIEW,
                    verdict_reasons=[f"Processing error: {result!s}"]
                )
            else:
                documents[doc_type] = result
        
        # Determine overall verdict
        verdict_service = get_verdict_service()
        individual_verdicts = {
            doc_type: result.verdict 
            for doc_type, result in documents.items()
        }
        overall_verdict = verdict_service.get_overall_verdict(individual_verdicts)
        
        # Check if cross-reference was used
        crossref_service = get_crossref_service()
        cross_reference_used = crossref_service.get_reference_data(user_id) is not None
        
        # Calculate processing time
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        logger.info(f"Completed verification request {request_id} in {processing_time:.2f}s - overall verdict: {overall_verdict}")
        
        return VerificationResponse(
            request_id=request_id,
            overall_verdict=overall_verdict,
            processing_time_seconds=processing_time,
            documents=documents,
            cross_reference_used=cross_reference_used,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in verify endpoint: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e!s}"
        )


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with structured responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP_ERROR",
            detail=exc.detail,
            request_id=None
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions with structured responses."""
    logger.error(f"Unhandled exception: {exc!s}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            detail="An unexpected error occurred",
            request_id=None
        ).model_dump()
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        reload=False
    )