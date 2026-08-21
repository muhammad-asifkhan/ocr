# Government Document Verification System

An enterprise-grade document classification, extraction, and verification system for Pakistani government documents (CNIC, Domicile, Transcript). Built with accuracy, auditability, and security as top priorities.

## 🏗️ Architecture Overview

This system uses a **deterministic-first approach** with LLM as a fallback only for variable layouts:

- **CNIC/Domicile**: 100% deterministic anchor-based extraction (no LLM)
- **Transcript**: Anchor-based extraction first, LLM fallback if insufficient fields found
- **Classification**: Keyword rules only (no LLM fallback)
- **Validation**: Three-state verdicts (verified/needs_review/rejected) with full evidence preservation

## 📁 Project Structure

```
D:\Projects\ocr\
├── api/
│   ├── __init__.py
│   └── main.py                    # FastAPI application with auth & rate limiting
├── services/
│   ├── __init__.py
│   ├── preprocess.py              # Image quality gates & preprocessing
│   ├── ocr.py                     # PaddleOCR with bounding boxes
│   ├── classify.py                # Deterministic document classification
│   ├── extract_cnic.py            # CNIC anchor-based extraction
│   ├── extract_domicile.py        # Domicile anchor-based extraction
│   ├── extract_transcript.py      # Transcript anchor + LLM extraction
│   ├── llm_service.py             # LLM service for transcript fallback
│   ├── validate_format.py         # Schema validation
│   ├── crossref.py                # Fuzzy matching against reference DB
│   ├── verdict.py                 # Three-state verdict logic
│   └── review_store.py            # Review evidence persistence
├── data/
│   ├── reference_db.csv           # User reference database
│   └── reviews.json               # Needs_review evidence storage
├── models/
│   └── phi-3-mini-4k-instruct-q4.gguf  # LLM model (transcript fallback only)
├── logs/
│   └── verification.log           # Application logs
├── config.py                      # Centralized configuration
├── requirements.txt              # Pinned dependencies
└── README.md                      # This file
```

## 🔧 Configuration Thresholds

### Image Quality Gates
- **MIN_BLUR_SCORE (100.0)**: Laplacian variance threshold for blur detection. Below this, images are rejected as too blurry.
- **MIN_WIDTH/MIN_HEIGHT (300px)**: Minimum resolution requirements.
- **MAX_FILE_SIZE_MB (10.0)**: Maximum uploaded file size.
- **MIN_FILE_SIZE_KB (10)**: Minimum file size to prevent empty uploads.

### OCR Configuration
- **MIN_WORD_CONFIDENCE (0.5)**: Minimum OCR confidence score to include a word in results.

### Classification
- **MIN_KEYWORD_MATCHES (1)**: Minimum keyword matches required for classification.
- Keywords are checked in priority: DOMICILE → CNIC → TRANSCRIPT

### Extraction
- **SEARCH_RADIUS (300px)**: Maximum distance to search for values near labels.
- **MAX_HORIZONTAL_DISTANCE (200px)**: Maximum horizontal distance for value extraction.
- **MAX_VERTICAL_DISTANCE (50px)**: Maximum vertical distance for value extraction.
- **MIN_ANCHOR_FIELDS (3)**: Minimum fields required from anchor extraction before LLM fallback (transcript only).

### Format Validation
- **CNIC_LENGTH (13)**: Required CNIC digit count.
- **MIN_NAME_LENGTH (2)**: Minimum characters for name fields.
- **MAX_NAME_LENGTH (100)**: Maximum characters for name fields.

### Cross-Reference Matching
- **FUZZY_NAME_THRESHOLD (88)**: Minimum similarity score for name matching (rapidfuzz).
- **FUZZY_PROVINCE_THRESHOLD (85)**: Minimum similarity score for province matching.
- **HIGH_MATCH_THRESHOLD (95)**: Above this = clear match.
- **LOW_MATCH_THRESHOLD (70)**: Below this = clear mismatch.
- **MIDDLE_BAND (70-95)**: Between these = needs review.

### Verdict Logic
- **MIN_EXTRACTION_CONFIDENCE (0.6)**: Below this = needs review.
- **MIN_OVERALL_CONFIDENCE (0.7)**: Below this = needs review.

### API Configuration
- **RATE_LIMIT_TIMES (10)**: Requests allowed per time window.
- **RATE_LIMIT_SECONDS (60)**: Time window for rate limiting.
- **REQUEST_TIMEOUT_SECONDS (120)**: Maximum processing time per request.

## 🚀 Setup Instructions

### 1. Prerequisites
- Python 3.8+
- 4GB RAM minimum
- Windows/Linux (CPU-only operation)

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Models

```bash
# Download PaddleOCR models (automatic on first run)
# The system will download required models automatically

# Download LLM model for transcript fallback (optional)
# Place phi-3-mini-4k-instruct-q4.gguf in models/ directory
# Current system has the model already downloaded
```

### 4. Configure Reference Database

Edit `data/reference_db.csv` with your user data:

```csv
user_id,cnic,name,province
user001,1234567890123,Ali Khan,Punjab
user002,2345667890124,Sara Ahmed,Sindh
user003,3456789012345,Usman Ali,KPK
```

### 5. Set API Key

```bash
# Set environment variable for production
# Windows:
set API_KEY=your-secure-api-key-here

# Linux/Mac:
export API_KEY=your-secure-api-key-here
```

### 6. Start the Server

```bash
python -m api.main
```

The API will start on `http://0.0.0.0:8000`

## 📡 API Usage

### Authentication

All requests require an API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/health
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Main Verification Endpoint

**POST** `/verify`

**Headers:**
- `X-API-Key`: Your API key

**Form Data:**
- `user_id`: User identifier (string)
- `cnic_img`: CNIC document image (file)
- `domicile_img`: Domicile document image (file)
- `transcript_img`: Transcript document image (file)

**Example Request:**

```bash
curl -X POST "http://localhost:8000/verify" \
  -H "X-API-Key: your-api-key" \
  -F "user_id=user001" \
  -F "cnic_img=@cnic.jpg" \
  -F "domicile_img=@domicile.jpg" \
  -F "transcript_img=@transcript.jpg"
```

**Example Response:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "overall_verdict": "verified",
  "processing_time_seconds": 8.5,
  "documents": {
    "cnic": {
      "document_type": "CNIC",
      "classification_confidence": 0.95,
      "extracted_fields": {
        "cnic_number": {
          "field_name": "cnic_number",
          "value": "1234567890123",
          "source_bbox": [100, 200, 300, 220],
          "extraction_confidence": 0.92,
          "extraction_method": "anchor"
        },
        "full_name": {
          "field_name": "full_name",
          "value": "Ali Khan",
          "source_bbox": [100, 250, 200, 270],
          "extraction_confidence": 0.88,
          "extraction_method": "anchor"
        }
      },
      "extraction_summary": {
        "total_fields": 5,
        "extracted_count": 4,
        "required_fields": 2,
        "required_extracted": 2,
        "avg_confidence": 0.85,
        "extraction_rate": 0.8,
        "required_satisfied": true
      },
      "format_valid": true,
      "format_errors": [],
      "crossref_match": true,
      "match_results": [
        {
          "field_name": "cnic_number",
          "matched": true,
          "similarity_score": 100.0,
          "extracted_value": "1234567890123",
          "reference_value": "1234567890123"
        },
        {
          "field_name": "full_name",
          "matched": true,
          "similarity_score": 95.0,
          "extracted_value": "Ali Khan",
          "reference_value": "Ali Khan"
        }
      ],
      "verdict": "verified",
      "verdict_reasons": ["All validation checks passed"],
      "evidence_id": null
    },
    "domicile": { /* similar structure */ },
    "transcript": { /* similar structure */ }
  },
  "cross_reference_used": true,
  "timestamp": "2026-08-19T22:30:00.000Z"
}
```

## 🔍 Verdict States

### **verified**
All required fields matched above threshold AND extraction confidence sufficient.

### **needs_review**
- Format validation failed
- Extraction confidence below threshold
- Fuzzy match score in ambiguous middle band (70-95%)
- Classification was uncertain
- Insufficient fields extracted (transcript LLM fallback)

### **rejected**
Required fields clearly do not match (similarity < 70%) with high extraction confidence.

## 🔒 Security Features

- **API Key Authentication**: Required for all endpoints
- **Rate Limiting**: 10 requests per 60 seconds per IP
- **PII Redaction**: Logs contain redacted CNIC numbers and names
- **File Validation**: MIME type and size validation before processing
- **Image Quality Gates**: Rejects poor quality images before processing
- **Audit Trail**: Full evidence preservation for all needs_review cases

## 📊 Audit Trail

All `needs_review` verdicts are automatically stored in `data/reviews.json` with:

- Complete extraction evidence
- Match results with similarity scores
- Format validation errors
- Verdict reasoning
- Timestamps and request IDs

This evidence can be retrieved for human review via a separate admin interface (to be implemented).

## 🧪 Testing the System

### Test with Sample Images

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test verification with sample images
curl -X POST "http://localhost:8000/verify" \
  -H "X-API-Key: dev-secret-key-change-in-production" \
  -F "user_id=user001" \
  -F "cnic_img=@test_cnic.jpg" \
  -F "domicile_img=@test_domicile.jpg" \
  -F "transcript_img=@test_transcript.jpg"
```

### Monitor Logs

```bash
# View application logs
tail -f logs/verification.log
```

## 🛠️ Configuration

All configuration is centralized in `config.py`. Key sections:

- **ImageQualityConfig**: Image processing thresholds
- **OCRConfig**: OCR engine settings
- **ClassificationConfig**: Document classification rules
- **CNICExtractionConfig**: CNIC field extraction parameters
- **DomicileExtractionConfig**: Domicile field extraction parameters
- **TranscriptExtractionConfig**: Transcript field extraction parameters
- **LLMConfig**: LLM fallback settings
- **FormatValidationConfig**: Schema validation rules
- **CrossReferenceConfig**: Fuzzy matching thresholds
- **VerdictConfig**: Decision logic parameters
- **APIConfig**: Server and authentication settings
- **DataHandlingConfig**: Logging and data retention

## 🚨 Error Handling

The system provides structured error responses:

```json
{
  "error": "IMAGE_QUALITY_ERROR",
  "detail": "Image quality check failed: image_too_blurry",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

Common error types:
- `IMAGE_QUALITY_ERROR`: Image failed quality gates
- `INVALID_MIME_TYPE`: Unsupported file type
- `FILE_TOO_LARGE`: File exceeds size limit
- `AUTHENTICATION_ERROR`: Invalid API key
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `PROCESSING_ERROR`: Document processing failure

## 📈 Performance

- **Processing Time**: ~5-10 seconds per document (3 documents processed concurrently)
- **Memory Usage**: ~4GB RAM
- **CPU**: Multi-core recommended (PaddleOCR uses 2 threads, LLM uses 4 threads)
- **Concurrency**: Processes 3 documents simultaneously per request

## 🔮 Future Enhancements

- Database backend for reference data (currently CSV)
- Admin interface for review management
- Real database for review storage (currently JSON)
- Encryption at rest for retained images
- Additional document types
- Webhook notifications for review cases
- Advanced image preprocessing
- Multi-language support

## 📝 License

This system is designed for government document verification and should be used in compliance with applicable data protection regulations.

## 🆘 Support

For issues or questions:
1. Check logs in `logs/verification.log`
2. Verify configuration in `config.py`
3. Ensure reference database is properly formatted
4. Confirm API key is set correctly

---

**Built for accuracy, auditability, and security in government document verification.**