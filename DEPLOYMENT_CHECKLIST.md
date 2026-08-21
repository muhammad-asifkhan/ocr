# Production Deployment Checklist

## Critical Security Configuration

### Required Environment Variables
- **API_KEY**: Production API key for authentication (REQUIRED - DO NOT use default)
  - Current default: `dev-secret-key-change-in-production`
  - Action: Set a strong, randomly generated API key before deployment
  - The system will log a CRITICAL warning if the default key is detected

### CORS Configuration
- **Current state**: `allow_origins=["*"]` (allows all origins)
- **Action**: Restrict to your actual domain before production deployment
- Location: `api/main.py` line 108
- Example: `allow_origins=["https://your-actual-domain.com"]`

## Server Requirements

### Minimum Server Specs (Estimated)
- **CPU**: 4 cores minimum (PaddleOCR and LLM fallback are CPU-intensive)
- **RAM**: 8 GB minimum (PaddleOCR models in memory + concurrent processing)
- **Storage**: 20 GB (model files + logs + reference data)
- **Python Version**: 3.13 (PaddleOCR not available for 3.14)

### Runtime Configuration
- **Workers**: Start with 1 worker, scale based on load
- **Max concurrent documents**: 3 (configurable in `APIConfig.MAX_CONCURRENT_DOCUMENTS`)
- **Rate limit**: 10 requests per 60 seconds per IP (configurable in `APIConfig`)

## Monitoring Requirements

### Essential Metrics (Post-Launch)
1. **Error Rate**: Track HTTP 4xx/5xx responses
2. **Verdict Distribution**: Monitor ratio of verified/needs_review/rejected
3. **Average Processing Time**: Time per document (target: <30 seconds)
4. **OCR Success Rate**: Percentage of documents where OCR returns results
5. **Cross-Reference Match Rate**: Percentage of documents matching reference database
6. **API Key Authentication Failures**: Detect unauthorized access attempts

### Recommended Monitoring Tools
- Prometheus + Grafana for metrics collection
- Application logging to centralized log aggregation (ELK, Loki, etc.)
- Alert on error rate >5% or average processing time >60 seconds

## Data Privacy & Compliance

### PII Handling
- **CNICs**: Redacted in logs (show first 3 and last 4 digits only)
- **Names**: Redacted in logs (show first letter only)
- **Images**: Processed in-memory, not persisted (no disk storage)
- **Reference Data**: Stored in `data/reference_db.csv` (contains user PII)

### Audit Trail
- All `needs_review` verdicts stored in `data/reviews.json`
- Logs include request IDs for tracing
- Verdict evidence preserved for all non-verified cases

## Known Limitations

### Platform Compatibility
- **Python 3.14**: NOT supported - PaddleOCR unavailable
- **Windows**: Development only - deploy on Linux for production
- **GPU**: CPU-only deployment (GPU support available but not configured)

### OCR Engine
- **PaddleOCR**: Primary OCR engine
- **No Alternative**: EasyOCR was adapted for development but not in production code
- **Dependencies**: Requires system libraries (libgomp1, libgl1-mesa-glx)

### LLM Fallback
- **Model**: Phi-3 Mini 4K instruct (GGUF format)
- **Trigger**: Only used when anchor-based extraction fails for transcripts
- **Validation**: LLM output validated against Pydantic schema before acceptance

## Pre-Deployment Validation

### Must Complete Before Deploy
1. [ ] Set production `API_KEY` environment variable
2. [ ] Restrict `allow_origins` to actual domain
3. [ ] Verify `data/reference_db.csv` contains production data
4. [ ] Run unit tests: `pytest tests/unit/`
5. [ ] Test `/health` endpoint responds
6. [ ] Test `/verify` with sample documents (using non-production data)
7. [ ] Verify rate limiting works (11th request returns 429)
8. [ ] Verify wrong API key returns 401
9. [ ] Verify missing files return 422
10. [ ] Verify logs do not contain full CNICs or names

### Docker Deployment
```bash
# Build image
docker build -t ocr-verification:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e API_KEY=your-production-key-here \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  ocr-verification:latest
```

## Rollback Plan

### In Case of Critical Failure
1. Scale to 0 workers to stop accepting new requests
2. Preserve logs and review data for analysis
3. Review `data/reviews.json` for failed verifications
4. Revert to previous version if available
5. Investigate root cause before redeploying

## Post-Launch

### Day 1 Tasks
- Monitor error rates closely
- Check OCR success rate for first 100 documents
- Verify no PII in logs
- Confirm rate limiting prevents abuse

### Week 1 Tasks
- Review verdict distribution (expecting most verified)
- Adjust thresholds if too many/review cases
- Optimize server resources based on actual load
- Set up automated alerts

### Month 1 Tasks
- Analyze common failure patterns
- Update reference database as needed
- Consider adding more document types
- Optimize OCR configuration based on document types received
