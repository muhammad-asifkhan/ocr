"""
Review storage service for persisting needs_review verdicts.
Stores evidence for audit trail and human review.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from config import REVIEW_STORE_PATH, ReviewStoreConfig
from services.verdict import VerdictEvidence

logger = logging.getLogger(__name__)


class ReviewRecord:
    """Represents a review record for a needs_review verdict."""
    def __init__(self, evidence: VerdictEvidence, request_id: str):
        self.review_id = str(uuid4())
        self.request_id = request_id
        self.evidence = evidence.to_dict()
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.status = "pending"  # pending, reviewed, resolved
        self.reviewed_by: str | None = None
        self.reviewed_at: str | None = None
        self.resolution_notes: str | None = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "review_id": self.review_id,
            "request_id": self.request_id,
            "evidence": self.evidence,
            "created_at": self.created_at,
            "status": self.status,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "resolution_notes": self.resolution_notes
        }


class ReviewStore:
    """Service for storing and retrieving review records."""
    
    def __init__(self):
        """Initialize review store."""
        self.storage_path = REVIEW_STORE_PATH
        self._ensure_storage_file()
    
    def _ensure_storage_file(self):
        """Ensure storage file exists."""
        if not self.storage_path.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump([], f)
            logger.info(f"Created review storage file: {self.storage_path}")
    
    def _load_reviews(self) -> list[dict]:
        """Load all reviews from storage."""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load reviews: {e!s}")
            return []
    
    def _save_reviews(self, reviews: list[dict]):
        """Save reviews to storage."""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(reviews, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save reviews: {e!s}")
    
    def store_review(self, evidence: VerdictEvidence, request_id: str) -> str:
        """
        Store a review record for a needs_review verdict.
        
        Args:
            evidence: VerdictEvidence object
            request_id: Request identifier
            
        Returns:
            Review ID
        """
        review_record = ReviewRecord(evidence, request_id)
        
        reviews = self._load_reviews()
        reviews.append(review_record.to_dict())
        
        self._save_reviews(reviews)
        
        logger.info(f"Stored review record: {review_record.review_id} for request: {request_id}")
        return review_record.review_id
    
    def get_review(self, review_id: str) -> dict | None:
        """
        Retrieve a review record by ID.
        
        Args:
            review_id: Review identifier
            
        Returns:
            Review record dictionary or None if not found
        """
        reviews = self._load_reviews()
        for review in reviews:
            if review.get('review_id') == review_id:
                return review
        return None
    
    def get_reviews_by_user(self, user_id: str) -> list[dict]:
        """
        Retrieve all review records for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of review record dictionaries
        """
        reviews = self._load_reviews()
        return [
            review for review in reviews 
            if review.get('evidence', {}).get('user_id') == user_id
        ]
    
    def get_reviews_by_status(self, status: str) -> list[dict]:
        """
        Retrieve all review records with a specific status.
        
        Args:
            status: Status to filter by (pending, reviewed, resolved)
            
        Returns:
            List of review record dictionaries
        """
        reviews = self._load_reviews()
        return [
            review for review in reviews 
            if review.get('status') == status
        ]
    
    def get_pending_reviews(self) -> list[dict]:
        """
        Retrieve all pending review records.
        
        Returns:
            List of pending review record dictionaries
        """
        return self.get_reviews_by_status("pending")
    
    def update_review_status(self, review_id: str, status: str, 
                           reviewed_by: str, resolution_notes: str = "") -> bool:
        """
        Update the status of a review record.
        
        Args:
            review_id: Review identifier
            status: New status (reviewed, resolved)
            reviewed_by: Who is reviewing
            resolution_notes: Notes about the resolution
            
        Returns:
            True if update successful, False otherwise
        """
        reviews = self._load_reviews()
        
        for i, review in enumerate(reviews):
            if review.get('review_id') == review_id:
                reviews[i]['status'] = status
                reviews[i]['reviewed_by'] = reviewed_by
                reviews[i]['reviewed_at'] = datetime.now(timezone.utc).isoformat()
                reviews[i]['resolution_notes'] = resolution_notes
                
                self._save_reviews(reviews)
                logger.info(f"Updated review {review_id} to status: {status}")
                return True
        
        logger.warning(f"Review not found: {review_id}")
        return False
    
    def cleanup_old_reviews(self, days: int | None = None) -> int:
        """
        Remove review records older than specified days.
        
        Args:
            days: Number of days to retain (uses config default if not specified)
            
        Returns:
            Number of reviews removed
        """
        if days is None:
            days = ReviewStoreConfig.REVIEW_RETENTION_DAYS
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        reviews = self._load_reviews()
        
        original_count = len(reviews)
        reviews = [
            review for review in reviews 
            if datetime.fromisoformat(review['created_at']) >= cutoff_date
        ]
        
        removed_count = original_count - len(reviews)
        
        if removed_count > 0:
            self._save_reviews(reviews)
            logger.info(f"Cleaned up {removed_count} old reviews (older than {days} days)")
        
        return removed_count
    
    def get_statistics(self) -> dict:
        """
        Get statistics about review records.
        
        Returns:
            Dictionary with review statistics
        """
        reviews = self._load_reviews()
        
        total = len(reviews)
        pending = len([r for r in reviews if r.get('status') == 'pending'])
        reviewed = len([r for r in reviews if r.get('status') == 'reviewed'])
        resolved = len([r for r in reviews if r.get('status') == 'resolved'])
        
        # Count by document type
        by_document_type = {}
        for review in reviews:
            doc_type = review.get('evidence', {}).get('document_type', 'unknown')
            by_document_type[doc_type] = by_document_type.get(doc_type, 0) + 1
        
        return {
            "total_reviews": total,
            "pending": pending,
            "reviewed": reviewed,
            "resolved": resolved,
            "by_document_type": by_document_type
        }


# Singleton instance
_review_store = None

def get_review_store() -> ReviewStore:
    """Get the singleton review store instance."""
    global _review_store
    if _review_store is None:
        _review_store = ReviewStore()
    return _review_store