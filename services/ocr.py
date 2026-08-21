"""
OCR service using PaddleOCR with bounding boxes and confidence scores.
Returns word-level OCR results with spatial information for anchor-based extraction.
"""

import importlib
import logging

import numpy as np

# IMPORT ORDER IS LOAD-BEARING - do not move below the paddleocr import.
#
# paddlepaddle's libpaddle.so interposes zlib symbols process-wide once it is
# loaded. Any compiled extension imported AFTER it that routes through zlib
# then fails with "zlib.error: Error -2 while decompressing data: inconsistent
# stream state". Verified in CI: importing paddle first breaks scipy; importing
# scipy (or skimage) first makes the whole paddleocr chain import cleanly.
# It surfaced as a pyclipper failure on some builds and scipy on others,
# because it hits whichever extension happens to load after paddle.
#
# importlib is used rather than plain imports so the statement below separates
# the two import blocks and isort cannot reorder paddleocr above this.
# Best-effort: in environments where these are absent (e.g. unit tests that mock
# paddleocr) there is nothing to protect, and the paddleocr import below fails
# loudly on its own if the real dependency chain is genuinely broken.
for _preload in ("scipy._lib._ccallback", "skimage.morphology"):
    try:
        importlib.import_module(_preload)
    except ImportError:
        logging.getLogger(__name__).debug("zlib-order preload skipped: %s", _preload)

from paddleocr import PaddleOCR  # noqa: E402  (see import-order note above)

from config import OCRConfig  # noqa: E402

logger = logging.getLogger(__name__)


class OCRResult:
    """Represents a single OCR result with spatial and confidence information."""
    
    def __init__(self, text: str, bbox: list[int], confidence: float):
        """
        Initialize OCR result.
        
        Args:
            text: Extracted text
            bbox: Bounding box [x1, y1, x2, y2] (top-left, bottom-right)
            confidence: Confidence score (0-1)
        """
        self.text = text
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
    
    @property
    def center(self) -> tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def width(self) -> int:
        """Get width of bounding box."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        """Get height of bounding box."""
        return self.bbox[3] - self.bbox[1]
    
    def distance_to(self, other: 'OCRResult') -> float:
        """Calculate Euclidean distance to another OCR result."""
        center1 = np.array(self.center)
        center2 = np.array(other.center)
        return np.linalg.norm(center1 - center2)
    
    def is_to_right_of(self, other: 'OCRResult', max_distance: int = 200) -> bool:
        """Check if this result is to the right of another within max distance."""
        return (self.bbox[0] > other.bbox[2] and 
                self.bbox[0] - other.bbox[2] <= max_distance and
                abs(self.center[1] - other.center[1]) <= 50)  # Similar vertical position
    
    def is_below(self, other: 'OCRResult', max_distance: int = 50) -> bool:
        """Check if this result is below another within max distance."""
        return (self.bbox[1] > other.bbox[3] and 
                self.bbox[1] - other.bbox[3] <= max_distance and
                abs(self.center[0] - other.center[0]) <= 50)  # Similar horizontal position
    
    def overlaps_vertically(self, other: 'OCRResult') -> bool:
        """Check if bounding boxes overlap vertically."""
        return not (self.bbox[3] < other.bbox[1] or other.bbox[3] < self.bbox[1])
    
    def overlaps_horizontally(self, other: 'OCRResult') -> bool:
        """Check if bounding boxes overlap horizontally."""
        return not (self.bbox[2] < other.bbox[0] or other.bbox[2] < self.bbox[0])


class OCRService:
    """Singleton OCR service using PaddleOCR."""
    
    _instance = None
    _ocr_engine = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize OCR engine (lazy loading)."""
        if self._ocr_engine is None:
            self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize PaddleOCR engine."""
        try:
            self._ocr_engine = PaddleOCR(
                use_angle_cls=OCRConfig.USE_ANGLE_CLASSIFIER,
                lang=OCRConfig.LANGUAGES[0],
                use_gpu=OCRConfig.USE_GPU,
                cpu_threads=OCRConfig.CPU_THREADS,
                show_log=OCRConfig.SHOW_LOG
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e!s}")
            raise RuntimeError(f"OCR initialization failed: {e!s}")
    
    def process_image(self, image: np.ndarray) -> list[OCRResult]:
        """
        Process image with OCR and return structured results.
        
        Args:
            image: Preprocessed numpy array image
            
        Returns:
            List of OCRResult objects with text, bbox, and confidence
        """
        try:
            # Run PaddleOCR
            result = self._ocr_engine.ocr(image, cls=False)
            
            if not result or not result[0]:
                logger.warning("OCR returned no results")
                return []
            
            # Convert PaddleOCR format to OCRResult objects
            ocr_results = []
            for line in result[0]:
                # PaddleOCR returns: [[bbox], (text, confidence)]
                bbox_points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line[1]    # (text, confidence)
                
                # Convert polygon bbox to rectangular bbox [x1, y1, x2, y2]
                x_coords = [point[0] for point in bbox_points]
                y_coords = [point[1] for point in bbox_points]
                bbox = [int(min(x_coords)), int(min(y_coords)), 
                        int(max(x_coords)), int(max(y_coords))]
                
                text = text_info[0]
                confidence = float(text_info[1])
                
                # Filter by minimum confidence
                if confidence >= OCRConfig.MIN_WORD_CONFIDENCE:
                    ocr_results.append(OCRResult(text, bbox, confidence))
            
            logger.info(f"OCR completed: {len(ocr_results)} words detected")
            return ocr_results
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e!s}")
            raise RuntimeError(f"OCR processing failed: {e!s}")
    
    def get_full_text(self, ocr_results: list[OCRResult]) -> str:
        """
        Get concatenated text from OCR results.
        
        Args:
            ocr_results: List of OCRResult objects
            
        Returns:
            Concatenated text string
        """
        return " ".join([result.text for result in ocr_results])
    
    def find_text_by_keywords(self, ocr_results: list[OCRResult], 
                             keywords: set, case_sensitive: bool = False) -> list[OCRResult]:
        """
        Find OCR results containing specific keywords.
        
        Args:
            ocr_results: List of OCRResult objects
            keywords: Set of keywords to search for
            case_sensitive: Whether search is case sensitive
            
        Returns:
            List of matching OCRResult objects
        """
        matches = []
        search_keywords = keywords if case_sensitive else {k.lower() for k in keywords}
        
        for result in ocr_results:
            text_to_search = result.text if case_sensitive else result.text.lower()
            if any(keyword in text_to_search for keyword in search_keywords):
                matches.append(result)
        
        return matches
    
    def find_text_nearby(self, ocr_results: list[OCRResult], 
                        anchor_result: OCRResult, 
                        max_distance: int = 300,
                        direction: str = 'any') -> list[OCRResult]:
        """
        Find OCR results near an anchor result.
        
        Args:
            ocr_results: List of OCRResult objects
            anchor_result: Anchor OCRResult to search around
            max_distance: Maximum distance in pixels
            direction: 'any', 'right', 'below', 'right_or_below'
            
        Returns:
            List of nearby OCRResult objects, sorted by distance
        """
        nearby = []
        
        for result in ocr_results:
            if result is anchor_result:
                continue
            
            distance = result.distance_to(anchor_result)
            
            if distance <= max_distance:
                # Apply direction filters
                if direction == 'any' or direction == 'right' and result.is_to_right_of(anchor_result, max_distance) or direction == 'below' and result.is_below(anchor_result, max_distance) or direction == 'right_or_below' and (result.is_to_right_of(anchor_result, max_distance) or result.is_below(anchor_result, max_distance)):
                    nearby.append((result, distance))
        
        # Sort by distance
        nearby.sort(key=lambda x: x[1])
        return [result for result, _ in nearby]


# Singleton instance
_ocr_service = None

def get_ocr_service() -> OCRService:
    """Get the singleton OCR service instance."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service