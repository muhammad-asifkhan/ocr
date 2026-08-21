"""
Image preprocessing and quality validation service.
Handles image loading, orientation correction, deskewing, denoising, and quality gates.
"""

import io
import logging

import cv2
import numpy as np
from PIL import Image, ImageOps

from config import ImageQualityConfig

logger = logging.getLogger(__name__)


class ImageQualityError(Exception):
    """Raised when image fails quality gates."""
    def __init__(self, reason: str, details: dict | None = None):
        self.reason = reason
        self.details = details or {}
        super().__init__(reason)


class Preprocessor:
    """Handles image preprocessing and quality validation."""
    
    @staticmethod
    def validate_file_upload(file_size: int, mime_type: str) -> None:
        """
        Validate uploaded file before processing.
        
        Args:
            file_size: File size in bytes
            mime_type: MIME type of the file
            
        Raises:
            ImageQualityError: If file fails validation
        """
        # Check file size
        max_size = ImageQualityConfig.MAX_FILE_SIZE_MB * 1024 * 1024
        min_size = ImageQualityConfig.MIN_FILE_SIZE_KB * 1024
        
        if file_size > max_size:
            raise ImageQualityError(
                "file_too_large",
                details={
                    "max_size_mb": ImageQualityConfig.MAX_FILE_SIZE_MB,
                    "actual_size_mb": file_size / (1024 * 1024)
                }
            )
        
        if file_size < min_size:
            raise ImageQualityError(
                "file_too_small",
                details={
                    "min_size_kb": ImageQualityConfig.MIN_FILE_SIZE_KB,
                    "actual_size_kb": file_size / 1024
                }
            )
        
        # Check MIME type
        if mime_type.lower() not in ImageQualityConfig.ALLOWED_MIME_TYPES:
            raise ImageQualityError(
                "invalid_mime_type",
                details={
                    "allowed_types": list(ImageQualityConfig.ALLOWED_MIME_TYPES),
                    "actual_type": mime_type
                }
            )
    
    @staticmethod
    def load_image(image_bytes: bytes) -> np.ndarray:
        """
        Load image from bytes with EXIF orientation correction.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            numpy array of the image
            
        Raises:
            ImageQualityError: If image cannot be loaded
        """
        try:
            # Use PIL for EXIF orientation correction
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image = ImageOps.exif_transpose(pil_image)
            
            # Convert to OpenCV format
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            logger.info("Image loaded successfully with EXIF correction")
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image: {e!s}")
            raise ImageQualityError("invalid_image_format", details={"error": str(e)})
    
    @staticmethod
    def check_resolution(image: np.ndarray) -> None:
        """
        Check if image meets minimum resolution requirements.
        
        Args:
            image: numpy array of the image
            
        Raises:
            ImageQualityError: If resolution is too low
        """
        height, width = image.shape[:2]
        
        if width < ImageQualityConfig.MIN_WIDTH or height < ImageQualityConfig.MIN_HEIGHT:
            raise ImageQualityError(
                "resolution_too_low",
                details={
                    "min_width": ImageQualityConfig.MIN_WIDTH,
                    "min_height": ImageQualityConfig.MIN_HEIGHT,
                    "actual_width": width,
                    "actual_height": height
                }
            )
        
        logger.info(f"Resolution check passed: {width}x{height}")
    
    @staticmethod
    def compute_blur_score(image: np.ndarray) -> float:
        """
        Compute blur score using Laplacian variance.
        Higher values indicate sharper images.
        
        Args:
            image: numpy array of the image (grayscale)
            
        Returns:
            Blur score (variance of Laplacian)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Compute Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return variance
    
    @staticmethod
    def check_blur(image: np.ndarray) -> float:
        """
        Check if image is too blurry.
        
        Args:
            image: numpy array of the image
            
        Returns:
            Blur score
            
        Raises:
            ImageQualityError: If image is too blurry
        """
        blur_score = Preprocessor.compute_blur_score(image)
        
        if blur_score < ImageQualityConfig.MIN_BLUR_SCORE:
            raise ImageQualityError(
                "poor_image_quality",
                details={
                    "blur_score": blur_score,
                    "min_required": ImageQualityConfig.MIN_BLUR_SCORE,
                    "reason": "image_too_blurry"
                }
            )
        
        logger.info(f"Blur check passed: score={blur_score:.2f}")
        return blur_score
    
    @staticmethod
    def deskew_image(image: np.ndarray) -> np.ndarray:
        """
        Correct image skew using Hough line transform.
        
        Args:
            image: numpy array of the image
            
        Returns:
            Deskewed image
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find all contours
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.warning("No contours found for deskewing, returning original")
                return image
            
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get minimum area rectangle
            min_area_rect = cv2.minAreaRect(largest_contour)
            angle = min_area_rect[-1]
            
            # Adjust angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Rotate image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            logger.info(f"Image deskewed by {angle:.2f} degrees")
            return rotated
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {e!s}, returning original")
            return image
    
    @staticmethod
    def denoise_image(image: np.ndarray) -> np.ndarray:
        """
        Apply denoising to improve OCR accuracy.
        
        Args:
            image: numpy array of the image
            
        Returns:
            Denoised image
        """
        try:
            # Use Non-local Means Denoising
            if len(image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            
            logger.info("Image denoising applied")
            return denoised
            
        except Exception as e:
            logger.warning(f"Denoising failed: {e!s}, returning original")
            return image
    
    @staticmethod
    def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, dict]:
        """
        Complete preprocessing pipeline with quality gates.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (preprocessed_image, quality_metrics)
            
        Raises:
            ImageQualityError: If image fails any quality gate
        """
        quality_metrics = {}
        
        # Load image with EXIF correction
        image = Preprocessor.load_image(image_bytes)
        quality_metrics['original_size'] = image.shape[:2]
        
        # Check resolution
        Preprocessor.check_resolution(image)
        
        # Check blur
        blur_score = Preprocessor.check_blur(image)
        quality_metrics['blur_score'] = blur_score
        
        # Apply preprocessing
        image = Preprocessor.deskew_image(image)
        image = Preprocessor.denoise_image(image)
        
        quality_metrics['preprocessed_size'] = image.shape[:2]
        quality_metrics['status'] = 'passed'
        
        logger.info("Image preprocessing completed successfully")
        return image, quality_metrics


def validate_and_preprocess(image_bytes: bytes, file_size: int, mime_type: str) -> tuple[np.ndarray, dict]:
    """
    Validate file upload and preprocess image with quality gates.
    
    Args:
        image_bytes: Raw image bytes
        file_size: File size in bytes
        mime_type: MIME type of the file
        
    Returns:
        Tuple of (preprocessed_image, quality_metrics)
        
    Raises:
        ImageQualityError: If validation fails
    """
    # Validate file upload
    Preprocessor.validate_file_upload(file_size, mime_type)
    
    # Preprocess with quality gates
    return Preprocessor.preprocess_image(image_bytes)