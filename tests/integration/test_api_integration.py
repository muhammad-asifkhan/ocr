"""
Integration test for the verification API.
NOTE: This test cannot run on Python 3.14/Windows because PaddleOCR is not available.
To run this test, the system must use Python 3.10-3.13 with PaddleOCR installed.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
import io
from PIL import Image


def create_test_image(width=400, height=300, color=(255, 255, 255)):
    """Create a simple test image."""
    img = Image.new('RGB', (width, height), color)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_verify_endpoint_with_valid_images():
    """
    Test POST /verify with real images.
    This would have caught uuid4, tuple-unpacking, and Image.BytesIO bugs.
    BLOCKED: Requires PaddleOCR which is unavailable on Python 3.14.
    """
    pytest.skip("Integration test blocked - PaddleOCR not available on Python 3.14")


def test_verify_with_existing_user_id():
    """Test verification with user_id matching reference_db.csv."""
    pytest.skip("Integration test blocked - PaddleOCR not available on Python 3.14")


def test_verify_with_nonexistent_user_id():
    """Test verification with user_id not in reference_db.csv."""
    pytest.skip("Integration test blocked - PaddleOCR not available on Python 3.14")


def test_rate_limiting():
    """Test that 11th request gets 429."""
    pytest.skip("Integration test blocked - PaddleOCR not available on Python 3.14")


def test_wrong_api_key():
    """Test wrong API key returns 401."""
    client = TestClient(app)
    # /verify endpoint requires auth
    response = client.post(
        "/verify",
        headers={"X-API-Key": "wrong-key"},
        data={"user_id": "user001"},
        files={
            "cnic_img": ("cnic.jpg", create_test_image(), "image/jpeg"),
            "domicile_img": ("domicile.jpg", create_test_image(), "image/jpeg"),
            "transcript_img": ("transcript.jpg", create_test_image(), "image/jpeg")
        }
    )
    assert response.status_code == 401


def test_missing_required_file():
    """Test missing required file returns 422."""
    client = TestClient(app)
    # Missing domicile_img
    response = client.post(
        "/verify",
        headers={"X-API-Key": "dev-secret-key-change-in-production"},
        data={"user_id": "user001"},
        files={
            "cnic_img": ("cnic.jpg", create_test_image(), "image/jpeg"),
            "transcript_img": ("transcript.jpg", create_test_image(), "image/jpeg")
        }
    )
    assert response.status_code == 422


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
