"""
Pytest configuration with mocks for problematic dependencies.
"""

import sys
from unittest.mock import MagicMock, Mock

# Mock paddleocr since it can't be installed on Python 3.14
mock_paddleocr = MagicMock()
mock_paddleocr_module = MagicMock()
mock_paddleocr_module.PaddleOCR = Mock
sys.modules['paddleocr'] = mock_paddleocr_module

# Mock llama_cpp for the same reason
mock_llama_cpp = MagicMock()
mock_llama_cpp_module = MagicMock()
mock_llama_cpp_module.Llama = Mock
sys.modules['llama_cpp'] = mock_llama_cpp_module
