"""
LLM service for transcript extraction fallback.
Lazy-loaded singleton that only initializes when needed.
"""

import json
import logging

from llama_cpp import Llama

from config import LLMConfig

logger = logging.getLogger(__name__)


class LLMService:
    """Singleton LLM service for transcript extraction fallback."""
    
    _instance = None
    _llm_engine = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize LLM engine (lazy loading)."""
        if self._llm_engine is None and LLMConfig.ENABLED:
            self._initialize_llm()
    
    def _initialize_llm(self):
        """
        Initialize Llama.cpp engine.

        The LLM is an OPTIONAL fallback used only for transcripts whose layout
        defeats anchor-based extraction. It must never take the whole API down:
        if the model is missing or fails to load we log it and leave the engine
        unset, so is_available() returns False and transcript extraction
        degrades to anchor-only rather than the service failing to start.
        """
        if not LLMConfig.MODEL_PATH.exists():
            logger.error(
                f"LLM model not found at {LLMConfig.MODEL_PATH}. "
                "Transcript extraction will run anchor-only with no LLM fallback."
            )
            self._llm_engine = None
            return

        try:
            self._llm_engine = Llama(
                model_path=str(LLMConfig.MODEL_PATH),
                n_ctx=LLMConfig.N_CTX,
                n_threads=LLMConfig.N_THREADS,
                verbose=False
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to initialize LLM: {e!s}. "
                "Transcript extraction will run anchor-only with no LLM fallback."
            )
            self._llm_engine = None
    
    def extract_transcript_fields(self, raw_text: str) -> dict | None:
        """
        Extract transcript fields using LLM.
        
        Args:
            raw_text: Raw text from OCR
            
        Returns:
            Dictionary of extracted fields or None if failed
        """
        if not LLMConfig.ENABLED:
            logger.warning("LLM is disabled in configuration")
            return None
        
        if self._llm_engine is None:
            logger.error("LLM engine not initialized")
            return None
        
        try:
            prompt = f"""Extract the following fields as JSON from this Academic Transcript text. 
Fields: roll_number, student_name, father_name, degree_program, total_marks, obtained_marks, grade, semester.
Return ONLY valid JSON. No markdown.
Text: {raw_text[:2000]}
JSON:"""
            
            response = self._llm_engine(
                prompt,
                max_tokens=LLMConfig.MAX_TOKENS,
                temperature=LLMConfig.TEMPERATURE
            )
            
            if not response or not response.get('choices'):
                logger.error("LLM returned no response")
                return None
            
            extracted_text = response['choices'][0]['text']
            
            # Parse JSON response
            try:
                extracted_data = json.loads(extracted_text)
                logger.info("LLM extraction successful")
                return extracted_data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {e!s}")
                return None
                
        except Exception as e:
            logger.error(f"LLM extraction failed: {e!s}")
            return None
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return LLMConfig.ENABLED and self._llm_engine is not None


# Singleton instance
_llm_service = None

def get_llm_service() -> LLMService | None:
    """Get the singleton LLM service instance."""
    global _llm_service
    if _llm_service is None and LLMConfig.ENABLED:
        _llm_service = LLMService()
    return _llm_service