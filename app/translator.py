"""
Import the GPU server translator to avoid code duplication.
This allows the GPU server to use the same translator module.
"""

from app.nlp.translator import TranslatorM2M100

def get_translator() -> TranslatorM2M100:
    """
    Get or create a singleton instance of the translator.
    
    Returns:
        TranslatorM2M100 instance
    """
    global _translator
    if '_translator' not in globals() or _translator is None:
        # Use CPU by default for translator if GPU is busy with BigBird, 
        # or let it manage its own device (it defaults to cuda if available)
        _translator = TranslatorM2M100()
    return _translator

_translator = None
