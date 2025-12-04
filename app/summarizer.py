"""
Import the GPU server summarizer to avoid code duplication.
This allows the GPU server to use the same summarizer module.
"""

from app.nlp.summarizer import SummarizerBigBirdPegasus

def get_summarizer() -> SummarizerBigBirdPegasus:
    """
    Get or create a singleton instance of the summarizer.
    
    Returns:
        SummarizerBigBirdPegasus instance
    """
    global _summarizer
    if '_summarizer' not in globals() or _summarizer is None:
        _summarizer = SummarizerBigBirdPegasus()
    return _summarizer

_summarizer = None
