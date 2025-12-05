# app/nlp/translator.py

import re
import logging
from typing import Optional, List
import torch
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration

from app.config import M2M100_MODEL_PATH

logger = logging.getLogger(__name__)

class TranslatorM2M100:

    def __init__(
        self,
        model_name: str = M2M100_MODEL_PATH,
        device: str | None = None,
        use_fp16: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16
        
        logger.info(f"[Translator] Loading M2M100 from {model_name}")
        logger.info(f"[Translator] Using device: {self.device}")

        try:
            self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)

            if self.device == "cuda":
                if self.use_fp16:
                    self.model = self.model.half()
                self.model.to(self.device)
            else:
                self.model.to(self.device)
                self.use_fp16 = False 
            
            self.src_lang = "en"
            self.tgt_lang = "ko"
            self.tokenizer.src_lang = self.src_lang
            
            logger.info("[Translator] Model loaded successfully")
            
        except Exception as e:
            logger.error(f"[Translator] Failed to load model: {e}")
            raise

    def _translate_one(self, sentence: str, max_length: int = 512) -> str:
        sentence = sentence.strip()
        if not sentence:
            return ""

        self.tokenizer.src_lang = self.src_lang
        encoded = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            # fp16 autocast if applicable
            with torch.amp.autocast('cuda', enabled=self.use_fp16 and self.device == "cuda"):
                output = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang),
                    num_beams=4,             
                    max_length=max_length,
                    no_repeat_ngram_size=3,
                    use_cache=True,
                    early_stopping=False,
                )

        ko = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return ko

    def translate_batch(self, sentences: list[str], max_length: int = 512) -> list[str]:
        """배치 번역 (greedy decoding으로 최적화)"""
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return []

        self.tokenizer.src_lang = self.src_lang
        enc = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=self.use_fp16 and self.device == "cuda"):
                output = self.model.generate(
                    **enc,
                    forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang),
                    num_beams=1,  # Greedy decoding (최대 속도)
                    do_sample=False,
                    max_length=max_length,
                    use_cache=True,
                )

        decoded = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return decoded


    def translate(self, text: str) -> str:
        if not text:
            return ""

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        ko_list = self.translate_batch(sentences)
        ko_full = " ".join(ko_list)
        return ko_full

    def translate_parallel_text(self, texts: list[str], batch_size: int = 32) -> list[str]:
        results: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            ko_list = self.translate_batch(batch)
            results.extend(ko_list)
        return results


# 싱글톤 인스턴스 및 접근 함수
_translator: Optional[TranslatorM2M100] = None

def get_translator() -> TranslatorM2M100:
    """
    TranslatorM2M100 싱글톤 인스턴스 반환.
    FP16 기본 활성화.
    """
    global _translator
    if _translator is None:
        _translator = TranslatorM2M100(use_fp16=True)
    return _translator

