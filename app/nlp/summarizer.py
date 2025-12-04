# app/nlp/summarizer.py

import re
import logging
from typing import Optional, List
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app.config import BIGBIRD_MODEL_PATH

logger = logging.getLogger(__name__)

def _basic_clean(text: str) -> str:
    """
    영문 논문 텍스트 정리.
    
    LaTeX 명령어, 특수 문자, 이상 패턴 등을 제거합니다.
    백엔드 서버의 clean_summary_en 로직을 통합하여 구현.
    """
    if not text:
        return ""

    # <n>을 공백으로 변환
    text = text.replace("<n>", " ")

    # @ 멘션, LaTeX 명령어, $ 기호 제거
    text = re.sub(r"@[a-zA-Z0-9_]+", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"\$+", " ", text)

    # 섹션 번호 제거
    text = re.sub(r"#\s*\d+", " ", text)

    # LaTeX 그래픽 관련 패턴 제거
    text = re.sub(r"epsf\.tex[^)]*\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\([^)]*width[^)]*\)", " ", text, flags=re.IGNORECASE)

    # "section" 단어 제거
    if "section" in text.lower():
        text = re.sub(r"\b[Ss]ection\b", " ", text)

    # 특수 문자 제거
    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("*", " ")
    text = text.replace(",", " ")

    # 빈 괄호 제거
    text = re.sub(r"\(\s*\)", " ", text)

    # 중복 공백 제거
    text = re.sub(r"\s+", " ", text).strip()

    return text


class SummarizerBigBirdPegasus:
    """
    BigBird-Pegasus 요약 모델 래퍼.
    싱글톤 패턴으로 모델을 한 번만 로딩하고 재사용합니다.
    """

    def __init__(
        self,
        model_path: str = BIGBIRD_MODEL_PATH,
        device: Optional[str] = None,
        max_input_length: int = 4096,
        max_output_length: int = 256,
    ):
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        logger.info(f"[Summarizer] Loading BigBird-Pegasus from {self.model_path}")
        logger.info(f"[Summarizer] Using device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                local_files_only=True,
            ).to(self.device)

            logger.info("[Summarizer] Model loaded successfully")

        except Exception as e:
            logger.error(f"[Summarizer] Failed to load model: {e}")
            raise

    def _summarize_once(self, text: str) -> str:
        if not text:
            return ""

        text = _basic_clean(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

        summary_ids = self.model.generate(
            **inputs,
            max_length=self.max_output_length,
            min_length=int(self.max_output_length * 0.4),
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            length_penalty=1.0,
            early_stopping=True,
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return _basic_clean(summary)

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 4000) -> List[str]:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks: List[str] = []
        buf = ""

        for p in paragraphs:
            if len(buf) + len(p) + 1 > max_chars:
                if buf:
                    chunks.append(buf.strip())
                buf = p
            else:
                buf += ("\n" + p) if buf else p

        if buf:
            chunks.append(buf.strip())

        return chunks

    def summarize(self, text: str) -> str:
        text = _basic_clean(text)
        if not text:
            return ""

        chunks = self._chunk_text(text, max_chars=4000)
        if not chunks:
            return ""

        chunk_summaries: List[str] = []
        for i, ch in enumerate(chunks):
            s = self._summarize_once(ch)
            if s:
                chunk_summaries.append(s)
            logger.debug(f"[Summarizer] Summarized chunk {i+1}/{len(chunks)}")

        if not chunk_summaries:
            return ""

        joined = " ".join(chunk_summaries)
        final_summary = self._summarize_once(joined)

        logger.info(f"[Summarizer] Final summary length: {len(final_summary)} chars")
        return final_summary


# 싱글톤 인스턴스 및 접근 함수
_summarizer: Optional[SummarizerBigBirdPegasus] = None

def get_summarizer() -> SummarizerBigBirdPegasus:
    """
    SummarizerBigBirdPegasus 싱글톤 인스턴스 반환.
    """
    global _summarizer
    if _summarizer is None:
        _summarizer = SummarizerBigBirdPegasus()
    return _summarizer
