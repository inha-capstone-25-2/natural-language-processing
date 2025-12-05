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


def _clean_output(text: str) -> str:
    """
    모델 출력에서 특수 토큰 제거.
    BigBird의 <sep_0>, <sep_1> 등의 특수 토큰을 제거합니다.
    """
    if not text:
        return ""
    
    # BigBird 특수 토큰 제거 (<sep_0>, <sep_1>, ..., <sep_N>)
    text = re.sub(r"<sep_\d+>", "", text)
    
    # 기타 특수 토큰 제거
    text = re.sub(r"<[a-z_]+>", "", text)
    
    # 중복 공백 제거
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


class SummarizerBigBirdPegasus:
    """
    BigBird-Pegasus 요약 모델 래퍼.
    싱글톤 패턴으로 모델을 한 번만 로딩하고 재사용합니다.
    FP16 및 배치 추론을 지원합니다.
    """

    def __init__(
        self,
        model_path: str = BIGBIRD_MODEL_PATH,
        device: Optional[str] = None,
        max_input_length: int = 4096,
        max_output_length: int = 256,
        use_fp16: bool = True,  # FP16 기본 활성화
    ):
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.use_fp16 = use_fp16 and self.device == "cuda"

        logger.info(f"[Summarizer] Loading BigBird-Pegasus from {self.model_path}")
        logger.info(f"[Summarizer] Using device: {self.device}, FP16: {self.use_fp16}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            
            # FP16으로 모델 로드 (메모리 절약 + 속도 향상)
            if self.use_fp16:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    dtype=torch.float16,
                ).to(self.device)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                ).to(self.device)

            logger.info("[Summarizer] Model loaded successfully")

        except Exception as e:
            logger.error(f"[Summarizer] Failed to load model: {e}")
            raise

    def _summarize_once(self, text: str) -> str:
        """단일 텍스트 요약 (greedy decoding으로 최적화)"""
        if not text:
            return ""

        text = _basic_clean(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=self.use_fp16):
                summary_ids = self.model.generate(
                    **inputs,
                    max_length=self.max_output_length,
                    min_length=50,
                    num_beams=1,  # Greedy decoding (최대 속도)
                    do_sample=False,
                    no_repeat_ngram_size=3,
                    use_cache=True,
                )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return _clean_output(summary)

    def _summarize_batch_raw(self, texts: List[str]) -> List[str]:
        """
        여러 텍스트를 GPU에서 배치로 한 번에 요약.
        Greedy decoding으로 최대 속도 달성.
        """
        if not texts:
            return []

        # 텍스트 정리
        cleaned_texts = [_basic_clean(t) for t in texts]
        # 빈 텍스트 필터링 (인덱스 유지)
        valid_indices = [i for i, t in enumerate(cleaned_texts) if t.strip()]
        valid_texts = [cleaned_texts[i] for i in valid_indices]

        if not valid_texts:
            return [""] * len(texts)

        # 배치 토크나이징 (패딩 적용)
        inputs = self.tokenizer(
            valid_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_input_length,
        ).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=self.use_fp16):
                summary_ids = self.model.generate(
                    **inputs,
                    max_length=self.max_output_length,
                    min_length=50,
                    num_beams=1,  # Greedy decoding (최대 속도)
                    do_sample=False,
                    no_repeat_ngram_size=3,
                    use_cache=True,
                )

        # 배치 디코딩 및 특수 토큰 제거
        summaries = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        summaries = [_clean_output(s) for s in summaries]

        # 원래 인덱스에 맞게 결과 재배치
        results = [""] * len(texts)
        for idx, summary in zip(valid_indices, summaries):
            results[idx] = summary

        return results

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
        """단일 텍스트 요약 (최적화: 짧은 텍스트는 재요약 생략)"""
        text = _basic_clean(text)
        if not text:
            return ""

        # 4000자 이하면 바로 요약 (재요약 생략으로 2배 빠름)
        if len(text) <= 4000:
            return self._summarize_once(text)

        # 긴 텍스트만 청크 분할
        chunks = self._chunk_text(text, max_chars=4000)
        if len(chunks) == 1:
            return self._summarize_once(chunks[0])

        # 여러 청크: 각각 요약 후 합쳐서 재요약
        chunk_summaries: List[str] = []
        for ch in chunks:
            s = self._summarize_once(ch)
            if s:
                chunk_summaries.append(s)

        if not chunk_summaries:
            return ""

        joined = " ".join(chunk_summaries)
        return self._summarize_once(joined)

    def summarize_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """
        여러 텍스트를 배치로 요약.
        모든 텍스트를 직접 배치 처리 (truncation 사용).
        BigBird는 4096 토큰까지 처리 가능하므로 대부분의 논문 초록은 한 번에 처리됨.
        """
        if not texts:
            return []

        all_results: List[str] = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # 직접 배치 처리 (truncation으로 긴 텍스트도 처리)
            summaries = self._summarize_batch_raw(batch_texts)
            all_results.extend(summaries)
            
            if (i // batch_size + 1) % 5 == 0 or i + batch_size >= len(texts):
                logger.info(f"[Summarizer] Batch progress: {min(i + batch_size, len(texts))}/{len(texts)} processed")

        return all_results


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

