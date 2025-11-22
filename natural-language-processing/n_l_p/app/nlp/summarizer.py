# app/nlp/summarizer.py

from typing import Optional, List
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def _basic_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SummarizerBigBirdPegasus:

    def __init__(
        self,
        model_name: str = "google/bigbird-pegasus-large-arxiv",
        device: Optional[str] = None,
        max_input_length: int = 4096,
        max_output_length: int = 256,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        print(f"[INFO] Loading BigBird-Pegasus on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

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

        summary = self.tokenizer.decode(
            summary_ids[0], skip_special_tokens=True
        )
        return _basic_clean(summary)

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
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

        # 1단계: 청크별 요약
        chunk_summaries = []
        for i, ch in enumerate(chunks):
            print(f"[DEBUG] summarizing chunk {i+1}/{len(chunks)} (len={len(ch)})")
            s = self._summarize_once(ch)
            if s:
                chunk_summaries.append(s)

        if not chunk_summaries:
            return ""

        # 2단계: 청크 요약들을 다시 한 번 요약
        joined = " ".join(chunk_summaries)
        final_summary = self._summarize_once(joined)
        return final_summary
