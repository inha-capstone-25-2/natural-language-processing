# app/nlp/refiner.py

from __future__ import annotations
from typing import Optional, List

import os
import re
from openai import OpenAI


class LocalKoreanRefiner:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_new_tokens: int = 350,
        api_key: Optional[str] = None,
    ) -> None:
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다. "
                "export/open 설정 후 다시 실행해주세요."
            )

        self.client = OpenAI(api_key=api_key)

        print(f"[INFO] Using OpenAI model for refiner: {self.model_name}")

 
    def _build_prompt_both(
        self,
        summary_en_raw: str,
        summary_ko_raw: str,
        keywords_en: Optional[List[str]],
    ) -> str:
        keywords_en = keywords_en or []
        keywords_en_str = ", ".join(keywords_en[:5])
        prompt = f"""
너는 컴퓨터 과학(cs) 분야 영문 논문의 **영어·한국어 요약을 동시에 다듬는 전문 연구 요약가**이다.

아래 입력을 참고하여,
1) 영어 요약(summary_en_raw)을 자연스럽고 간결한 학술 영어로 다듬고,
2) 한국어 초벌 요약(summary_ko_raw)을 자연스럽고 정확한 한국어 논문 요약으로 다듬어라.

[SUMMARY_EN_RAW]
{summary_en_raw}

[SUMMARY_KO_RAW]
{summary_ko_raw}

[KEYWORDS_EN]
{keywords_en_str}

---

## 도메인 용어 번역/표기 규칙 (한국어/영어 공통)

- support vector machine → 한국어: 서포트 벡터 머신(SVM) / 영어: support vector machine (SVM)
- SVM → 서포트 벡터 머신(SVM) / SVM
- perceptron → 퍼셉트론 / perceptron
- neural network / neural networks → 신경망 / neural network(s)
- deep learning → 딥러닝 / deep learning
- machine learning → 머신러닝 / machine learning
- ordinal regression → 순서 회귀(ordinal regression)
- regression → 회귀 / regression
- classification → 분류 / classification
- pattern matching → 패턴 매칭 / pattern matching
- nondestructive testing → 비파괴 검사 / nondestructive testing
- acoustic emission → 음향 방출(acoustic emission)
- triangulation → 삼각측량 / triangulation
- NP-hard → NP-어려운(NP-hard) 문제 / NP-hard problem
- pseudocausality → 의사인과성(pseudocausality)

가능한 한 위 용어들을 그대로 사용하고,
- 한국어에서는 처음 등장 시 한/영 병기,
- 영어에서는 자연스러운 학술 영어 표현을 사용하라.

---

## 임무(Task)

1. SUMMARY_EN_RAW를 기반으로, 의미와 사실을 보존하면서 자연스러운 학술 영어 요약으로 다듬어라.
2. SUMMARY_KO_RAW를 같은 의미가 되도록, 자연스럽고 간결한 한국어 논문 요약체(~다, ~이다)로 다듬어라.
3. KEYWORDS_EN에 포함된 핵심 개념은 가능하면 두 요약에 반영하되, 문맥에 어울리게 자연스럽게 포함하라.
4. 영어와 한국어 요역은 최대 2-3문장 이내로 간결하게 작성하라.
## 규칙(Rules)

- 새로운 내용, 근거 없는 추론, 과장은 추가하지 마라.
- 저자 이름, 이메일, 주소, 코드 조각 등 메타데이터는 포함하지 마라.
- 문장은 중간에 끊기지 않고 완전한 문장으로 끝나야 한다.
- 'Human:', 'Assistant:' 등의 대화 형식 문구는 출력하지 마라.
- '�'와 같은 깨진 문자는 출력하지 마라.

## 출력 형식(Output)

다음 형식 **그대로** 출력하라:

[SUMMARY_EN_REFINED]
(여기에 다듬어진 영어 요약)

[SUMMARY_KO_REFINED]
(여기에 다듬어진 한국어 요약)

- 마크다운 헤더(###), 리스트(-, *), 코드블록(```), JSON 등은 사용하지 마라.
"""
        return prompt
    
    def _extract_answer(self, full_text: str) -> str:
        text = full_text.strip()

        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        text = re.sub(r"^\s*\[SUMMARY_KO\]\s*", "", text)

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        text = " ".join(lines).strip()

        while "  " in text:
            text = text.replace("  ", " ")

        return text.strip()
    
    def _parse_both(self, full_text: str) -> tuple[str, str]:
        text = full_text.strip()

        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        en_part = ""
        ko_part = ""
        current = None
        buf = []

        for line in text.splitlines():
            line_stripped = line.strip()
            if line_stripped.startswith("[SUMMARY_EN_REFINED]"):
                if current == "ko":
                    ko_part = "\n".join(buf).strip()
                buf = []
                current = "en"
                continue
            if line_stripped.startswith("[SUMMARY_KO_REFINED]"):
                if current == "en":
                    en_part = "\n".join(buf).strip()
                buf = []
                current = "ko"
                continue

            if current in ("en", "ko") and line_stripped:
                buf.append(line_stripped)

        if current == "en":
            en_part = "\n".join(buf).strip()
        elif current == "ko":
            ko_part = "\n".join(buf).strip()

        def _compact(s: str) -> str:
            lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
            s = " ".join(lines)
            while "  " in s:
                s = s.replace("  ", " ")
            return s.strip()

        return _compact(en_part), _compact(ko_part)
    
    def refine_both(
        self,
        summary_en_raw: str,
        summary_ko_raw: str,
        keywords_en: Optional[List[str]] = None,
    ) -> tuple[str, str]:
        keywords_en = keywords_en or []
        summary_en_raw = (summary_en_raw or "").strip()
        summary_ko_raw = (summary_ko_raw or "").strip()

        if not summary_en_raw and not summary_ko_raw:
            return "", ""

        prompt = self._build_prompt_both(summary_en_raw, summary_ko_raw, keywords_en)

        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
            max_tokens=self.max_new_tokens,
            frequency_penalty=0.3,
            presence_penalty=0.0,
        )

        raw_text = res.choices[0].message.content or ""
        refined_en, refined_ko = self._parse_both(raw_text)
        return refined_en, refined_ko
    
    def refine(
        self,
        summary_en: str,
        summary_ko_raw: str,
        keywords_en: Optional[List[str]] = None,
    ) -> str:
        keywords_en = keywords_en or []
        summary_en = (summary_en or "").strip()
        summary_ko_raw = (summary_ko_raw or "").strip()

        if not summary_ko_raw and not summary_en:
            return ""

        prompt = self._build_prompt(summary_en, summary_ko_raw, keywords_en)

        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,       
            max_tokens=self.max_new_tokens,
            frequency_penalty=0.3,   
            presence_penalty=0.0,
        )

        raw_text = res.choices[0].message.content or ""
        refined = self._extract_answer(raw_text)
        return refined

    def refine_batch(
        self,
        summaries_en: List[str],
        summaries_ko_raw: List[str],
        keywords_en_list: Optional[List[Optional[List[str]]]] = None,
    ) -> List[str]:
        assert len(summaries_en) == len(summaries_ko_raw)
        if keywords_en_list is not None:
            assert len(keywords_en_list) == len(summaries_en)

        results: List[str] = []
        for idx, (en, ko) in enumerate(zip(summaries_en, summaries_ko_raw)):
            kw = keywords_en_list[idx] if keywords_en_list is not None else None
            refined = self.refine(en, ko, keywords_en=kw)
            results.append(refined)
        return results
