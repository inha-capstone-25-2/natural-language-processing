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
        max_new_tokens: int = 300,
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

 
    def _build_prompt(
        self,
        summary_en: str,
        summary_ko_raw: str,
        keywords_en: Optional[List[str]],
    ) -> str:
        keywords_en_str = ", ".join(keywords_en or [])

        prompt = f"""
너는 컴퓨터 과학(cs) 분야 영문 논문의 한국어 요약을 자연스럽고 정확하게 다듬는 **전문 연구 요약가**이다.

아래 입력을 참고하여, 사실을 보존하면서 자연스럽고 간결한 한국어 논문 요약을 작성하라.

[SUMMARY_EN]
{summary_en}

[SUMMARY_KO_RAW]
{summary_ko_raw}

[KEYWORDS_EN]
{keywords_en_str}

---

## 도메인 용어 번역 규칙 (반드시 우선 적용)

- support vector machine → **서포트 벡터 머신(SVM)**
- SVM → **서포트 벡터 머신(SVM)**
- perceptron → **퍼셉트론**
- neural network / neural networks → **신경망**
- deep learning → **딥러닝**
- machine learning → **머신러닝**
- ordinal regression → **순서 회귀(ordinal regression)**
- regression → **회귀**
- classification → **분류**
- pattern matching → **패턴 매칭**
- nondestructive testing / nondestructive test → **비파괴 검사**
- acoustic emission → **음향 방출(acoustic emission)**
- triangulation → **삼각측량**
- NP-hard → **NP-어려운(NP-hard)** 문제
- pseudocausality → **의사인과성(pseudocausality)**

가능한 한 위 용어들을 그대로 사용하고, 처음 등장할 때는 영어 약어/원어를 괄호로 병기하라.

---

## 임무(Task)
- SUMMARY_EN을 사실 기준으로 삼고,
- SUMMARY_KO_RAW의 어색한 번역, 비문, 반복을 수정하여
- 한국어 논문 요약체(~다, ~이다)로 자연스럽게 정리하라.

## 규칙(Rules)
1. SUMMARY_EN의 의미와 사실을 절대 왜곡하지 마라.
2. SUMMARY_KO_RAW의 오역, 문법 오류, 문장 반복은 모두 교정하라.
3. KEYWORDS_EN에 포함된 핵심 개념은 가능한 한 요약에 반영하되, 문맥에 어울리게 자연스럽게 포함하라.
4. 새로운 내용, 근거 없는 추론, 과장된 해석을 추가하지 마라.
5. 저자 이름, 이메일, 주소, PACS 코드, OCIS 코드, LaTeX 조각 등 메타데이터는 포함하지 마라.
6. 'Human:', 'Assistant:', 'User:' 같은 대화 형식 문구는 출력하지 마라.
7. 영어 문장을 길게 복사하지 말고, 필요한 약어(CNN, SVM 등)만 한국어 문장 안에서 사용하라.
8. '�'와 같은 깨진 문자는 출력하지 마라.
9. 문장은 중간에 끊기지 않고, 완전한 문장으로 끝나야 한다.
10. 요약은 한 단락 또는 두 단락 이내로 작성하라.

## 출력 형식(Output)
- 출력에는 **다듬어진 한국어 요약 문단만** 포함한다.
- '[SUMMARY_KO]' 같은 레이블이나 추가 설명은 붙이지 마라.
- 마크다운 헤더(###), 리스트(-, *), 코드블록(```), JSON 등은 사용하지 마라.

이제 위 규칙을 모두 지켜서, 다듬어진 한국어 논문 요약만 작성하라.
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
