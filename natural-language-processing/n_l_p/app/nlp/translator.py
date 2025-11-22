# app/nlp/translator.py

import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TranslatorM2M100:

    def __init__(self, model_name: str = "facebook/m2m100_1.2B", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading M2M100 1.2B on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        self.src_lang = "en"
        self.tgt_lang = "ko"

    def _translate_one(self, sentence: str) -> str:
        sentence = sentence.strip()
        if not sentence:
            return ""

        self.tokenizer.src_lang = self.src_lang
        encoded = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        output = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang),
            num_beams=5,
            max_length=512,
            no_repeat_ngram_size=3,
        )

        ko = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return ko

    
    def _postprocess(self, text: str) -> str:
       
      
        if not text:
            return ""

        # --- 1) 긴 영어 문장 제거 ---
        text = re.sub(
            r"(?:[A-Za-z]{1,}\s+){6,}[A-Za-z]{1,}",
            " ",
            text,
        )

        # --- 2) 물리 / 수학 / QCD / 표기 관련 용어 ---
        physics_terms = {
            # QCD / QGP / SM / LHC 표기 통일
            "lhc": "LHC",
            "Lhc": "LHC",
            "qcd": "QCD",
            "Qcd": "QCD",
            "sm)": "SM)",
            "sm )": "SM )",
            "sm의": "SM의",
            "Sm의": "SM의",

            "방해적인 양자 염색 역학": "섭동적 양자색역학(pQCD)",
            "방해적인 양자 염색역학": "섭동적 양자색역학(pQCD)",
            "방해적인 양자 크로모디나믹": "섭동적 양자색역학(pQCD)",
            "방해적인 양자 크로모다이내믹스": "섭동적 양자색역학(pQCD)",

            "완전 미분 계산": "완전 미분 형태의 계산",
            "광자 하드론": "광자-하드론",
            "포톤 하드론": "광자-하드론",

            "킬라 콘데스탄": "카이럴 콘덴세이트(chiral condensate)",
            "카이럴 콘데스탄": "카이럴 콘덴세이트(chiral condensate)",
            "콘덴서트": "콘덴세이트",
            "콘데스탄": "콘덴세이트",

            "quark-글루온": "쿼크-글루온",
            "quark 글루온": "쿼크-글루온",
            "쿼크 글루온": "쿼크-글루온",
            "quark gluons": "쿼크와 글루온",
            "hadrons": "하드론",
            "Qgp": "QGP",
            "qgp": "QGP",
            "녹색 QCD": "글루온 QCD",
            "Hadron Pairs에 대한 리뷰": "하드론 쌍",
            "사진 HADRON": "광자-하드론",
            # collider / colliders 번역 통일
            "충돌자": "충돌기",
            "글루언": "글루온",
            "글루안": "글루온",
            "1 ~ 1의 일치가 있다": "일대일 대응이 존재한다",
       }

        # --- 3) Computer Science / ML / NLP / 그래프 이론 용어 ---
        cs_terms = {
            # ML/NLP 핵심 용어
            "주의 메커니즘": "어텐션 메커니즘",
            "주의 메커니즘을": "어텐션 메커니즘을",
            "주의": "어텐션",  # 대부분 attention 의미로 쓰이는 경우

            "변압기 모델": "트랜스포머 모델",
            "변압기 기반": "트랜스포머 기반",
            "변압기": "트랜스포머",

            "머리": "헤드",  # multi-head
            "기능 공간": "특징 공간",
            "기능 표현": "특징 표현",
            "기능": "특징",

            "매립": "임베딩",
            "매립된": "임베딩된",
            "삽입": "임베딩",
            "삽입된": "임베딩된",

            "교육 데이터": "학습 데이터",
            "교육 세트": "학습 세트",
            "교육 손실": "학습 손실",

            "표현 공간": "표현 공간(embedding space)",

            # 그래프 이론 / 그래프 ML
            "차트들": "그래프들",
            "차트": "그래프",
            "희귀 그래프": "희소 그래프",
            "희귀한 그래프": "희소 그래프",
            "희귀한 차트": "희소 그래프",
            "희귀 차트": "희소 그래프",

            "스파르스 그래프": "희소 그래프",
            "스파스 그래프": "희소 그래프",

            "양파티트 그래프": "이분 그래프",
            "양파티트": "이분(bipartite)",
            "양자파티트": "이분(bipartite)",

            "Graphs 완료": "완전 그래프",
            "Graph 완료": "완전 그래프",

            # IR / DB / System
            "색인": "인덱스",
            "순위 매기기": "랭킹",
            "지연": "레이턴시(latency)",
        }

        replacements = {}
        replacements.update(physics_terms)
        replacements.update(cs_terms)

        for src, tgt in replacements.items():
            if src in text:
                text = text.replace(src, tgt)

        # --- 4) 문체: ~습니다 / ~입니다 → ~다 / ~이다 로 정리 (학술체) ---
        # 문장 끝 패턴 위주로 처리
        style_map = {
            "입니다.": "이다.",
            "입니다 .": "이다.",
            "입니다 . ": "이다. ",

            "됩니다.": "된다.",
            "됩니다 .": "된다.",
            "됩니다 . ": "된다. ",

            "이루어집니다.": "이루어진다.",
            "제시됩니다.": "제시된다.",
            "비교됩니다.": "비교된다.",
            "사용됩니다.": "사용된다.",
        }
        for src, tgt in style_map.items():
            if src in text:
                text = text.replace(src, tgt)

        # 보다 일반적인 "~습니다." → "~다." 처리 (위에서 못 잡은 것들)
        text = re.sub(r"([가-힣])습니다\.", r"\1다.", text)
        text = re.sub(r"([가-힣])하였다\.", r"\1했다.", text)

        # --- 5) 괄호/마침표/공백 정리 ---
        # 빈 괄호 제거
        text = re.sub(r"\(\s*\)", " ", text)

        # 연속된 점들("..", ". .", "...") → 하나의 "."으로 만들기
        text = re.sub(r"\.\s*\.+", ".", text)   # ". ." / "..." -> "."
        text = re.sub(r"\s*\.\s*", ". ", text)  # 점 주변 공백: " . 결과" → ". 결과"

        # 여분 공백 정리
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def translate(self, text: str) -> str:
        if not text:
            return ""

        # 문장 단위로 나누기 (., ?, ! 기준)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        ko_sentences: list[str] = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            ko = self._translate_one(sent)
            ko_sentences.append(ko)

        ko_full = " ".join(ko_sentences)
        ko_full = self._postprocess(ko_full)
        return ko_full
