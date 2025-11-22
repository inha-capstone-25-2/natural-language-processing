# app/pipeline/text_utils.py

import re
from typing import List, Dict


def clean_summary_en(text: str) -> str:
    """
    BigBird-Pegasus 요약 결과에서 LaTeX/TeX 잡음 제거용 클리너.
    - @xcite, @xmath0 같은 매크로
    - #1, #2 자리표시자
    - epsf.tex, width=... 포함된 괄호
    - section 토큰
    등을 최대한 제거해서 깔끔한 영어 요약만 남긴다.
    """
    if not text:
        return ""

    # 0) 줄바꿈 토큰 치환
    text = text.replace("<n>", " ")

    # 1) LaTeX 매크로 / 수식 기호 제거
    text = re.sub(r"@[a-zA-Z0-9_]+", " ", text)   # @xcite, @xmath0 등
    text = re.sub(r"\\[a-zA-Z]+", " ", text)      # \cite, \ref 등
    text = re.sub(r"\$+", " ", text)              # $, $$

    # 2) #1, #2, #3 같은 자리표시자 제거
    text = re.sub(r"#\s*\d+", " ", text)          # "# 1", "#2" 등 싹 지우기

    # 3) epsf.tex부터 괄호 끝까지 날리기 (그림 관련 LaTeX)
    text = re.sub(r"epsf\.tex[^)]*\)", " ", text, flags=re.IGNORECASE)

    # 4) width= ... 이 들어간 괄호 전체 제거 (혹시 남은 것들 대비)
    text = re.sub(r"\([^)]*width[^)]*\)", " ", text, flags=re.IGNORECASE)

    # 5) 'section' 있으면 전부 삭제
    if "section" in text.lower():
        text = re.sub(r"\b[Ss]ection\b", " ", text)

    # 6) 남은 대괄호/별표/쉼표 정리
    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("*", " ")
    text = text.replace(",", " ")
    text = re.sub(r"\(\s*\)", " ", text)
    # 7) 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    """
    아주 단순한 청크 분할:
    - 문단 기준으로 잘라서 max_chars 넘지 않게 묶어줌
    """
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


def build_raw_text(doc: Dict) -> str:
    """
    MongoDB 문서에서 abstract + body(text)를 하나로 합쳐서
    요약 입력으로 사용할 raw 텍스트 생성.
    """
    abstract = doc.get("abstract") or ""
    body = doc.get("text") or ""
    raw_text = (abstract + "\n" + body).strip()
    return raw_text

def postprocess_ko_summary(text: str) -> str:
    """
    M2M100 한국어 번역 결과를 논문 스타일에 맞게 가볍게 교정.
    - 자주 나오는 오역/어색한 표현만 치환.
    """
    if not text:
        return ""

    repl = {
        "방해적인 양자 크로모디나믹": "섭동적 양자색역학(pQCD)",
        "완전히 차별적인 계산": "완전 미분 계산",
        "포톤": "광자",
        "쿨라이더": "충돌기",
        "퀘르크": "쿼크",
        "콘덴서트": "콘덴세이트",
        "상대에 의해 결정된다": "상호작용에 의해 결정된다",
        "hadrons": "하드론",
        "quark gluons": "쿼크와 글루온",
        "Qgp": "QGP",
    }

    for src, tgt in repl.items():
        text = text.replace(src, tgt)

    # 공백/마침표 정리
    text = text.replace("  ", " ").strip()
    return text