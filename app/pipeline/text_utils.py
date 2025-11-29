# app/pipeline/text_utils.py

import re
from typing import List, Dict


def clean_summary_en(text: str) -> str:
    if not text:
        return ""

    text = text.replace("<n>", " ")

    text = re.sub(r"@[a-zA-Z0-9_]+", " ", text)   
    text = re.sub(r"\\[a-zA-Z]+", " ", text)     
    text = re.sub(r"\$+", " ", text)           

    text = re.sub(r"#\s*\d+", " ", text)        

    text = re.sub(r"epsf\.tex[^)]*\)", " ", text, flags=re.IGNORECASE)

    text = re.sub(r"\([^)]*width[^)]*\)", " ", text, flags=re.IGNORECASE)

    if "section" in text.lower():
        text = re.sub(r"\b[Ss]ection\b", " ", text)

    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("*", " ")
    text = text.replace(",", " ")
    text = re.sub(r"\(\s*\)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, max_chars: int = 4000) -> List[str]:
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
   
    abstract = doc.get("abstract") or ""
    body = doc.get("text") or ""
    raw_text = (abstract + "\n" + body).strip()
    return raw_text
