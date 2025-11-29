# clean_cs_en.py

from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


LATEX_INLINE_PATTERNS = [
    r"\$[^$]*\$",           
    r"\\\([^\)]*\\\)",      
    r"\\\[[^\]]*\\\]",          
]

LATEX_ENV_PATTERNS = [
    r"\\begin\{equation\}.*?\\end\{equation\}",
    r"\\begin\{align\}.*?\\end\{align\}",
    r"\\begin\{eqnarray\}.*?\\end\{eqnarray\}",
]

def clean_line(text: str) -> str:
    if not text:
        return ""

    for pat in LATEX_INLINE_PATTERNS + LATEX_ENV_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.DOTALL)

    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text)
    text = re.sub(r"@[a-zA-Z0-9_]+", " ", text) 

    text = text.replace("~", " ")
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("|", " ")
    text = text.replace("*", " ")
    text = text.replace("`", " ")

    text = re.sub(r"\.\s*\.+", ".", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_to_sentences(text: str) -> list[str]:

    if not text:
        return []

    sents = re.split(r'(?<=[.!?])\s+', text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents


def main(
    in_path: Path = DATA_DIR / "cs_raw_en.txt",
    out_path: Path = DATA_DIR / "cs_train_en.txt",
    min_chars: int = 30,
    max_chars: int = 400,
):


    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            n_in += 1
            line = line.strip()
            if not line:
                continue

            cleaned = clean_line(line)
            if not cleaned:
                continue

            sents = split_to_sentences(cleaned)
            for s in sents:
                if len(s) < min_chars:
                    continue
                if len(s) > max_chars:
                    continue
                fout.write(s + "\n")
                n_out += 1

    print(f"[DONE] Input lines: {n_in}")
    print(f"[DONE] Output sentences: {n_out}")
    print(f"[DONE] Saved cleaned sentences to {out_path}")


if __name__ == "__main__":
    main()
