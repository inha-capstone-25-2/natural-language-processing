# auto_translate_finetune_corpus.py

from pathlib import Path
from app.nlp.translator import TranslatorM2M100  
import time

DATA_DIR = Path("data")
EN_PATH = DATA_DIR / "cs_train_en.txt"
KO_PATH = DATA_DIR / "cs_train_ko.txt"

def main():
    assert EN_PATH.exists(), f"{EN_PATH} not found. 먼저 build_cs_finetune_corpus.py를 실행해줘."

    translator = TranslatorM2M100(
        model_name="facebook/m2m100_418M"
    )

    lines = EN_PATH.read_text(encoding="utf-8").splitlines()
    print(f"[INFO] Total sentences: {len(lines)}")

    ko_lines = []
    start = time.time()
    for i, en in enumerate(lines, start=1):
        ko = translator.translate(en)
        ko_lines.append(ko)
        if i % 50 == 0:
            elapsed = time.time() - start
            print(f"[PROGRESS] {i}/{len(lines)} done, elapsed {elapsed/60:.1f} min")

    with KO_PATH.open("w", encoding="utf-8") as f:
        for ko in ko_lines:
            f.write(ko.strip() + "\n")

    print(f"[DONE] Saved ko translations to {KO_PATH.resolve()}")


if __name__ == "__main__":
    main()
