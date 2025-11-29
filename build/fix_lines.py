# fix_parallel_lines.py

from pathlib import Path

DATA_DIR = Path("data")  
EN_PATH = DATA_DIR / "cs_train_en.txt"
KO_PATH = DATA_DIR / "cs_train_ko.txt"

OUT_EN = DATA_DIR / "cs_train_en.aligned.txt"
OUT_KO = DATA_DIR / "cs_train_ko.aligned.txt"

def main():
    en_lines = EN_PATH.read_text(encoding="utf-8").splitlines()
    ko_lines = KO_PATH.read_text(encoding="utf-8").splitlines()

    print(f"[INFO] EN lines = {len(en_lines)}")
    print(f"[INFO] KO lines = {len(ko_lines)}")

    en_lines = [l.strip() for l in en_lines]
    ko_lines = [l.strip() for l in ko_lines]

    pairs = []
    for i, (e, k) in enumerate(zip(en_lines, ko_lines)):
        if e and k:
            pairs.append((e, k))

    print(f"[INFO] Non-empty aligned pairs (min(len(en), len(ko))) = {len(pairs)}")

    with OUT_EN.open("w", encoding="utf-8") as f_en, OUT_KO.open("w", encoding="utf-8") as f_ko:
        for e, k in pairs:
            f_en.write(e + "\n")
            f_ko.write(k + "\n")

    print(f"[DONE] Saved aligned files:")
    print(f"  - {OUT_EN}")
    print(f"  - {OUT_KO}")

if __name__ == "__main__":
    main()
