# build_cs_parallel_by_translate_resume.py


from pathlib import Path
from tqdm.auto import tqdm
import torch

from app.nlp.translator import TranslatorM2M100


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def main(
    in_path: Path = DATA_DIR / "cs_train_en.txt",
    out_path: Path = DATA_DIR / "cs_train_ko.txt",
    max_lines: int | None = None,
    batch_size: int = 64,
    write_buffer: int = 2000,
    resume: bool = True,
):
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found.")

    out_path.parent.mkdir(parents=True, exist_ok=True)


    already_done = 0
    if resume and out_path.exists():
        with out_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    already_done += 1

        print(f"[RESUME] Found {already_done:,} translated lines already.")

    translator = TranslatorM2M100(
        model_name="facebook/m2m100_418M",
        device="cuda",
        use_fp16=True,
    )
    use_amp = (translator.device == "cuda" and translator.use_fp16)

    lines_en = []
    with in_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if line:
                lines_en.append(line)

    total = len(lines_en)
    print(f"[INFO] Total EN lines: {total:,}")

    if already_done >= total:
        print("[INFO] All lines already translated. Nothing to do.")
        return

    start_idx = already_done
    print(f"[INFO] Resume from index: {start_idx:,} (0-based)")

    file_mode = "a" if already_done > 0 else "w"

    buffer = []

    with out_path.open(file_mode, encoding="utf-8") as fout:
        for i in tqdm(range(start_idx, total, batch_size), desc="Batch Translating"):
            batch = lines_en[i:i + batch_size]

            if use_amp:
                with torch.cuda.amp.autocast():
                    ko_list = translator.translate_batch(batch)
            else:
                ko_list = translator.translate_batch(batch)

            for s in ko_list:
                buffer.append(s.replace("\n", " ").strip())

            if len(buffer) >= write_buffer:
                fout.write("\n".join(buffer) + "\n")
                buffer.clear()

        if buffer:
            fout.write("\n".join(buffer) + "\n")

    print(f"[DONE] Saved translated file → {out_path}")


if __name__ == "__main__":
    main()
