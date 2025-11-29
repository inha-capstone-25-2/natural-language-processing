# build_cs_en_from_mongo_fast.py
from pathlib import Path
from app.mongodb import papers_col


def build_cs_raw_en(limit: int | None = None, batch_size=10000):
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    out_path = data_dir / "cs_raw_en.txt"

    query = {
        "categories": {"$regex": r"^cs"}  
    }
    

    cursor = papers_col.find(query, {"title": 1, "abstract": 1})

    cursor = cursor.batch_size(5000)

    total = papers_col.count_documents(query)
    print(f"[INFO] Total cs.* papers in DB: {total}")

    if limit is not None:
        cursor = cursor.limit(limit)
        print(f"[INFO] Limiting to first {limit} documents")

    n_written = 0
    buffer = []

    with out_path.open("w", encoding="utf-8") as f:
        for idx, doc in enumerate(cursor):
            title = doc.get("title")
            abstract = doc.get("abstract")

            if title:
                title = title.strip()
            if abstract:
                abstract = abstract.strip()

            if not title and not abstract:
                continue

            if title and abstract:
                text = f"{title} {abstract}"
            else:
                text = title or abstract

            buffer.append(text.replace("\n", " "))

            if len(buffer) >= batch_size:
                f.write("\n".join(buffer) + "\n")
                buffer.clear()

            n_written += 1

        if buffer:
            f.write("\n".join(buffer) + "\n")

    print(f"[DONE] Wrote {n_written} lines to {out_path}")


if __name__ == "__main__":
    build_cs_raw_en(limit=None)
