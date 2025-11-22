# app/mongodb.py
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

# ---- .env 로드 ----
ROOT_DIR = Path(__file__).resolve().parents[2]  # RSRS 폴더
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "arxiv")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "papers")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
papers_col = db[MONGO_COLLECTION_NAME]


def get_unprocessed_papers(limit: int | None = None):
    """
    아직 요약 안 된 논문 가져오기.
    summary_en 필드가 없는 문서 + 값이 빈 경우도 포함.
    """
    query = {
        "$or": [
            {"summary_en": {"$exists": False}},
            {"summary_en": None},
            {"summary_en": ""},
        ]
    }
    if limit is not None:
        return papers_col.find(query).limit(limit)
    return papers_col.find(query)


# def update_summary(arxiv_id: str, summary_en: str, summary_ko: str | None = None):
#     """
#     요약/번역 결과를 해당 논문에 업데이트.
#     """
#     update_doc = {"summary_en": summary_en}
#     if summary_ko is not None:
#         update_doc["summary_ko"] = summary_ko

#     papers_col.update_one(
#         {"id": arxiv_id},
#         {"$set": update_doc},
#     )


def debug_print():
    """
    현재 코드가 어떤 서버/DB/컬렉션/문서 수를 보고 있는지 출력
    """
    print("=== MongoDB Debug Info ===")
    print("MONGO_URI         :", MONGO_URI)
    print("MONGO_DB_NAME     :", MONGO_DB_NAME)
    print("MONGO_COLLECTION  :", MONGO_COLLECTION_NAME)
    print("All DBs           :", client.list_database_names())
    print("Collections in DB :", db.list_collection_names())
    total = papers_col.count_documents({})
    no_sum = papers_col.count_documents(
        {
            "$or": [
                {"summary_en": {"$exists": False}},
                {"summary_en": None},
                {"summary_en": ""},
            ]
        }
    )
    print("Total docs        :", total)
    print("Unprocessed docs  :", no_sum)
    print("==========================")
