# app/mongodb.py
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

# ---- .env 로드 ----
ROOT_DIR = Path(__file__).resolve().parents[2]  
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://rsrs-root:KIQu3jebjHNhTEE6mm5tgj2oNjYr7J805k2JLbE0AVo@35.87.92.19:27017/arxiv?authSource=admin")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "arxiv")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "papers")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
papers_col = db[MONGO_COLLECTION_NAME]


def get_unprocessed_papers(limit: int | None = None):
    
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

def debug_print():
   
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
