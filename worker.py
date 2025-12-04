import os
import logging
from celery import Celery
from pymongo import MongoClient, UpdateOne
from bson.objectid import ObjectId
from dotenv import load_dotenv

from app.nlp.summarizer import get_summarizer
from app.nlp.translator import get_translator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# MongoDB 설정
MONGO_URI = os.getenv("MONGO_URI", "mongodb://rsrs-root:KIQu3jebjHNhTEE6mm5tgj2oNjYr7J805k2JLbE0AVo@35.87.92.19:27017/arxiv?authSource=admin")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "arxiv")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "papers")

# Celery 설정
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("worker", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=True,
    worker_concurrency=1,  # GPU 메모리 제한으로 인해 동시성 1로 제한
    worker_prefetch_multiplier=1, # 한 번에 하나의 작업만 가져옴
)

def get_db_collection():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    return db[MONGO_COLLECTION_NAME]

@celery_app.task(name="process_paper", bind=True, max_retries=3)
def process_paper(self, paper_id: str, text: str):
    """
    논문 처리 작업: 요약 -> 번역 -> DB 업데이트
    """
    logger.info(f"[Task] Processing paper {paper_id}")
    
    try:
        # 1. 요약 (BigBird)
        summarizer = get_summarizer()
        summary_en = summarizer.summarize(text)
        logger.info(f"[Task] Summary generated for {paper_id} (len: {len(summary_en)})")
        
        # 2. 번역 (M2M100)
        translator = get_translator()
        summary_ko = translator.translate(summary_en)
        logger.info(f"[Task] Translation generated for {paper_id} (len: {len(summary_ko)})")
        
        # 3. DB 업데이트
        collection = get_db_collection()
        
        # ObjectId 변환 시도 (실패 시 문자열 그대로 사용)
        try:
            oid = ObjectId(paper_id)
        except:
            oid = paper_id
            
        result = collection.update_one(
            {"_id": oid},
            {
                "$set": {
                    "summary.en": summary_en,
                    "summary.ko": summary_ko,
                    "summary_refined": True, # 처리 완료 플래그
                    "processing_status": "completed",
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if result.matched_count == 0:
            logger.warning(f"[Task] Paper {paper_id} not found in DB")
            return {"status": "not_found", "paper_id": paper_id}
            
        logger.info(f"[Task] DB updated for {paper_id}")
        return {"status": "success", "paper_id": paper_id}

    except Exception as e:
        logger.error(f"[Task] Failed to process paper {paper_id}: {e}")
        
        # DB에 에러 상태 기록
        try:
            collection = get_db_collection()
            try:
                oid = ObjectId(paper_id)
            except:
                oid = paper_id
                
            collection.update_one(
                {"_id": oid},
                {"$set": {"processing_status": "failed", "error_message": str(e)}}
            )
        except:
            pass
            
        raise self.retry(exc=e, countdown=60) # 1분 후 재시도

from datetime import datetime
