"""
GPU 요약 서버 - FastAPI 메인 파일.

BigBird-Pegasus + M2M100 모델을 사용한 논문 요약 및 번역 API를 제공합니다.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from app.nlp.summarizer import get_summarizer
from app.nlp.translator import get_translator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --- Schemas ---


class SummarizeBatchRequest(BaseModel):
    """배치 요약 요청 스키마"""
    texts: List[str]


class SummaryResult(BaseModel):
    """개별 요약 결과"""
    summary_en: str
    summary_ko: str


class SummarizeBatchResponse(BaseModel):
    """배치 요약 응답 스키마"""
    results: List[SummaryResult]


class HealthResponse(BaseModel):
    """헬스 체크 응답 스키마"""
    status: str
    model_loaded: bool
    device: str
    gpu_memory_allocated: str = "N/A"
    gpu_memory_reserved: str = "N/A"


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 및 종료 시 모델 로딩/언로딩"""
    logger.info("[Startup] Loading models...")
    try:
        summarizer = get_summarizer()
        logger.info(f"[Startup] BigBird loaded on {summarizer.device}")
        
        translator = get_translator()
        logger.info(f"[Startup] M2M100 loaded on {translator.device}")
    except Exception as e:
        logger.error(f"[Startup] Failed to load models: {e}")
        raise

    yield

    logger.info("[Shutdown] Cleaning up...")


app = FastAPI(
    title="GPU Summary Server",
    description="BigBird-Pegasus + M2M100 기반 논문 요약 및 번역 API",
    version="2.0.0",
    lifespan=lifespan,
)


# --- API Endpoints ---


@app.post("/summarize/batch", response_model=SummarizeBatchResponse)
async def summarize_batch(request: SummarizeBatchRequest):
    """
    배치 텍스트 요약 및 번역.
    GPU 배치 추론을 사용하여 처리량 극대화.

    Args:
        request: 요약할 텍스트 리스트

    Returns:
        요약 결과 리스트 (영문 + 한글)

    Raises:
        HTTPException: 요약 생성 실패 시
    """
    import time
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts는 비어있을 수 없습니다")

    total_start = time.time()
    batch_size = len(request.texts)
    logger.info(f"[API] ========== 배치 요약 요청 시작 ==========")
    logger.info(f"[API] 요청 텍스트 수: {batch_size}")

    try:
        summarizer = get_summarizer()
        translator = get_translator()
        
        # 1. 배치 요약 (GPU 병렬 처리)
        summarize_start = time.time()
        if batch_size == 1:
            # 단일 텍스트는 기존 방식
            summaries_en = [summarizer.summarize(request.texts[0])]
        else:
            # 여러 텍스트는 배치 처리
            summaries_en = summarizer.summarize_batch(request.texts)
        summarize_time = time.time() - summarize_start
        logger.info(f"[API] 배치 요약 완료 | 소요: {summarize_time:.2f}s | 평균: {summarize_time/batch_size:.2f}s/건")
        
        # 2. 배치 번역 (GPU 병렬 처리)
        translate_start = time.time()
        summaries_ko = translator.translate_batch(summaries_en)
        translate_time = time.time() - translate_start
        logger.info(f"[API] 배치 번역 완료 | 소요: {translate_time:.2f}s | 평균: {translate_time/batch_size:.2f}s/건")
        
        # 결과 생성
        results = []
        for i, (en, ko) in enumerate(zip(summaries_en, summaries_ko)):
            results.append(SummaryResult(summary_en=en, summary_ko=ko))
            if i < 3:  # 처음 3개만 로깅
                en_preview = en[:60] + "..." if len(en) > 60 else en
                ko_preview = ko[:60] + "..." if len(ko) > 60 else ko
                logger.info(f"[API] [{i+1}] EN: {en_preview}")
                logger.info(f"[API] [{i+1}] KO: {ko_preview}")

        total_time = time.time() - total_start
        logger.info(f"[API] ========== 배치 요약 완료 ==========")
        logger.info(f"[API] 총 처리: {len(results)}건 | 총 소요: {total_time:.2f}s | 평균: {total_time/len(results):.2f}s/건")
        return SummarizeBatchResponse(results=results)

    except Exception as e:
        logger.error(f"[API] Batch summarize error: {e}")
        raise HTTPException(status_code=500, detail=f"요약 생성 실패: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    헬스 체크 및 GPU 메모리 정보.

    Returns:
        서버 및 모델 상태
    """
    try:
        summarizer = get_summarizer()
        device = summarizer.device

        gpu_memory_allocated = "N/A"
        gpu_memory_reserved = "N/A"

        if device == "cuda":
            try:
                import torch
                gpu_memory_allocated = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
                gpu_memory_reserved = f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            except Exception as e:
                logger.warning(f"[Health] Failed to get GPU memory info: {e}")

        return HealthResponse(
            status="ok",
            model_loaded=True,
            device=device,
            gpu_memory_allocated=gpu_memory_allocated,
            gpu_memory_reserved=gpu_memory_reserved,
        )

    except Exception as e:
        logger.error(f"[Health] Health check failed: {e}")
        return HealthResponse(status="error", model_loaded=False, device="unknown")


@app.get("/")
def root():
    """루트 엔드포인트"""
    return {"message": "GPU Summary Server", "version": "2.0.0"}
