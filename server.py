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

    Args:
        request: 요약할 텍스트 리스트

    Returns:
        요약 결과 리스트 (영문 + 한글)

    Raises:
        HTTPException: 요약 생성 실패 시
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts는 비어있을 수 없습니다")

    logger.info(f"[API] Batch summarize request: {len(request.texts)} texts")

    try:
        summarizer = get_summarizer()
        translator = get_translator()
        results = []

        for i, text in enumerate(request.texts):
            try:
                # 1. 영문 요약
                summary_en = summarizer.summarize(text)
                logger.debug(f"[API] Summarized text {i+1}/{len(request.texts)}")
                
                # 2. 한글 번역
                summary_ko = translator.translate(summary_en) if summary_en else ""
                logger.debug(f"[API] Translated text {i+1}/{len(request.texts)}")
                
                results.append(SummaryResult(
                    summary_en=summary_en,
                    summary_ko=summary_ko
                ))
            except Exception as e:
                logger.error(f"[API] Failed to process text {i+1}: {e}")
                results.append(SummaryResult(summary_en="", summary_ko=""))

        logger.info(f"[API] Batch summarize completed: {len(results)} results")
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
