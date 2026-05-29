"""FastAPI-приложение сервиса кредитного скоринга.

Endpoints:
    GET  /health           — liveness + статус загруженной модели
    POST /predict          — одна заявка → вероятность дефолта и решение
    POST /predict/batch    — массив заявок
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src.service.predictor import Predictor
from src.service.schemas import (
    ApplicationFeatures,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictResponse,
)
from src.utils.config import ServiceSettings
from src.utils.logging import get_logger


settings = ServiceSettings()
logger = get_logger("src.service", level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = settings.resolved_model_path()
    try:
        app.state.predictor = Predictor.load(model_path)
        logger.info(
            "Сервис готов. model=%s trained_at=%s threshold=%.3f",
            app.state.predictor.model_name,
            app.state.predictor.trained_at,
            settings.decision_threshold,
        )
    except FileNotFoundError as exc:
        logger.error("Не удалось загрузить модель: %s", exc)
        app.state.predictor = None
    yield


app = FastAPI(
    title="Credit Scoring API",
    description="Home Credit Default Risk — предсказание вероятности дефолта по заявке на кредит.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "request_id=%s method=%s path=%s status=%d duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


def _get_predictor(request: Request) -> Predictor:
    predictor: Predictor | None = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Сначала обучите её (`python -m src.models.train`).",
        )
    return predictor


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    predictor: Predictor | None = request.app.state.predictor
    return HealthResponse(
        status="ok" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        model_name=predictor.model_name if predictor else None,
        model_trained_at=predictor.trained_at if predictor else None,
    )


def _predict_one(predictor: Predictor, payload: ApplicationFeatures) -> PredictResponse:
    proba_list = predictor.predict_proba([payload.model_dump()])
    proba = proba_list[0]
    decision = predictor.decide(proba, threshold=settings.decision_threshold)
    return PredictResponse(
        default_probability=proba,
        decision=decision,
        threshold=settings.decision_threshold,
        model_name=predictor.model_name,
        model_trained_at=predictor.trained_at,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: ApplicationFeatures, request: Request):
    predictor = _get_predictor(request)
    return _predict_one(predictor, payload)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest, request: Request):
    predictor = _get_predictor(request)
    if not payload.items:
        raise HTTPException(status_code=400, detail="Поле `items` не должно быть пустым.")
    proba = predictor.predict_proba(item.model_dump() for item in payload.items)
    predictions = [
        PredictResponse(
            default_probability=p,
            decision=predictor.decide(p, threshold=settings.decision_threshold),
            threshold=settings.decision_threshold,
            model_name=predictor.model_name,
            model_trained_at=predictor.trained_at,
        )
        for p in proba
    ]
    return BatchPredictResponse(predictions=predictions)


@app.get("/")
def root():
    return {
        "service": "credit-scoring",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }
