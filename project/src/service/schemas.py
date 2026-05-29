"""Pydantic-схемы для API кредитного скоринга.

Все числовые признаки опциональны — клиент может отправить неполную заявку;
пайплайн препроцессинга сам импьютирует отсутствующие значения.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ApplicationFeatures(BaseModel):
    """Входные признаки одной кредитной заявки."""

    model_config = ConfigDict(extra="ignore")

    # Числовые признаки
    AMT_INCOME_TOTAL: Optional[float] = Field(None, ge=0, description="Годовой доход клиента.")
    AMT_CREDIT: Optional[float] = Field(None, ge=0, description="Запрашиваемая сумма кредита.")
    AMT_ANNUITY: Optional[float] = Field(None, ge=0, description="Аннуитетный платёж по кредиту.")
    AMT_GOODS_PRICE: Optional[float] = Field(None, ge=0, description="Цена приобретаемого товара.")
    DAYS_BIRTH: Optional[float] = Field(
        None, le=0, description="Возраст в днях, отрицательное число (например, -12000 ≈ 33 года)."
    )
    DAYS_EMPLOYED: Optional[float] = Field(
        None, description="Стаж в днях до подачи заявки; отрицательное число или sentinel 365243."
    )
    DAYS_REGISTRATION: Optional[float] = Field(None, le=0)
    DAYS_ID_PUBLISH: Optional[float] = Field(None, le=0)
    CNT_CHILDREN: Optional[int] = Field(None, ge=0)
    CNT_FAM_MEMBERS: Optional[float] = Field(None, ge=0)
    REGION_POPULATION_RELATIVE: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1)

    # Категориальные признаки
    NAME_CONTRACT_TYPE: Optional[str] = None
    CODE_GENDER: Optional[str] = None
    FLAG_OWN_CAR: Optional[str] = None
    FLAG_OWN_REALTY: Optional[str] = None
    NAME_INCOME_TYPE: Optional[str] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    NAME_HOUSING_TYPE: Optional[str] = None
    OCCUPATION_TYPE: Optional[str] = None


class PredictResponse(BaseModel):
    default_probability: float = Field(..., ge=0, le=1)
    decision: Literal["approve", "reject"]
    threshold: float = Field(..., ge=0, le=1)
    model_name: str
    model_trained_at: Optional[str] = None


class BatchPredictRequest(BaseModel):
    items: List[ApplicationFeatures]


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: Optional[str] = None
    model_trained_at: Optional[str] = None
