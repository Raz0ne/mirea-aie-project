"""Pydantic schemas for the credit-scoring API.

All numeric features are Optional so callers can submit incomplete applications;
the preprocessing pipeline imputes missing values.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ApplicationFeatures(BaseModel):
    """Input features for a single credit application."""

    model_config = ConfigDict(extra="ignore")

    # Numeric features
    AMT_INCOME_TOTAL: Optional[float] = Field(None, ge=0, description="Annual income of the client.")
    AMT_CREDIT: Optional[float] = Field(None, ge=0, description="Loan amount requested.")
    AMT_ANNUITY: Optional[float] = Field(None, ge=0, description="Loan annuity.")
    AMT_GOODS_PRICE: Optional[float] = Field(None, ge=0, description="Price of the goods.")
    DAYS_BIRTH: Optional[float] = Field(
        None, le=0, description="Age in days, negative (e.g. -12000 ≈ 33 years old)."
    )
    DAYS_EMPLOYED: Optional[float] = Field(
        None, description="Days employed before application; negative or sentinel 365243."
    )
    DAYS_REGISTRATION: Optional[float] = Field(None, le=0)
    DAYS_ID_PUBLISH: Optional[float] = Field(None, le=0)
    CNT_CHILDREN: Optional[int] = Field(None, ge=0)
    CNT_FAM_MEMBERS: Optional[float] = Field(None, ge=0)
    REGION_POPULATION_RELATIVE: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1)
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1)

    # Categorical features
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
