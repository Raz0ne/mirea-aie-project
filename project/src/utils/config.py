"""Утилиты конфигурации проекта.

Два слоя конфигурации:

1. `TrainConfig` — YAML-конфиг обучения (пути, фичи, гиперпараметры моделей).
2. `ServiceSettings` — конфиг сервиса на pydantic-settings, читается из окружения.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"


@dataclass
class DataConfig:
    raw_path: str
    use_synthetic_if_missing: bool
    synthetic_n_rows: int
    test_size: float
    val_size: float
    random_state: int


@dataclass
class FeaturesConfig:
    numeric: List[str]
    categorical: List[str]


@dataclass
class ModelSpec:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    data: DataConfig
    features: FeaturesConfig
    target: str
    baseline: ModelSpec
    improved: ModelSpec

    @classmethod
    def load(cls, path: str | Path = DEFAULT_TRAIN_CONFIG) -> "TrainConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            data=DataConfig(**raw["data"]),
            features=FeaturesConfig(**raw["features"]),
            target=raw["target"],
            baseline=ModelSpec(**raw["models"]["baseline"]),
            improved=ModelSpec(**raw["models"]["improved"]),
        )


class ServiceSettings(BaseSettings):
    """Runtime-настройки сервиса, читаются из окружения (поддерживается .env)."""

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    model_path: str = "artifacts/model.joblib"
    decision_threshold: float = 0.5
    random_seed: int = 42

    model_config = SettingsConfigDict(
        env_file=os.environ.get("ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def resolved_model_path(self) -> Path:
        p = Path(self.model_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p
