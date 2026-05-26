# Credit Scoring — Home Credit Default Risk

Итоговый мини-проект по курсу «Инженерия Искусственного Интеллекта»: REST-сервис кредитного скоринга на базе датасета **Home Credit Default Risk** (Kaggle). По анкете заявителя сервис предсказывает вероятность дефолта и возвращает решение `approve` / `reject` по настраиваемому порогу.

---

## 1. Паспорт проекта

- **Название:** Credit Scoring Service
- **Автор:** `Разон Владислав Юрьевич`
- **Группа:** `ИНБО-07-22`
- **Контакт:** `@raz0ne`

**Краткое описание.** Сервис обучает две модели — Logistic Regression (baseline) и LightGBM (улучшенная) — на признаках заявителя из основной таблицы датасета Home Credit (`application_train.csv`). Финальная модель выбирается по ROC-AUC на отложенной валидации и упаковывается в joblib-артефакт. FastAPI-сервис принимает заявку и возвращает вероятность дефолта.

---

## 2. Структура проекта

```
project/
├── README.md                  # этот файл
├── report.md                  # отчёт по проекту
├── self-checklist.md          # чеклист самопроверки
├── requirements.txt           # зависимости
├── Dockerfile                 # сборка образа сервиса
├── docker-compose.yml         # запуск контейнера
├── .dockerignore              # игнор-листы
├── .gitignore
├── configs/
│   ├── env.example            # пример переменных окружения (копируется в .env)
│   └── model.yaml             # конфиг обучения (фичи, гиперпараметры)
├── data/
│   ├── README.md              # как готовятся данные
│   └── raw/                   # CSV (реальный из Kaggle или синтетический fallback)
├── notebooks/
│   ├── 01_eda.ipynb           # разведочный анализ
│   └── 02_models_comparison.ipynb  # сравнение baseline vs LightGBM
├── src/
│   ├── data/                  # загрузка, синтетический генератор, препроцессинг
│   ├── models/                # baseline, LightGBM, train.py, метрики
│   ├── service/               # FastAPI app, схемы, предиктор
│   └── utils/                 # конфиги, логирование
├── tests/                     # pytest: данные, модель, сервис
└── artifacts/
    ├── model.joblib           # обученная модель (создаётся train.py)
    └── model.metrics.json     # метрики baseline / improved / final test
```

---

## 3. Требования и установка

- Python 3.10+ (тестировано на 3.9 и 3.11).
- На macOS для LightGBM нужен `libomp`: `brew install libomp`.

```bash
cd project

python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt

# Конфигурация (опционально): скопировать пример и поправить значения.
cp configs/env.example .env
```

---

## 4. Как запустить проект

### 4.1. Обучение модели

```bash
python -m src.models.train
```

Что произойдёт:

1. Загружается `data/raw/application_train.csv`. Если файла нет (типичный случай в репозитории), пайплайн сгенерирует **синтетический Home Credit-подобный датасет** (50 000 строк) — это поведение включено в `configs/model.yaml` через `data.use_synthetic_if_missing: true`. Никакого ручного скачивания не требуется.
2. Стратифицированный сплит train/val/test (70/10/20).
3. Обучаются две модели: Logistic Regression и LightGBM.
4. Выбирается лучшая по ROC-AUC на val.
5. На отложенном тесте считаются финальные метрики (ROC-AUC, PR-AUC, KS, F1, log-loss, Brier).
6. Артефакт сохраняется в `artifacts/model.joblib` (+ `artifacts/model.metrics.json` для удобства чтения).

### 4.2. Запуск сервиса

```bash
python -m src.service
```

По умолчанию сервис поднимается на `http://0.0.0.0:8000`. Конфигурация (порт, путь к модели, threshold, log_level) — через переменные окружения или `.env`.

Документация Swagger UI: `http://127.0.0.1:8000/docs`.

### 4.3. Запуск через Docker

```bash
# Сначала собрать артефакт модели локально (или внутри стадии CI).
python -m src.models.train

docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api

# Или с docker-compose:
docker compose up --build
```

### 4.4. Тесты

```bash
pytest tests -v
```

Покрытие:

- генератор синтетических данных (схема + реалистичный positive rate);
- предиктор загружается, выдаёт вероятности в [0, 1], корректно ранжирует risky/safe заявки;
- сервис: `/health`, `/predict`, `/predict/batch`, валидация входных данных, корневой endpoint.

---

## 5. Endpoints сервиса

| Method | Path             | Описание                                                                      |
|--------|------------------|-------------------------------------------------------------------------------|
| GET    | `/health`        | Liveness + статус загруженной модели (имя, время обучения).                   |
| GET    | `/`              | Краткая информация о сервисе.                                                 |
| POST   | `/predict`       | Скоринг одной заявки.                                                         |
| POST   | `/predict/batch` | Скоринг массива заявок.                                                       |
| GET    | `/docs`          | Swagger UI.                                                                   |

**Пример запроса:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "AMT_INCOME_TOTAL": 180000,
    "AMT_CREDIT": 500000,
    "AMT_ANNUITY": 25000,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -3000,
    "EXT_SOURCE_1": 0.45,
    "EXT_SOURCE_2": 0.55,
    "EXT_SOURCE_3": 0.50,
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_INCOME_TYPE": "Working"
  }'
```

**Ответ:**

```json
{
  "default_probability": 0.346,
  "decision": "approve",
  "threshold": 0.5,
  "model_name": "baseline (logreg)",
  "model_trained_at": "2026-05-26T09:50:38Z"
}
```

Все поля заявки опциональны — недостающие признаки имитируются как пропуски и обрабатываются импутерами/категориальной обработкой пайплайна.

---

## 6. Данные

- **Источник:** [Home Credit Default Risk на Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data).
- **Файл, который мы используем:** `application_train.csv` — основная таблица заявок (~300k строк, ~120 колонок).
- **Подмножество признаков:** см. `configs/model.yaml` — 14 числовых и 9 категориальных колонок (демография, финансы, занятость, жильё, внешние скоры).
- **Целевая переменная:** `TARGET` ∈ {0, 1} — клиент допустил серьёзную просрочку.
- **Если реального CSV нет:** автоматически генерируется синтетический Home Credit-подобный датасет в `src/data/synthetic.py` (та же схема, реалистичные диапазоны, ~6-8% положительного класса). Это позволяет воспроизвести проект без аккаунта Kaggle.
- **Подробности и инструкция по скачиванию реального датасета:** см. [`data/README.md`](data/README.md).

В репозитории не хранятся ни большие CSV, ни конкретные обучающие данные — папка `data/raw/` находится в `.gitignore`.

---

## 7. Конфигурация и безопасность

- Параметры обучения — в `configs/model.yaml` (фичи, гиперпараметры, размеры сплитов).
- Параметры сервиса — через переменные окружения / `.env` (см. `configs/env.example`).
- Реальный `.env` исключён через `.gitignore`. Никаких реальных секретов в репозитории нет.
- Логирование: structured access log на каждый запрос (request_id, method, path, status, duration_ms).

---

## 8. Демонстрация на защите

1. Показать структуру проекта и `report.md`.
2. Запустить `python -m src.models.train` (или показать готовый артефакт + `model.metrics.json`).
3. Запустить `python -m src.service`, открыть Swagger UI, выполнить пару запросов:
   - «хороший» клиент с высокими EXT_SOURCE → низкая вероятность дефолта, `approve`;
   - «плохой» клиент с низкими EXT_SOURCE и большим кредитом относительно дохода → высокая вероятность, `reject`.
4. Показать `notebooks/01_eda.ipynb` (EDA) и `notebooks/02_models_comparison.ipynb` (сравнение моделей, ROC/PR кривые, важность признаков).
5. Прогнать `pytest tests -v`.

---

## 9. Ограничения и дальнейшая работа

- В репозитории по умолчанию обучаемся на **синтетике** (если не подложить реальный CSV). Метрики реалистичны по форме, но не сопоставимы с публичными результатами соревнования.
- Используется только `application_train.csv` — без агрегатов из `bureau.csv`, `previous_application.csv` и т.п. С ними реальная ROC-AUC поднимается до ~0.80+.
- Нет интерпретации онлайн через SHAP — артефакт хранится, можно добавить endpoint `/predict/explain`.
- Нет MLflow / DVC — артефакт версионируется по имени файла и времени обучения внутри bundle.

Возможные следующие шаги: feature engineering из вспомогательных таблиц, калибровка вероятностей, threshold tuning под бизнес-метрику (стоимость FP vs FN), Prometheus-метрики, SHAP-эндпоинт.
