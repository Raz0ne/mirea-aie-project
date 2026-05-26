# Данные проекта

Используется датасет **Home Credit Default Risk** (Kaggle), таблица `application_train.csv`.

## Где должен лежать CSV

```
project/data/raw/application_train.csv
```

Папка `data/raw/` находится в `.gitignore`, в репозитории CSV не хранится.

## Вариант A — реальный Kaggle-датасет

1. Скачать через Kaggle CLI или вручную: <https://www.kaggle.com/c/home-credit-default-risk/data>.
2. Распаковать `application_train.csv` в `data/raw/`.
3. Запустить обучение:

   ```bash
   python -m src.models.train
   ```

С Kaggle CLI:

```bash
kaggle competitions download -c home-credit-default-risk -f application_train.csv -p data/raw/
unzip -o data/raw/application_train.csv.zip -d data/raw/
```

## Вариант B — синтетический fallback (по умолчанию)

Если CSV в `data/raw/` нет, при запуске `python -m src.models.train` модуль `src/data/dataset.py` автоматически сгенерирует синтетический Home Credit-подобный датасет (`data/raw/application_train.csv`, 50 000 строк) через `src/data/synthetic.py`:

- те же названия колонок и типы, что в реальном датасете;
- реалистичные диапазоны (доход — лог-нормальный, days_* — отрицательные целые, EXT_SOURCE_* — beta-распределения с пропусками);
- positive rate ≈ 6-8%, как в оригинале;
- целевая переменная сгенерирована из логистической функции по EXT_SOURCE_*, отношениям credit/income и annuity/income — модели будут учить реальный сигнал, но никаких персональных данных в файле нет.

Чтобы сгенерировать датасет вручную:

```bash
python -m src.data.synthetic --out data/raw/application_train.csv --rows 50000
```

## Что нельзя класть сюда

- реальные персональные данные третьих лиц;
- конфиденциальные/служебные выгрузки;
- большие raw-дампы.

Любые такие файлы должны оставаться вне репозитория.
