# ArgueFlow — MLOps-пайплайн для классификации аргументов

## Содержание

1. [Описание проекта](#описание-проекта)
2. [Инструкции по установке](#инструкции-по-установке)
   - [Предварительные требования](#предварительные-требования)
   - [Установка зависимостей](#установка-зависимостей)
   - [Настройка DVC (Google Drive)](#настройка-dvc-google-drive)
3. [Данные](#данные)
   - [Описание](#описание)
   - [Формат данных](#формат-данных)
4. [Обучение модели](#обучение-модели)
   - [Train](#train)
   - [Кастомизация параметров](#кастомизация-параметров)
5. [Архитектура проекта](#архитектура-проекта)
6. [Технические детали](#технические-детали)
   - [Используемые технологии](#используемые-технологии)
7. [Дополнительно (разработка)](#дополнительно-разработка)
   - [pytest](#pytest)
   - [Проверка качества кода](#проверка-качества-кода)
   - [Sphinx-документация](#sphynx-документация-кода)
   - [Добавление новых зависимостей](#добавление-новых-зависимостей)
   - [Использование CLI](#использование)
8. [TODO](#todo)

---

## Описание проекта

**ArgueFlow** — это MLOps-пайплайн для задачи классификации аргументов в эссе
школьников, построенный по мотивам соревнования
[Feedback Prize - Predicting Effective Arguments (Kaggle)](https://www.kaggle.com/competitions/feedback-prize-effectiveness).

Цель проекта — определить эффективность отдельных фрагментов аргументированного
текста. Каждый фрагмент (дискурс) должен быть отнесён к одной из категорий:

- **Effective** – убедительный и хорошо структурированный аргумент
- **Adequate** – допустимый, но менее убедительный аргумент
- **Ineffective** – слабый или неаргументированный фрагмент

Модель обучается на размеченных данных, содержащих фрагменты эссе и их типы
(введение, утверждение, доказательство, контраргумент и т.д.).

Модель основана на подходе **классификации токенов**, аналогичном решению одного
из победителей соревнования
([5-е место](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347369)).

Документация проекта доступна по адресу: \
[vsweatherfield.github.io/argueflow](https://vsweatherfield.github.io/argueflow)

## Инструкции по установке

### Предварительные требования

- python 3.11 или выше
- conda (miniconda) для управления окружениями

### Установка зависимостей

```bash
# Клонирование репозитория
git clone https://github.com/VSWeatherfield/argueflow.git
cd argueflow

# Создание и активация виртуального окружения
conda create -n argueflow python=3.12
conda activate argueflow

# Установка зависимостей
pip install uv
uv pip install -e .[dev] --system

# Установка pre-commit hooks
pre-commit install
```

### Настройка DVC (Google Drive)

При первом запуске train данные автоматически загрузятся с Google Drive. Если вы
хотите загрузить их заранее:

```bash
mkdir -p logs
argueflow download_data
```

💡 Для скачивания данных через DVC с Google Drive потребуется приватный ключ
gdrive-creds.json, путь к которому указывается в поле
gdrive_service_account_json_file_path в .dvc/config. Напишите мне в telegram —
@vsweatherfield, чтобы его получить.

Альтернативно - данные доступны по
[ссылке](https://drive.google.com/file/d/1l9c_M5okqe16SX_OTwmbbNSJXN_jgaRZ/view?usp=sharing).
После распоковки архива, данные необходимо переместить в директорию
**data/raw**.

## Данные

### Описание

Входные данные — эссе школьников 6–12 классов США, размеченные вручную по
основным элементам аргументированного письма:

- **Lead** – введение, привлекающее внимание читателя
- **Position** – мнение или вывод по главному вопросу
- **Claim** – утверждение в поддержку позиции
- **Counterclaim** – утверждение, опровергающее другое утверждение
- **Rebuttal** – опровержение контраргумента
- **Evidence** – примеры и идеи, подтверждающие аргументы
- **Concluding Statement** – заключительное заявление

### Формат данных

Обучающая выборка (`train.csv`):

- `discourse_id` – ID элемента текста
- `essay_id` – ID эссе
- `discourse_text` – текст элемента
- `discourse_type` – тип элемента
- `discourse_effectiveness` – целевой рейтинг качества

Пример данных:

```csv
discourse_id,essay_id,discourse_text,discourse_type,discourse_effectiveness
0013cc385424,007ACE74B050,"Hi, i'm Isaac, i'm going to be writing about..",Lead,Adequate
9704a709b505,007ACE74B050,"On my perspective, I think that the face is..",Position,Adequate
c22adee811b6,007ACE74B050,"I think that the face is a natural landform..",Claim,Adequate
```

**Общий размер датасета** – 4195 файлов, 20.64 MB.

## Обучение модели

### Train

Запуск обучения с дефолтными параметрами:

```bash
argueflow train
```

Альтернативно, можно указать путь к конфигу:

```bash
argueflow train --cfg_path=configs --cfg_name=config
```

### Кастомизация параметров

Основные параметры обучения находятся в configs/train/default.yaml:

```yaml
batch_size: 64
num_workers: 4
nepochs: 10
lr: 1e-5
```

## Инференс модели

### Infer

Инференс лучшей модели:

```bash
argueflow infer
```

## Архитектура проекта

```bash
> tree -L 3 -I docs -I argueflow.egg-info/ -I logs .

.
├── README.md                       # описание проекта, более подробно - https://vsweatherfield.github.io/argueflow/
├── argueflow                       # основной python модуль
│   ├── __init__.py
│   ├── cli
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   └── commands.py             # сборный файл со всеми коммандами
│   ├── data
│   │   ├── argue_module.py         # файл с lightning data module
│   │   ├── data_preparation.py     # препоцессинг данных для подачи в датасет
│   │   └── tokenized_dataset.py    # dataset class, прогон данных через токенизатор
│   ├── infer
│   │   └── infer.py                # инференс модели
│   ├── models
│   │   ├── fp2_base_model.py       # класс модели
│   │   └── fp2_lightning_module.py # файл с model lightning module
│   ├── train
│   │   └── train.py                # обучение модели
│   └── utils
│       ├── dvc_utils.py            # загрузка данных (альтернативно - dvc pull)
│       ├── logging_utils.py        # настройка python logger
│       └── tokenizer.py            # загрузка токенайзера
├── configs
│   ├── config.yaml                 # сборный файл конфигов
│   ├── data
│   │   └── default_paths.yaml      # конфигурация путей к данным
│   ├── logger
│   │   └── csv.yaml                # конфигурация питоновского логгера
│   ├── model
│   │   ├── deberta.yaml            # параметры деберты
│   │   └── distilbert.yaml         # параметры берта
│   ├── python_logging
│   │   └── setup.yaml              # настройка логгера
│   ├── train
│   │   └── default.yaml            # параметры обучения
│   └── trainer
│       └── default.yaml            # конфигурация обучения (pt lightning)
├── data
│   ├── processed.dvc               # dvc файл необработанных данных
│   └── raw.dvc                     # dvc файл подготовленных данных
├── notebooks
├── pyproject.toml                  # конфигурационный файл зависимостей проекта
├── models
│   └── best_model.ckpt.dvc         # dvc файл чекпоинта модели
├── tests
│   ├── test_cli.py                 # тест вызова корректных команд
│   └── test_download_data.py       # тест загрузки данных в случае их отсутствия
└── uv.lock
```

## Технические детали

### Используемые технологии

ml стек:

- pytorch lightning

конфигуринг и менеджмент экспериментов:

- hydra
- mlflow

менеджмент данных:

- dvc: версионирование данных с интеграцией через fire + hydra

инструменты качества кода:

- uv
- pre-commit: проверка кода с помощью black, isort, flake8, prettier

## Дополнительно (разработка)

### pytest:

```bash
pytest tests/

# с покрытием кода
pytest tests/ --cov=argueflow --cov-report=xml -v
```

### Проверка качества кода

```bash
# разом все проверки
pre-commit run -a

# или по отдельности
black ./
isort ./
flake8 argueflow/
```

### Sphinx-документация

Доступна по [ссылке](https://vsweatherfield.github.io/argueflow). Иначе, можно
пересобрать как:

```bash
cd docs/
make html
```

### Добавление новых зависимостей

Зависимости в **pyproject.toml** - скорее декларативные, поэтому лучше -
добаввить зависимость в файл и запустить

```bash
uv pip install -e .[dev] --system # аргумент --system специфичен для conda окружения
```

### Использование CLI

```bash
> argueflow

NAME
    argueflow

SYNOPSIS
    argueflow COMMAND

COMMANDS
    COMMAND is one of the following:

     download_data
       Download data using DVC

     infer
       Run inference

     prepare_data
       Format raw data into prepared one

     train
       Train the model
```

## TODO

Планирую далее доделать проект в течение ближайшего времени. Основные задачи:

- Провести серию экспериментов, сравненить модели
- Разработать интерфейс для инференса с помощью FastAPI
- Интегрировать Triton Inference Server для ускоренного инференса
- Настроить полноценное развёртывание с использованием Docker Compose
