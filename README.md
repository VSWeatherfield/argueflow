# ArgueFlow - MLOps pipeline for argument classification

## Постановка задачи

Выбранное соревнование: **Feedback Prize - Predicting Effective Arguments**
(Kaggle).

Задача: оценивать аргументы школьников 6–12 классов по трем категориям:

- **Effective**
- **Adequate**
- **Ineffective**

Цель — помочь учащимся в подготовке к SAT и другим письменным экзаменам,
поступлению в университеты, а также развить навыки письма.

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

## Валидация

Так как обучение занимает значительное время, **не использую K-Fold
Cross-Validation**. Вместо этого:

- **80/20** разделение данных для стандартного пайплайна обучения.
- Если потребуется оценить качество модели в соревновании, обучу её на **всей
  выборке** и сравню результат по public/private score.

## Метрики

Оценка решений основана на **multi-class logarithmic loss** (log loss).

## Моделирование

### Бейзлайн

- **RoBERTa-base** + линейный head без доработок.
- Альтернативы: **DistilBERT, DeBERTa-v3-small**.

### Основная модель

- Подход на основе **классификации токенов (Token Classification)**.
- **DeBERTa-v3-large + GRU + multi-dropout head**.
- **AWP (Adversarial Weight Perturbation)** после 2-х эпох.

Код обучения и инференса реплицируемого решения -
[ссылка](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347369).
\
Sphynx документация кода - [ссылка](https://vsweatherfield.github.io/argueflow/).

## Внедрение

Первичное видение модели:

- Пакет с кодом для **обучения, валидации и инференса**.
- **CLI-интерфейс** для запуска предсказаний.
- **Docker-контейнер** для удобного развертывания.
