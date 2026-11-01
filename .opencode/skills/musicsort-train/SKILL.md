---
name: musicsort-train
description: Hand-label tracks and retrain the CLAP genre head for musicsort — edit genres in labels_template.csv, fill blanks from classifier predictions, run train_genre_head.py, and restart sorting with the new head. Use when the user wants to fix genres, add custom genres, fill or regenerate labels_template.csv, or retrain the classifier.
---

# musicsort — разметка и переобучение

Пайплайн меток: `labels_template.csv` (`path,genre`, мультилейбл через `;`) → `train_genre_head.py` → `clap_genre_head.joblib` → сортировка `musicsort.py --classifier clap-head`. Полный контекст — `README.md` в корне проекта.

## Ключевой факт

Правка жанров в CSV **ничего не переобучает сама по себе**. Переобучение — всегда отдельный ручной шаг, и оно каждый раз идёт с нуля по текущему содержимому CSV (перезаписывает joblib). Эмбеддинги при этом не пересчитываются — они в `.clap_cache.pkl` по ключу `(path, mtime, size)`, так что обучение занимает секунды.

## Шаг 1 — работа с метками

```bash
# автозаполнение пустых строк предсказаниями последнего прогона
.venv/bin/python generate_label_template.py --fill-from-log sorted/musicsort_log.csv
```

Правила fill-from-log: заполняет ТОЛЬКО пустые строки; заполненные пользователем никогда не трогает; ремапит устаревшие пути (после переезда папки) на текущую библиотеку по хвосту пути / совпадению `(папка, имя файла)`.

```bash
# сгенерировать новый шаблон (перезапишет CSV!)
.venv/bin/python generate_label_template.py --source music --out labels_template.csv --count 200 --seed 42
```

Осторожно: генерация затирает существующие метки — перед ней предложи пользователю проверку/бэкап.

## Шаг 2 — обучение

```bash
.venv/bin/python train_genre_head.py --labels-csv labels_template.csv --cache .clap_cache.pkl --out clap_genre_head.joblib
```

Пороги: минимум 20 размеченных треков; классы с <5 примерами дают warning и предсказывают слабо; для стабильности ~20–30 треков на жанр. Словарь открытый — новый жанр из CSV становится классом. Обучение на ID3-тегах (`--source music` без `--labels-csv`) в этой библиотеке не вариант: usable тегов 0.

## Шаг 3 — применить новую голову

```bash
.venv/bin/python musicsort.py --classifier clap-head --workers 8 --yes
```

(sorted/ зачистится и перезапишется; CLAP мгновенно из кэша, DSP ~5 мин — подробности в README и в скиле `musicsort`.)

## Типичные запросы пользователя

- «я поправил жанры, примени» → шаг 2 затем шаг 3; напомни, что обучение пересоздаёт голову по CSV.
- «заполни лейблы сам» → шаг 1 fill-from-log, покажи сколько строк заполнено и что пользовательские не тронуты.
- «добавлю жанр X везде» → предложи grep/правку CSV, потом шаги 2–3; предупреди про <5 примеров.
