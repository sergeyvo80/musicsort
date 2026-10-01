# musicsort — сортировка музыки по жанру и настроению

Утилита раскладывает аудиотеку `music/` в `organized/<жанр>/<настроение>/`. Жанр — из ID3-тега или CLAP-классификатора, настроение — из DSP-анализа (arousal/valence по перцентилям коллекции).

## Установка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt              # librosa, numpy, mutagen
pip install -r requirements-clap.txt        # torch, laion-clap, sklearn, joblib — только для CLAP
```

## Обучение (`train_genre_head.py`)

Голова — `OneVsRest(LogisticRegression)` поверх CLAP-эмбеддингов (512d), мультилейбл (`;`), открытый словарь жанров. Эмбеддинги кэшируются в `.clap_cache.pkl` по `(path, mtime, size)`.

### Вариант 1 — только ID3-теги (как было запущено)

```bash
.venv/bin/python train_genre_head.py --source music --cache .clap_cache.pkl --out clap_genre_head.joblib
# Reading tags from 672 file(s)...
# 0 of 672 tracks have a usable genre tag to train from
# Need at least 20 labeled tracks — exit 1
```

В текущей библиотеке 0 usable тегов (`musicsort.py:119` `read_tag_genre` + `GENRE_TAG_ALIASES:42` + фильтр спама) — вариант нерабочий. Требуется либо разметить CSV, либо использовать `clap-zeroshot`.

### Вариант 2 — ручная разметка CSV (успешный запуск)

```bash
.venv/bin/python train_genre_head.py --labels-csv labels_template.csv --cache .clap_cache.pkl --out clap_genre_head.joblib
# 44 labeled track(s) loaded from labels_template.csv
# Genres: 90s (2), acid (1), alternative_rock (1), ambient (1), breakbeat (4), dream (2),
#   drum_and_bass (3), euro_dance (1), eurodance (3), hard_techno (1), hard_trance (1),
#   house (9), jazz (1), kolbasa (3), liquid_drum_and_bass (1), pop (1), progressive (1),
#   psy_folk (2), psybient (2), rap (2), rock (1), russian (6), russian_rock (1),
#   speed_garage (1), tech_trance (1), techno (9), trance (7), trance_classic (6)
# Extracting CLAP embeddings for the labeled tracks...  # 44/44 hits из .clap_cache.pkl, без пере-эмбеддинга
# Training classifier head on 44 embedded tracks...
# WARNING: 23 жанра <5 примеров — будут предсказывать плохо
# Trained on 44 tracks, saved head to clap_genre_head.joblib (67KB, 28 классов)
```

Источник — `labels_template.csv` (200 строк из `generate_label_template.py`, заполнено 44). Проверка: `joblib.load` → `binarizer.classes_` 28 шт., `predict_with_head` работает. Для стабильности нужно 20–30 треков на жанр.

### Подготовка CSV (если нужно расширить разметку)

```bash
.venv/bin/python generate_label_template.py --source music --out labels_template.csv --count 200 --seed 42
# пиши жанры в колонку genre, мультилейбл через ; — напр. electronic;ambient
```

## Классификация (`musicsort.py`)

Настроение всегда считается DSP: `arousal = 0.5*percentile(tempo)+0.5*percentile(rms)`, `valence = 0.5*percentile(centroid)+0.5*(1-percentile(percussive_ratio))` → 9 настроений `MOOD_GRID:69`. Жанр: `tag > CLAP > эвристика` (`classify_tracks:254`).

### Вариант 1 — dry-run эвристика (дефолт, без CLAP)

```bash
.venv/bin/python musicsort.py --classifier heuristic --dry-run --workers 4
# быстро, без модели, правило guess_genre_from_features:174
```

### Вариант 2 — dry-run clap-head, полный анализ (первый прогон, скачка модели)

```bash
.venv/bin/python musicsort.py --classifier clap-head --dry-run --workers 4
# Found 672 audio file(s) under music
# Reading tags and analyzing audio (duration=60s, workers=4)... 672/672 analyzed (~6-7 мин)
# Running CLAP genre classification (clap-head) on 672 untagged...
# Loading CLAP model (first run downloads ~2GB checkpoint + roberta-base)
#   embedding 16/628 ... 628/628  (~10-15 мин, первый раз)
# Would copy 672 file(s) into organized + разбивка genre/mood
# Таймаут 600s недостаточен — нужен ~1200s; кэш .clap_cache.pkl растёт с 95KB → 1.4MB
```

### Вариант 3 — dry-run clap-head, кэш уже полный (быстро)

```bash
.venv/bin/python musicsort.py --classifier clap-head --dry-run --workers 8 --duration 30
# Found 672 ... 672/672 analyzed (~3 мин при 30s)
# Running CLAP genre classification (clap-head) on 672 untagged...  # 672/672 hits, мгновенно
# Would copy 672 file(s) into organized
#   house/atmospheric:29, trance/atmospheric:19, hard_techno/driving:18, ...
```

### Вариант 4 — реальная сортировка (выполненный прогон)

```bash
.venv/bin/python musicsort.py --classifier clap-head --workers 8
# Found 672 audio file(s) under .../music
# Reading tags and analyzing audio (duration=60s, workers=8)... 672/672 analyzed
# Running CLAP genre classification (clap-head) on 672 untagged...  # кэш полон, ~0s
# Log written to .../organized/musicsort_log.csv
# Copied 672 file(s) into .../organized
# organized/ 9.3GB, 23 жанра: 90s, acid, alternative_rock, ambient, breakbeat, dream,
#   drum_and_bass, euro_dance, eurodance, hard_techno, hard_trance, house, jazz,
#   kolbasa, pop, psy_folk, psybient, rap, russian, techno, trance, trance_classic (+ др.)
# Лог: organized/musicsort_log.csv — source_path,genre,genres,mood,dest_path,status (все ok)
```

Копирование (`--move` для перемещения, без флага — копия). `duration` — секунд на трек для анализа, `workers` — параллельность librosa.

### Вариант 5 — zero-shot CLAP (без обучения)

```bash
.venv/bin/python musicsort.py --classifier clap-zeroshot --dry-run --workers 8
# Не требует labels_template.csv / clap_genre_head.joblib, сравнивает эмбеддинг с текстами DEFAULT_PROMPTS:34
# Пороги: top <0.2 → other, z-score >1.0 — genre_clap.py:52
```

## Результат текущего прогона

```
music/ (672 файла, 8 папок New/new2..new8) → organized/ (672 скопировано)
.clap_cache.pkl 1.4MB (672 эмбеддинга)
clap_genre_head.joblib 67KB (28 классов, обучено на 44)
```

## Примечания

* `Need at least 20` — минимум для головы; `<5` на класс — ворнинг `genre_clap.py:212`.
* `Xing stream size off` — битый mp3-header, игнорируется.
* Повторная классификация после заполнения кэша занимает только время DSP-анализа; CLAP-часть мгновенна.
