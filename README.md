# musicsort — сортировка музыки по жанру и настроению

> **Языки / Languages:** [🇷🇺 Русский](#ru) · [🇬🇧 English](#en)

<a id="ru"></a>

Утилита раскладывает аудиотеку `music/` в `sorted/`.

**Ключевые свойства текущей версии:**

* **Плоская структура**: папки жанров и папки настроений лежат рядом на одном уровне — `sorted/techno/`, `sorted/mood_driving/` и т.д. Папки настроений несут префикс `mood_` (`MOOD_FOLDER_PREFIX`), чтобы не путались с жанрами.
* **Мульти-размещение**: один файл лежит в нескольких папках сразу — по копии в каждом его жанре (CLAP даёт мультилейбл) плюс в папке настроения.
* **Зачистка перед записью**: перед каждой реальной сортировкой `sorted/` очищается — скрипт печатает предупреждение со списком удаляемого и просит подтвердить (`y`); `--yes` пропускает вопрос; ответ не `y` ничего не трогает. Затереть источник нельзя: если dest совпадает с `music/` или содержит его, запуск прервётся с ошибкой.
* Жанр — из ID3-тега или CLAP-классификатора (каскад `tag > CLAP > эвристика`), настроение — из DSP-анализа (arousal/valence по перцентилям коллекции).
* Словарь жанров открытый: свои жанры вписываются в `labels_template.csv`, модель переобучается одной командой (см. ниже).

## Установка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt              # librosa, numpy, mutagen
pip install -r requirements-clap.txt        # torch, laion-clap, sklearn, joblib — только для CLAP
```

## Просто отсортировать

```bash
.venv/bin/python musicsort.py --classifier clap-head --workers 8 --yes
```

Что происходит: DSP-анализ 672 файлов ~5 минут (8 воркеров) → CLAP-классификация мгновенно из кэша `.clap_cache.pkl` → `sorted/` зачистка + запись (разложение по плоским папкам, лог `sorted/musicsort_log.csv`).

* Без `--yes` скрипт покажет, что будет удалено, и спросит подтверждение.
* Сначала посмотреть прикидку, ничего не трогая: добавить `--dry-run` (напечатает план placements и разбивку по папкам).
* Dry-run печатает breakdown по **плоским папкам** уже с учётом мультиразмещения.
* Копирование по умолчанию (`--move` — перемещение). Из-за мультиразмещения `--move` копирует файл во все папки кроме последней, куда перемещает; источник освобождается.
* В CSV-логе (`source_path, genre, genres, mood, dest_path, status`) все назначения файла идут через `;` в `dest_path`, колонка `mood` — без префикса.
* Смена dest — `--dest <папка>` (всё то же применяется к нему).

## Режимы жанрового классификатора (`--classifier`)

| Режим | Чем предсказывает | Когда использовать |
|---|---|---|
| `heuristic` (дефолт) | DSP-правила по tempo/flatness/percussive_ratio | быстро, без модели, грубые buckets (`electronic`, `ambient`,...) |
| `clap-zeroshot` | эмбеддинг трека vs текстовые промпты жанров | без обучения; пороги: top <0.2 → `other`, z-score >1.0 (`genre_clap.py`) |
| `clap-head` | CLAP + обученная голова из твоих меток | точнее всех; требует `clap_genre_head.joblib` (см. переобучение) |

Мультилейбл: трек получает все жанры с вероятностью ≥0.5, первый — «основной» в логе. Хэш-кэш эмбеддингов ведётся по `(path, mtime, size)`: переезд библиотеки обнуляет совпадения и вызывает одноразовый пересчёт (~10–15 мин), после чего кэш снова мгновенный.

## Классификация до и после обучения

Приоритет жанра всегда одинаков и не зависит от того, обучена ли голова: **ID3-тег → CLAP → эвристика**. Разница — то, что находится на шаге CLAP.

### До обучения (файл `clap_genre_head.joblib` отсутствует)

Первый прогон не требует обучения — работают два пути:

**1. `--classifier heuristic` (дефолт — модель вообще не нужна).** `guess_genre_from_features` применяет фиксированные пороги к DSP-фичам каждого трека:

| Условие | Жанр |
|---|---|
| tempo ≥ 118 **и** spectral flatness > 0.015 | `electronic` |
| tempo ≤ 110 **и** percussive_ratio < 0.3 **и** flatness < 0.015 | `acoustic_classical` |
| percussive_ratio > 0.35 **и** 75 ≤ tempo ≤ 160 **и** zcr > 0.09 **и** flatness > 0.02 | `rock_metal` |
| tempo ≤ 100 **и** percussive_ratio > 0.3 **и** centroid < 2200 | `hiphop_beat` |
| rms < 0.05 **и** tempo < 100 | `ambient` |
| ничего не подошло | `other` |

Правила «на глазок», не модель (комментарий в коде честно это оговаривает: four-on-the-floor электроника из-за синтезаторовых пэдов HPSS считает «гармоничной», поэтому flatness — лучший признак электроники, а не percussive_ratio).

**2. `--classifier clap-zeroshot` (без обучения, но с моделью CLAP).**

1. Аудио → эмбеддинг 512d (LAION-CLAP HTSAT-tiny), кэшируется в `.clap_cache.pkl`.
2. CLAP сравнивает его по косинусной близости с текстовыми эмбеддингами промптов жанров (`DEFAULT_PROMPTS` в `genre_clap.py`): `electronic` → *"electronic dance music, house, techno or trance"*, `hiphop_beat` → *"hip hop or rap music"* и т.д.
3. Двухчастный фильтр вместо фиксированного порога: если верхняя близость < **0.2** → треку ставится `other` (ни один жанр словаря не подходит); остальные жанры включаются в мультилейбл, только если их z-score относительно средней похожести этого трека ≥ **1.0** — это отсекает "не худший, но и не стоящий" варианты.

**3. `--classifier clap-head` без файла головы** — корректно завершится ошибкой `No trained head at ... - run train_genre_head.py first, or use --classifier clap-zeroshot` и вернёт код 1. Это и есть точка, где нужно обучение из раздела «Переобучение».

Настроение во всех этих режимах считается одинаково и независимо от жанровой ветки (см. «Как считается настроение»).

### После обучения

**`--classifier clap-head`:**

1. Эмбеддинг трека (те же 512d, тот же кэш) подаётся в голову.
2. Голова — **OneVsRest LogisticRegression**: по одному бинарному классификатору на жанр; каждый выдаёт вероятность «трек относится к жанру X».
3. Жанры с вероятностью ≥ **0.5** (`HEAD_PROBABILITY_THRESHOLD`) становятся аудио-мультилейблом трека; если ни один не дотянул — берётся argmax (самая вероятная) как основной жанр.
4. Дальше всё как всегда: каскад (теги перекрывают CLAP), плоские папки по каждому жанру + `mood_*`, CSV-лог.

Ключевое отличие режимов: zero-shot сравнивает эмбеддинг с *текстовыми описаниями* жанров, head — с *границами, выученными на твоих примерах*. Отсюда и качество: head понимает твои специфичные классы (`kolbasa`, `trance_classic`), zero-shot — только то, что умеет описывать текстом.

### Где хранится дообученная модель

* **`clap_genre_head.joblib` в корне проекта** — дефолт для `--out` в `train_genre_head.py` и для `--clap-head` в `musicsort.py`. Переобучение **перезаписывает файл целиком** — голова каждый раз учится с нуля, инкрементального «дообучения поверх» нет.
* Внутри joblib — словарь `{"classifier": OneVsRestClassifier(...), "binarizer": MultiLabelBinarizer()}`; список жанров хранится в `binarizer.classes_` (открытый словарь из CSV — сколько жанров вметили, столько классов).
* Положение можно поменять: `--out путь/файл` при обучении и `--clap-head путь/файл` при сортировке.
* Эмбеддинги лежат отдельно в `.clap_cache.pkl` и при переобучении **не пересчитываются** — переобучать голову можно сколько угодно часто и дёшево.

## Разметка (`labels_template.csv`)

CSV с колонками `path,genre`; жанр можно оставить пустым — тогда его можно дозаполнить.

### Генерация нового шаблона

```bash
.venv/bin/python generate_label_template.py --source music --out labels_template.csv --count 200 --seed 42
# пиши жанры в колонку genre, мультилейбл через ; — напр. house;ambient
```

### Автозаполнение пустых строк предсказаниями классификатора

```bash
.venv/bin/python generate_label_template.py --fill-from-log sorted/musicsort_log.csv
# Filled 156 blank genre cell(s) from .../musicsort_log.csv
```

Правила:

* пустые строки заполняются полным мультилейблом классификатора (`house;kolbasa;progressive;techno`) — каждую запись можно потом править/убирать;
* заполненные тобой строки **не перезаписываются никогда**, дозаполнять можно между любыми прогонами;
* пути из старого расположения библиотеки автоматически ремапятся на текущее (по хвосту пути / совпадению имени файла и папки), так что CSV живёт даже после переезда проекта.

## Переобучение под свои данные (пошагово)

**Шаг 1. Правка жанров.** Открой `labels_template.csv` и правь колонку `genre`: мультилейбл через `;`, свои жанры в любом виде (`hardstyle`, `russian_rock`). Сама правка ничего не переобучает.

**Шаг 2. Переобучение головы.**

```bash
.venv/bin/python train_genre_head.py --labels-csv labels_template.csv --cache .clap_cache.pkl --out clap_genre_head.joblib
```

* голова (OneVsRest LogisticRegression поверх CLAP-эмбеддингов 512d) обучается **с нуля** по текущему содержимому CSV и перезаписывает `clap_genre_head.joblib`; это занимает секунды — эмбеддинги из кэша;
* минимум 20 размеченных треков (`Need at least 20`), жанры с <5 примерами дадут warning и будут предсказывать слабо; для стабильного качества — ~20–30 треков на жанр;
* альтернативный источник меток — ID3-теги (`--source music` без `--labels-csv`), но в этой библиотеке usable тегов 0.

**Шаг 3. Перезапуск сортировки с новой моделью.**

```bash
.venv/bin/python musicsort.py --classifier clap-head --workers 8 --yes
```

`sorted/` зачистится и запишется заново по новым предсказаниям. CLAP мгновенен, DSP ~5 мин.

Цикл целиком: правка CSV → переобучение (секунды) → перезапуск (~5 мин).

## Как считается настроение

Всегда DSP, для всей коллекции сразу: `arousal = 0.5*percentile(tempo) + 0.5*percentile(rms)`, `valence = 0.5*percentile(centroid) + 0.5*(1−percentile(percussive_ratio))` → сетка 3×3 `MOOD_GRID` → 9 настроений: melancholic, contemplative, peaceful, mysterious, atmospheric, uplifting, tension_building, driving, euphoric. Пороги относительные — «высокая энергия» значит «высокая относительно твоей коллекции».

## Результат последнего прогона (актуальная версия)

```
music/ (672 файла) → sorted/ (плоско): 37 жанровых папок + 9 mood_* папок
672/672 ok, 2367 folder placements, ~33GB (мульти-копии)
Топ: techno 217, trance 209, mood_atmospheric 208, trance_classic 173, house 152
Лог: sorted/musicsort_log.csv
```

Исторические прогоны до редизайна писали в `organized/` вложенной раскладкой `<жанр>/<настроение>`.

## Примечания

* `Warning: Xing stream size off by more than 1%` — битый mp3-header, игнорируется.
* Первое использование CLAP скачивает чекпоинт ~2GB (+ roberta-base).
* `.clap_cache.pkl`, `clap_genre_head.joblib`, `labels_template.csv`, `music/`, `organized/`, `sorted/` — в `.gitignore`.
* Скрипты: `musicsort.py` (сортировка), `genre_clap.py` (эмбеддинги/zero-shot/голова), `train_genre_head.py` (обучение), `generate_label_template.py` (шаблон/автозаполнение).

* * *

<a id="en"></a>

# musicsort — sort your music library by genre and mood

> **Languages / Языки:** [🇷🇺 Русский](#ru) · [🇬🇧 English](#en)

The utility organizes the audio library from `music/` into `sorted/`.

**Key properties of the current version:**

* **Flat structure**: genre folders and mood folders sit side by side on the same level — `sorted/techno/`, `sorted/mood_driving/`, etc. Mood folders carry a `mood_` prefix (`MOOD_FOLDER_PREFIX`) so they never collide with genres.
* **Multi-placement**: one file lives in several folders at once — a copy in each of its genres (CLAP is multi-label) plus the mood folder.
* **Wipe before writing**: before every real sorting run `sorted/` is cleared — the script prints a warning listing what will be removed and asks to confirm (`y`); `--yes` skips the prompt; anything other than `y` touches nothing. You cannot nuke the source: if dest equals `music/` or contains it, the run aborts with an error.
* Genre comes from an ID3 tag or the CLAP classifier (cascade `tag > CLAP > heuristic`), mood comes from DSP analysis (arousal/valence percentiles of the collection).
* The genre vocabulary is open: your genres go into `labels_template.csv`, and the model retrains with a single command (see below).

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt              # librosa, numpy, mutagen
pip install -r requirements-clap.txt        # torch, laion-clap, sklearn, joblib — CLAP only
```

## Just sort it

```bash
.venv/bin/python musicsort.py --classifier clap-head --workers 8 --yes
```

What happens: DSP analysis of 672 files in ~5 minutes (8 workers) → CLAP classification instantly from the `.clap_cache.pkl` cache → wipe + write of `sorted/` (flat folder layout, log at `sorted/musicsort_log.csv`).

* Without `--yes` the script shows what will be deleted and asks for confirmation.
* To preview first, add `--dry-run` (prints the placement plan and per-folder breakdown).
* The dry-run breakdown counts **flat folders** and already accounts for multi-placement.
* Copying is the default (`--move` to move instead). Because of multi-placement, `--move` copies the file to all folders except the last one, which receives the move; the source is freed.
* In the CSV log (`source_path, genre, genres, mood, dest_path, status`) all of a file's destinations are joined with `;` in `dest_path`; the `mood` column carries no prefix.
* Change dest with `--dest <folder>` (everything above applies to it).

## Genre classifier modes (`--classifier`)

| Mode | What it predicts with | When to use |
|---|---|---|
| `heuristic` (default) | DSP rules over tempo/flatness/percussive_ratio | fast, no model, coarse buckets (`electronic`, `ambient`, ...) |
| `clap-zeroshot` | track embedding vs genre text prompts | no training needed; thresholds: top <0.2 → `other`, z-score >1.0 (`genre_clap.py`) |
| `clap-head` | CLAP + a head trained on your labels | the most accurate; requires `clap_genre_head.joblib` (see retraining) |

Multi-label: a track gets every genre with probability ≥0.5; the first one is the "primary" in the log. The embedding cache is keyed by `(path, mtime, size)`: moving the library invalidates key matches and triggers a one-time recompute (~10–15 min), after which the cache is instant again.

## Classification before and after training

The genre priority is always the same and does not depend on whether a head is trained: **ID3 tag → CLAP → heuristic**. The difference is what happens at the CLAP step.

### Before training (no `clap_genre_head.joblib` file)

The first run requires no training — two paths work:

**1. `--classifier heuristic` (default — no model needed at all).** `guess_genre_from_features` applies fixed thresholds to each track's DSP features:

| Condition | Genre |
|---|---|
| tempo ≥ 118 **and** spectral flatness > 0.015 | `electronic` |
| tempo ≤ 110 **and** percussive_ratio < 0.3 **and** flatness < 0.015 | `acoustic_classical` |
| percussive_ratio > 0.35 **and** 75 ≤ tempo ≤ 160 **and** zcr > 0.09 **and** flatness > 0.02 | `rock_metal` |
| tempo ≤ 100 **and** percussive_ratio > 0.3 **and** centroid < 2200 | `hiphop_beat` |
| rms < 0.05 **and** tempo < 100 | `ambient` |
| nothing fits | `other` |

These are rules of thumb, not a model (the code comment says this honestly: four-on-the-floor electronics uses sustained synth pads, so HPSS scores it "harmonic" — flatness is the better electronic signal, not percussive_ratio).

**2. `--classifier clap-zeroshot` (no training, but uses the CLAP model).**

1. Audio → 512d embedding (LAION-CLAP HTSAT-tiny), cached in `.clap_cache.pkl`.
2. CLAP compares it by cosine similarity against text embeddings of the genre prompts (`DEFAULT_PROMPTS` in `genre_clap.py`): `electronic` → *"electronic dance music, house, techno or trance"*, `hiphop_beat` → *"hip hop or rap music"*, etc.
3. A two-part filter instead of a fixed threshold: if the top similarity < **0.2** → the track is labeled `other` (no vocabulary genre fits); other genres only join the multi-label if their z-score relative to that track's mean similarity ≥ **1.0** — this rejects options that are "not the worst but not standing out either".

**3. `--classifier clap-head` without a head file** — fails cleanly with `No trained head at ... - run train_genre_head.py first, or use --classifier clap-zeroshot` and exit code 1. This is exactly the point where the "Retraining" section comes in.

Mood is computed the same way in all of these modes and independently of the genre branch (see "How mood is computed").

### After training

**`--classifier clap-head`:**

1. The track's embedding (the same 512d, the same cache) is fed into the head.
2. The head is a **OneVsRest LogisticRegression**: one binary classifier per genre; each outputs a probability "this track belongs to genre X".
3. Genres with probability ≥ **0.5** (`HEAD_PROBABILITY_THRESHOLD`) become the track's multi-label; if none reaches it, the argmax (most probable) genre is used as primary.
4. From there the usual cascade applies (tags override CLAP), flat folders per genre + `mood_*`, CSV log.

The key difference between the modes: zero-shot compares the embedding with *text descriptions* of genres, the head compares it with *decision boundaries learned from your examples*. That's where quality comes from: the head understands your specific classes (`kolbasa`, `trance_classic`), zero-shot only knows what it can describe in text.

### Where the retrained model is stored

* **`clap_genre_head.joblib` in the project root** — the default for `--out` in `train_genre_head.py` and for `--clap-head` in `musicsort.py`. Retraining **overwrites the file entirely** — the head is trained from scratch every time; there is no incremental "fine-tuning on top".
* Inside the joblib there is a dict `{"classifier": OneVsRestClassifier(...), "binarizer": MultiLabelBinarizer()}`; the genre list lives in `binarizer.classes_` (an open vocabulary from the CSV — as many genres as you labeled, that many classes).
* The location is configurable: `--out path/file` when training and `--clap-head path/file` when sorting.
* Embeddings live separately in `.clap_cache.pkl` and are **not recomputed** on retraining — you can retrain the head as often as you like, cheaply.

## Labeling (`labels_template.csv`)

A CSV with `path,genre` columns; the genre cell may be left empty — it can be filled in later.

### Generating a new template

```bash
.venv/bin/python generate_label_template.py --source music --out labels_template.csv --count 200 --seed 42
# write genres in the genre column, multi-label with ';' — e.g. house;ambient
```

### Autofilling blank rows from classifier predictions

```bash
.venv/bin/python generate_label_template.py --fill-from-log sorted/musicsort_log.csv
# Filled 156 blank genre cell(s) from .../musicsort_log.csv
```

Rules:

* blank rows are filled with the classifier's full multi-label (`house;kolbasa;progressive;techno`) — every entry can be edited or removed later;
* rows you filled yourself are **never overwritten**, and you can top up at any time between runs;
* paths from the library's previous location are automatically remapped to the current one (by path tail / folder-and-filename match), so the CSV survives project relocation.

## Retraining on your own data (step by step)

**Step 1. Edit genres.** Open `labels_template.csv` and edit the `genre` column: multi-label with `;`, your own genres in any form (`hardstyle`, `russian_rock`). Editing alone does not retrain anything.

**Step 2. Retrain the head.**

```bash
.venv/bin/python train_genre_head.py --labels-csv labels_template.csv --cache .clap_cache.pkl --out clap_genre_head.joblib
```

* the head (OneVsRest LogisticRegression over 512d CLAP embeddings) is trained **from scratch** on the current CSV contents and overwrites `clap_genre_head.joblib`; this takes seconds — embeddings come from the cache;
* at least 20 labeled tracks are required (`Need at least 20`); genres with <5 examples get a warning and will predict poorly; for stable quality aim for ~20–30 tracks per genre;
* an alternative label source is ID3 tags (`--source music` without `--labels-csv`), but this library has 0 usable tags.

**Step 3. Restart sorting with the new model.**

```bash
.venv/bin/python musicsort.py --classifier clap-head --workers 8 --yes
```

`sorted/` is wiped and rewritten with the new predictions. CLAP is instant, DSP takes ~5 min.

The full loop: edit the CSV → retrain (seconds) → restart (~5 min).

## How mood is computed

Always DSP, over the whole collection at once: `arousal = 0.5*percentile(tempo) + 0.5*percentile(rms)`, `valence = 0.5*percentile(centroid) + 0.5*(1−percentile(percussive_ratio))` → the 3×3 `MOOD_GRID` → 9 moods: melancholic, contemplative, peaceful, mysterious, atmospheric, uplifting, tension_building, driving, euphoric. Thresholds are relative — "high energy" means "high relative to your own collection".

## Latest run result (current version)

```
music/ (672 files) → sorted/ (flat): 37 genre folders + 9 mood_* folders
672/672 ok, 2367 folder placements, ~33GB (multi-copies)
Top: techno 217, trance 209, mood_atmospheric 208, trance_classic 173, house 152
Log: sorted/musicsort_log.csv
```

Historical runs before the redesign wrote to `organized/` with the nested `<genre>/<mood>` layout.

## Notes

* `Warning: Xing stream size off by more than 1%` — a broken mp3 header, ignored.
* First CLAP use downloads a ~2GB checkpoint (+ roberta-base).
* `.clap_cache.pkl`, `clap_genre_head.joblib`, `labels_template.csv`, `music/`, `organized/`, `sorted/` are in `.gitignore`.
* Scripts: `musicsort.py` (sorting), `genre_clap.py` (embeddings/zero-shot/head), `train_genre_head.py` (training), `generate_label_template.py` (template/autofill).
