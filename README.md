# musicsort — сортировка музыки по жанру и настроению

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
