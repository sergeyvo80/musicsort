---
name: musicsort
description: Run and manage the musicsort sorting pipeline — sort the music/ library into flat sorted/ folders, dry-run previews, wipe confirmation, and interpreting musicsort_log.csv. Use when the user asks to sort/resort music, run musicsort.py, or change where/how sorted output is produced.
---

# musicsort — запуск сортировки

## Команды

Всегда через проектный интерпретатор: `.venv/bin/python musicsort.py ...` (полный контекст — `README.md` в корне проекта).

```bash
# просто отсортировать (dest=sorted/, зачистка подтверждается автоматически)
.venv/bin/python musicsort.py --classifier clap-head --workers 8 --yes

# прикидка без изменений файлов
.venv/bin/python musicsort.py --classifier clap-head --dry-run --workers 8

# переместить вместо копирования
.venv/bin/python musicsort.py --classifier clap-head --workers 8 --yes --move
```

## Что важно знать до запуска

- **Времена на 672 файлах (8 workers)**: DSP-анализ `--duration 60` ~5–7 мин; CLAP-эмбеддинги из тёплого `.clap_cache.pkl` — мгновенно, холодные (переезд библиотеки сбрасывает кэш по путям) — ~10–15 мин + первый чекпоинт ~2GB.
- **Долгие прогоны** запускай в background (`background=true`), прогресс смотри tail'ом выходного файла, не поллингом.
- **Зачистка**: перед записью `sorted/` очищается полностью; с `--yes` без вопросов. Никогда не запускай реальную (не dry-run) сортировку с `--dest` равным или содержащим `music/` — скрипт сам откажет, но не полагайся только на это.
- **Плоская раскладка**: `dest/<жанр>/` и `dest/mood_<настроение>/` на одном уровне; файл копируется в КАЖДЫЙ свой жанр + свою mood-папку. Ожидай ~3.5 placements на файл (672 файла ≈ 2400 placements ≈ 33GB).
- **`--move`** с мульти-размещением: копии во все папки кроме последней, move в последнюю.
- **Лог**: `sorted/musicsort_log.csv` (`source_path, genre, genres, mood, dest_path, status`; dest_path — пути через `;`). Проверять результат: `status==ok` должно быть == числу файлов.
- **Остановка**: kill по `pkill -f "musicsort.py"` безопасен ДО фазы записи; после начала wipe/copy прерывание оставит только что очищенный dest.

## Типичные запросы пользователя

- «отсортируй/перезапусти сортировку» → команда выше с `--yes` (если пользователь явно не хочет вопросов), в фоне.
- «что будет, если пересортировать» → `--dry-run`, покажи top папок из breakdown.
- «запиши в другую папку» → `--dest <путь>`; напомни о зачистке этой папки.
