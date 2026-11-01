#!/usr/bin/env python3
"""Generate a CSV template for hand-labeling tracks by genre.

Picks a random sample of tracks from the library and writes a "path,genre"
CSV with the genre column left blank. Fill it in and feed it to
train_genre_head.py --labels-csv to train a CLAP classifier head.

A track can belong to more than one genre - separate them with ";"
(e.g. "electronic;ambient"). The genre vocabulary is open: you aren't
limited to the categories GENRE_TAG_ALIASES already knows about (below) -
introduce a new one anytime, it just needs a handful of labeled examples to
be learnable. Leave a row blank/delete it to skip that track - "other" isn't
a real genre CLAP can be trained to recognize, so don't use it as a label.

Categories used elsewhere in this project, as a starting point:
acoustic_classical, ambient, electronic, hiphop_beat, jazz_blues, pop,
rock_metal.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path

from musicsort import DEFAULT_EXTENSIONS, discover_files


def _name_indexes(paths) -> tuple[dict, dict]:
    """Index the library for lookup: filenames alone can repeat across
    subfolders (release-group duplicates etc.), so prefer the
    (parent_folder, filename) pair and fall back to filename only.
    """
    by_pair: dict = {}
    by_name: dict = {}
    for p in paths:
        path = Path(p)
        by_pair.setdefault((path.parent.name, path.name), path)
        by_name.setdefault(path.name, path)
    return by_pair, by_name


def locate_file(path_str: str, source: Path, by_pair: dict, by_name: dict) -> Path | None:
    """Resolve a CSV path against the current location of the library.

    Labels may have been written when the library lived elsewhere (or under
    a different folder name), so try the tail of the old path relative to
    the current source root first, then name-based fallbacks.
    """
    parts = Path(path_str).parts
    if source.name in parts:
        idx = len(parts) - 1 - parts[::-1].index(source.name)
        candidate = Path(source, *parts[idx + 1 :])
        if candidate.is_file():
            return candidate
    raw = Path(path_str)
    hit = by_pair.get((raw.parent.name, raw.name))
    if hit is not None and hit.is_file():
        return hit
    hit = by_name.get(raw.name)
    if hit is not None and hit.is_file():
        return hit
    return None


def fill_from_log(out_path: Path, log_path: Path, source: Path, log) -> int:
    """Fill blank genre cells in an existing labels CSV with the genres the
    classifier itself predicted (from musicsort_log.csv). Rows the user
    already filled are never touched, and stale paths from an older library
    location are remapped onto the current one so retraining works.
    """
    if not log_path.is_file():
        log.error("Log file not found: %s", log_path)
        return 1
    if not out_path.is_file():
        log.error("Labels CSV not found: %s - run this script without --fill-from-log to generate one", out_path)
        return 1

    # Log paths may be stale too, so index the log the same name-based way.
    log_genres: dict[tuple[str, str], str] = {}
    with open(log_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["genres"].strip():
                p = Path(row["source_path"])
                log_genres.setdefault((p.parent.name, p.name), row["genres"])

    with open(out_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    by_pair, by_name = _name_indexes(str(p) for p in discover_files(source, DEFAULT_EXTENSIONS))

    filled = 0
    relocated = 0
    unresolved = 0
    for row in rows:
        old_key = (Path(row["path"]).parent.name, Path(row["path"]).name)
        located = locate_file(row["path"], source, by_pair, by_name)
        if located is not None:
            if str(located) != row["path"]:
                row["path"] = str(located)
                relocated += 1
            key = (located.parent.name, located.name)
        else:
            unresolved += 1
            key = old_key  # still try to fill the label from the log

        if not row.get("genre", "").strip():
            predicted = log_genres.get(key)
            if predicted:
                row["genre"] = predicted
                filled += 1

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "genre"])
        writer.writeheader()
        writer.writerows(rows)

    log.info("Filled %d blank genre cell(s) from %s", filled, log_path)
    if relocated:
        log.info("Remapped %d path(s) to the current library location", relocated)
    if unresolved:
        log.warning("%d row(s) could not be matched to a current file but the genre was filled anyway; check them", unresolved)
    log.info("Your own filled rows were kept as-is - keep editing them, then retrain with:")
    log.info("  .venv/bin/python train_genre_head.py --labels-csv %s --cache .clap_cache.pkl --out clap_genre_head.joblib", out_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="music", help="Folder with the source audio files (default: music)")
    parser.add_argument("--out", default="labels_template.csv", help="Where to write the CSV template")
    parser.add_argument("--count", type=int, default=200, help="How many tracks to sample (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, for a reproducible sample")
    parser.add_argument(
        "--extensions", default=",".join(sorted(DEFAULT_EXTENSIONS)), help="Comma-separated list of extensions to include"
    )
    parser.add_argument(
        "--fill-from-log",
        default=None,
        metavar="LOG",
        help="Instead of generating a new template, fill blank genre cells in --out with the "
        "classifier's own genres from a musicsort_log.csv. Rows you filled yourself, "
        "and their paths, are preserved; run this again after reclassifying to fill rows you haven't touched.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("generate_label_template")

    source = Path(args.source).expanduser().resolve()
    extensions = {e if e.startswith(".") else f".{e}" for e in args.extensions.split(",") if e}

    if args.fill_from_log:
        if not source.is_dir():
            log.error("Source folder not found: %s", source)
            return 1
        return fill_from_log(
            Path(args.out).expanduser().resolve(), Path(args.fill_from_log).expanduser().resolve(), source, log
        )

    if not source.is_dir():
        log.error("Source folder not found: %s", source)
        return 1

    files = discover_files(source, extensions)
    if not files:
        log.error("No audio files found under %s", source)
        return 1

    count = min(args.count, len(files))
    sample = random.Random(args.seed).sample(files, count)
    sample.sort()

    out_path = Path(args.out).expanduser().resolve()
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "genre"])
        for p in sample:
            writer.writerow([str(p), ""])

    log.info("Wrote %d track(s) to %s", count, out_path)
    log.info("Fill in the 'genre' column - separate multiple genres with ';' (e.g. 'electronic;ambient').")
    log.info("New genres are fine, not just: acoustic_classical, ambient, electronic, hiphop_beat, jazz_blues, pop, rock_metal.")
    log.info("Then run: python3 train_genre_head.py --labels-csv %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
