#!/usr/bin/env python3
"""Train a genre-classification head on top of CLAP embeddings.

A track can have more than one genre - separate them with ";" in the CSV
(e.g. "electronic;ambient"). The genre vocabulary is open: any label you use
becomes a class the head can predict, so you can introduce new genres (not
just the ones GENRE_TAG_ALIASES knows about) just by labeling a few tracks
with them.

Ground truth normally comes for free from tracks that already carry a
usable ID3 genre tag (mapped through musicsort.GENRE_TAG_ALIASES). If
your library has few or none of those (common for libraries pulled from
release-group/pirate sources, where the genre field is missing or stuffed
with a promo URL), pass --labels-csv instead: a CSV with "path,genre"
columns for a small hand-labeled sample (~20-30 tracks per genre is enough
to start; see generate_label_template.py). Point musicsort.py at the
trained result with --classifier clap-head to classify the rest of the
library.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import genre_clap
from musicsort import DEFAULT_EXTENSIONS, discover_files, read_tag_genre, sanitize_name


def load_labels_from_csv(csv_path: Path) -> dict[str, list[str]]:
    labeled: dict[str, list[str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            path = str(Path(row["path"]).expanduser().resolve())
            genres = [sanitize_name(g) for g in row["genre"].split(";") if g.strip()]
            if genres:
                labeled[path] = genres
    return labeled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="music", help="Folder with the source audio files (default: music)")
    parser.add_argument("--cache", default=".clap_cache.pkl", help="CLAP embedding cache file")
    parser.add_argument("--out", default="clap_genre_head.joblib", help="Where to save the trained head")
    parser.add_argument(
        "--extensions", default=",".join(sorted(DEFAULT_EXTENSIONS)), help="Comma-separated list of extensions to include"
    )
    parser.add_argument(
        "--labels-csv",
        default=None,
        help="CSV with 'path,genre' columns for hand-labeled tracks, used instead of ID3 tags. "
        "Separate multiple genres for one track with ';' (e.g. 'electronic;ambient').",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("train_genre_head")

    if args.labels_csv:
        labeled = load_labels_from_csv(Path(args.labels_csv).expanduser().resolve())
        log.info("%d labeled track(s) loaded from %s", len(labeled), args.labels_csv)
    else:
        source = Path(args.source).expanduser().resolve()
        extensions = {e if e.startswith(".") else f".{e}" for e in args.extensions.split(",") if e}

        if not source.is_dir():
            log.error("Source folder not found: %s", source)
            return 1

        files = discover_files(source, extensions)
        log.info("Reading tags from %d file(s)...", len(files))
        tag_genres = {str(p): read_tag_genre(p) for p in files}
        labeled = {p: [g] for p, g in tag_genres.items() if g}
        log.info("%d of %d tracks have a usable genre tag to train from", len(labeled), len(files))

    if len(labeled) < 20:
        log.error(
            "Need at least 20 labeled tracks to train a head (have %d). "
            "Either hand-label more tracks into a --labels-csv, or use --classifier clap-zeroshot instead.",
            len(labeled),
        )
        return 1

    genre_counts: dict[str, int] = {}
    for genres in labeled.values():
        for g in genres:
            genre_counts[g] = genre_counts.get(g, 0) + 1
    log.info("Genres in the label set: %s", ", ".join(f"{g} ({n})" for g, n in sorted(genre_counts.items())))

    log.info("Extracting CLAP embeddings for the labeled tracks...")
    embeddings = genre_clap.embed_files(list(labeled.keys()), cache_path=Path(args.cache))

    log.info("Training classifier head on %d embedded tracks...", len(embeddings))
    used = genre_clap.train_head(embeddings, labeled, Path(args.out))
    log.info("Trained on %d tracks, saved head to %s", used, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
