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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="music", help="Folder with the source audio files (default: music)")
    parser.add_argument("--out", default="labels_template.csv", help="Where to write the CSV template")
    parser.add_argument("--count", type=int, default=200, help="How many tracks to sample (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, for a reproducible sample")
    parser.add_argument(
        "--extensions", default=",".join(sorted(DEFAULT_EXTENSIONS)), help="Comma-separated list of extensions to include"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("generate_label_template")

    source = Path(args.source).expanduser().resolve()
    extensions = {e if e.startswith(".") else f".{e}" for e in args.extensions.split(",") if e}

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
