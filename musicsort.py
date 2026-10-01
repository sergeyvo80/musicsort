#!/usr/bin/env python3
"""
Organize an audio library into folders by genre and mood.

Genre comes from existing tags when present. For untagged tracks, pick a
classifier with --classifier:
  - heuristic (default): rule-based guess from tempo/percussiveness/timbre.
    Not a trained model - DSP rules of thumb, useful for sorting but not
    perfectly accurate for every track.
  - clap-zeroshot: LAION-CLAP audio embeddings matched against text
    descriptions of each genre. No training data needed. See genre_clap.py.
  - clap-head: CLAP embeddings + a small classifier trained on your own
    tagged tracks (train_genre_head.py). Most accurate once you have enough
    tagged tracks to train from.

Mood is always derived from the audio itself, using a simplified
arousal/valence model (the same idea used in music-emotion-recognition
research): tempo + loudness give the "energy" axis, brightness +
harmonic-vs-percussive balance give the "positive/negative" axis. Thresholds
for both axes are computed from the percentiles of the library being
processed, so "high energy" means "high relative to your own collection",
not some fixed number tuned on someone else's music.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import logging
import os
import shutil
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")

DEFAULT_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"}

GENRE_TAG_ALIASES = {
    "hip hop": "hiphop_beat",
    "hip-hop": "hiphop_beat",
    "rap": "hiphop_beat",
    "trance": "electronic",
    "house": "electronic",
    "techno": "electronic",
    "edm": "electronic",
    "dance": "electronic",
    "drum and bass": "electronic",
    "dubstep": "electronic",
    "electronica": "electronic",
    "rock": "rock_metal",
    "metal": "rock_metal",
    "punk": "rock_metal",
    "classical": "acoustic_classical",
    "orchestral": "acoustic_classical",
    "acoustic": "acoustic_classical",
    "folk": "acoustic_classical",
    "jazz": "jazz_blues",
    "blues": "jazz_blues",
    "pop": "pop",
    "ambient": "ambient",
    "chillout": "ambient",
    "downtempo": "ambient",
}

MOOD_GRID = {
    ("low", "low"): "melancholic",
    ("low", "medium"): "contemplative",
    ("low", "high"): "peaceful",
    ("medium", "low"): "mysterious",
    ("medium", "medium"): "atmospheric",
    ("medium", "high"): "uplifting",
    ("high", "low"): "tension_building",
    ("high", "medium"): "driving",
    ("high", "high"): "euphoric",
}


@dataclass
class TrackFeatures:
    path: Path
    tempo: float
    rms: float
    centroid: float
    zcr: float
    flatness: float
    percussive_ratio: float
    tag_genre: str | None = None
    error: str | None = None


@dataclass
class TrackResult:
    path: Path
    genre: str
    mood: str
    dest: Path | None = None
    error: str | None = None


def discover_files(source: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def looks_like_spam_tag(value: str) -> bool:
    """Pirated-release packs often stuff a promo URL into the genre field."""
    lowered = value.lower()
    if any(token in lowered for token in (".com", ".net", ".org", ".ru", "www.", "http")):
        return True
    return len(value) > 30


def read_tag_genre(path: Path) -> str | None:
    try:
        from mutagen import File as MutagenFile

        tags = MutagenFile(path, easy=True)
        if not tags:
            return None
        raw = tags.get("genre")
        if not raw:
            return None
        value = str(raw[0]).strip().lower()
        if not value or looks_like_spam_tag(value):
            return None
        return GENRE_TAG_ALIASES.get(value, sanitize_name(value))
    except Exception:
        return None


def analyze_file(path_str: str, duration: float) -> dict:
    """Runs in a worker process - must only take/return picklable plain data."""
    import numpy as np
    import librosa

    path = Path(path_str)
    try:
        y, sr = librosa.load(path, sr=22050, mono=True, duration=duration)
        if y.size == 0:
            raise ValueError("empty audio stream")

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])
        rms = float(librosa.feature.rms(y=y).mean())
        centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        zcr = float(librosa.feature.zero_crossing_rate(y=y).mean())
        flatness = float(librosa.feature.spectral_flatness(y=y).mean())

        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = float(np.sum(y_harmonic ** 2))
        percussive_energy = float(np.sum(y_percussive ** 2))
        percussive_ratio = percussive_energy / (harmonic_energy + percussive_energy + 1e-9)

        return {
            "path": path_str,
            "tempo": tempo,
            "rms": rms,
            "centroid": centroid,
            "zcr": zcr,
            "flatness": flatness,
            "percussive_ratio": percussive_ratio,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - report, don't crash the batch
        return {"path": path_str, "error": str(exc)}


def guess_genre_from_features(f: TrackFeatures) -> str:
    """Rough, absolute-threshold rules of thumb - not a trained model.

    Note: four-on-the-floor electronic music is mostly sustained synth
    pads/leads, so HPSS scores it as "harmonic", not "percussive" - a fast
    tempo plus a noisy/synthetic spectrum (spectral flatness) is a much
    better electronic signal here than percussive_ratio.
    """
    if f.tempo >= 118 and f.flatness > 0.015:
        return "electronic"
    if f.tempo <= 110 and f.percussive_ratio < 0.3 and f.flatness < 0.015:
        return "acoustic_classical"
    if f.percussive_ratio > 0.35 and 75 <= f.tempo <= 160 and f.zcr > 0.09 and f.flatness > 0.02:
        return "rock_metal"
    if f.tempo <= 100 and f.percussive_ratio > 0.3 and f.centroid < 2200:
        return "hiphop_beat"
    if f.rms < 0.05 and f.tempo < 100:
        return "ambient"
    return "other"


def percentile_rank(values: list[float], value: float) -> float:
    if len(values) <= 1:
        return 0.5
    below = sum(1 for v in values if v <= value)
    return below / len(values)


def sanitize_name(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
    return "_".join(keep.strip().lower().split()) or "unknown"


def unique_destination(dest_dir: Path, filename: str) -> Path:
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    stem, suffix = Path(filename).stem, Path(filename).suffix
    counter = 2
    while True:
        candidate = dest_dir / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def classify_tracks(
    analyzed: list[TrackFeatures], clap_predictions: dict[str, list[str]] | None = None
) -> dict[str, tuple[str, str]]:
    """Returns {path_str: (genre, mood)} using library-relative thresholds.

    Genre priority: ID3 tag > CLAP prediction (if provided) > DSP heuristic.
    A CLAP classifier can predict more than one genre per track (see
    genre_clap.py) - the first one is used as the primary genre here, for
    picking a single destination folder. The full set is logged separately
    in main()'s CSV report.
    """
    clap_predictions = clap_predictions or {}
    valid = [f for f in analyzed if f.error is None]

    tempos = [f.tempo for f in valid]
    rmses = [f.rms for f in valid]
    centroids = [f.centroid for f in valid]
    perc_ratios = [f.percussive_ratio for f in valid]

    result: dict[str, tuple[str, str]] = {}
    for f in analyzed:
        if f.error is not None:
            continue

        arousal = 0.5 * percentile_rank(tempos, f.tempo) + 0.5 * percentile_rank(rmses, f.rms)
        valence = 0.5 * percentile_rank(centroids, f.centroid) + 0.5 * (
            1 - percentile_rank(perc_ratios, f.percussive_ratio)
        )

        arousal_bucket = "low" if arousal < 0.33 else "high" if arousal > 0.66 else "medium"
        valence_bucket = "low" if valence < 0.33 else "high" if valence > 0.66 else "medium"
        mood = MOOD_GRID[(arousal_bucket, valence_bucket)]

        clap_genres = clap_predictions.get(str(f.path))
        genre = f.tag_genre or (clap_genres[0] if clap_genres else None) or guess_genre_from_features(f)
        result[str(f.path)] = (genre, mood)

    return result


def format_eta(done: int, total: int, elapsed: float) -> str:
    if done == 0:
        return "?"
    remaining = (elapsed / done) * (total - done)
    return f"{remaining:5.0f}s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="music", help="Folder with the source audio files (default: music)")
    parser.add_argument("--dest", default="organized", help="Folder to write the sorted library into (default: organized)")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them")
    parser.add_argument("--dry-run", action="store_true", help="Analyze and print the plan, but don't touch any files")
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds of audio to analyze per file (default: 60)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Parallel analysis workers")
    parser.add_argument("--extensions", default=",".join(sorted(DEFAULT_EXTENSIONS)), help="Comma-separated list of extensions to include")
    parser.add_argument("--log", default=None, help="CSV report path (default: <dest>/musicsort_log.csv)")
    parser.add_argument(
        "--classifier",
        choices=["heuristic", "clap-zeroshot", "clap-head"],
        default="heuristic",
        help="Genre classifier for untagged tracks: DSP rules of thumb (default), "
        "zero-shot CLAP, or a CLAP embedding + trained head (see train_genre_head.py)",
    )
    parser.add_argument("--clap-cache", default=".clap_cache.pkl", help="CLAP embedding cache file")
    parser.add_argument("--clap-head", default="clap_genre_head.joblib", help="Trained head file (for --classifier clap-head)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("musicsort")

    source = Path(args.source).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    extensions = {e if e.startswith(".") else f".{e}" for e in args.extensions.split(",") if e}
    log_path = Path(args.log).expanduser().resolve() if args.log else dest / "musicsort_log.csv"

    if not source.is_dir():
        log.error("Source folder not found: %s", source)
        return 1

    files = discover_files(source, extensions)
    if not files:
        log.error("No audio files (%s) found under %s", ", ".join(sorted(extensions)), source)
        return 1

    log.info("Found %d audio file(s) under %s", len(files), source)
    log.info("Reading tags and analyzing audio (duration=%.0fs, workers=%d)...", args.duration, args.workers)

    tag_genres: dict[str, str | None] = {str(p): read_tag_genre(p) for p in files}

    analyzed: list[TrackFeatures] = []
    errors: list[tuple[Path, str]] = []
    total = len(files)
    done = 0
    import time
    start = time.time()

    with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_to_path = {
            pool.submit(analyze_file, str(p), args.duration): p for p in files
        }
        for fut in futures.as_completed(future_to_path):
            path = future_to_path[fut]
            done += 1
            data = fut.result()
            if data.get("error"):
                errors.append((path, data["error"]))
            else:
                analyzed.append(
                    TrackFeatures(
                        path=path,
                        tempo=data["tempo"],
                        rms=data["rms"],
                        centroid=data["centroid"],
                        zcr=data["zcr"],
                        flatness=data["flatness"],
                        percussive_ratio=data["percussive_ratio"],
                        tag_genre=tag_genres.get(str(path)),
                    )
                )
            elapsed = time.time() - start
            print(
                f"\r  {done}/{total} analyzed  (eta {format_eta(done, total, elapsed)})",
                end="",
                file=sys.stderr,
            )
    print(file=sys.stderr)

    if errors:
        log.warning("%d file(s) failed to analyze and will be skipped:", len(errors))
        for path, reason in errors[:10]:
            log.warning("  %s -> %s", path, reason)
        if len(errors) > 10:
            log.warning("  ... and %d more", len(errors) - 10)

    clap_predictions: dict[str, list[str]] = {}
    if args.classifier != "heuristic":
        import genre_clap

        untagged = [str(f.path) for f in analyzed if f.tag_genre is None]
        if untagged:
            log.info(
                "Running CLAP genre classification (%s) on %d untagged track(s)...",
                args.classifier,
                len(untagged),
            )
            embeddings = genre_clap.embed_files(untagged, cache_path=Path(args.clap_cache))
            if args.classifier == "clap-zeroshot":
                text_embeddings = genre_clap.get_text_embeddings(sorted(genre_clap.DEFAULT_PROMPTS))
                clap_predictions = {
                    p: genre_clap.zero_shot_genres(emb, text_embeddings) for p, emb in embeddings.items()
                }
            else:
                head_path = Path(args.clap_head)
                if not head_path.is_file():
                    log.error(
                        "No trained head at %s - run train_genre_head.py first, or use --classifier clap-zeroshot",
                        head_path,
                    )
                    return 1
                head = genre_clap.load_head(head_path)
                clap_predictions = {p: genre_clap.predict_with_head(head, emb) for p, emb in embeddings.items()}

    classification = classify_tracks(analyzed, clap_predictions)

    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in analyzed:
        genre, mood = classification[str(f.path)]
        genre_dir = sanitize_name(genre)
        mood_dir = sanitize_name(mood)
        all_genres = clap_predictions.get(str(f.path)) or [genre]
        genres_field = ";".join(sanitize_name(g) for g in all_genres)
        target_dir = dest / genre_dir / mood_dir

        if args.dry_run:
            rows.append((f.path, genre_dir, genres_field, mood_dir, None, "dry-run"))
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = unique_destination(target_dir, f.path.name)
        try:
            if args.move:
                shutil.move(str(f.path), str(target_path))
            else:
                shutil.copy2(str(f.path), str(target_path))
            rows.append((f.path, genre_dir, genres_field, mood_dir, target_path, "ok"))
        except Exception as exc:  # noqa: BLE001
            rows.append((f.path, genre_dir, genres_field, mood_dir, None, f"copy_failed: {exc}"))

    for path, reason in errors:
        rows.append((path, "", "", "", None, f"analysis_failed: {reason}"))

    if not args.dry_run:
        with open(log_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["source_path", "genre", "genres", "mood", "dest_path", "status"])
            for path, genre_dir, genres_field, mood_dir, target_path, status in rows:
                writer.writerow(
                    [str(path), genre_dir, genres_field, mood_dir, str(target_path) if target_path else "", status]
                )
        log.info("Log written to %s", log_path)

    ok = sum(1 for r in rows if r[5] == "ok")
    log.info(
        "%s %d file(s)%s into %s",
        "Would move" if args.dry_run and args.move else "Would copy" if args.dry_run else ("Moved" if args.move else "Copied"),
        len(analyzed) if args.dry_run else ok,
        "" if not errors else f" ({len(errors)} skipped due to errors)",
        dest,
    )

    if args.dry_run:
        from collections import Counter

        breakdown = Counter((g, m) for _, g, _, m, _, _ in rows)
        for (g, m), count in sorted(breakdown.items()):
            log.info("  %s/%s: %d", g, m, count)

    return 0


if __name__ == "__main__":
    sys.exit(main())
