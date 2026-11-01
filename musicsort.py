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

Layout: the destination tree is flat - <dest>/<name> folders for every genre
and every mood (mood folders carry a "mood_" prefix), side by side, one level
deep. A track is placed in every genre folder it was assigned (CLAP can
predict several) plus its mood folder, so one track may live in several
folders at once. The destination folder is wiped clean before every real
(non-dry-run) run - you get a warning and a confirmation prompt first, so
stale copies from a previous layout can't linger.
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

# Mood folders get this prefix in the flat destination tree, so genre
# folders (techno, ambient, ...) and mood folders (mood_driving, ...)
# never collide and are visually distinguished at a glance.
MOOD_FOLDER_PREFIX = "mood_"


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
    genre_clap.py) - the first one serves as the primary genre in the CSV
    report, the full set is used for the flat destination layout and is also
    logged in main()'s CSV report.
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


def list_dir_contents(dest: Path) -> list[Path]:
    return sorted(dest.iterdir()) if dest.is_dir() else []


def wipe_dest(dest: Path, source: Path, assume_yes: bool, log) -> bool:
    """Clears the destination folder before a fresh sorting run, so stale
    folders or copies from a previous layout can't linger. Warns and asks
    for confirmation first (--yes skips the prompt); returns False without
    touching anything on abort or failure. Refuses when wiping would take
    out the source library (dest equal to or containing it).
    """
    if dest == source or source.is_relative_to(dest):
        log.error("Refusing to wipe %s: it contains the source folder %s", dest, source)
        return False

    existing = list_dir_contents(dest)
    if existing:
        log.warning("Destination %s will be WIPED before writing:", dest)
        log.warning("  %d item(s) will be deleted permanently, e.g.:", len(existing))
        for entry in existing[:10]:
            log.warning("    - %s", entry.name)
        if len(existing) > 10:
            log.warning("    ... and %d more", len(existing) - 10)
        if not assume_yes:
            answer = input("Proceed with wiping? Type y to continue, anything else to abort: ")
            if answer.strip().lower() not in ("y", "yes"):
                log.info("Aborted - destination left untouched.")
                return False
    else:
        log.info("Destination %s is empty or does not exist yet - nothing to wipe.", dest)

    deleted = 0
    for entry in existing:
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            deleted += 1
        except OSError as exc:
            log.error("Failed to delete %s: %s - aborting to avoid a partial wipe", entry, exc)
            return False
    if deleted:
        log.info("Wiped %d item(s) from %s", deleted, dest)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="music", help="Folder with the source audio files (default: music)")
    parser.add_argument("--dest", default="sorted", help="Folder to write the sorted library into (default: sorted). Wiped clean before a real run")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt when wiping the destination folder")
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

    if not args.dry_run and not wipe_dest(dest, source, args.yes, log):
        return 1

    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[Path, str, str, str, list[str], list[Path], str]] = []
    for f in analyzed:
        genre, mood = classification[str(f.path)]
        mood_dir = MOOD_FOLDER_PREFIX + sanitize_name(mood)
        all_genres = clap_predictions.get(str(f.path)) or [genre]
        genre_dirs = list(dict.fromkeys(sanitize_name(g) for g in all_genres))
        genres_field = ";".join(genre_dirs)

        # Flat layout: every genre folder and the mood-prefixed mood folder
        # sit directly under dest, so genres and mood are browsable on the
        # same level. A multi-genre track gets one copy per genre on top of
        # its mood copy.
        target_names = genre_dirs + [mood_dir]
        target_dirs = [dest / name for name in target_names]

        if args.dry_run:
            rows.append((f.path, genre_dirs[0], genres_field, mood_dir, target_names, [], "dry-run"))
            continue

        dest_paths: list[Path] = []
        last_dir = target_dirs[-1]
        try:  # noqa: BLE001
            for target_dir in target_dirs:
                # With several destinations plain --move can't happen for
                # all of them: extras are copied, only the final one gets
                # the move so the source ends up with a single remaining
                # reference.
                is_last = target_dir is last_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = unique_destination(target_dir, f.path.name)
                if args.move and is_last:
                    shutil.move(str(f.path), str(target_path))
                else:
                    shutil.copy2(str(f.path), str(target_path))
                dest_paths.append(target_path)
            rows.append((f.path, genre_dirs[0], genres_field, mood_dir, target_names, dest_paths, "ok"))
        except Exception as exc:
            rows.append((f.path, genre_dirs[0], genres_field, mood_dir, target_names, dest_paths, f"copy_failed: {exc}"))

    for path, reason in errors:
        rows.append((path, "", "", "", [], [], f"analysis_failed: {reason}"))

    ok = sum(1 for r in rows if r[6] == "ok")
    placements = sum(len(r[5]) if r[6] == "ok" else len(r[4]) if r[6] == "dry-run" else 0 for r in rows)

    if not args.dry_run:
        with open(log_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["source_path", "genre", "genres", "mood", "dest_path", "status"])
            for path, primary, genres_field, mood_dir, target_names, dest_paths, status in rows:
                writer.writerow(
                    [str(path), primary, genres_field, mood_dir, ";".join(str(p) for p in dest_paths), status]
                )
        log.info("Log written to %s", log_path)

    log.info(
        "%s %d file(s)%s into %s (%d folder placements)",
        "Would move" if args.dry_run and args.move else "Would copy" if args.dry_run else ("Moved" if args.move else "Copied"),
        len(analyzed) if args.dry_run else ok,
        "" if not errors else f" ({len(errors)} skipped due to errors)",
        dest,
        placements,
    )

    if args.dry_run:
        from collections import Counter

        breakdown = Counter()
        for row in rows:
            for name in row[4]:
                breakdown[name] += 1
        for name, count in sorted(breakdown.items()):
            log.info("  %s: %d", name, count)

    return 0


if __name__ == "__main__":
    sys.exit(main())
