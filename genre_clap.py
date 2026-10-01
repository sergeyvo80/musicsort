"""CLAP-based genre classification.

Two modes, both built on the same LAION-CLAP audio embeddings, and both
multi-label - a track can land in more than one genre (e.g. "electronic" and
"ambient" for a chillout trance track):

- zero-shot: compare a track's embedding against text embeddings of each
  target genre's description. No training data needed, works out of the box.
- head: a one-vs-rest logistic-regression classifier (one binary model per
  genre) trained on top of CLAP embeddings, using hand-labeled or tagged
  tracks as ground truth. The set of genres isn't fixed - whatever labels
  show up in the training data become the classes, so introducing a new
  genre is just a matter of labeling a few tracks with it and retraining.

Embeddings are cached on disk keyed by (path, mtime, size), since extracting
them is the slow part and re-runs (e.g. after tweaking the head) shouldn't
redo it.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

log = logging.getLogger("musicsort.clap")

# Default, human-readable prompts for the genre buckets GENRE_TAG_ALIASES
# maps into. These just give zero-shot matching a better starting point -
# any other genre name works too, including ones introduced later while
# hand-labeling tracks (see prompt_for_genre's fallback).
DEFAULT_PROMPTS = {
    "hiphop_beat": "hip hop or rap music",
    "electronic": "electronic dance music, house, techno or trance",
    "rock_metal": "rock or metal music",
    "acoustic_classical": "acoustic, classical or folk music",
    "jazz_blues": "jazz or blues music",
    "pop": "pop music",
    "ambient": "ambient, chillout or downtempo music",
}

OTHER_LABEL = "other"
# Raw CLAP cosine similarities sit in a compressed, per-track-shifted range
# (0.46 for a clean genre match on one track, 0.32 on another), so a fixed
# absolute bar either lets everything through or nothing. Two-part rule
# instead: the top match must clear an absolute floor (else nothing in the
# vocabulary fits well - "other"), and any *other* genre is only included
# if it's notably above that track's own mean similarity (z-score), not
# just "not the worst option".
ZERO_SHOT_MIN_TOP_SIMILARITY = 0.2
ZERO_SHOT_Z_THRESHOLD = 1.0
HEAD_PROBABILITY_THRESHOLD = 0.5

_model = None


def _get_model():
    global _model
    if _model is None:
        import laion_clap

        log.info("Loading CLAP model (first run downloads the ~2GB checkpoint)...")
        # Default amodel (HTSAT-tiny) matches the architecture of the
        # pretrained checkpoint load_ckpt() downloads - passing a different
        # amodel here causes a state_dict shape mismatch at load time.
        _model = laion_clap.CLAP_Module(enable_fusion=False)
        _model.load_ckpt()
    return _model


def _cache_key(path: str) -> str:
    st = Path(path).stat()
    return f"{path}|{st.st_mtime_ns}|{st.st_size}"


def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as fh:
        pickle.dump(cache, fh)
    tmp_path.replace(cache_path)


def embed_files(
    paths: list[str], cache_path: Path | None = None, batch_size: int = 16
) -> dict[str, np.ndarray]:
    """Returns {path: embedding} for the given audio files."""
    cache = _load_cache(cache_path) if cache_path else {}
    key_by_path = {p: _cache_key(p) for p in paths}

    result: dict[str, np.ndarray] = {}
    to_compute = []
    for p in paths:
        cached = cache.get(key_by_path[p])
        if cached is not None:
            result[p] = cached
        else:
            to_compute.append(p)

    if to_compute:
        model = _get_model()
        for i in range(0, len(to_compute), batch_size):
            batch = to_compute[i : i + batch_size]
            log.info("  embedding %d/%d...", min(i + batch_size, len(to_compute)), len(to_compute))
            try:
                embeddings = model.get_audio_embedding_from_filelist(x=batch, use_tensor=False)
            except Exception:
                # Fall back to one-at-a-time so a single bad file doesn't
                # drop the whole batch.
                embeddings = []
                for p in batch:
                    try:
                        embeddings.append(
                            model.get_audio_embedding_from_filelist(x=[p], use_tensor=False)[0]
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.warning("  failed to embed %s: %s", p, exc)
                        embeddings.append(None)
            for p, emb in zip(batch, embeddings):
                if emb is None:
                    continue
                emb = np.asarray(emb, dtype=np.float32)
                result[p] = emb
                cache[key_by_path[p]] = emb

    if cache_path and to_compute:
        _save_cache(cache_path, cache)

    return result


def prompt_for_genre(genre: str) -> str:
    return DEFAULT_PROMPTS.get(genre, f"{genre.replace('_', ' ')} music")


def get_text_embeddings(genres: list[str]) -> dict[str, np.ndarray]:
    """Text embeddings for an arbitrary set of genre names - not limited to
    DEFAULT_PROMPTS, so genres introduced later via hand-labeling work too.
    """
    model = _get_model()
    prompts = [prompt_for_genre(g) for g in genres]
    embeddings = model.get_text_embedding(prompts, use_tensor=False)
    return {g: np.asarray(e, dtype=np.float32) for g, e in zip(genres, embeddings)}


def zero_shot_genres(
    embedding: np.ndarray,
    text_embeddings: dict[str, np.ndarray],
    min_top_similarity: float = ZERO_SHOT_MIN_TOP_SIMILARITY,
    z_threshold: float = ZERO_SHOT_Z_THRESHOLD,
) -> list[str]:
    """A track can plausibly belong to more than one genre, so this returns
    every genre that stands out for this track, most similar first - not
    just the single best match. See the ZERO_SHOT_* constants for why this
    isn't a plain "similarity >= constant" filter.
    """
    genres = list(text_embeddings.keys())
    sims = np.array(
        [
            float(
                np.dot(embedding, text_embeddings[g])
                / (np.linalg.norm(embedding) * np.linalg.norm(text_embeddings[g]) + 1e-9)
            )
            for g in genres
        ]
    )
    order = np.argsort(-sims)
    if sims[order[0]] < min_top_similarity:
        return [OTHER_LABEL]

    std = float(sims.std())
    if std < 1e-6:
        return [genres[order[0]]]

    z = (sims - sims.mean()) / std
    matches = [genres[i] for i in order if z[i] >= z_threshold]
    return matches or [genres[order[0]]]


def train_head(embeddings: dict[str, np.ndarray], labels: dict[str, list[str]], out_path: Path) -> int:
    """Trains a multi-label head (one-vs-rest logistic regression, one
    binary model per genre) on {path: embedding} / {path: [genres]}.

    Returns the number of tracks used for training. Raises ValueError if
    there isn't enough labeled data to fit anything meaningful. The genre
    vocabulary is whatever appears in `labels` - there's no fixed class list.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    import joblib

    paths = [p for p in embeddings if labels.get(p)]
    if len(paths) < 20:
        raise ValueError(f"only {len(paths)} labeled tracks available - need at least 20 to train a head")

    x = np.stack([embeddings[p] for p in paths])
    y_raw = [labels[p] for p in paths]

    binarizer = MultiLabelBinarizer()
    y = binarizer.fit_transform(y_raw)

    counts = y.sum(axis=0)
    sparse = [g for g, c in zip(binarizer.classes_, counts) if c < 5]
    if sparse:
        log.warning("These genres have fewer than 5 examples and will predict poorly: %s", ", ".join(sparse))

    clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(x, y)

    joblib.dump({"classifier": clf, "binarizer": binarizer}, out_path)
    return len(paths)


def load_head(path: Path):
    import joblib

    return joblib.load(path)


def predict_with_head(head, embedding: np.ndarray, threshold: float = HEAD_PROBABILITY_THRESHOLD) -> list[str]:
    clf, binarizer = head["classifier"], head["binarizer"]
    proba = np.asarray(clf.predict_proba(embedding.reshape(1, -1)))[0]
    matches = [g for g, p in zip(binarizer.classes_, proba) if p >= threshold]
    if not matches:
        matches = [str(binarizer.classes_[int(np.argmax(proba))])]
    return matches
