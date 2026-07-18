#!/usr/bin/env python3
"""
compute_ci_from_csv.py
----------------------
Standalone script: loads per-sample score CSVs and computes EER + CI.
Writes *_eer.txt in the same format as run_inference(). No GPU needed.

Usage:
    python compute_ci_from_csv.py [scores_text_outputs_dir] [--overwrite] [--bootstraps N]
                                  [--dataset NAME] [--dev-protocol PATH] [--eval-protocol PATH]

Environment variables:
    EER_NUM_BOOTSTRAPS        number of bootstrap iterations (default: 1000)
    EER_ALPHA                 CI percentile, e.g. 5 → [2.5%, 97.5%] (default: 5)
    EER_MAX_RESAMPLE_RETRIES  retries when a bootstrap draw is single-class (default: 50)
    CONFIDENCE_INTERVALS_DIR  path to the confidence_intervals package if not installed
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_curve

# =============================================================================
# CI library — sys.path injection + conditional import
# =============================================================================
_CONFIDENCE_INTERVALS_DIR = os.environ.get("CONFIDENCE_INTERVALS_DIR", "").strip()
if _CONFIDENCE_INTERVALS_DIR:
    _ci_dir = os.path.abspath(os.path.expanduser(_CONFIDENCE_INTERVALS_DIR))
    if os.path.basename(_ci_dir).lower() == "confidence_intervals":
        _ci_dir = os.path.dirname(_ci_dir)
    if _ci_dir not in sys.path:
        sys.path.insert(0, _ci_dir)

_CI_IMPORT_ERROR = None
try:
    from confidence_intervals import evaluate_with_conf_int, get_bootstrap_indices
    _CI_AVAILABLE = True
except Exception as _e:
    evaluate_with_conf_int = None
    get_bootstrap_indices  = None
    _CI_AVAILABLE          = False
    _CI_IMPORT_ERROR       = _e

# =============================================================================
# Constants  (edit here to change defaults)
# =============================================================================

# Bootstrap CI parameters — also overridable via env-vars or --bootstraps flag
EER_NUM_BOOTSTRAPS       = int(os.environ.get("EER_NUM_BOOTSTRAPS",      "1000"))
EER_ALPHA                = float(os.environ.get("EER_ALPHA",              "5"))
EER_MAX_RESAMPLE_RETRIES = int(os.environ.get("EER_MAX_RESAMPLE_RETRIES", "50"))

# Base filesystem paths used when building protocol paths below
_BASE_U = "/path/to"
_BASE_P = "/path/to/projects"

# Default scores directory when no positional argument is given on the CLI
_DEFAULT_SCORES_DIR = (
    f"{_BASE_U}/voxtral_clean/"
    "voxtral_clean-generation-asvspoof5-aug-bfloat16-2-constant-classification_as_generation-with-aug-no-margin/scores_text_outputs"
)

# Per-dataset column indices inside the protocol file: (file_id_col, attack_col, codec_col)
# codec_col=None means the dataset has no dedicated codec column.
_PROTOCOL_COLS: Dict[str, Tuple] = {
    "asvspoof5":    (1, 7, 6),    # tab-separated: spk file_id gender … codec attack label …
    "asvspoof2019": (1, 4, None), # space-separated: spk file_id - env attack label  (attack@4, label@5)
    "asvspoof2021": (1, 4, None), # space-separated: spk file_id codec env attack label …  (attack@4, label@5)
}

# Protocol file paths per (dataset, variant).
# Each entry: (variant_keywords, dev_path, eval_path)
#   variant_keywords=[]  →  default, matched when no other keyword hits
#   keywords are matched case-insensitively against the scores_dir path string
_PROTOCOL_DB: Dict[str, list] = {
    "asvspoof5": [
        (
            [],
            f"{_BASE_U}/projects/ASVspoof5/ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv",
            f"{_BASE_U}/projects/ASVspoof5/ASVspoof5_protocols/ASVspoof5.eval.track_1.tsv",
        ),
    ],
    "asvspoof2019": [
        (
            ["2021df", "2021-df", "df"],       # model trained on 2019, evaluated on 2021-DF
            f"{_BASE_U}/projects/ASVspoof2019/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
            f"{_BASE_P}/ASVspoof2021/DF-keys-full/keys/DF/CM/trial_metadata.txt",
        ),
        (
            ["2021la", "2021-la", "2021"],     # model trained on 2019, evaluated on 2021-LA
            f"{_BASE_U}/projects/ASVspoof2019/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
            f"{_BASE_P}/ASVspoof2021/LA-keys-full/keys/LA/CM/trial_metadata.txt",
        ),
        (
            [],                                # pure 2019 train+eval
            f"{_BASE_U}/projects/ASVspoof2019/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
            f"{_BASE_U}/projects/ASVspoof2019/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
        ),
    ],
    "asvspoof2021": [
        (
            ["df"],
            f"{_BASE_U}/projects/ASVspoof2019/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
            f"{_BASE_P}/ASVspoof2021/DF-keys-full/keys/DF/CM/trial_metadata.txt",
        ),
        (
            [],                                # default → LA
            f"{_BASE_U}/projects/ASVspoof2019/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
            f"{_BASE_P}/ASVspoof2021/LA-keys-full/keys/LA/CM/trial_metadata.txt",
        ),
    ],
}


# =============================================================================
# Protocol helpers
# =============================================================================

def _auto_detect_protocols(
    dataset_name: str,
    scores_dir: Path,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve protocol file paths automatically from the dataset name and the
    scores directory path, without requiring CLI arguments.

    Walks _PROTOCOL_DB[dataset_name] in order.  For each candidate entry the
    variant keywords are checked against scores_dir (case-insensitive substring
    match).  The first entry where ANY keyword appears in the path wins (OR
    semantics — keywords are treated as alternatives, not conjunctions); an
    entry with an empty keyword list acts as the catch-all default.

    Args:
        dataset_name (str):
            Lowercase dataset identifier, e.g. "asvspoof5", "asvspoof2019".
            Matched as a key in _PROTOCOL_DB.
        scores_dir (Path):
            Path to the scores_text_outputs directory being processed.
            Its string representation is searched for variant keywords
            (e.g. "df", "2021la").

    Returns:
        Tuple[Optional[str], Optional[str]]:
            (dev_protocol_path, eval_protocol_path).
            Each element is an absolute path string if the file exists on disk,
            or None if the file was not found or the dataset is unknown.
    """
    key = dataset_name.strip().lower()
    candidates = _PROTOCOL_DB.get(key)
    if not candidates:
        print(f"    [Protocol] Dataset '{dataset_name}' not in _PROTOCOL_DB — skipping enrichment")
        return None, None

    dir_str = str(scores_dir).lower()
    for keywords, dev_path, eval_path in candidates:
        # Empty keyword list = default catch-all entry; always matches.
        # Non-empty list = match if ANY keyword appears in the path (OR semantics).
        # "all" would require every keyword simultaneously, which is impossible
        # when the list contains both "2021df" and "2021-df" as alternatives.
        if keywords and not any(kw in dir_str for kw in keywords):
            continue
        label = f"keywords={keywords}" if keywords else "default"
        print(f"    [Protocol] Matched {label} for dataset '{dataset_name}'")
        dev_ok  = Path(dev_path).is_file()  if dev_path  else False
        eval_ok = Path(eval_path).is_file() if eval_path else False
        if dev_ok:
            print(f"    [Protocol] dev  ✓ {dev_path}")
        else:
            print(f"    [Protocol] dev  ✗ NOT FOUND: {dev_path}")
            dev_path = None
        if eval_ok:
            print(f"    [Protocol] eval ✓ {eval_path}")
        else:
            print(f"    [Protocol] eval ✗ NOT FOUND: {eval_path}")
            eval_path = None
        return dev_path, eval_path

    print(f"    [Protocol] No matching variant for '{dataset_name}' — skipping enrichment")
    return None, None


def _load_protocol_map(
    protocol_path: str,
    file_id_col: int,
    attack_col: int,
    codec_col: Optional[int],
) -> Dict[str, dict]:
    """
    Parse a protocol file (TSV or space-separated TXT) into a lookup map.

    Tab-split is attempted first; if a line has fewer than 2 tab-separated
    fields it is re-split on whitespace.  Rows that do not have enough columns
    to satisfy the requested indices are silently counted and skipped.

    Args:
        protocol_path (str):
            Absolute path to the protocol file on disk.
        file_id_col (int):
            Zero-based column index of the utterance/file ID field.
            Typical values: 1 for ASVspoof5 (tab-sep), 1 for ASVspoof2019.
        attack_col (int):
            Zero-based column index of the attack-type field.
            Typical values: 7 for ASVspoof5, 5 for ASVspoof2019/2021.
        codec_col (Optional[int]):
            Zero-based column index of the codec field, or None when the
            dataset has no codec column (ASVspoof2019/2021).

    Returns:
        Dict[str, dict]:
            Mapping  {file_id: {"attack": str, "codec": str | None}}.
            Returns an empty dict if the file does not exist.
    """
    result: Dict[str, dict] = {}
    path = Path(protocol_path)
    if not path.is_file():
        print(f"    [Protocol] WARN: file not found: {protocol_path}")
        return result

    needed = max([file_id_col, attack_col] + ([codec_col] if codec_col is not None else [])) + 1
    skipped = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < needed:
                skipped += 1
                continue
            file_id = parts[file_id_col].strip()
            attack  = parts[attack_col].strip()
            codec   = parts[codec_col].strip() if codec_col is not None else None
            if file_id:
                result[file_id] = {"attack": attack, "codec": codec}

    print(f"    [Protocol] Loaded {len(result):,} entries from {path.name}"
          + (f" ({skipped} rows skipped — too few columns)" if skipped else ""))
    return result


def _enrich_from_protocol(
    loaded: Dict[str, np.ndarray],
    protocol_map: Dict[str, dict],
    has_codec: bool,
) -> Dict[str, np.ndarray]:
    """
    Overwrite the 'attack' (and optionally 'codec') arrays in loaded using
    the protocol lookup map.

    The attack field is updated only for rows whose current value is a
    placeholder ('-', '', or 'nan'), preserving any meaningful labels already
    written by the inference script.  The codec field is always overwritten
    from the protocol regardless of its current value, because CSVs never
    carry codec information.

    Args:
        loaded (Dict[str, np.ndarray]):
            Sample dict as returned by _load_scores_csv.  Must contain
            'file_id' and 'attack' arrays; 'codec' is optional (defaults to
            all '-' if absent).
        protocol_map (Dict[str, dict]):
            Lookup map as returned by _load_protocol_map:
            {file_id: {"attack": str, "codec": str|None}}.
        has_codec (bool):
            Whether to enrich the 'codec' column.  Should be True only when
            codec_col was not None in _PROTOCOL_COLS for this dataset.

    Returns:
        Dict[str, np.ndarray]:
            A new dict (shallow copy of loaded) with 'attack' and optionally
            'codec' replaced.  The original dict is not mutated.
    """
    if not protocol_map:
        return loaded

    file_ids = loaded.get("file_id")
    attacks  = loaded.get("attack")
    if file_ids is None or attacks is None:
        return loaded

    new_attacks = attacks.copy()
    new_codecs  = loaded.get("codec", np.full(len(file_ids), "-", dtype=object)).copy()
    n_atk = n_cod = 0

    for i, fid in enumerate(file_ids):
        entry = protocol_map.get(str(fid).strip())
        if entry is None:
            continue
        if str(attacks[i]).strip() in ("-", "", "nan"):
            new_attacks[i] = entry["attack"]
            n_atk += 1
        if has_codec and entry.get("codec") is not None:
            new_codecs[i] = entry["codec"]
            n_cod += 1

    total = len(file_ids)
    if n_atk:
        print(f"    [Enrich] attack : {n_atk:,} / {total:,} rows enriched from protocol")
    else:
        print(f"    [Enrich] attack : no rows updated (all already labelled or no match)")
    if has_codec:
        if n_cod:
            print(f"    [Enrich] codec  : {n_cod:,} / {total:,} rows enriched from protocol")
        else:
            print(f"    [Enrich] codec  : no rows updated")

    result = dict(loaded)
    result["attack"] = new_attacks
    if has_codec:
        result["codec"] = new_codecs
    return result


def _label_bonafide_attacks(loaded: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Replace the attack label with the string 'bonafide' for every genuine
    (label=0) sample.

    ASVspoof protocols store '-' in the attack column for bonafide utterances.
    The eer.txt format uses 'bonafide' as the display name in the Attack table,
    so this relabelling must happen after protocol enrichment and before stats
    computation.

    Args:
        loaded (Dict[str, np.ndarray]):
            Sample dict containing at minimum 'labels' (int64 array, 0=bonafide
            / 1=spoof) and 'attack' (object array of attack-type strings).

    Returns:
        Dict[str, np.ndarray]:
            A new dict (shallow copy) with 'attack' updated.
            Rows with label=1 (spoof) are not touched.
            Returns the original dict unchanged if 'labels' or 'attack' is absent.
    """
    labels  = loaded.get("labels")
    attacks = loaded.get("attack")
    if labels is None or attacks is None:
        return loaded
    mask = labels == 0
    n = int(np.sum(mask))
    if n == 0:
        return loaded
    new_attacks = attacks.copy()
    new_attacks[mask] = "bonafide"
    print(f"    [Enrich] bonafide relabel: {n:,} genuine samples renamed to 'bonafide'")
    result = dict(loaded)
    result["attack"] = new_attacks
    return result


# =============================================================================
# CSV loading
# =============================================================================

def _load_scores_csv(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Read a per-sample scores CSV produced by the inference script.

    Expected CSV columns (header row required):
        file_id, speaker_id, attack, label, score

    The 'label' column is interpreted as 0 (bonafide) when the value is
    "bonafide" or "0", and as 1 (spoof) otherwise.
    A 'codec' column initialised to '-' for every row is added to the
    returned dict; it is later populated by _enrich_from_protocol.

    Args:
        path (Path):
            Absolute path to the CSV file.

    Returns:
        Optional[Dict[str, np.ndarray]]:
            Dict with keys:
                'scores'     – float64 array, spoof probability scores
                'labels'     – int64 array, 0=bonafide / 1=spoof
                'file_id'    – object array of utterance IDs
                'speaker_id' – object array of speaker IDs
                'attack'     – object array of attack-type strings
                'codec'      – object array, all '-' (enriched later)
            Returns None if the file does not exist or is empty.
    """
    if not path.is_file():
        return None
    file_ids, speaker_ids, attacks, labels_list, scores_list = [], [], [], [], []
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                file_ids.append(row.get("file_id", ""))
                speaker_ids.append(row.get("speaker_id", "unknown"))
                attacks.append(row.get("attack", "-"))
                lv = row.get("label", "0")
                labels_list.append(0 if str(lv).strip().lower() in ("bonafide", "0") else 1)
                scores_list.append(float(row.get("score", 0.0)))
    except Exception as exc:
        print(f"    [CSV] WARN: could not load {path.name}: {exc}")
        return None
    if not scores_list:
        print(f"    [CSV] WARN: {path.name} is empty")
        return None
    labels = np.array(labels_list, dtype=np.int64)
    print(f"    [CSV] {path.name}: {len(scores_list):,} rows  "
          f"({int(np.sum(labels==0)):,} bonafide / {int(np.sum(labels==1)):,} spoof)")
    return {
        "scores":     np.array(scores_list, dtype=np.float64),
        "labels":     labels,
        "file_id":    np.array(file_ids,    dtype=object),
        "speaker_id": np.array(speaker_ids, dtype=object),
        "attack":     np.array(attacks,     dtype=object),
        "codec":      np.full(len(scores_list), "-", dtype=object),
    }


# =============================================================================
# EER + CI computation
# =============================================================================

def _compute_raw_eer(
    labels: np.ndarray,
    scores: np.ndarray,
    return_threshold: bool = False,
):
    """
    Compute the Equal Error Rate (EER) from a set of binary labels and scores.

    Uses sklearn's roc_curve with pos_label=1 (spoof is the positive class).
    The EER is the point on the ROC curve where FPR ≈ FNR = argmin|FPR - FNR|.
    If the resulting threshold is non-finite it is replaced by the median score.

    Args:
        labels (np.ndarray):
            Integer array of ground-truth labels (0=bonafide, 1=spoof).
        scores (np.ndarray):
            Float array of model scores (higher = more likely spoof).
            Must be the same length as labels.
        return_threshold (bool):
            If True, also return the score threshold at which EER occurs.
            Default: False.

    Returns:
        float | Tuple[float, float]:
            If return_threshold=False → eer (float, in [0, 1]).
            If return_threshold=True  → (eer, threshold) both as floats.
            Returns nan (or (nan, nan)) when labels is empty or single-class.
    """
    if len(labels) == 0 or np.unique(labels).size < 2:
        return (np.nan, np.nan) if return_threshold else np.nan
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thr = float(thresholds[idx]) if idx < len(thresholds) else np.nan
    if not np.isfinite(thr):
        thr = float(np.median(scores))
    return (eer, thr) if return_threshold else eer


def _compute_eer_with_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    conditions: Optional[np.ndarray],
    context: str = "",
) -> Dict[str, float]:
    """
    Compute EER and a bootstrap confidence interval around it.

    Wraps _compute_raw_eer inside evaluate_with_conf_int from the
    confidence_intervals library.  When that library is unavailable, CI values
    are returned as nan.

    When a bootstrap draw accidentally contains only one class, up to
    EER_MAX_RESAMPLE_RETRIES redraws are attempted using get_bootstrap_indices
    before falling back to the full-dataset EER.

    Args:
        labels (np.ndarray):
            Integer array, 0=bonafide / 1=spoof.
        scores (np.ndarray):
            Float array of model scores aligned with labels.
        conditions (Optional[np.ndarray]):
            Stratification array for conditioned bootstrapping (e.g. speaker
            IDs or attack types).  Pass None for unconditional bootstrapping.
        context (str):
            Human-readable label used in warning messages (e.g. "dev/speaker").

    Returns:
        Dict[str, float]:
            {
                'eer_percent'     : EER in percent (0–100),
                'ci_low_percent'  : lower CI bound in percent,
                'ci_high_percent' : upper CI bound in percent,
                'threshold'       : score threshold at which EER occurs,
            }
            CI values are nan when the library is unavailable or bootstrap fails.
    """
    eer_raw, threshold = _compute_raw_eer(labels, scores, return_threshold=True)

    if not _CI_AVAILABLE:
        return {
            "eer_percent":     float(eer_raw) * 100.0 if np.isfinite(eer_raw) else np.nan,
            "ci_low_percent":  np.nan,
            "ci_high_percent": np.nan,
            "threshold":       threshold,
        }

    full_has_two_classes = (np.unique(labels).size >= 2)
    retry_seed = {"v": 0}

    def _eer_metric(lbl: np.ndarray, scr: np.ndarray) -> float:
        if not full_has_two_classes:
            return float(eer_raw)
        if np.unique(lbl).size >= 2:
            return float(_compute_raw_eer(lbl, scr))
        for _ in range(max(1, EER_MAX_RESAMPLE_RETRIES)):
            idx = get_bootstrap_indices(
                num_samples=len(labels),
                conditions=conditions,
                random_state=retry_seed["v"],
            )
            retry_seed["v"] += 1
            if np.unique(labels[idx]).size >= 2:
                return float(_compute_raw_eer(labels[idx], scores[idx]))
        return float(eer_raw)

    try:
        center, conf_int = evaluate_with_conf_int(
            scores, _eer_metric, labels, conditions,
            num_bootstraps=max(1, EER_NUM_BOOTSTRAPS),
            alpha=EER_ALPHA,
        )
        eer_out = float(center) if np.isfinite(float(center)) else float(eer_raw)
        ci_arr  = np.asarray(conf_int).reshape(-1)
        ci_low  = float(ci_arr[0]) if ci_arr.size >= 1 else np.nan
        ci_high = float(ci_arr[1]) if ci_arr.size >= 2 else np.nan
    except Exception as exc:
        print(f"    [CI] WARN: bootstrap failed ({context}): {exc}")
        eer_out, ci_low, ci_high = float(eer_raw), np.nan, np.nan

    return {
        "eer_percent":     eer_out * 100.0,
        "ci_low_percent":  ci_low  * 100.0 if np.isfinite(ci_low)  else np.nan,
        "ci_high_percent": ci_high * 100.0 if np.isfinite(ci_high) else np.nan,
        "threshold":       threshold,
    }


def _compute_triplet(
    loaded: Dict[str, np.ndarray],
    split_name: str,
) -> Dict[str, Dict]:
    """
    Run _compute_eer_with_ci three times for the standard CI triplet:
    unconditional / speaker-conditioned / attack-conditioned.

    Prints a progress line before and after each of the three bootstrap runs.

    Args:
        loaded (Dict[str, np.ndarray]):
            Sample dict as returned by _load_scores_csv (after enrichment).
            Must contain 'labels' and 'scores'.
            'speaker_id' and 'attack' are used as conditioning arrays when
            present.
        split_name (str):
            Label used in print statements, e.g. "dev" or "eval".

    Returns:
        Dict[str, Dict]:
            {
                'none'    : result dict from _compute_eer_with_ci (unconditional),
                'speaker' : result dict (speaker-conditioned),
                'attack'  : result dict (attack-conditioned),
            }
            Each inner dict has keys: eer_percent, ci_low_percent,
            ci_high_percent, threshold.  Speaker/attack entries have nan CI
            when the corresponding metadata is absent.
    """
    labels   = loaded["labels"]
    scores   = loaded["scores"]
    speakers = loaded.get("speaker_id")
    attacks  = loaded.get("attack")

    n_spk = len(np.unique(speakers)) if speakers is not None else 0
    n_atk = len(np.unique(attacks))  if attacks  is not None else 0

    print(f"\n  --- CI [{split_name}] ---", flush=True)
    print(f"    samples : {len(scores):,}  "
          f"({int(np.sum(labels==0)):,} bonafide / {int(np.sum(labels==1)):,} spoof)  "
          f"| speakers: {n_spk}  | attack types: {n_atk}", flush=True)

    # 1. none
    print(f"    [CI] none  — {EER_NUM_BOOTSTRAPS} bootstraps ...", flush=True)
    none_res = _compute_eer_with_ci(labels, scores, None, f"{split_name}/none")
    print(f"    [CI] none  → EER={none_res['eer_percent']:.4f}%"
          f"  CI=[{none_res['ci_low_percent']:.4f}, {none_res['ci_high_percent']:.4f}]", flush=True)

    # 2. speaker-conditioned  (skip when all speakers share the same ID, e.g. "unknown")
    if speakers is not None and n_spk > 1:
        print(f"    [CI] speaker-conditioned — {EER_NUM_BOOTSTRAPS} bootstraps ...", flush=True)
        spk_res = _compute_eer_with_ci(labels, scores, speakers, f"{split_name}/speaker")
        print(f"    [CI] speaker → EER={spk_res['eer_percent']:.4f}%"
              f"  CI=[{spk_res['ci_low_percent']:.4f}, {spk_res['ci_high_percent']:.4f}]", flush=True)
    else:
        reason = "no speaker metadata" if speakers is None else f"all speakers identical (n_spk={n_spk})"
        print(f"    [CI] speaker-conditioned — skipped ({reason})", flush=True)
        spk_res = {"eer_percent": np.nan, "ci_low_percent": np.nan, "ci_high_percent": np.nan, "threshold": np.nan}

    # 3. attack-conditioned
    if attacks is not None:
        print(f"    [CI] attack-conditioned  — {EER_NUM_BOOTSTRAPS} bootstraps ...", flush=True)
        atk_res = _compute_eer_with_ci(labels, scores, attacks, f"{split_name}/attack")
        print(f"    [CI] attack  → EER={atk_res['eer_percent']:.4f}%"
              f"  CI=[{atk_res['ci_low_percent']:.4f}, {atk_res['ci_high_percent']:.4f}]", flush=True)
    else:
        print(f"    [CI] attack-conditioned  — skipped (no attack metadata)", flush=True)
        atk_res = {"eer_percent": np.nan, "ci_low_percent": np.nan, "ci_high_percent": np.nan, "threshold": np.nan}

    return {"none": none_res, "speaker": spk_res, "attack": atk_res}


# =============================================================================
# Per-class accuracy stats
# =============================================================================

def _compute_attack_stats(
    loaded: Dict[str, np.ndarray],
    threshold: float,
) -> Dict[str, Dict[str, int]]:
    """
    Count correct predictions for each attack type at a fixed score threshold.

    A sample is predicted spoof when score >= threshold (pred=1), bonafide
    otherwise (pred=0).  Correct means pred == label.

    Args:
        loaded (Dict[str, np.ndarray]):
            Sample dict containing 'scores' (float64), 'labels' (int64),
            and 'attack' (object).
        threshold (float):
            Score threshold at which predictions are binarised.
            Must be finite; returns {} otherwise.

    Returns:
        Dict[str, Dict[str, int]]:
            {attack_name: {"correct": int, "total": int}, …}
            Returns an empty dict when 'attack' is absent or threshold is nan.
    """
    scores  = loaded["scores"]
    labels  = loaded["labels"]
    attacks = loaded.get("attack")
    if attacks is None or not np.isfinite(threshold):
        return {}
    preds = (scores >= threshold).astype(int)
    stats: Dict[str, Dict[str, int]] = {}
    for atk in np.unique(attacks):
        mask    = attacks == atk
        correct = int(np.sum(preds[mask] == labels[mask]))
        total   = int(np.sum(mask))
        stats[str(atk)] = {"correct": correct, "total": total}
    n_types = len(stats)
    acc = int(np.sum(preds == labels))
    print(f"    [Stats] attack  : {n_types} attack types  "
          f"({acc:,} / {len(labels):,} correct overall at threshold {threshold:.6f})")
    return stats


def _compute_codec_stats(
    loaded: Dict[str, np.ndarray],
    threshold: float,
) -> Dict[str, Dict[str, int]]:
    """
    Count correct predictions for each codec type at a fixed score threshold.

    Uses the same binarisation rule as _compute_attack_stats (score >= threshold
    → pred=1).  Skips computation when all codec values are '-' (no codec info).

    Args:
        loaded (Dict[str, np.ndarray]):
            Sample dict containing 'scores' (float64), 'labels' (int64),
            and 'codec' (object array populated by _enrich_from_protocol).
        threshold (float):
            Score threshold at which predictions are binarised.
            Must be finite; returns {} otherwise.

    Returns:
        Dict[str, Dict[str, int]]:
            {codec_name: {"correct": int, "total": int}, …}
            Returns an empty dict when 'codec' is absent, threshold is nan,
            or all codec values are '-'.
    """
    scores = loaded["scores"]
    labels = loaded["labels"]
    codecs = loaded.get("codec")
    if codecs is None or not np.isfinite(threshold):
        return {}
    unique = np.unique(codecs)
    if len(unique) == 1 and str(unique[0]) == "-":
        print(f"    [Stats] codec   : skipped (all entries are '-', no codec info)")
        return {}
    preds = (scores >= threshold).astype(int)
    stats: Dict[str, Dict[str, int]] = {}
    for c in unique:
        mask    = codecs == c
        correct = int(np.sum(preds[mask] == labels[mask]))
        total   = int(np.sum(mask))
        stats[str(c)] = {"correct": correct, "total": total}
    print(f"    [Stats] codec   : {len(stats)} codec types")
    return stats


# =============================================================================
# eer.txt formatting helpers
# =============================================================================

def _fmt_eer_ci(entry: Dict[str, float]) -> str:
    """
    Format an EER+CI result dict as a single display string.

    Args:
        entry (Dict[str, float]):
            Dict with keys 'eer_percent', 'ci_low_percent', 'ci_high_percent'
            as returned by _compute_eer_with_ci.

    Returns:
        str:
            "X.XXXX [lo.XXXX, hi.XXXX]" when all three values are finite,
            "X.XXXX [n/a, n/a]"          when EER is finite but CI is not,
            "n/a"                          when EER is also non-finite.
    """
    eer   = entry.get("eer_percent",    np.nan)
    c_lo  = entry.get("ci_low_percent", np.nan)
    c_hi  = entry.get("ci_high_percent", np.nan)
    if np.isfinite(eer) and np.isfinite(c_lo) and np.isfinite(c_hi):
        return f"{eer:.4f} [{c_lo:.4f}, {c_hi:.4f}]"
    if np.isfinite(eer):
        return f"{eer:.4f} [n/a, n/a]"
    return "n/a"


def _fmt_float(val: Any) -> str:
    """
    Convert an arbitrary value to a 4-decimal float string.

    Args:
        val (Any):
            Any value that can be cast with float().

    Returns:
        str:
            "X.XXXX" when val is a finite number,
            "n/a"    when val is non-finite or cannot be converted.
    """
    try:
        f = float(val)
        return f"{f:.4f}" if np.isfinite(f) else "n/a"
    except Exception:
        return "n/a"


def _append_stats_block(
    lines: List[str],
    loaded: Dict[str, np.ndarray],
    threshold: float,
    split_label: str,
) -> None:
    """
    Build and append the per-class accuracy block to the eer.txt line list.

    The block format is:
        <split_label> eval stats
        Threshold@EER: X.XXXXXX
        Attack  | Correct / Total
        <attack> | <correct> / <total>
        …
        bonafide | <correct> / <total>
        [blank line]
        Codec | Correct / Total          ← only when codec data is available
        <codec> | <correct> / <total>
        …
        [blank line]

    Attacks are sorted alphabetically with 'bonafide' forced to last.
    Codecs are sorted alphabetically with '-' forced to first.
    If _compute_attack_stats returns an empty dict the entire block is omitted.

    Args:
        lines (List[str]):
            The in-progress list of eer.txt lines.  Modified in-place.
        loaded (Dict[str, np.ndarray]):
            Enriched + relabelled sample dict from _load_and_enrich.
        threshold (float):
            Score threshold at EER, used to binarise predictions.
        split_label (str):
            Prefix for the stats header, e.g. "Evaluation set" or "Dev set".

    Returns:
        None  (lines is modified in-place).
    """
    atk_stats   = _compute_attack_stats(loaded, threshold)
    codec_stats = _compute_codec_stats(loaded, threshold)
    if not atk_stats:
        print(f"    [Stats] no attack stats computed — skipping stats block")
        return

    lines.append(f"{split_label} eval stats")
    lines.append(f"Threshold@EER: {threshold:.6f}")
    lines.append("Attack  | Correct / Total")
    for atk in sorted(atk_stats, key=lambda x: "~" if x == "bonafide" else x):
        s = atk_stats[atk]
        lines.append(f"{atk} | {s['correct']} / {s['total']}")

    if codec_stats:
        lines.append("")
        lines.append("Codec | Correct / Total")
        for c in sorted(codec_stats, key=lambda x: "" if x == "-" else x):
            s = codec_stats[c]
            lines.append(f"{c} | {s['correct']} / {s['total']}")

    lines.append("")


# =============================================================================
# Completion checks and file parsing
# =============================================================================

def _has_eer_ci(eer_txt: Path, has_dev: bool, has_eval: bool) -> bool:
    """
    Check whether the EER+CI metric lines have already been written.

    Searches for the sentinel strings "Dev set EER+CI none (%)" and
    "Evaluation set EER+CI none (%)" in the existing file.

    Args:
        eer_txt (Path):   Path to the candidate *_eer.txt file.
        has_dev (bool):   Whether a dev CSV exists for this checkpoint.
        has_eval (bool):  Whether an eval CSV exists for this checkpoint.

    Returns:
        bool:
            True when the file exists AND contains the expected EER+CI line(s)
            for every split that has a CSV.  False otherwise.
    """
    if not eer_txt.is_file():
        return False
    content = eer_txt.read_text(encoding="utf-8")
    dev_ok  = (not has_dev)  or ("Dev set EER+CI none (%)" in content)
    eval_ok = (not has_eval) or ("Evaluation set EER+CI none (%)" in content)
    return dev_ok and eval_ok


def _has_eval_stats(eer_txt: Path, has_eval: bool) -> bool:
    """
    Check whether the evaluation attack-stats block has already been written.

    Searches for the sentinel string "Evaluation set eval stats".

    Args:
        eer_txt (Path):   Path to the candidate *_eer.txt file.
        has_eval (bool):  Whether an eval CSV exists for this checkpoint.
                          When False the function immediately returns True
                          (nothing to check).

    Returns:
        bool:
            True when no eval set exists, OR when the file already contains
            the "Evaluation set eval stats" header.
    """
    if not has_eval:
        return True
    if not eer_txt.is_file():
        return False
    return "Evaluation set eval stats" in eer_txt.read_text(encoding="utf-8")


def _is_complete(eer_txt: Path, has_dev: bool, has_eval: bool) -> bool:
    """
    Return True when the eer.txt is fully complete: both EER+CI lines AND the
    attack-stats block are present (for every split that has a CSV).

    This is the top-level gate: when True the checkpoint is skipped entirely.

    Args:
        eer_txt (Path):   Path to the candidate *_eer.txt file.
        has_dev (bool):   Whether a dev CSV exists for this checkpoint.
        has_eval (bool):  Whether an eval CSV exists for this checkpoint.

    Returns:
        bool:
            True only when _has_eer_ci AND _has_eval_stats both return True.
    """
    return _has_eer_ci(eer_txt, has_dev, has_eval) and _has_eval_stats(eer_txt, has_eval)


def _extract_metric_lines(eer_txt: Path) -> Tuple[List[str], Optional[float]]:
    """
    Parse an existing eer.txt and extract the metric header plus the threshold.

    Used by the fast path when EER+CI is already present but the stats block
    is missing: we preserve the existing metric lines verbatim and append only
    the newly-computed stats.

    The "metric header" is defined as all lines up to and including the last
    line that starts with "Evaluation scores file:" or "Dev scores file:".
    Any existing stats block that appears after that line is discarded.

    The Threshold@EER is parsed from a "Threshold@EER: X" line if one is
    present in the existing file; it may appear inside the old stats block.

    Args:
        eer_txt (Path):
            Path to the existing *_eer.txt file.

    Returns:
        Tuple[List[str], Optional[float]]:
            (preserved_lines, threshold)
            preserved_lines – list of raw line strings (no trailing newlines)
                              up to and including the last scores-file line.
            threshold       – float value parsed from Threshold@EER, or None
                              when that line is absent or unparseable.
            Returns ([], None) when the file does not exist.
    """
    if not eer_txt.is_file():
        return [], None
    raw = eer_txt.read_text(encoding="utf-8").splitlines()
    cutoff = 0
    threshold = None
    for i, line in enumerate(raw):
        if line.startswith("Evaluation scores file:") or line.startswith("Dev scores file:"):
            cutoff = i + 1
        if line.startswith("Threshold@EER:"):
            try:
                threshold = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return raw[:cutoff], threshold


# =============================================================================
# Per-checkpoint pipeline
# =============================================================================

def _load_and_enrich(
    csv_path: Path,
    protocol_path: Optional[str],
    file_id_col: int,
    attack_col: int,
    codec_col: Optional[int],
    split_label: str,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load a scores CSV and run the full enrichment pipeline on it:
    protocol map injection → bonafide relabelling.

    This helper consolidates the three steps (_load_scores_csv,
    _enrich_from_protocol, _label_bonafide_attacks) that are applied
    identically to both the dev and eval splits.

    Args:
        csv_path (Path):
            Path to the *_scores_dev.csv or *_scores_eval.csv file.
        protocol_path (Optional[str]):
            Absolute path to the protocol file, or None to skip enrichment.
        file_id_col (int):
            Column index of the file ID in the protocol file.
        attack_col (int):
            Column index of the attack type in the protocol file.
        codec_col (Optional[int]):
            Column index of the codec in the protocol file, or None.
        split_label (str):
            Human-readable label for print statements ("dev" or "eval").

    Returns:
        Optional[Dict[str, np.ndarray]]:
            Fully enriched sample dict ready for CI computation and stats,
            or None if the CSV could not be loaded.
    """
    print(f"\n  --- Load + Enrich [{split_label}] ---")

    print(f"    [CSV] Loading {csv_path.name} ...")
    loaded = _load_scores_csv(csv_path)
    if loaded is None:
        print(f"    [CSV] WARN: failed to load — skipping {split_label}")
        return None

    if protocol_path:
        print(f"    [Protocol] Loading {Path(protocol_path).name} ...")
        proto_map = _load_protocol_map(protocol_path, file_id_col, attack_col, codec_col)
        loaded = _enrich_from_protocol(loaded, proto_map, has_codec=(codec_col is not None))
    else:
        print(f"    [Protocol] No protocol path — attack/codec columns unchanged")

    loaded = _label_bonafide_attacks(loaded)
    return loaded


def process_checkpoint(
    ckpt_name: str,
    scores_dir: Path,
    overwrite: bool,
    dev_protocol: Optional[str] = None,
    eval_protocol: Optional[str] = None,
    dataset_name_hint: Optional[str] = None,
) -> None:
    """
    Full processing pipeline for a single checkpoint.

    Determines which work needs to be done (skip / fast path / full path),
    runs the appropriate steps, and writes the *_eer.txt file.

    Decision logic:
        1. Both CSVs missing                       → skip entirely.
        2. eer.txt is fully complete, no overwrite → skip entirely.
        3. EER+CI already done, stats missing      → FAST PATH:
             read threshold from file if present, otherwise compute it
             cheaply from eval scores (raw argmin, no bootstrap);
             then compute stats only — no bootstrap at all.
        4. Otherwise                               → FULL PATH:
             load CSVs, enrich, run 3×bootstrap CI, compute stats, write.

    Args:
        ckpt_name (str):
            Checkpoint name, e.g. "checkpoint-epoch8".  Used to build the CSV
            and eer.txt filenames (<ckpt_name>_scores_dev.csv, etc.).
        scores_dir (Path):
            Directory that contains the CSV files and will receive the eer.txt.
        overwrite (bool):
            When True, always run the full path even if the file is complete.
        dev_protocol (Optional[str]):
            Explicit path to the dev protocol file.  When None (default),
            _auto_detect_protocols is called to find it automatically.
        eval_protocol (Optional[str]):
            Explicit path to the eval protocol file.  Same auto-detect rule.
        dataset_name_hint (Optional[str]):
            Override the dataset name used for column-index lookup in
            _PROTOCOL_COLS.  When None, resolution order is:
            (1) existing eer.txt "Dataset:" line;
            (2) heuristic — "asvspoof2019" or "asvspoof2021" substring
                found in scores_dir path;
            (3) hard-coded default "asvspoof5".

    Returns:
        None.  Side effect: writes or updates <ckpt_name>_eer.txt.
    """
    dev_csv  = scores_dir / f"{ckpt_name}_scores_dev.csv"
    eval_csv = scores_dir / f"{ckpt_name}_scores_eval.csv"
    eer_txt  = scores_dir / f"{ckpt_name}_eer.txt"

    has_dev  = dev_csv.is_file()
    has_eval = eval_csv.is_file()

    # ---- pre-flight checks ----
    if not has_dev and not has_eval:
        print(f"  [SKIP] {ckpt_name} — no CSV files found")
        return

    if not overwrite and _is_complete(eer_txt, has_dev, has_eval):
        print(f"  [SKIP] {ckpt_name} — eer.txt already complete  (use --overwrite to rerun)")
        return

    eer_done   = _has_eer_ci(eer_txt, has_dev, has_eval)
    stats_done = _has_eval_stats(eer_txt, has_eval)
    fast_path  = (not overwrite) and eer_done and (not stats_done) and has_eval

    # ---- resolve dataset name ----
    # Priority: CLI --dataset hint > existing eer.txt "Dataset:" line
    #         > heuristic from directory name > "asvspoof5" default
    dataset_name = dataset_name_hint  # None until resolved below
    if dataset_name_hint is None:
        # 1. Try to recover from the existing file (most reliable)
        if eer_txt.is_file():
            for line in eer_txt.read_text(encoding="utf-8").splitlines():
                if line.startswith("Dataset:"):
                    dataset_name = line.split(":", 1)[1].strip()
                    break
        # 2. If not found in file (missing or no Dataset: line), use directory heuristic
        if dataset_name is None:
            dir_lower = str(scores_dir).lower()
            if "asvspoof2019" in dir_lower:
                dataset_name = "asvspoof2019"
                print(f"  [Dataset] Auto-detected 'asvspoof2019' from directory name")
            elif "asvspoof2021" in dir_lower:
                dataset_name = "asvspoof2021"
                print(f"  [Dataset] Auto-detected 'asvspoof2021' from directory name")
            else:
                dataset_name = "asvspoof5"
                print(f"  [Dataset] Using default 'asvspoof5'  (pass --dataset to override)")

    # ---- header print ----
    print(f"\n{'='*62}")
    mode = "FAST (stats only)" if fast_path else "FULL (EER+CI+stats)"
    print(f"  {ckpt_name}  [{mode}]")
    print(f"  dataset    : {dataset_name}")
    print(f"  scores_dir : {scores_dir}")
    print(f"  has_dev    : {has_dev}  ({dev_csv.name})")
    print(f"  has_eval   : {has_eval}  ({eval_csv.name})")
    print(f"  eer_done   : {eer_done}  |  stats_done: {stats_done}  |  overwrite: {overwrite}")

    # ---- protocol columns ----
    file_id_col, attack_col, codec_col = _PROTOCOL_COLS.get(
        dataset_name.lower(), (1, 7, None)
    )
    print(f"  proto cols : file_id={file_id_col}  attack={attack_col}  codec={codec_col}")

    # ---- protocol path resolution ----
    if dev_protocol is None and eval_protocol is None:
        print(f"\n  --- Protocol auto-detection ---")
        dev_protocol, eval_protocol = _auto_detect_protocols(dataset_name, scores_dir)
    else:
        print(f"\n  --- Protocol (CLI-supplied) ---")
        print(f"    dev  : {dev_protocol or '(none)'}")
        print(f"    eval : {eval_protocol or '(none)'}")

    # =========================================================================
    # FAST PATH — EER+CI already done, only recompute missing stats block
    # =========================================================================
    if fast_path:
        print(f"\n  --- Fast path: parse threshold from existing file ---")
        preserved_lines, existing_thr = _extract_metric_lines(eer_txt)
        if not preserved_lines:
            print(f"    [WARN] Could not extract metric lines from existing eer.txt — falling back to FULL path")
            fast_path = False
        elif existing_thr is None:
            # Threshold@EER is written only inside the stats block.  When stats
            # were never computed (the exact scenario that triggers fast path),
            # the line is absent.  We compute it cheaply after loading the CSV.
            print(f"    [INFO] Threshold@EER not in file — will compute from eval scores  "
                  f"({len(preserved_lines)} header lines preserved)")
        else:
            print(f"    Threshold@EER (from file) = {existing_thr:.6f}  "
                  f"({len(preserved_lines)} header lines preserved)")

    if fast_path:
        eval_loaded = _load_and_enrich(
            eval_csv, eval_protocol, file_id_col, attack_col, codec_col, "eval"
        )
        if eval_loaded is None:
            print(f"  [WARN] Could not load eval CSV — aborting")
            print(f"{'='*62}")
            return

        # Compute threshold from raw EER when it was absent in the existing file
        # (no bootstrap — just the argmin|FPR-FNR| computation, very fast).
        if existing_thr is None or not np.isfinite(existing_thr):
            print(f"\n  --- Fast path: computing threshold from eval scores (no bootstrap) ---")
            _, existing_thr = _compute_raw_eer(
                eval_loaded["labels"], eval_loaded["scores"], return_threshold=True
            )
            if not np.isfinite(existing_thr):
                print(f"    [WARN] Could not compute threshold (empty or single-class?) — skipping")
                print(f"{'='*62}")
                return
            print(f"    Threshold@EER (computed) = {existing_thr:.6f}")

        print(f"\n  --- Computing stats (no bootstrap) ---")
        lines = list(preserved_lines)
        _append_stats_block(lines, eval_loaded, existing_thr, "Evaluation set")

        print(f"\n  --- Writing eer.txt ---")
        print(f"    path  : {eer_txt}")
        print(f"    lines : {len(lines)}")
        eer_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"    [OK] written")
        print(f"{'='*62}")
        return

    # =========================================================================
    # FULL PATH — load CSVs, enrich, run bootstrap CI, write complete file
    # =========================================================================

    # ---- load + enrich ----
    dev_loaded  = _load_and_enrich(dev_csv,  dev_protocol,  file_id_col, attack_col, codec_col, "dev")  if has_dev  else None
    eval_loaded = _load_and_enrich(eval_csv, eval_protocol, file_id_col, attack_col, codec_col, "eval") if has_eval else None

    if dev_loaded is None and eval_loaded is None:
        print(f"  [WARN] All CSV loads failed — nothing to write, skipping")
        print(f"{'='*62}")
        return

    # ---- bootstrap CI ----
    print(f"\n  --- Bootstrap CI ---")
    dev_triplet  = _compute_triplet(dev_loaded,  "dev")  if dev_loaded  is not None else None
    eval_triplet = _compute_triplet(eval_loaded, "eval") if eval_loaded is not None else None

    # ---- build file content ----
    print(f"\n  --- Building eer.txt ---")
    lines: List[str] = [
        f"Checkpoint: {ckpt_name}",
        f"Dataset: {dataset_name}",
        f"Device: cpu",
    ]

    if dev_loaded is not None and dev_triplet is not None:
        n   = len(dev_loaded["scores"])
        thr = dev_triplet["none"].get("threshold", np.nan)
        lines.append(f"Dev set EER (%): {_fmt_float(dev_triplet['none']['eer_percent'])}")
        lines.append(f"Dev set EER+CI none (%): {_fmt_eer_ci(dev_triplet['none'])}")
        lines.append(f"Dev set EER+CI speaker-conditioned (%): {_fmt_eer_ci(dev_triplet['speaker'])}")
        lines.append(f"Dev set EER+CI attack-conditioned (%): {_fmt_eer_ci(dev_triplet['attack'])}")
        lines.append(f"Dev scores file: {dev_csv} (rows={n})")
        print(f"    dev  EER: {_fmt_float(dev_triplet['none']['eer_percent'])}%  "
              f"threshold={thr:.6f}")
        print(f"\n  --- Computing dev stats ---")
        _append_stats_block(lines, dev_loaded, thr, "Dev set")

    if eval_loaded is not None and eval_triplet is not None:
        n   = len(eval_loaded["scores"])
        thr = eval_triplet["none"].get("threshold", np.nan)
        lines.append(f"Evaluation set EER (%): {_fmt_float(eval_triplet['none']['eer_percent'])}")
        lines.append(f"Evaluation set EER+CI none (%): {_fmt_eer_ci(eval_triplet['none'])}")
        lines.append(f"Evaluation set EER+CI speaker-conditioned (%): {_fmt_eer_ci(eval_triplet['speaker'])}")
        lines.append(f"Evaluation set EER+CI attack-conditioned (%): {_fmt_eer_ci(eval_triplet['attack'])}")
        lines.append(f"Evaluation scores file: {eval_csv} (rows={n})")
        print(f"    eval EER: {_fmt_float(eval_triplet['none']['eer_percent'])}%  "
              f"threshold={thr:.6f}")
        print(f"\n  --- Computing stats ---")
        _append_stats_block(lines, eval_loaded, thr, "Evaluation set")

    # ---- write ----
    print(f"\n  --- Writing eer.txt ---")
    print(f"    path  : {eer_txt}")
    print(f"    lines : {len(lines)}")
    eer_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    [OK] written")
    print(f"{'='*62}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Entry point: parse CLI arguments, discover checkpoints, and process each one.

    Steps:
        1. Parse arguments (scores_dir, --overwrite, --bootstraps,
           --dev-protocol, --eval-protocol).
        2. Print a startup banner summarising the run configuration.
        3. Glob scores_dir for *_scores_dev.csv and *_scores_eval.csv files
           to build the list of checkpoint names.
        4. Print the status of each checkpoint (complete / EER done / pending).
        5. Call process_checkpoint() for each checkpoint in sorted order.

    CLI arguments:
        scores_dir (positional, optional):
            Path to the scores_text_outputs directory.
            Defaults to _DEFAULT_SCORES_DIR.
        --overwrite:
            Force recomputation even when eer.txt is already complete.
        --bootstraps N:
            Override EER_NUM_BOOTSTRAPS for this run.
        --dev-protocol PATH:
            Explicit dev protocol file path (disables auto-detection).
        --eval-protocol PATH:
            Explicit eval protocol file path (disables auto-detection).
        --dataset NAME:
            Dataset name for protocol column-index lookup and path
            auto-detection (e.g. asvspoof5, asvspoof2019, asvspoof2021).
            When omitted the name is read from an existing eer.txt
            'Dataset:' line; defaults to 'asvspoof5' for new files.
            Must be supplied explicitly when running on a non-ASVspoof5
            experiment directory that has no existing eer.txt.

    Returns:
        None.  Exits with sys.exit(1) on invalid arguments,
        sys.exit(0) when no CSVs are found.
    """
    parser = argparse.ArgumentParser(
        description="Compute EER+CI from score CSVs and write *_eer.txt files. No GPU needed."
    )
    parser.add_argument(
        "scores_dir", nargs="?", default=_DEFAULT_SCORES_DIR,
        help=f"Path to scores_text_outputs directory (default: {_DEFAULT_SCORES_DIR})",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute even if eer.txt is already complete",
    )
    parser.add_argument(
        "--bootstraps", type=int, default=None,
        help="Override EER_NUM_BOOTSTRAPS (e.g. --bootstraps 200 for a quick test)",
    )
    parser.add_argument(
        "--dev-protocol", default=None, metavar="PATH",
        help="Protocol file for the dev split (overrides auto-detection)",
    )
    parser.add_argument(
        "--eval-protocol", default=None, metavar="PATH",
        help="Protocol file for the eval split (overrides auto-detection)",
    )
    parser.add_argument(
        "--dataset", default=None, metavar="NAME",
        help=(
            "Dataset name for protocol column-index lookup and auto-detection "
            "(e.g. asvspoof5, asvspoof2019, asvspoof2021).  "
            "When omitted, the name is read from an existing eer.txt 'Dataset:' line; "
            "falls back to 'asvspoof5' for new files."
        ),
    )
    args = parser.parse_args()

    global EER_NUM_BOOTSTRAPS
    if args.bootstraps is not None:
        EER_NUM_BOOTSTRAPS = args.bootstraps

    scores_dir = Path(args.scores_dir)
    if not scores_dir.is_dir():
        print(f"ERROR: '{scores_dir}' is not a directory")
        sys.exit(1)

    # ---- startup banner ----
    print("=" * 62)
    print("compute_ci_from_csv.py")
    print(f"  scores_dir  : {scores_dir}")
    print(f"  dataset     : {args.dataset or '(auto-detect from eer.txt / default asvspoof5)'}")
    print(f"  bootstraps  : {EER_NUM_BOOTSTRAPS}")
    print(f"  alpha       : {EER_ALPHA}  → [{EER_ALPHA/2:.1f}%, {100-EER_ALPHA/2:.1f}%] CI")
    print(f"  overwrite   : {args.overwrite}")
    if _CI_AVAILABLE:
        print(f"  CI library  : available")
    else:
        print(f"  CI library  : NOT FOUND — CI values will be n/a")
        print(f"    error     : {_CI_IMPORT_ERROR}")
        print(f"    fix       : set CONFIDENCE_INTERVALS_DIR to the package directory")
    print("=" * 62)

    # ---- discover checkpoints ----
    ckpt_names: set = set()
    for f in scores_dir.glob("*_scores_dev.csv"):
        ckpt_names.add(f.name.replace("_scores_dev.csv", ""))
    for f in scores_dir.glob("*_scores_eval.csv"):
        ckpt_names.add(f.name.replace("_scores_eval.csv", ""))

    if not ckpt_names:
        print(f"No *_scores_dev.csv / *_scores_eval.csv files found in {scores_dir}")
        sys.exit(0)

    sorted_names = sorted(ckpt_names)
    print(f"\nFound {len(sorted_names)} checkpoint(s):")
    for name in sorted_names:
        eer_txt  = scores_dir / f"{name}_eer.txt"
        has_dev  = (scores_dir / f"{name}_scores_dev.csv").is_file()
        has_eval = (scores_dir / f"{name}_scores_eval.csv").is_file()
        status = (
            "complete"              if _is_complete(eer_txt, has_dev, has_eval) else
            "EER done/stats missing" if _has_eer_ci(eer_txt, has_dev, has_eval) else
            "pending"
        )
        print(f"  {name}  [{status}]")
    print()

    # ---- process ----
    for ckpt_name in sorted_names:
        process_checkpoint(
            ckpt_name, scores_dir,
            overwrite=args.overwrite,
            dev_protocol=args.dev_protocol,
            eval_protocol=args.eval_protocol,
            dataset_name_hint=args.dataset,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
