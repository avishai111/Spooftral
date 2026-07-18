#!/usr/bin/env python3
"""
Verify embeddings stored in an NPZ bundle created by aggregate_embeddings_to_npz.py.

Checks:
  - "embeddings" exists and is a 2D float array
  - no NaN/Inf values
  - metadata arrays (labels/attacks/speaker_ids/codecs/file_ids) have matching length
  - optional expected embedding dimension / count
  - basic sanity stats and warnings
"""

import os
import sys

import numpy as np


# ------------ PARAMETERS (EDIT HERE) ------------
NPZ_PATH = "/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama/total_train_embeddings.npz"  # Path to the .npz file to verify
EXPECTED_DIM = None  # e.g., 512
EXPECTED_COUNT = None  # e.g., 12345
REQUIRE_FILE_IDS = False
STRICT = False  # Treat warnings as errors
NO_STATS = False  # Skip printing min/mean/std stats
# -----------------------------------------------


def _is_string_array(arr):
    return arr.dtype.kind in {"U", "S", "O"}


def _count_missing_strings(arr):
    if arr.dtype.kind in {"U", "S"}:
        return int(np.sum(arr == ""))
    if arr.dtype.kind == "O":
        try:
            return int(np.sum((arr == None) | (arr == "")))  # noqa: E711
        except Exception:
            return 0
    return 0


def _check_meta_array(
    data,
    key,
    expected_len,
    required,
    errors,
    warnings,
):
    if key not in data.files:
        if required:
            errors.append(f"Missing required key: {key}")
        else:
            warnings.append(f"Missing optional key: {key}")
        return None
    arr = data[key]
    if arr.ndim != 1:
        errors.append(f"Key {key} expected 1D array, got shape {arr.shape}")
    if arr.shape[0] != expected_len:
        errors.append(
            f"Key {key} length mismatch: expected {expected_len}, got {arr.shape[0]}"
        )
    if not _is_string_array(arr):
        warnings.append(f"Key {key} dtype is not string-like: {arr.dtype}")
    missing = _count_missing_strings(arr)
    if missing:
        warnings.append(f"Key {key} has {missing} empty entries")
    return arr


def main():
    npz_path = NPZ_PATH
    expected_dim = EXPECTED_DIM
    expected_count = EXPECTED_COUNT
    require_file_ids = REQUIRE_FILE_IDS
    strict = STRICT
    no_stats = NO_STATS

    if not npz_path:
        raise ValueError("Set NPZ_PATH to the .npz file path.")

    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    errors = []
    warnings = []
    embeddings = None
    emb_min = emb_max = emb_mean = emb_std = None

    with np.load(npz_path, allow_pickle=False) as data:
        if "embeddings" not in data.files:
            errors.append("Missing required key: embeddings")
        else:
            embeddings = data["embeddings"]
            if embeddings.ndim != 2:
                errors.append(
                    f"Embeddings expected 2D array, got shape {embeddings.shape}"
                )
            else:
                n, d = embeddings.shape
                if n == 0 or d == 0:
                    errors.append(f"Embeddings shape is invalid: {embeddings.shape}")
                if not np.issubdtype(embeddings.dtype, np.floating):
                    errors.append(
                        f"Embeddings dtype is not float: {embeddings.dtype}"
                    )
                if expected_dim is not None and d != expected_dim:
                    errors.append(
                        f"Embedding dim mismatch: expected {expected_dim}, got {d}"
                    )
                if expected_count is not None and n != expected_count:
                    errors.append(
                        f"Embedding count mismatch: expected {expected_count}, got {n}"
                    )
                if np.issubdtype(embeddings.dtype, np.floating):
                    finite_mask = np.isfinite(embeddings)
                    if not np.all(finite_mask):
                        bad = int(np.size(finite_mask) - np.sum(finite_mask))
                        errors.append(f"Embeddings contain {bad} NaN/Inf values")

                if n > 0 and d > 0:
                    zero_rows = int(np.sum(np.all(embeddings == 0, axis=1)))
                    if zero_rows:
                        warnings.append(
                            f"Embeddings contain {zero_rows} all-zero rows"
                        )

                if not no_stats and n > 0 and d > 0:
                    emb_min = float(np.min(embeddings))
                    emb_max = float(np.max(embeddings))
                    emb_mean = float(np.mean(embeddings))
                    emb_std = float(np.std(embeddings))

        if embeddings is not None and embeddings.ndim == 2:
            expected_len = embeddings.shape[0]
        else:
            expected_len = None

        if expected_len is not None:
            _check_meta_array(
                data,
                "labels",
                expected_len,
                required=True,
                errors=errors,
                warnings=warnings,
            )
            _check_meta_array(
                data,
                "attacks",
                expected_len,
                required=True,
                errors=errors,
                warnings=warnings,
            )
            _check_meta_array(
                data,
                "speaker_ids",
                expected_len,
                required=True,
                errors=errors,
                warnings=warnings,
            )
            _check_meta_array(
                data,
                "codecs",
                expected_len,
                required=True,
                errors=errors,
                warnings=warnings,
            )
            file_ids = _check_meta_array(
                data,
                "file_ids",
                expected_len,
                required=require_file_ids,
                errors=errors,
                warnings=warnings,
            )
            if file_ids is not None:
                try:
                    unique_count = np.unique(file_ids).size
                    if unique_count != expected_len:
                        warnings.append(
                            f"file_ids contain duplicates: {expected_len - unique_count} duplicates"
                        )
                except Exception:
                    warnings.append("Could not check file_ids for duplicates")

        keys = ", ".join(sorted(data.files))
        print(f"NPZ: {npz_path}")
        print(f"Keys: {keys}")
        if embeddings is not None and embeddings.ndim == 2:
            print(f"Embeddings shape: {embeddings.shape}")
            print(f"Embeddings dtype: {embeddings.dtype}")
            if not no_stats and emb_min is not None:
                print(
                    "Embeddings stats: "
                    f"min={emb_min:.6g}, max={emb_max:.6g}, "
                    f"mean={emb_mean:.6g}, std={emb_std:.6g}"
                )

    for msg in warnings:
        print(f"WARNING: {msg}")
    for msg in errors:
        print(f"ERROR: {msg}", file=sys.stderr)

    if errors:
        sys.exit(1)
    if strict and warnings:
        sys.exit(2)
    print("OK: NPZ embeddings look consistent.")


if __name__ == "__main__":
    main()
