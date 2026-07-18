#!/usr/bin/env python3
"""
Aggregate per-file embedding .npy files into a single .npz bundle.

The script reads a metadata CSV that points to embedding files and merges
metadata fields (label/attack/speaker_id/codec). Missing fields can be
filled from the official ASVspoof protocol files to keep a consistent
schema across ASVspoof2019/2021/5. The output NPZ contains:
  - embeddings
  - labels
  - attacks
  - speaker_ids
  - codecs
  - file_ids (optional)
"""

import csv
import os
import re

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def _infer_set_name(metadata_path):
    """Infer set name (train/dev/eval) from metadata filename."""
    base = os.path.basename(metadata_path)
    m = re.search(r"_(train|dev|eval)_voxtral_all_embeddings_metadata\.csv$", base)
    if m:
        return m.group(1)
    return None


def _infer_track_from_file_id(file_id):
    """Infer track name (LA/PA/DF) from file_id prefix, if present."""
    if not file_id:
        return None
    prefix = file_id.split('_', 1)[0]
    if prefix in {'LA', 'PA', 'DF'}:
        return prefix
    return None


def _infer_track_from_path(path, pattern):
    """Infer track name using a regex pattern on a path."""
    if not path:
        return None
    m = re.search(pattern, path)
    if m:
        return m.group(1)
    return None


def _infer_dataset(metadata_path, audio_path, file_id):
    """Infer dataset family (asvspoof2019/2021/5) and track, if possible."""
    track = _infer_track_from_file_id(file_id)
    for value in (audio_path, metadata_path):
        if value and "ASVspoof2019" in value:
            track = track or _infer_track_from_path(value, r"ASVspoof2019_(LA|PA)")
            return "asvspoof2019", track
        if value and "ASVspoof2021" in value:
            track = track or _infer_track_from_path(value, r"ASVspoof2021_(LA|PA|DF)")
            return "asvspoof2021", track
        if value and "ASVspoof5" in value:
            return "asvspoof5", None
    if file_id and re.match(r"^[TDE]_", file_id):
        return "asvspoof5", None
    return None, track


def _infer_dataset_from_protocol_path(protocol_path):
    """Infer dataset family from a protocol file path."""
    if not protocol_path:
        return None
    if "ASVspoof2019" in protocol_path:
        return "asvspoof2019"
    if "ASVspoof2021" in protocol_path or protocol_path.endswith("trial_metadata.txt"):
        return "asvspoof2021"
    if "ASVspoof5" in protocol_path or protocol_path.endswith(".tsv"):
        return "asvspoof5"
    return None


def _infer_dataset_root(audio_path, dataset):
    """Infer dataset root path from an audio path and dataset name."""
    markers = {
        "asvspoof2019": "/ASVspoof2019",
        "asvspoof2021": "/ASVspoof2021",
        "asvspoof5": "/ASVspoof5",
    }
    marker = markers.get(dataset)
    if not marker or not audio_path or marker not in audio_path:
        return None
    return audio_path.split(marker)[0] + marker


def _infer_protocol_path(dataset, track, set_name, audio_path, metadata_path):
    """Construct the expected protocol path for a given dataset/track/set."""
    if not dataset:
        return None
    root = _infer_dataset_root(audio_path, dataset)
    if dataset == "asvspoof2019":
        if not track or not set_name:
            return None
        suffix = "trn" if set_name == "train" else "trl"
        filename = f"ASVspoof2019.{track}.cm.{set_name}.{suffix}.txt"
        if root:
            return os.path.join(
                root,
                track,
                track,
                f"ASVspoof2019_{track}_cm_protocols",
                filename,
            )
    elif dataset == "asvspoof2021":
        if not track:
            return None
        if root:
            return os.path.join(
                root,
                f"{track}-keys-full",
                "keys",
                track,
                "CM",
                "trial_metadata.txt",
            )
    elif dataset == "asvspoof5":
        if not set_name:
            return None
        if set_name == "train":
            filename = "ASVspoof5.train.tsv"
        elif set_name in {"dev", "eval"}:
            filename = f"ASVspoof5.{set_name}.track_1.tsv"
        else:
            return None
        if root:
            return os.path.join(root, "ASVspoof5_protocols", filename)
    if metadata_path and root is None:
        return None
    return None


def _load_protocol_map_2019(protocol_path):
    """Parse ASVspoof2019 CM protocol file into a dict keyed by file_id."""
    mapping = {}
    with open(protocol_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker_id, file_id, _, attack, label = parts[:5]
            mapping[file_id] = {
                "speaker_id": speaker_id,
                "attack": attack,
                "label": label,
            }
    return mapping


def _load_protocol_map_2021(protocol_path):
    """Parse ASVspoof2021 CM trial_metadata into a dict keyed by file_id."""
    mapping = {}
    with open(protocol_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            speaker_id = parts[0]
            file_id = parts[1]
            codec = parts[2]
            attack = parts[4]
            label = parts[5]
            mapping[file_id] = {
                "speaker_id": speaker_id,
                "attack": attack,
                "codec": codec,
                "label": label,
            }
    return mapping


def _load_protocol_map_5(protocol_path):
    """Parse ASVspoof5 track_1 TSV protocol into a dict keyed by file_id."""
    mapping = {}
    with open(protocol_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            speaker_id = parts[0]
            file_id = parts[1]
            codec = parts[3]
            attack = parts[7]
            label = parts[8]
            mapping[file_id] = {
                "speaker_id": speaker_id,
                "attack": attack,
                "codec": codec,
                "label": label,
            }
    return mapping


def _load_protocol_map(protocol_path, dataset):
    """Dispatch protocol parsing based on dataset family."""
    if dataset == "asvspoof2019":
        return _load_protocol_map_2019(protocol_path)
    if dataset == "asvspoof2021":
        return _load_protocol_map_2021(protocol_path)
    if dataset == "asvspoof5":
        return _load_protocol_map_5(protocol_path)
    raise ValueError(f"Unknown dataset for protocol parsing: {dataset}")


def _load_metadata_rows(metadata_path):
    """Load metadata rows and return rows plus flags about missing fields."""
    rows = []
    needs_attack = False
    needs_speaker_id = False
    needs_codec = False
    sample_audio_path = ""
    with open(metadata_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        if not reader.fieldnames:
            raise ValueError(f"Metadata CSV has no header: {metadata_path}")
        required_columns = {'embedding_path', 'label'}
        missing_columns = sorted(required_columns - set(reader.fieldnames))
        if missing_columns:
            raise ValueError(
                "Metadata CSV missing required columns "
                f"{missing_columns}: {metadata_path}"
            )
        fieldnames = set(reader.fieldnames)
        has_attack = 'attack' in fieldnames
        has_speaker_id = 'speaker_id' in fieldnames
        has_codec = 'codec' in fieldnames
        has_file_id = 'file_id' in fieldnames
        has_audio_path = 'audio_path' in fieldnames

        needs_attack = not has_attack
        needs_speaker_id = not has_speaker_id
        needs_codec = not has_codec

        for row in reader:
            emb_path = row.get('embedding_path')
            label = row.get('label')
            if not emb_path or label is None:
                continue
            if not os.path.exists(emb_path):
                raise FileNotFoundError(f"Missing embedding file: {emb_path}")

            attack = row.get('attack') if has_attack else None
            speaker_id = row.get('speaker_id') if has_speaker_id else None
            codec = row.get('codec') if has_codec else None
            audio_path = row.get('audio_path') if has_audio_path else ''

            if has_attack and (attack is None or attack == ''):
                needs_attack = True
            if has_speaker_id and (speaker_id is None or speaker_id == ''):
                needs_speaker_id = True
            if has_codec and (codec is None or codec == ''):
                needs_codec = True

            file_id = row.get('file_id') if has_file_id else ''
            if not file_id:
                if audio_path:
                    file_id = os.path.splitext(os.path.basename(audio_path))[0]
                else:
                    file_id = os.path.splitext(os.path.basename(emb_path))[0]

            if audio_path and not sample_audio_path:
                sample_audio_path = audio_path

            rows.append((emb_path, label, attack, speaker_id, codec, file_id, audio_path))
    if not rows:
        raise ValueError(f"No valid rows found in metadata CSV: {metadata_path}")
    meta = {
        "needs_attack": needs_attack,
        "needs_speaker_id": needs_speaker_id,
        "needs_codec": needs_codec,
        "sample_audio_path": sample_audio_path,
    }
    return rows, meta


def _iter_rows(rows, desc, use_progress):
    """Yield rows with an optional progress bar."""
    if use_progress and tqdm is not None:
        return tqdm(rows, total=len(rows), desc=desc, unit="file", dynamic_ncols=True)
    return rows


# ------------ PARAMETERS (EDIT HERE) ------------
METADATA_PATH = "/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/llama3/llama3_dev_voxtral_all_embeddings_metadata.csv"
OUTPUT_PATH = "/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/llama3/total_dev_embeddings.npz"
SET_NAME = "dev"  # Expected: train/dev/eval (used only if OUTPUT_PATH is empty)
DTYPE = "float32"
INCLUDE_FILE_IDS = True
NO_PROGRESS = False
PROTOCOL_PATH = "/gpfs0/bgu-benshimo/users/wavishay/projects/ASVspoof5/ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv"
# -----------------------------------------------


def main():
    """Run aggregation using module-level configuration."""
    metadata_path = METADATA_PATH
    output_path = OUTPUT_PATH
    set_name = SET_NAME
    dtype = DTYPE
    include_file_ids = INCLUDE_FILE_IDS
    no_progress = NO_PROGRESS

    if not metadata_path:
        raise ValueError("Set METADATA_PATH to the metadata CSV path.")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    valid_sets = {"train", "dev", "eval"}
    if set_name not in valid_sets:
        inferred_set_name = _infer_set_name(metadata_path)
        if inferred_set_name:
            set_name = inferred_set_name
        else:
            set_name = None
    if not output_path:
        if not set_name:
            raise ValueError(
                "Could not infer set name from metadata filename. "
                "Set SET_NAME or OUTPUT_PATH."
            )
        output_path = os.path.join(
            os.path.dirname(metadata_path),
            f"total_{set_name}_embeddings.npz",
        )

    rows, meta = _load_metadata_rows(metadata_path)

    sample_audio_path = meta["sample_audio_path"]
    first_file_id = rows[0][5]
    dataset, track = _infer_dataset(metadata_path, sample_audio_path, first_file_id)
    protocol_path = PROTOCOL_PATH or ""
    if protocol_path and not dataset:
        dataset = _infer_dataset_from_protocol_path(protocol_path)

    # Determine embedding dimension from first item.
    first_emb = np.load(rows[0][0])
    first_emb = np.asarray(first_emb).reshape(-1)
    emb_dim = first_emb.shape[0]
    dtype = np.dtype(dtype)

    embeddings = np.empty((len(rows), emb_dim), dtype=dtype)
    labels = []
    attacks = []
    speaker_ids = []
    codecs = []
    file_ids = []

    needs_protocol = (
        meta["needs_attack"]
        or meta["needs_speaker_id"]
        or (meta["needs_codec"] and dataset != "asvspoof2019")
    )
    if needs_protocol:
        if not dataset:
            raise ValueError(
                "Could not infer dataset type for protocol lookup. "
                "Set PROTOCOL_PATH explicitly."
            )
        if not protocol_path:
            if dataset in {"asvspoof2019", "asvspoof5"} and not set_name:
                raise ValueError(
                    "Set SET_NAME to train/dev/eval to locate protocol files."
                )
            protocol_path = _infer_protocol_path(
                dataset, track, set_name, sample_audio_path, metadata_path
            )
        if not protocol_path or not os.path.isfile(protocol_path):
            raise FileNotFoundError(
                "Protocol file not found. "
                "Set PROTOCOL_PATH to the correct protocol file."
            )
        protocol_map = _load_protocol_map(protocol_path, dataset)
    else:
        protocol_map = {}

    missing_attack = []
    missing_speaker = []
    missing_codec = []
    missing_label = []

    desc = f"Packing {os.path.basename(output_path)}"
    for idx, (emb_path, label, attack, speaker_id, codec, file_id, _) in enumerate(
        _iter_rows(rows, desc=desc, use_progress=not no_progress)
    ):
        emb = np.load(emb_path)
        emb = np.asarray(emb).reshape(-1)
        if emb.shape[0] != emb_dim:
            raise ValueError(
                f"Embedding dimension mismatch in {emb_path}: "
                f"expected {emb_dim}, got {emb.shape[0]}"
            )
        proto = protocol_map.get(file_id) if protocol_map else None

        if label is None or label == "":
            if proto:
                label = proto.get("label")
        if attack is None or attack == "":
            if proto:
                attack = proto.get("attack")
        if speaker_id is None or speaker_id == "":
            if proto:
                speaker_id = proto.get("speaker_id")

        if dataset == "asvspoof2019":
            codec = "-"
        else:
            if codec is None or codec == "":
                if proto:
                    codec = proto.get("codec")

        if label is None or label == "":
            missing_label.append(file_id)
        if attack is None or attack == "":
            missing_attack.append(file_id)
        if speaker_id is None or speaker_id == "":
            missing_speaker.append(file_id)
        if dataset != "asvspoof2019" and (codec is None or codec == ""):
            missing_codec.append(file_id)

        embeddings[idx] = emb.astype(dtype, copy=False)
        labels.append(label)
        attacks.append(attack)
        speaker_ids.append(speaker_id)
        codecs.append(codec)
        if include_file_ids:
            file_ids.append(file_id)

    if missing_label or missing_attack or missing_speaker or missing_codec:
        def _fmt_missing(name, items):
            if not items:
                return None
            sample = ", ".join(items[:5])
            return f"{name}: {len(items)} (e.g., {sample})"

        details = list(
            filter(
                None,
                [
                    _fmt_missing("label", missing_label),
                    _fmt_missing("attack", missing_attack),
                    _fmt_missing("speaker_id", missing_speaker),
                    _fmt_missing("codec", missing_codec),
                ],
            )
        )
        raise ValueError(
            "Missing fields after CSV/protocol merge. " + " | ".join(details)
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if include_file_ids:
        np.savez(
            output_path,
            embeddings=embeddings,
            labels=np.array(labels, dtype=str),
            attacks=np.array(attacks, dtype=str),
            speaker_ids=np.array(speaker_ids, dtype=str),
            codecs=np.array(codecs, dtype=str),
            file_ids=np.array(file_ids, dtype=str),
        )
    else:
        np.savez(
            output_path,
            embeddings=embeddings,
            labels=np.array(labels, dtype=str),
            attacks=np.array(attacks, dtype=str),
            speaker_ids=np.array(speaker_ids, dtype=str),
            codecs=np.array(codecs, dtype=str),
        )

    print(f"Saved NPZ: {output_path}")
    print(f"Embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
