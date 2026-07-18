import contextlib
import csv
import gc
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_curve
from transformers import AutoProcessor

from utils.build_model import VoxtralStageWrapper_classification, build_voxtral_model, CACHE_DIR as _MODEL_CACHE_DIR
from utils.build_model_for_classification_head import build_classification_head_model
from utils.config_helper_functions import extract_label_texts_from_gen_cfg
from utils.dataset import ASVspoofIndexDataset, collate_fn, make_dataloader
from utils.eval_classification import evaluate_epoch_classification_as_generation
from utils.freezing import verify_model_frozen
from utils.helper_function import load_text_prompts, maybe_log_eval_stats


# confidence_intervals settings (configurable via environment variables)
CONFIDENCE_INTERVALS_DIR = os.environ.get("CONFIDENCE_INTERVALS_DIR", "").strip()
EER_NUM_BOOTSTRAPS = int(os.environ.get("EER_NUM_BOOTSTRAPS", "1000"))
EER_ALPHA = float(os.environ.get("EER_ALPHA", "5"))
EER_MAX_RESAMPLE_RETRIES = int(os.environ.get("EER_MAX_RESAMPLE_RETRIES", "50"))

if CONFIDENCE_INTERVALS_DIR:
    _ci_dir = os.path.abspath(os.path.expanduser(CONFIDENCE_INTERVALS_DIR))
    # Python needs the parent directory in sys.path, not the package folder itself
    if os.path.basename(_ci_dir).lower() == "confidence_intervals":
        _ci_dir = os.path.dirname(_ci_dir)
    if _ci_dir not in sys.path:
        sys.path.insert(0, _ci_dir)

_CONF_INT_IMPORT_ERROR = None
try:
    # confidence_intervals is an optional dependency for bootstrap CIs on EER;
    from confidence_intervals import evaluate_with_conf_int, get_bootstrap_indices
except Exception as _import_error:
    evaluate_with_conf_int = None
    get_bootstrap_indices = None
    _CONF_INT_IMPORT_ERROR = _import_error

_SCORE_KEYS = ("scores", "score", "y_score", "y_scores", "probabilities", "probability", "probs", "spoof_score")
_LABEL_KEYS = ("labels", "label", "y_true", "targets", "target", "truth")
_FILE_ID_KEYS = ("file_id", "file_ids", "file", "files", "utt_id", "utt_ids", "sample_id", "sample_ids", "id", "ids")
_SPEAKER_KEYS = ("speaker_id", "speaker_ids", "speaker", "speakers")
_ATTACK_KEYS = ("attack", "attacks")
_ROWS_KEYS = ("rows", "samples", "records", "entries", "items", "per_sample")


def _resolve_cfg(cfg: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    """Return the first matching key from a config dict, trying exact/upper/lower variants."""
    for key in list(keys):
        for cand in (key, str(key).upper(), str(key).lower()):
            if isinstance(cfg, dict) and cand in cfg:
                return cfg[cand]
    return default


def _normalize_encoder_type(encoder_type: Any) -> Optional[str]:
    """Normalize encoder type values coming from YAML/CLI/checkpoints."""
    if encoder_type is None:
        return None
    if isinstance(encoder_type, str):
        enc = encoder_type.strip().lower()
        if enc in ("", "none", "null"):
            return None
        return enc
    return None


def _load_config_dict(ckpt_payload: Any, cfg_path: Optional[str]) -> Dict[str, Any]:
    """Build a config dict by merging checkpoint-embedded configs with an optional YAML file."""
    cfg: Dict[str, Any] = {}
    if isinstance(ckpt_payload, dict) and "configs" in ckpt_payload and isinstance(ckpt_payload["configs"], dict):
        cfg = dict(ckpt_payload["configs"])
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        if isinstance(file_cfg, dict):
            cfg.update(file_cfg)
    return cfg


def _ensure_config_env(cfg_path: Optional[str]) -> str:
    """Resolve a valid config path and ensure VOXTRAL_CONFIG is set consistently."""
    if cfg_path:
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"config_path not found: {cfg_path}")
        os.environ["VOXTRAL_CONFIG"] = cfg_path
        return cfg_path
    env_path = os.environ.get("VOXTRAL_CONFIG")
    if env_path:
        if not os.path.exists(env_path):
            raise FileNotFoundError(f"VOXTRAL_CONFIG points to missing file: {env_path}")
        return env_path
    raise ValueError(
        "VOXTRAL_CONFIG is not set. Set 'config_path' in run_args (e.g., configs/hyperparameters_05.yaml) "
        "or export VOXTRAL_CONFIG."
    )


def _load_model_weights(model: torch.nn.Module, ckpt_payload: Any) -> None:
    """Load model weights from a checkpoint, resolving the state-dict key then loading strict=True."""
    state = None
    if isinstance(ckpt_payload, dict):
        for key in ("wrapper_state_dict", "model_state_dict", "state_dict"):
            if key in ckpt_payload:
                state = ckpt_payload[key]
                break
    if state is None:
        state = ckpt_payload
    # strict=True is mandatory — mismatched architectures must raise immediately (CLAUDE.md invariant 2)
    model.load_state_dict(state, strict=True)


def _detect_adapter_head_checkpoint(ckpt_payload: Any) -> bool:
    """
    Detect whether a checkpoint was saved from AudioAdapterHeadClassifier (adapter-head training)
    vs VoxtralStageWrapper_classification (classification-as-generation training).
    Adapter-head checkpoints have 'classification_head.*' keys in the state dict.
    """
    state = None
    if isinstance(ckpt_payload, dict):
        for key in ("wrapper_state_dict", "model_state_dict", "state_dict"):
            if key in ckpt_payload:
                state = ckpt_payload[key]
                break
    if state is None:
        state = ckpt_payload
    if not isinstance(state, dict):
        return False
    return any(k.startswith("classification_head.") for k in state.keys())


def _evaluate_adapter_head(
    model: torch.nn.Module,
    loader,
    device: str,
    use_amp: bool = False,
    amp_dtype: str = "fp16",
    encoder_type: Optional[str] = None,
    return_per_sample: bool = False,
    include_speaker_ids: bool = False,
    tqdm_desc: str = "Evaluating (adapter-head)",
) -> Dict[str, Any]:
    """
    Evaluate an AudioAdapterHeadClassifier model on a dataloader.
    Returns metrics dict with loss, eer (in percent), accuracy, avg_spoof_prob,
    and optionally per-sample scores/labels/file_ids/speaker_ids/attacks.
    """
    from tqdm import tqdm

    resolved_encoder_type = _normalize_encoder_type(encoder_type)
    if resolved_encoder_type is None and hasattr(model, "encoder_type"):
        resolved_encoder_type = _normalize_encoder_type(getattr(model, "encoder_type"))
    use_raw_audio = resolved_encoder_type is not None
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    losses, probs_list, labels_list = [], [], []
    file_ids_list, speaker_ids_list, attacks_list = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=tqdm_desc, leave=False):
            # Move only required tensors to device
            keys_to_device = {"classification_labels"}
            if use_raw_audio:
                keys_to_device.add("raw_audio_input")
            else:
                keys_to_device.add("input_features")

            device_batch = {}
            for k, v in batch.items():
                if torch.is_tensor(v) and k in keys_to_device:
                    device_batch[k] = v.to(device, non_blocking=True)
                else:
                    device_batch[k] = v

            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16 if amp_dtype == "bf16" else torch.float16)
                if use_amp and device.startswith("cuda")
                else contextlib.nullcontext()
            )
            with amp_context:
                if use_raw_audio:
                    raw_audio_lengths = device_batch.get("raw_audio_length")
                    if raw_audio_lengths is None:
                        raw_audio_lengths = device_batch.get("raw_audio_lengths")
                    outputs = model(
                        raw_audio_input=device_batch.get("raw_audio_input"),
                        raw_audio_lengths=raw_audio_lengths,
                    )
                else:
                    outputs = model(input_features=device_batch.get("input_features"))

                logits = outputs["logits"]
                loss = criterion(logits, device_batch["classification_labels"])

            losses.append(loss.detach().item())
            # Converts the 2-class logits to probabilities and keeps the spoof-class (index 1)
            # probability as the per-sample detection score used for EER.
            p_spoof = torch.softmax(logits, dim=-1)[:, 1]
            probs_list.extend(p_spoof.detach().cpu().tolist())
            labels_list.extend(device_batch["classification_labels"].detach().cpu().tolist())

            if return_per_sample:
                batch_file_ids = device_batch.get("file_id") or device_batch.get("file_ids")
                if batch_file_ids is not None:
                    if isinstance(batch_file_ids, torch.Tensor):
                        file_ids_list.extend([str(x) for x in batch_file_ids.cpu().tolist()])
                    else:
                        file_ids_list.extend([str(x) for x in batch_file_ids])
                if include_speaker_ids:
                    batch_speakers = device_batch.get("speaker_id") or device_batch.get("speaker_ids")
                    if batch_speakers is not None:
                        speaker_ids_list.extend([str(x) for x in batch_speakers])
                    batch_attacks = device_batch.get("attack") or device_batch.get("attacks")
                    if batch_attacks is not None:
                        attacks_list.extend([str(x) for x in batch_attacks])

            del outputs, logits, loss

    eer = compute_eer(labels_list, probs_list) if labels_list else np.nan
    # Accuracy uses a fixed 0.5 probability cutoff to binarize predictions; this is
    # independent of the separately computed EER-optimal threshold.
    preds = [1 if p >= 0.5 else 0 for p in probs_list]
    acc = float(np.mean(np.equal(preds, labels_list))) if labels_list else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_prob = float(np.mean(probs_list)) if probs_list else 0.0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result: Dict[str, Any] = {
        "loss": avg_loss,
        "eer": float(eer),
        "accuracy": acc,
        "avg_spoof_prob": avg_prob,
    }

    if return_per_sample:
        result["scores"] = np.asarray(probs_list, dtype=np.float64)
        result["labels"] = np.asarray(labels_list, dtype=np.int64)
        if file_ids_list:
            result["file_id"] = np.asarray(file_ids_list, dtype=object)
        if speaker_ids_list:
            result["speaker_id"] = np.asarray(speaker_ids_list, dtype=object)
        if attacks_list:
            result["attack"] = np.asarray(attacks_list, dtype=object)

    return result


def normalize_label_to_int(label_value: Any) -> int:
    """
    Normalize a label value to binary class ID: bonafide/genuine/0 -> 0, spoof/1 -> 1.
    Accepts bytes, int, float, or text representations.
    """
    if isinstance(label_value, (bytes, bytearray)):
        label_value = label_value.decode("utf-8", errors="ignore")
    if isinstance(label_value, (int, np.integer)):
        v = int(label_value)
        if v in (0, 1):
            return v
    if isinstance(label_value, (float, np.floating)):
        if float(label_value) in (0.0, 1.0):
            return int(label_value)
    txt = str(label_value).strip().lower()
    if txt in {"bonafide", "genuine", "0", "0.0"}:
        return 0
    if txt in {"spoof", "1", "1.0"}:
        return 1
    raise ValueError(f"Unsupported label value: {label_value}")


def _parse_conf_int_bounds(conf_int: Any) -> tuple[float, float]:
    """Parse confidence-interval output into (low, high) bounds, returning NaN on failure."""
    low = np.nan
    high = np.nan
    if conf_int is None:
        return low, high
    if isinstance(conf_int, dict):
        for k in ("low", "lower", "ci_low", "left"):
            if k in conf_int:
                low = float(conf_int[k])
                break
        for k in ("high", "upper", "ci_high", "right"):
            if k in conf_int:
                high = float(conf_int[k])
                break
        return low, high
    arr = np.asarray(conf_int).reshape(-1)
    if arr.size >= 2:
        low = float(arr[0])
        high = float(arr[1])
    return low, high


def _append_text_log_line(log_path: Optional[Union[str, Path]], message: str) -> None:
    """Append one UTF-8 line to log_path if provided."""
    if not log_path:
        return
    out_path = Path(log_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(str(message).rstrip("\n") + "\n")


def _compute_raw_eer(labels: np.ndarray, scores: np.ndarray, with_threshold: bool = False):
    """Compute raw EER from labels/scores. Returns (eer, threshold) or just eer."""
    if len(labels) == 0 or len(scores) == 0:
        return (np.nan, np.nan) if with_threshold else np.nan
    if np.unique(labels).size < 2:
        return (np.nan, np.nan) if with_threshold else np.nan

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    diff = np.abs(fpr - fnr)
    if diff.size == 0:
        return (np.nan, np.nan) if with_threshold else np.nan

    # EER is the ROC point where the false-positive rate equals the false-negative
    # rate; locate it by minimizing |fpr - fnr|, then report the score threshold there.
    eer_idx = int(np.argmin(diff))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    threshold = thresholds[eer_idx] if eer_idx < len(thresholds) else 0.5
    if not np.isfinite(threshold):
        threshold = 0.5
    return (eer, float(threshold)) if with_threshold else eer


def compute_eer(
    y_true: Any,
    y_score: Any,
    conditions: Optional[Any] = None,
    return_threshold: bool = False,
    return_conf_int: bool = False,
    use_conf_int: bool = False,
    fallback_log_path: Optional[Union[str, Path]] = None,
    fallback_context: str = "",
):
    """
    Compute EER with optional confidence intervals and condition-aware bootstrapping.
    When CI is enabled, the returned EER is the bootstrap center from CI evaluation.
    For single-class bootstrap draws, falls back to the full-set raw EER point estimate.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) != len(y_score):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} elements, y_score has {len(y_score)}")
    if conditions is not None:
        conditions = np.asarray(conditions)

    eer_raw, threshold = _compute_raw_eer(y_true, y_score, with_threshold=True)
    eer_out = float(eer_raw)
    conf_low = np.nan
    conf_high = np.nan

    if use_conf_int:
        if evaluate_with_conf_int is None or get_bootstrap_indices is None:
            raise ImportError(
                "Could not import confidence_intervals utilities. "
                "Set CONFIDENCE_INTERVALS_DIR to your local confidence_intervals folder."
            ) from _CONF_INT_IMPORT_ERROR

        full_has_two_classes = np.unique(y_true).size >= 2
        retry_seed_state = {"value": 0}
        fallback_stats = {"full_single_class": 0, "retry_exhausted": 0}

        def _eer_metric(labels: np.ndarray, scores: np.ndarray) -> float:
            """Bootstrap EER metric with class-balance retry and fallback to full-set EER."""
            if not full_has_two_classes:
                fallback_stats["full_single_class"] += 1
                return float(eer_raw)
            if np.unique(labels).size >= 2:
                return float(_compute_raw_eer(labels, scores, with_threshold=False))
            # Retry sampling until both classes appear
            for _ in range(max(1, int(EER_MAX_RESAMPLE_RETRIES))):
                sel_indices = get_bootstrap_indices(
                    num_samples=len(y_true),
                    conditions=conditions,
                    random_state=retry_seed_state["value"],
                )
                retry_seed_state["value"] += 1
                retry_labels = y_true[sel_indices]
                if np.unique(retry_labels).size < 2:
                    continue
                retry_scores = y_score[sel_indices]
                return _compute_raw_eer(retry_labels, retry_scores, with_threshold=False)
            fallback_stats["retry_exhausted"] += 1
            return float(eer_raw)

        # Runs the EER computation over many bootstrap resamples to produce a center
        # estimate plus a (1 - alpha%) confidence interval.
        eer_center, conf_int = evaluate_with_conf_int(
            y_score,
            _eer_metric,
            y_true,
            conditions,
            num_bootstraps=max(1, int(EER_NUM_BOOTSTRAPS)),
            alpha=float(EER_ALPHA),
        )
        if fallback_stats["full_single_class"] > 0 or fallback_stats["retry_exhausted"] > 0:
            context_prefix = f"[{fallback_context}] " if str(fallback_context).strip() else ""
            _append_text_log_line(
                fallback_log_path,
                (
                    f"{context_prefix}Single-class fallback used: returning full-set EER as center "
                    f"(center_eer={float(eer_raw):.10f}, "
                    f"full_single_class={int(fallback_stats['full_single_class'])}, "
                    f"retry_exhausted={int(fallback_stats['retry_exhausted'])})"
                ),
            )
        if np.isfinite(float(eer_center)):
            eer_out = float(eer_center)
        else:
            context_prefix = f"[{fallback_context}] " if str(fallback_context).strip() else ""
            _append_text_log_line(
                fallback_log_path,
                (
                    f"{context_prefix}CI computation returned non-finite center; using full-set EER "
                    f"(center_eer={float(eer_raw):.10f})"
                ),
            )
        conf_low, conf_high = _parse_conf_int_bounds(conf_int)

    eer_pct = float(eer_out) * 100.0
    conf_low_pct = float(conf_low) * 100.0
    conf_high_pct = float(conf_high) * 100.0
    if return_threshold and return_conf_int:
        return eer_pct, float(threshold), conf_low_pct, conf_high_pct
    if return_threshold:
        return eer_pct, float(threshold)
    if return_conf_int:
        return eer_pct, conf_low_pct, conf_high_pct
    return eer_pct


def _lookup_key_case_insensitive(mapping: Any, candidate_keys: Iterable[str]) -> Any:
    """Return the first matching value from a dict by trying candidate keys case-insensitively."""
    if not isinstance(mapping, dict):
        return None
    key_map = {str(k).lower(): k for k in mapping.keys()}
    for cand in candidate_keys:
        real_key = key_map.get(str(cand).lower())
        if real_key is not None:
            return mapping[real_key]
    return None


def _to_numpy_1d(value: Any) -> Optional[np.ndarray]:
    """Convert input to a flattened 1D NumPy array, returning None for None input."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _extract_rows_payload(container: Any) -> Optional[list]:
    """Extract per-sample rows payload from a metrics container by known alias keys."""
    rows = _lookup_key_case_insensitive(container, _ROWS_KEYS)
    if isinstance(rows, (list, tuple)):
        return list(rows)
    return None


def _extract_from_rows(rows: list) -> Optional[Dict[str, np.ndarray]]:
    """Parse a list of row dicts into aligned arrays for file_id/speaker/attack/label/score."""
    if not rows:
        return None
    file_ids: list[str] = []
    scores: list[float] = []
    labels: list[int] = []
    speakers: list[str] = []
    attacks: list[str] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        score_val = _lookup_key_case_insensitive(row, _SCORE_KEYS)
        label_val = _lookup_key_case_insensitive(row, _LABEL_KEYS)
        if score_val is None or label_val is None:
            continue
        try:
            score = float(score_val)
            label = normalize_label_to_int(label_val)
        except Exception:
            continue
        sample_idx = len(scores)
        file_id_val = _lookup_key_case_insensitive(row, _FILE_ID_KEYS)
        file_ids.append(f"sample_{sample_idx:07d}" if file_id_val is None else str(file_id_val))
        scores.append(score)
        labels.append(label)
        spk = _lookup_key_case_insensitive(row, _SPEAKER_KEYS)
        speakers.append("unknown" if spk is None else str(spk))
        atk = _lookup_key_case_insensitive(row, _ATTACK_KEYS)
        attacks.append("-" if atk is None else str(atk))

    if not scores:
        return None
    return {
        "file_id": np.asarray(file_ids, dtype=object).reshape(-1),
        "scores": np.asarray(scores, dtype=np.float64).reshape(-1),
        "labels": np.asarray(labels, dtype=np.int64).reshape(-1),
        "speaker": np.asarray(speakers, dtype=object).reshape(-1),
        "attack": np.asarray(attacks, dtype=object).reshape(-1),
    }


def _extract_eer_inputs(metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract aligned arrays (scores, labels, file_id, speaker, attack) from a metrics payload.
    Tries direct array fields first, then falls back to per-row parsing.
    """
    if not isinstance(metrics, dict):
        return None

    eval_stats = metrics.get("eval_stats")
    containers = [metrics]
    if isinstance(eval_stats, dict):
        containers.append(eval_stats)

    scores_arr = None
    labels_arr = None
    file_id_arr = None
    speaker_arr = None
    attack_arr = None

    for container in containers:
        if scores_arr is None:
            scores_arr = _to_numpy_1d(_lookup_key_case_insensitive(container, _SCORE_KEYS))
        if labels_arr is None:
            raw_labels = _to_numpy_1d(_lookup_key_case_insensitive(container, _LABEL_KEYS))
            if raw_labels is not None:
                try:
                    labels_arr = np.asarray(
                        [normalize_label_to_int(v) for v in raw_labels],
                        dtype=np.int64,
                    ).reshape(-1)
                except Exception:
                    labels_arr = None
        if file_id_arr is None:
            file_id_arr = _to_numpy_1d(_lookup_key_case_insensitive(container, _FILE_ID_KEYS))
        if speaker_arr is None:
            speaker_arr = _to_numpy_1d(_lookup_key_case_insensitive(container, _SPEAKER_KEYS))
        if attack_arr is None:
            attack_arr = _to_numpy_1d(_lookup_key_case_insensitive(container, _ATTACK_KEYS))

    if scores_arr is not None and labels_arr is not None and len(scores_arr) == len(labels_arr):
        n = len(scores_arr)
        file_id_out = (
            np.asarray([str(v) for v in file_id_arr], dtype=object).reshape(-1)
            if file_id_arr is not None and len(file_id_arr) == n
            else np.asarray([f"sample_{i:07d}" for i in range(n)], dtype=object).reshape(-1)
        )
        speaker_out = (
            np.asarray([str(v) for v in speaker_arr], dtype=object).reshape(-1)
            if speaker_arr is not None and len(speaker_arr) == n
            else None
        )
        attack_out = (
            np.asarray([str(v) for v in attack_arr], dtype=object).reshape(-1)
            if attack_arr is not None and len(attack_arr) == n
            else None
        )
        return {
            "file_id": file_id_out,
            "scores": np.asarray(scores_arr, dtype=np.float64).reshape(-1),
            "labels": np.asarray(labels_arr, dtype=np.int64).reshape(-1),
            "speaker": speaker_out,
            "attack": attack_out,
        }

    # Fallback: try row-based parsing
    for container in containers:
        rows = _extract_rows_payload(container)
        if rows is None:
            continue
        from_rows = _extract_from_rows(rows)
        if from_rows is not None:
            return from_rows
    return None


def _normalize_text_field(value: Any, default_value: str) -> str:
    """Normalize a text metadata field, replacing None/empty/nan with a default."""
    if value is None:
        return default_value
    txt = str(value).strip()
    if txt == "" or txt.lower() in {"nan", "none"}:
        return default_value
    return txt


def _label_int_to_text(label_value: Any) -> str:
    """Convert integer label to canonical text: 1 -> 'spoof', 0 -> 'bonafide'."""
    try:
        return "spoof" if int(label_value) == 1 else "bonafide"
    except Exception:
        return str(label_value)


def _export_scores_text_from_metrics(metrics: Optional[Dict[str, Any]], output_path: Union[str, Path]) -> tuple[Optional[int], Optional[str]]:
    """
    Export per-sample scores to a CSV file (columns: file_id, speaker_id, attack, label, score).
    Returns (row_count, None) on success or (None, error_message) on failure.
    """
    extracted = _extract_eer_inputs(metrics)
    if extracted is None:
        return None, "per-sample scores/labels were not found in metrics"

    scores = np.asarray(extracted.get("scores"), dtype=np.float64).reshape(-1)
    labels = np.asarray(extracted.get("labels"), dtype=np.int64).reshape(-1)
    if len(scores) == 0 or len(labels) == 0 or len(scores) != len(labels):
        return None, "invalid per-sample scores/labels shape in metrics"

    n = len(scores)
    file_ids = extracted.get("file_id")
    speakers = extracted.get("speaker")
    attacks = extracted.get("attack")

    file_ids_out = (
        np.asarray(file_ids, dtype=object).reshape(-1)
        if file_ids is not None and len(file_ids) == n
        else np.asarray([f"sample_{i:07d}" for i in range(n)], dtype=object)
    )
    speakers_out = (
        np.asarray(speakers, dtype=object).reshape(-1)
        if speakers is not None and len(speakers) == n
        else np.asarray(["unknown"] * n, dtype=object)
    )
    attacks_out = (
        np.asarray(attacks, dtype=object).reshape(-1)
        if attacks is not None and len(attacks) == n
        else np.asarray(["-"] * n, dtype=object)
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id", "speaker_id", "attack", "label", "score"])
        for idx in range(n):
            writer.writerow([
                _normalize_text_field(file_ids_out[idx], f"sample_{idx:07d}"),
                _normalize_text_field(speakers_out[idx], "unknown"),
                _normalize_text_field(attacks_out[idx], "-"),
                _label_int_to_text(labels[idx]),
                f"{float(scores[idx]):.10f}",
            ])
    return int(n), None


def _try_load_scores_csv(path: Path) -> Optional[Dict[str, Any]]:
    """
    Try to load an existing per-sample scores CSV into a metrics-compatible dict.
    Returns None if the file is missing, unreadable, or contains no data rows.
    Used by run_inference() to skip model inference for splits already completed.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_file():
        return None
    try:
        file_ids: list = []
        speaker_ids: list = []
        attacks: list = []
        labels_list: list = []
        scores_list: list = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_ids.append(row.get("file_id", ""))
                speaker_ids.append(row.get("speaker_id", "unknown"))
                attacks.append(row.get("attack", "-"))
                label_val = row.get("label", "0")
                labels_list.append(0 if str(label_val).strip().lower() in ("bonafide", "0") else 1)
                scores_list.append(float(row.get("score", 0.0)))
        if not scores_list:
            return None
        return {
            "scores":     np.array(scores_list, dtype=np.float64),
            "labels":     np.array(labels_list, dtype=np.int64),
            "file_id":    np.array(file_ids, dtype=object),
            "speaker_id": np.array(speaker_ids, dtype=object),
            "attack":     np.array(attacks, dtype=object),
        }
    except Exception as exc:
        print(f"[Eval] Warning: could not load existing scores CSV {path}: {exc}")
        return None


def _write_split_results_from_loaded(
    loaded_metrics: Dict[str, Any],
    eer_label: str,
    scores_label: str,
    scores_path: Path,
    out_path: Path,
    fallback_log_path: Path,
    split_name: str,
    dataset_name: str,
    ckpt_name: str,
    log_eval_stats_flag: bool = False,
) -> Dict[str, Any]:
    """
    Compute EER and write result lines for a split that was already evaluated in a
    previous run.  Loads scores/labels from the existing CSV (no model inference).
    Returns a result dict in the same format as _run_split_evaluation().
    """
    n = len(loaded_metrics.get("scores", []))
    print(
        f"[Eval] Split={split_name} | checkpoint={ckpt_name} | dataset={dataset_name}"
        f" | RESUMED ({n} rows from existing CSV)"
    )
    triplet = _compute_eer_triplet_from_metrics(
        loaded_metrics,
        fallback_log_path=fallback_log_path,
        fallback_context_prefix=split_name,
    )
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"{eer_label} EER (%): {_fmt_float(_pick_display_eer_percent(loaded_metrics, triplet))} [resumed]\n")
        _write_triplet_lines(f, eer_label, triplet)
        f.write(f"{scores_label} {scores_path} (rows={n}) [from existing file]\n")
    eval_stats = _load_eval_stats_json(
        scores_path.parent / f"{ckpt_name}_eval_stats_{split_name}.json"
    )
    maybe_log_eval_stats(
        enabled=log_eval_stats_flag,
        log_file=str(out_path),
        header=f"{eer_label} eval stats [resumed]",
        eval_stats=eval_stats,
    )
    return dict(metrics=loaded_metrics, triplet=triplet, scores_rows=n, scores_err=None, error=None)


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy scalars and arrays."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _save_eval_stats_json(eval_stats: Optional[Dict[str, Any]], json_path: Path) -> None:
    """Atomically save eval_stats dict to JSON for future resume use. No-op when eval_stats is empty."""
    if not eval_stats:
        return
    try:
        json_path = Path(json_path)
        tmp_path = json_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(eval_stats, f, indent=2, default=_json_default)
        tmp_path.replace(json_path)
    except Exception as exc:
        print(f"[Eval] Warning: could not save eval_stats JSON {json_path}: {exc}")


def _load_eval_stats_json(json_path: Path) -> Optional[Dict[str, Any]]:
    """Load eval_stats dict from JSON, returning None if file is missing or unreadable."""
    json_path = Path(json_path)
    if not json_path.is_file():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[Eval] Warning: could not load eval_stats JSON {json_path}: {exc}")
        return None


def _compute_eer_triplet_from_metrics(
    metrics: Optional[Dict[str, Any]],
    fallback_log_path: Optional[Union[str, Path]] = None,
    fallback_context_prefix: str = "",
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Compute EER+CI for three condition modes (none, speaker, attack) from a metrics payload.
    Returns a dict with keys 'none', 'speaker', 'attack', each containing eer_percent and CI bounds.
    """
    extracted = _extract_eer_inputs(metrics)
    if extracted is None:
        return None

    labels = extracted["labels"]
    scores = extracted["scores"]
    speaker_conditions = extracted.get("speaker")
    attack_conditions = extracted.get("attack")

    _ci_available = evaluate_with_conf_int is not None and get_bootstrap_indices is not None

    def _safe_compute(mode_name: str, conditions: Optional[np.ndarray], reason: Optional[str] = None) -> Dict[str, Any]:
        """Run one EER+CI computation, returning NaN values and an error string on failure."""
        if reason:
            return {"eer_percent": np.nan, "ci_low_percent": np.nan, "ci_high_percent": np.nan, "error": reason}
        try:
            context = f"{fallback_context_prefix}/{mode_name}" if str(fallback_context_prefix).strip() else str(mode_name)
            if _ci_available:
                eer, _, ci_low, ci_high = compute_eer(
                    labels, scores,
                    conditions=conditions,
                    return_threshold=True,
                    return_conf_int=True,
                    use_conf_int=True,
                    fallback_log_path=fallback_log_path,
                    fallback_context=context,
                )
            else:
                eer, _ = compute_eer(
                    labels, scores,
                    conditions=conditions,
                    return_threshold=True,
                    return_conf_int=False,
                    use_conf_int=False,
                )
                ci_low, ci_high = np.nan, np.nan
            return {
                "eer_percent": float(eer),
                "ci_low_percent": float(ci_low),
                "ci_high_percent": float(ci_high),
                "error": None,
            }
        except Exception as exc:
            return {"eer_percent": np.nan, "ci_low_percent": np.nan, "ci_high_percent": np.nan, "error": str(exc)}

    return {
        "none": _safe_compute("none", None),
        "speaker": _safe_compute("speaker", speaker_conditions, reason=None if speaker_conditions is not None else "speaker metadata not found in eval metrics"),
        "attack": _safe_compute("attack", attack_conditions, reason=None if attack_conditions is not None else "attack metadata not found in eval metrics"),
    }


def _fmt_eer_ci(entry: Optional[Dict[str, Any]]) -> str:
    """Format an EER+CI entry as 'eer [ci_low, ci_high]' or 'n/a' if values are missing."""
    if not isinstance(entry, dict):
        return "n/a"
    eer = entry.get("eer_percent", np.nan)
    ci_low = entry.get("ci_low_percent", np.nan)
    ci_high = entry.get("ci_high_percent", np.nan)
    error = entry.get("error")
    if np.isfinite(eer) and np.isfinite(ci_low) and np.isfinite(ci_high):
        return f"{eer:.4f} [{ci_low:.4f}, {ci_high:.4f}]"
    if np.isfinite(eer):
        return f"{eer:.4f} [n/a, n/a]"
    if error:
        return f"n/a ({error})"
    return "n/a"


def _write_triplet_lines(file_obj, split_name: str, triplet: Optional[Dict[str, Dict[str, Any]]]) -> None:
    """Write the three EER+CI summary lines (none/speaker/attack conditioned) to a file."""
    if not isinstance(triplet, dict):
        file_obj.write(f"{split_name} EER+CI none (%): n/a\n")
        file_obj.write(f"{split_name} EER+CI speaker-conditioned (%): n/a\n")
        file_obj.write(f"{split_name} EER+CI attack-conditioned (%): n/a\n")
        return
    file_obj.write(f"{split_name} EER+CI none (%): {_fmt_eer_ci(triplet.get('none'))}\n")
    file_obj.write(f"{split_name} EER+CI speaker-conditioned (%): {_fmt_eer_ci(triplet.get('speaker'))}\n")
    file_obj.write(f"{split_name} EER+CI attack-conditioned (%): {_fmt_eer_ci(triplet.get('attack'))}\n")


def _pick_display_eer_percent(metrics: Optional[Dict[str, Any]], triplet: Optional[Dict[str, Dict[str, Any]]]) -> Any:
    """
    Pick the best EER percent for display: prefers triplet none-conditioned value,
    falls back to metrics['eer'] (already in percent), then 'n/a'.
    """
    if isinstance(triplet, dict):
        none_entry = triplet.get("none")
        if isinstance(none_entry, dict):
            val = none_entry.get("eer_percent", np.nan)
            if np.isfinite(val):
                return float(val)
    # NOTE: eval_classification returns metrics["eer"] already in percent units
    if isinstance(metrics, dict) and "eer" in metrics:
        try:
            val = float(metrics.get("eer"))
        except Exception:
            val = np.nan
        if np.isfinite(val):
            return float(val)
    return "n/a"


def _fmt_float(val: Any) -> str:
    """Format a numeric value as a 4-decimal string, or 'n/a' for non-finite/invalid values."""
    try:
        f_val = float(val)
        if not np.isfinite(f_val):
            return "n/a"
        return f"{f_val:.4f}"
    except Exception:
        return "n/a"


# ---------------------------------------------------------------------------
# Config / model building helpers extracted from run_inference
# ---------------------------------------------------------------------------

def _parse_inference_config(args: SimpleNamespace, raw_ckpt: Any) -> Dict[str, Any]:
    """
    Parse and validate all inference configuration from run_args and the checkpoint.

    Loads the YAML (via _ensure_config_env / _load_config_dict), validates required
    fields, resolves all hyperparameters, instantiates the AutoProcessor, and builds
    the common dataset/loader/eval kwarg dicts consumed by the rest of run_inference.

    Returns a flat dict with every key needed downstream.
    """
    cfg_path = _ensure_config_env(args.config_path)
    cfg_dict = _load_config_dict(raw_ckpt, cfg_path)

    dataset_name = args.dataset or _resolve_cfg(cfg_dict, ("DATASET", "dataset"))
    if not dataset_name:
        raise ValueError("Dataset is required. Set 'dataset' in run_args or ensure DATASET exists in the checkpoint config.")

    skip_dev = bool(getattr(args, "skip_dev", False))

    dev_protocol = None if skip_dev else (args.dev_set or _resolve_cfg(cfg_dict, ("dev_protocol", "DEV_PROTOCOL")))
    eval_protocol = args.eval_set or _resolve_cfg(cfg_dict, ("eval_protocol", "EVAL_PROTOCOL"))
    if not eval_protocol:
        raise ValueError("eval_set path is required. Set 'eval_set' in run_args or ensure EVAL_PROTOCOL exists in the checkpoint config.")
    if not skip_dev and not dev_protocol:
        raise ValueError("dev_set path is required unless skip_dev=True. Set 'dev_set' in run_args or ensure DEV_PROTOCOL exists in the checkpoint config.")

    dev_audio_dir = None if skip_dev else (args.dev_audio_dir or _resolve_cfg(cfg_dict, ("dev_audio_dir", "DEV_AUDIO_DIR")))
    eval_audio_dir = args.eval_audio_dir or _resolve_cfg(cfg_dict, ("eval_audio_dir", "EVAL_AUDIO_DIR"))
    if not eval_audio_dir:
        raise ValueError("eval_audio_dir is required. Set 'eval_audio_dir' in run_args or in the config.")
    if not skip_dev and not dev_audio_dir:
        raise ValueError("dev_audio_dir is required unless skip_dev=True. Set 'dev_audio_dir' in run_args or in the config.")

    prompts_path = (
        args.prompts_path
        or os.environ.get("VOXTRAL_PROMPTS_PATH")
        or _resolve_cfg(cfg_dict, ("prompts_path",))
    )
    if not prompts_path or not os.path.exists(prompts_path):
        raise FileNotFoundError("Prompts file is missing. Set 'prompts_path' in run_args, export VOXTRAL_PROMPTS_PATH, or set prompts_path in the config.")
    prompts = load_text_prompts(prompts_path)
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_path}")

    pad_features_mode = args.pad_features_mode or _resolve_cfg(cfg_dict, ("PAD_FEATURES_MODE",), "zero")
    pad_waveform_mode = args.pad_waveform_mode or _resolve_cfg(cfg_dict, ("PAD_WAVEFORM_MODE",), "none")
    constant_prompts_flag = bool(_resolve_cfg(cfg_dict, ("constant_prompts_flag", "CONSTANT_PROMPTS_FLAG"), False))
    three_random_prompts = bool(_resolve_cfg(cfg_dict, ("3_random_prompts", "three_random_prompts"), False))
    random_prompt_variants = _resolve_cfg(
        cfg_dict,
        ("three_random_prompt_variants", "random_prompt_variants", "three_random_prompts_variants"),
        [],
    ) or []
    if isinstance(random_prompt_variants, str):
        random_prompt_variants = [random_prompt_variants]

    n_audio_tokens = int(_resolve_cfg(cfg_dict, ("N_audio_tokens", "N_AUDIO_TOKENS"), 375))
    augmentation_prob_raw = _resolve_cfg(cfg_dict, ("AUGMENTATION_PROB", "augmentation_prob"), 0.0)
    try:
        augmentation_prob = float(augmentation_prob_raw)
    except (TypeError, ValueError):
        augmentation_prob = 0.0

    precision = str(_resolve_cfg(cfg_dict, ("precision", "PRECISION"), "float32")).lower()
    precision_to_dtype = {
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    dtype = precision_to_dtype.get(precision, torch.float32)
    amp_dtype = "bf16" if dtype == torch.bfloat16 else ("fp16" if dtype == torch.float16 else "fp32")

    log_eval_stats_flag = True if args.log_eval_stats is None else bool(args.log_eval_stats)

    target_device = args.device
    if target_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        target_device = "cpu"

    base_model_name = _resolve_cfg(cfg_dict, ("model_id", "VOXTRAL_MODEL"))
    if not base_model_name:
        raise ValueError("model_id is missing. Add it to the checkpoint config or provide via YAML.")

    finetune_mode = _resolve_cfg(cfg_dict, ("finetune_mode", "FINETUNE_MODE"), "lora")
    target_modules = _resolve_cfg(cfg_dict, ("target_modules", "TARGET_MODULES"), ["q_proj", "k_proj", "v_proj", "o_proj"])
    use_4bit = bool(_resolve_cfg(cfg_dict, ("use_4bit", "USE_4BIT"), False))
    trainable_audio_encoder = _resolve_cfg(cfg_dict, ("trainable_audio_encoder", "TRAINABLE_AUDIO_ENCODER"), False)
    _audio_mode = str(trainable_audio_encoder).strip().lower() if isinstance(trainable_audio_encoder, str) else ("trainable" if trainable_audio_encoder else "frozen")
    language_model_only_lora = (_audio_mode != "dora")
    encoder_type = _normalize_encoder_type(_resolve_cfg(cfg_dict, ("encoder_type", "ENCODER_TYPE"), None))
    k_downsample = int(_resolve_cfg(cfg_dict, ("k_downsample", "K_DOWNSAMPLE"), 4))
    lora_rank = int(_resolve_cfg(cfg_dict, ("lora_rank", "LORA_RANK"), 16))
    lora_alpha = int(_resolve_cfg(cfg_dict, ("lora_alpha", "LORA_ALPHA"), 32))
    lora_dropout = float(_resolve_cfg(cfg_dict, ("lora_dropout", "LORA_DROPOUT"), 0.05))
    projector_peft = bool(_resolve_cfg(cfg_dict, ("projector_peft", "PROJECTOR_PEFT"), False))

    num_classes = 2
    model_cfg = _resolve_cfg(cfg_dict, ("model", "MODEL"), {})
    if isinstance(model_cfg, dict):
        num_classes = int(model_cfg.get("num_classes", num_classes))

    # With no explicit eval batch size, default to twice the training batch size:
    # inference needs no gradient/optimizer memory and can afford larger batches.
    batch_size_cfg = args.batch_size or _resolve_cfg(cfg_dict, ("eval_batch_size", "EVAL_BATCH_SIZE"))
    base_batch = int(_resolve_cfg(cfg_dict, ("BATCH_SIZE", "batch_size"), 1))
    eval_batch_size = int(batch_size_cfg) if batch_size_cfg and int(batch_size_cfg) > 0 else max(1, base_batch * 2)
    num_workers = int(args.num_workers if args.num_workers is not None else _resolve_cfg(cfg_dict, ("NUM_WORKERS", "num_workers"), 0))

    generation_cfg = _resolve_cfg(cfg_dict, ("generation_classification",), {}) or {}

    # Resolve the cached snapshot directory so MistralCommonTokenizer.from_pretrained
    # receives a local path instead of a repo ID.  When given a repo ID it calls
    # download_tokenizer_from_hf_hub (mistral_common) which requires listing the remote
    # repo even if the file is already cached — failing with HF_HUB_OFFLINE=1.
    # A directory path bypasses the HF hub lookup entirely.
    _processor_path = base_model_name
    _ref_file = Path(_MODEL_CACHE_DIR) / ("models--" + base_model_name.replace("/", "--")) / "refs" / "main"
    if _ref_file.is_file():
        _commit = _ref_file.read_text().strip()
        _snapshot = Path(_MODEL_CACHE_DIR) / ("models--" + base_model_name.replace("/", "--")) / "snapshots" / _commit
        if _snapshot.is_dir():
            _processor_path = str(_snapshot)
    processor = AutoProcessor.from_pretrained(_processor_path, cache_dir=_MODEL_CACHE_DIR)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    adapter_head_cfg = _resolve_cfg(cfg_dict, ("audio_adapter_head",), {}) or {}
    if not isinstance(adapter_head_cfg, dict) and hasattr(adapter_head_cfg, "__dict__"):
        adapter_head_cfg = dict(adapter_head_cfg.__dict__)

    use_amp = target_device.startswith("cuda") and dtype in (torch.float16, torch.bfloat16)

    _label_texts = extract_label_texts_from_gen_cfg(generation_cfg)
    _constant_prompt_override = _resolve_cfg(cfg_dict, ("constant_prompt_override",), None) or None
    print(f"[INFERENCE] label_texts: 0='{_label_texts.get(0, 'bonafide')}', 1='{_label_texts.get(1, 'spoof')}'")
    print(f"[INFERENCE] constant_prompt_override: {_constant_prompt_override!r}")

    common_dataset_kwargs = dict(
        dataset_type=dataset_name,
        processor=processor,
        prompts=prompts,
        AUGMENTATION_PROB=augmentation_prob,
        # is_train/augmentation_on are forced off so inference always runs on clean,
        # deterministic audio, not the training-time augmented version.
        is_train=False,
        augmentation_on=False,
        reasoning=False,
        SPECIAL_TOKEN_TASK="<SPECIAL_20>",
        dtype=dtype,
        CONFIG_AUGMENTATION=None,
        encoder_type=encoder_type,
        pad_features_mode=pad_features_mode,
        pad_waveform_mode=pad_waveform_mode,
        constant_prompts_flag=constant_prompts_flag,
        three_random_prompts=three_random_prompts,
        random_prompt_variants=random_prompt_variants,
        constant_prompt_override=_constant_prompt_override,
        label_texts=_label_texts,
        N_audio_tokens=n_audio_tokens,
    )
    common_loader_kwargs = dict(
        batch_size=eval_batch_size,
        shuffle=False,
        ensure_bonafide_in_batch=False,
        balance_bonafide_spoof=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=target_device.startswith("cuda"),
        persistent_workers=False,  # inference = single pass; persistent workers cause deadlock on cleanup
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_fn,
        worker_init_fn=None,
    )
    common_eval_kwargs = dict(
        processor=processor,
        device=target_device,
        generation_cfg=generation_cfg,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        enable_debug_logging=False,
        return_example=False,
        return_eval_stats=log_eval_stats_flag,
        return_per_sample=True,
        include_speaker_ids=True,
    )

    return dict(
        cfg_dict=cfg_dict,
        dataset_name=dataset_name,
        skip_dev=skip_dev,
        dev_protocol=dev_protocol,
        eval_protocol=eval_protocol,
        dev_audio_dir=dev_audio_dir,
        eval_audio_dir=eval_audio_dir,
        base_model_name=base_model_name,
        finetune_mode=finetune_mode,
        target_modules=target_modules,
        use_4bit=use_4bit,
        trainable_audio_encoder=trainable_audio_encoder,
        language_model_only_lora=language_model_only_lora,
        encoder_type=encoder_type,
        k_downsample=k_downsample,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        projector_peft=projector_peft,
        num_classes=num_classes,
        precision=precision,
        dtype=dtype,
        amp_dtype=amp_dtype,
        target_device=target_device,
        use_amp=use_amp,
        log_eval_stats_flag=log_eval_stats_flag,
        processor=processor,
        adapter_head_cfg=adapter_head_cfg,
        generation_cfg=generation_cfg,
        common_dataset_kwargs=common_dataset_kwargs,
        common_loader_kwargs=common_loader_kwargs,
        common_eval_kwargs=common_eval_kwargs,
    )


def _build_inference_model(cfg: Dict[str, Any], raw_ckpt: Any) -> tuple[torch.nn.Module, bool]:
    """
    Build the inference model, load checkpoint weights, and move to target device.

    Detects the checkpoint type (adapter-head vs CAG), constructs the appropriate
    model, loads weights with strict=True, moves to device, calls model.eval() and
    verify_model_frozen().

    Returns (model, is_adapter_head).
    """
    is_adapter_head = _detect_adapter_head_checkpoint(raw_ckpt)
    if is_adapter_head:
        print("[Eval] Detected adapter-head checkpoint (Whisper + classification head)")
    else:
        print("[Eval] Detected classification-as-generation checkpoint (full LLM)")

    target_device = cfg["target_device"]
    use_4bit = cfg["use_4bit"]
    load_device = target_device

    if is_adapter_head:
        adapter_head_cfg = cfg["adapter_head_cfg"]
        model = build_classification_head_model(
            base_model_name=cfg["base_model_name"],
            device=load_device,
            dtype=cfg["dtype"],
            finetune_mode=cfg["finetune_mode"],
            lora_rank=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            lora_target_modules=cfg["target_modules"],
            use_4bit=use_4bit,
            precision=cfg["precision"],
            load_model_path=None,
            load_pretrained_audio_stack=True,
            language_model_only_lora=cfg["language_model_only_lora"],
            encoder_type=cfg["encoder_type"],
            audio_encoder_tuning=adapter_head_cfg.get("audio_encoder_tuning", "frozen"),
            k_downsample=cfg["k_downsample"],
            num_classes=cfg["num_classes"],
            pooling_strategy=adapter_head_cfg.get("pooling_strategy", "mean"),
            head_hidden_dim=int(adapter_head_cfg.get("head_hidden_dim", 256)),
            head_dropout=float(adapter_head_cfg.get("head_dropout", 0.1)),
            train_audio_adapter=bool(adapter_head_cfg.get("train_audio_adapter", True)),
            drop_language_model=bool(adapter_head_cfg.get("drop_language_model", True)),
        )
    else:
        base_model = build_voxtral_model(
            finetune_mode=cfg["finetune_mode"],
            base_model_name=cfg["base_model_name"],
            dtype=cfg["dtype"],
            device=load_device,
            load_pretrained_base=True,
            lora_rank=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            lora_target_modules=cfg["target_modules"],
            use_4bit=use_4bit,
            precision=cfg["precision"],
            language_model_only_lora=cfg["language_model_only_lora"],
        )
        print("[Eval] Wrapping base model (PEFT/LoRA init) ...", flush=True)
        model = VoxtralStageWrapper_classification(
            base_model,
            encoder_type=cfg["encoder_type"],
            trainable_audio_encoder=cfg["trainable_audio_encoder"],
            k_downsample=cfg["k_downsample"],
            num_classes=cfg["num_classes"],
            finetune_mode=cfg["finetune_mode"],
            lora_rank=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            lora_target_modules=cfg["target_modules"],
            projector_peft=cfg["projector_peft"],
        )
        print("[Eval] Wrapper init done.", flush=True)

    # Disable the LM's KV-cache: CAG scoring only needs a single forward pass per
    # sample, not multi-step autoregressive generation.
    if hasattr(model, "model") and hasattr(model.model, "config"):
        try:
            model.model.config.use_cache = False
        except Exception:
            pass

    print("[Eval] Loading checkpoint weights (load_state_dict) ...", flush=True)
    _load_model_weights(model, raw_ckpt)
    print("[Eval] Checkpoint weights loaded.", flush=True)
    if target_device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True   # ~20% speedup on Ampere+; negligible precision loss
        torch.backends.cudnn.allow_tf32 = True
    model.eval()
    verify_model_frozen(model)
    return model, is_adapter_head


def _run_split_evaluation(
    split_name: str,
    eer_label: str,
    scores_label: str,
    protocol: str,
    audio_dir: str,
    scores_path: Path,
    out_path: Path,
    fallback_log_path: Path,
    model: torch.nn.Module,
    is_adapter_head: bool,
    device: str,
    use_amp: bool,
    amp_dtype: str,
    encoder_type: Optional[str],
    log_eval_stats_flag: bool,
    dataset_name: str,
    ckpt_name: str,
    common_dataset_kwargs: Dict[str, Any],
    common_loader_kwargs: Dict[str, Any],
    common_eval_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run evaluation for a single split (dev or eval) end-to-end.

    Constructs the dataset and DataLoader, dispatches to the appropriate evaluator,
    computes the EER triplet, exports per-sample scores to CSV, writes summary lines
    to out_path, logs eval_stats, cleans up the loader/dataset, and empties CUDA cache.

    Returns a result dict with keys: metrics, triplet, scores_rows, scores_err, error.
    """
    print(f"[Eval] Split={split_name} | checkpoint={ckpt_name} | dataset={dataset_name}")

    metrics: Optional[Dict[str, Any]] = None
    triplet: Optional[Dict[str, Any]] = None
    scores_rows: Optional[int] = None
    scores_err: Optional[str] = None
    error: Optional[Exception] = None
    loader = None
    dataset = None

    try:
        dataset = ASVspoofIndexDataset(
            path_to_protocol=protocol,
            data_dir=audio_dir,
            **common_dataset_kwargs,
        )
        loader = make_dataloader(dataset, **common_loader_kwargs)
        _bar_desc = f"Evaluating (step {ckpt_name} | {split_name})"
        if is_adapter_head:
            metrics = _evaluate_adapter_head(
                model, loader,
                device=device, use_amp=use_amp, amp_dtype=amp_dtype,
                encoder_type=encoder_type, return_per_sample=True, include_speaker_ids=True,
                tqdm_desc=_bar_desc,
            )
        else:
            metrics = evaluate_epoch_classification_as_generation(
                model, loader, **common_eval_kwargs, eval_step=f"{ckpt_name} | {split_name}",
            )
    except Exception as exc:
        error = exc
        print(f"[Warn] {split_name.capitalize()} evaluation failed; continuing. Error: {exc}")
    finally:
        if not error:
            try:
                triplet = _compute_eer_triplet_from_metrics(
                    metrics,
                    fallback_log_path=fallback_log_path,
                    fallback_context_prefix=split_name,
                )
                scores_rows, scores_err = _export_scores_text_from_metrics(metrics, scores_path)
                _save_eval_stats_json(
                    (metrics or {}).get("eval_stats"),
                    scores_path.parent / f"{ckpt_name}_eval_stats_{split_name}.json",
                )
            except Exception as _inner_exc:
                print(f"[Warn] Failed during {split_name} post-processing: {_inner_exc}")

        with open(out_path, "a", encoding="utf-8") as f:
            if error:
                f.write(f"{eer_label} EER (%): n/a ({split_name} evaluation failed)\n")
                f.write(f"{eer_label} EER+CI none (%): n/a ({split_name} evaluation failed)\n")
                f.write(f"{eer_label} EER+CI speaker-conditioned (%): n/a ({split_name} evaluation failed)\n")
                f.write(f"{eer_label} EER+CI attack-conditioned (%): n/a ({split_name} evaluation failed)\n")
                f.write(f"{scores_label} n/a ({split_name} evaluation failed)\n")
            else:
                f.write(f"{eer_label} EER (%): {_fmt_float(_pick_display_eer_percent(metrics, triplet))}\n")
                _write_triplet_lines(f, eer_label, triplet)
                if scores_rows is not None:
                    f.write(f"{scores_label} {scores_path} (rows={scores_rows})\n")
                else:
                    f.write(f"{scores_label} n/a ({scores_err or 'unknown error'})\n")

        if not error:
            maybe_log_eval_stats(
                enabled=log_eval_stats_flag,
                log_file=str(out_path),
                header=f"{eer_label} eval stats",
                eval_stats=(metrics or {}).get("eval_stats"),
            )
        try:
            del loader, dataset
        except Exception:
            pass
        # Explicitly free GPU/CPU memory after each split so running dev then eval
        # back-to-back doesn't accumulate memory.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return dict(metrics=metrics, triplet=triplet, scores_rows=scores_rows, scores_err=scores_err, error=error)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_inference(run_args: Dict[str, Any], prebuilt_model=None) -> Dict[str, Any]:
    """
    Run full checkpoint inference and persist EER/scores artifacts.
    Evaluates dev split (optional) and eval split, writes EER summaries and per-sample scores files.
    Returns a dict with metrics, errors, and output artifact paths.

    prebuilt_model : optional (model, is_adapter_head) tuple from a previous call.
        When supplied the expensive base-model build + CPU→GPU transfer is skipped;
        only the LoRA/adapter weights are reloaded via load_state_dict.  All
        checkpoints in a sweep share the same architecture, so this is safe.
    """
    _defaults = {
        "checkpoint_path": None,
        "config_path": None,
        "dataset": None,
        "dev_set": None,
        "eval_set": None,
        "dev_audio_dir": None,
        "eval_audio_dir": None,
        "prompts_path": None,
        "pad_features_mode": None,
        "pad_waveform_mode": None,
        "batch_size": None,
        "num_workers": None,
        "device": "cuda",
        "log_eval_stats": None,
        "skip_dev": False,
    }
    args = SimpleNamespace(**{**_defaults, **run_args})

    if args.checkpoint_path is None:
        raise ValueError("checkpoint_path is required in run_args")
    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    ckpt_name = ckpt_path.stem
    print(f"[Eval] Starting checkpoint: {ckpt_name} ({ckpt_path})")

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _parse_inference_config(args, raw_ckpt)

    # Set up output paths before model loading so we can check for existing CSVs.
    scores_out_dir = ckpt_path.parent / "scores_text_outputs"
    out_path = scores_out_dir / f"{ckpt_name}_eer.txt"
    dev_scores_path = scores_out_dir / f"{ckpt_name}_scores_dev.csv"
    eval_scores_path = scores_out_dir / f"{ckpt_name}_scores_eval.csv"
    fallback_log_path = scores_out_dir / f"{ckpt_name}_eer_single_class_fallback.txt"
    scores_out_dir.mkdir(parents=True, exist_ok=True)

    # Resume detection: load existing CSVs to skip already-completed splits.
    dev_loaded = _try_load_scores_csv(dev_scores_path) if not cfg["skip_dev"] else None
    eval_loaded = _try_load_scores_csv(eval_scores_path)
    if dev_loaded is not None:
        print(f"[Eval] Dev scores CSV found ({len(dev_loaded['scores'])} rows) — skipping dev model inference")
    if eval_loaded is not None:
        print(f"[Eval] Eval scores CSV found ({len(eval_loaded['scores'])} rows) — skipping eval model inference")

    # Only load the model if at least one split still needs to be evaluated.
    need_model = (not cfg["skip_dev"] and dev_loaded is None) or (eval_loaded is None)
    if need_model:
        if prebuilt_model is not None:
            model, is_adapter_head = prebuilt_model
            print(f"[Eval] Reusing prebuilt model — reloading weights for {ckpt_name}", flush=True)
            _load_model_weights(model, raw_ckpt)
            model.eval()
            verify_model_frozen(model)
        else:
            model, is_adapter_head = _build_inference_model(cfg, raw_ckpt)
    else:
        model, is_adapter_head = None, False
        print(f"[Eval] All splits already evaluated — skipping model load for {ckpt_name}")
    del raw_ckpt  # free CPU RAM after weights are loaded (or unused)

    try:
        fallback_log_size_before = fallback_log_path.stat().st_size if fallback_log_path.is_file() else 0
    except OSError:
        fallback_log_size_before = 0

    device = cfg["target_device"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {ckpt_name}\n")
        f.write(f"Dataset: {cfg['dataset_name']}\n")
        f.write(f"Device: {device}\n")

    # Shared kwargs for _run_split_evaluation (only used when model is not None).
    split_kwargs = dict(
        model=model,
        is_adapter_head=is_adapter_head,
        device=device,
        use_amp=cfg["use_amp"],
        amp_dtype=cfg["amp_dtype"],
        encoder_type=cfg["encoder_type"],
        log_eval_stats_flag=cfg["log_eval_stats_flag"],
        dataset_name=cfg["dataset_name"],
        ckpt_name=ckpt_name,
        out_path=out_path,
        fallback_log_path=fallback_log_path,
        common_dataset_kwargs=cfg["common_dataset_kwargs"],
        common_loader_kwargs=cfg["common_loader_kwargs"],
        common_eval_kwargs=cfg["common_eval_kwargs"],
    )

    # --- Dev split (optional) ---
    if not cfg["skip_dev"]:
        if dev_loaded is not None:
            dev_result = _write_split_results_from_loaded(
                dev_loaded,
                eer_label="Dev set",
                scores_label="Dev scores file:",
                scores_path=dev_scores_path,
                out_path=out_path,
                fallback_log_path=fallback_log_path,
                split_name="dev",
                dataset_name=cfg["dataset_name"],
                ckpt_name=ckpt_name,
                log_eval_stats_flag=cfg["log_eval_stats_flag"],
            )
        else:
            dev_result = _run_split_evaluation(
                split_name="dev",
                eer_label="Dev set",
                scores_label="Dev scores file:",
                protocol=cfg["dev_protocol"],
                audio_dir=cfg["dev_audio_dir"],
                scores_path=dev_scores_path,
                **split_kwargs,
            )
    else:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write("Dev set EER (%): skipped (skip_dev=True)\n")
            f.write("Dev set EER+CI none (%): skipped (skip_dev=True)\n")
            f.write("Dev set EER+CI speaker-conditioned (%): skipped (skip_dev=True)\n")
            f.write("Dev set EER+CI attack-conditioned (%): skipped (skip_dev=True)\n")
            f.write("Dev scores file: skipped (skip_dev=True)\n")
        dev_result = dict(metrics=None, triplet=None, scores_rows=None, scores_err=None, error=None)

    # --- Eval split ---
    if eval_loaded is not None:
        eval_result = _write_split_results_from_loaded(
            eval_loaded,
            eer_label="Evaluation set",
            scores_label="Evaluation scores file:",
            scores_path=eval_scores_path,
            out_path=out_path,
            fallback_log_path=fallback_log_path,
            split_name="eval",
            dataset_name=cfg["dataset_name"],
            ckpt_name=ckpt_name,
            log_eval_stats_flag=cfg["log_eval_stats_flag"],
        )
    else:
        eval_result = _run_split_evaluation(
            split_name="eval",
            eer_label="Evaluation set",
            scores_label="Evaluation scores file:",
            protocol=cfg["eval_protocol"],
            audio_dir=cfg["eval_audio_dir"],
            scores_path=eval_scores_path,
            **split_kwargs,
        )

    # Note any fallback log entries written during this run
    try:
        if fallback_log_path.is_file() and fallback_log_path.stat().st_size > fallback_log_size_before:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(f"One-class fallback notices file: {fallback_log_path}\n")
            print(f"Wrote one-class fallback notices to {fallback_log_path}")
    except OSError:
        pass

    print(f"Saved EER results to {out_path}")
    dev_eer_text = "skipped" if cfg["skip_dev"] else _fmt_float(_pick_display_eer_percent(dev_result["metrics"], dev_result["triplet"]))
    eval_eer_text = _fmt_float(_pick_display_eer_percent(eval_result["metrics"], eval_result["triplet"]))
    print(f"Dev EER: {dev_eer_text} | Eval EER: {eval_eer_text}")

    return {
        "dev_metrics": dev_result["metrics"],
        "eval_metrics": eval_result["metrics"],
        "dev_eer_triplet": dev_result["triplet"],
        "eval_eer_triplet": eval_result["triplet"],
        "dev_scores_path": str(dev_scores_path),
        "eval_scores_path": str(eval_scores_path),
        "dev_scores_rows": dev_result["scores_rows"],
        "eval_scores_rows": eval_result["scores_rows"],
        "dev_error": dev_result["error"],
        "eval_error": eval_result["error"],
        "output_path": str(out_path),
        # Pass model forward for sweep reuse; preserve prebuilt_model if we didn't load one.
        "_model": (model, is_adapter_head) if model is not None else prebuilt_model,
    }


def make_epoch_run_args_list(run_args: Dict[str, Any]) -> list:
    """Expand a sweep dict into per-checkpoint dicts.

    If run_args contains 'checkpoint_path', returns [run_args] unchanged.
    Otherwise expects:
      checkpoint_dir  — directory containing checkpoint-epochN.pt files
      epoch_start     — first epoch index (default: 1)
      epoch_end       — last epoch index (inclusive, required)
      epoch_step      — step between epochs (default: 1)
    plus all the usual run_args keys (config_path, dataset, etc.).
    """
    if "checkpoint_path" in run_args:
        return [run_args]

    checkpoint_dir = run_args.get("checkpoint_dir")
    epoch_end = run_args.get("epoch_end")
    if checkpoint_dir is None or epoch_end is None:
        raise ValueError("sweep run_args must have 'checkpoint_dir' and 'epoch_end'")

    epoch_start = int(run_args.get("epoch_start", 1))
    epoch_end = int(epoch_end)
    epoch_step = int(run_args.get("epoch_step", 1))
    if epoch_step == 0:
        raise ValueError("epoch_step must not be zero")
    if epoch_step < 0:
        raise ValueError(f"epoch_step must be positive (got {epoch_step}); to sweep in reverse order swap epoch_start and epoch_end")
    base = {k: v for k, v in run_args.items()
            if k not in ("checkpoint_dir", "epoch_start", "epoch_end", "epoch_step")}

    result = []
    for epoch in range(epoch_start, epoch_end + 1, epoch_step):
        ckpt_path = str(Path(checkpoint_dir) / f"checkpoint-epoch{epoch}.pt")
        result.append({"checkpoint_path": ckpt_path, **base})
    return result


DEFAULT_RUN_ARGS10 = {
    "checkpoint_path": "/path/to/voxtral_clean/voxtral_clean-generation-asvspoof5-aug-bfloat16-2-constant-classification_as_generation-with-aug21/checkpoint-epoch10.pt",
    "config_path": "/path/to/voxtral_clean/configs/hyperparameters_05_dif_margins21.yaml",
    "dataset": "asvspoof5",
    "dev_set": "/path/to/projects/ASVspoof5/ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv",
    "eval_set": "/path/to/projects/ASVspoof5/ASVspoof5_protocols/ASVspoof5.eval.track_1.tsv",
    "dev_audio_dir": "/path/to/projects/ASVspoof5/flac_D",
    "eval_audio_dir": "/path/to/projects/ASVspoof5/flac_E/flac_E_eval",
    "device": "cuda",
    "skip_dev": False,
}

DEFAULT_RUN_ARGS9 = {
    "checkpoint_path": "/path/to/voxtral_clean/voxtral_clean-generation-asvspoof5-aug-bfloat16-2-constant-classification_as_generation-with-aug21/checkpoint-epoch9.pt",
    "config_path": "/path/to/voxtral_clean/configs/hyperparameters_05_dif_margins21.yaml",
    "dataset": "asvspoof5",
    "dev_set": "/path/to/projects/ASVspoof5/ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv",
    "eval_set": "/path/to/projects/ASVspoof5/ASVspoof5_protocols/ASVspoof5.eval.track_1.tsv",
    "dev_audio_dir": "/path/to/projects/ASVspoof5/flac_D",
    "eval_audio_dir": "/path/to/projects/ASVspoof5/flac_E/flac_E_eval",
    "device": "cuda",
    "skip_dev": False,
}


# --- Sweep example: run epoch7..epoch15 ---
SWEEP_RUN_ARGS = {
    "checkpoint_dir": "/path/to/voxtral_clean/voxtral_clean-generation-asvspoof5-aug-bfloat16-2-constant-classification_as_generation-with-aug21",
    "epoch_start": 13,   # default 1
    "epoch_end": 15,
    "epoch_step": 1,    # default 1
    "config_path": "/path/to/voxtral_clean/configs/hyperparameters_05_dif_margins21.yaml",
    "dataset": "asvspoof5",
    "dev_set": "/path/to/projects/ASVspoof5/ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv",
    "eval_set": "/path/to/projects/ASVspoof5/ASVspoof5_protocols/ASVspoof5.eval.track_1.tsv",
    "dev_audio_dir": "/path/to/projects/ASVspoof5/flac_D",
    "eval_audio_dir": "/path/to/projects/ASVspoof5/flac_E/flac_E_eval",
    "device": "cuda",
    "skip_dev": False,
}


if __name__ == "__main__":
    run_args_list = [SWEEP_RUN_ARGS]
    #run_args_list = [DEFAULT_RUN_ARGS10, DEFAULT_RUN_ARGS9, SWEEP_RUN_ARGS]
    runnable_args: list[Dict[str, Any]] = []
    skipped_paths: list[str] = []
    for run_args in run_args_list:
        for expanded in make_epoch_run_args_list(run_args):
            if "checkpoint_path" not in expanded:
                raise KeyError(f"make_epoch_run_args_list returned a dict without 'checkpoint_path': {expanded}")
            ckpt_path = Path(str(expanded["checkpoint_path"]))
            if ckpt_path.exists():
                runnable_args.append(expanded)
            else:
                skipped_paths.append(str(ckpt_path))

    if skipped_paths:
        print("Skipping missing checkpoints at script start:")
        for path in skipped_paths:
            print(f"SKIP: {path}")
    else:
        print("No checkpoints skipped at script start.")

    failed_paths: list[str] = []
    _shared_model = None  # built once, reused across sweep iterations
    for run_args in runnable_args:
        try:
            result = run_inference(run_args, prebuilt_model=_shared_model)
            _shared_model = result.get("_model")  # carry forward for next checkpoint
        except Exception as _exc:
            ckpt = run_args.get("checkpoint_path", "<unknown>")
            print(f"[ERROR] Checkpoint failed, continuing sweep: {ckpt}\n  {type(_exc).__name__}: {_exc}")
            failed_paths.append(str(ckpt))
            _shared_model = None  # force full rebuild after a failure

    if failed_paths:
        print(f"\n{len(failed_paths)} checkpoint(s) failed during sweep:")
        for p in failed_paths:
            print(f"  FAILED: {p}")
