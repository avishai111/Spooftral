"""
utils/helper_function.py
========================
Shared utility functions used across training, evaluation, and inference.

Key responsibilities
--------------------
EER computation
    compute_eer / compute_eer_with_threshold — sklearn roc_curve-based EER
    with pos_label=1 (spoof).  Returns (nan, nan) on empty or single-class
    input; never raises.

CAG scoring
    compute_prob_for_sequence — computes the sum of log-probabilities of
    target token ids at each step (used to score "bonafide" vs "spoof").

Checkpoint selection
    get_resume_checkpoint_info — finds the most recently modified checkpoint
    in output_dir, preferring checkpoint-last.pt when mtime is equal.

Metrics logging
    append_metrics_to_file / log_metrics — append JSON-keyed metrics to
    per-epoch log files and optionally to W&B.

Miscellaneous helpers
    Config — thin wrapper turning a dict into an attribute-access object.
    autocast_ctx — returns the correct torch.autocast context for bf16/fp16/fp32.
    load_protocol_ASVspoof5 / load_protocol_ASVspoof2019 — read tab/space
    separated protocol files into lists of dicts.
    add_tekken_once_filter — suppresses repeated Tekken tokenizer log spam.
"""
import json
import math
import os
import re
import contextlib
import logging
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from torchaudio import transforms
from transformers import AutoProcessor, VoxtralForConditionalGeneration
try:
    from .augmentation import (
        WaveformAugmentationConfig,
        augment_waveform,
        get_default_augmentation_config,
        set_default_augmentation_config,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    from augmentation import (  # type: ignore
        WaveformAugmentationConfig,
        augment_waveform,
        get_default_augmentation_config,
        set_default_augmentation_config,
    )


class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        self.__dict__ = dictionary


def add_tekken_once_filter(logger_name: str = "mistral_common.tokens.tokenizers.tekken"):
    """
    Attach a once-only filter to suppress repeated Tekken tokenizer logs.
    - Adds logging.Filter that lets only the first record through for the given logger_name.
    - Returns the filter instance so callers can remove or inspect it.
    """
    class _TekkenOnceFilter(logging.Filter):
        def __init__(self):
            super().__init__()
            self._seen = False

        def filter(self, record):
            if self._seen:
                return False
            self._seen = True
            return True

    filt = _TekkenOnceFilter()
    logging.getLogger(logger_name).addFilter(filt)
    return filt

def load_text_prompts(file_path):
    """
    Load newline-delimited prompts from a text file.
    - Strips whitespace, ignores empty lines, returns list of prompt strings.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

# ========== Protocol Parsing ==========
def load_protocol_ASVspoof2019(protocol_path, data_dir, reasoning=False):
    """
    Parse ASVspoof2019 protocol file.
    - Returns list of dicts: audio_path (.flac), steps (optional), label ('bonafide'/'spoof').
    - Validates file existence under data_dir; skips malformed lines; raises on missing audio.
    """
    entries = []
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue  # skip malformed lines

            file_id = parts[1]
            label = parts[-1]
            label_text = "bonafide" if label == "bonafide" else "spoof"
            steps = parts[2] if reasoning and len(parts) > 2 else None

            audio_path = os.path.join(data_dir, file_id + ".flac")
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"File not found: {audio_path}")

            entry = {
                "audio_path": audio_path,
                "steps": steps,
                "label": label_text
            }
            entries.append(entry)

    print(f"Loaded {len(entries)} entries from {protocol_path}")
    return entries

# ========== Protocol Parsing ==========

def load_protocol_ASVspoof5(protocol_path, audio_dir, reasoning):
    """
    Parse ASVspoof5 protocol with optional reasoning text.
    - Returns list of dicts containing audio_path, steps (if reasoning=True), and label.
    - Skips malformed lines; builds .flac path under audio_dir without existence check.
    """
    entries = []
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 11:
                continue  # skip malformed lines

            # ASVspoof5 protocol is tab-separated; only select columns are used by position:
            # 0=speaker_id, 1=file_id, 2=gender, 6=codec, 7=attack_type, 8=label, 10=reasoning text.
            speaker_id = fields[0]
            file_id = fields[1]
            gender = fields[2]
            codec = fields[6]
            attack = fields[7]
            label = fields[8]
            if reasoning == True:
                steps = fields[10]

            audio_path = os.path.join(audio_dir, file_id + ".flac")
            if reasoning == True: 
                entry = {
                    "audio_path": audio_path,
                    "steps": steps,
                    "label": label
                }
            else:
                entry = {
                    "audio_path": audio_path,
                    "steps": None,
                    "label": label
                }
            entries.append(entry)
    return entries


def compute_eer_with_threshold(y_true, y_score):
    """
    Compute EER and the decision threshold at the EER operating point.
    Returns (nan, nan) on empty input or when only one class is present.
    The threshold is read from roc_curve; if it is non-finite (sklearn prepends
    max(score)+1 as thresholds[0]) it falls back to np.median(y_score).
    pos_label=1 is assumed (label 1 = spoof, higher score = more spoof).
    """
    import warnings

    if len(y_true) == 0 or len(y_score) == 0:
        warnings.warn("Cannot compute EER on empty arrays, returning nan")
        return float("nan"), float("nan")

    if len(y_true) != len(y_score):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} elements, y_score has {len(y_score)}")

    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_score, np.ndarray):
        y_score = np.array(y_score)

    if np.unique(y_true).size < 2:
        warnings.warn("Cannot compute EER with only one unique label, returning nan")
        return float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr

    diff = np.abs(fpr - fnr)
    if diff.size == 0:
        warnings.warn("Could not compute EER (no threshold points), returning nan")
        return float("nan"), float("nan")

    # EER is the operating point where FPR and FNR are equal; pick the threshold
    # that minimizes their gap and average the two rates there.
    eer_idx = int(np.argmin(diff))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    threshold = float("nan")
    if eer_idx < len(thresholds):
        threshold = float(thresholds[eer_idx])
    if not np.isfinite(threshold):
        # roc_curve prepends max(score)+1 as thresholds[0]; if EER lands there,
        # fall back to the median of y_score as a sensible decision boundary.
        warnings.warn(
            f"EER threshold is non-finite ({threshold}); falling back to median of scores.",
            stacklevel=2,
        )
        threshold = float(np.median(y_score))
    return float(eer), float(threshold)


def compute_eer(y_true, y_score):
    """
    Compute EER for binary classification. Returns nan on empty input or
    single-class input. Wraps compute_eer_with_threshold and discards the threshold.
    """
    eer, _ = compute_eer_with_threshold(y_true, y_score)
    return eer



def extract_label_from_output(text):
    """
    Extract last occurrence of spoof/bonafide label from text.
    - Case-insensitive search for bonafide/genuine/spoof tokens; returns 'unknown' if none found.
    - Uses last match to prefer final model decision in a string.
    """
    text = text.lower().strip()
    matches = list(re.finditer(r"\b(bonafide|bona-fide|genuine|spoof|spoofed)\b", text))
    if not matches:
        return "unknown"
    last_match = matches[-1].group(1)
    if last_match in ("bonafide", "bona-fide", "genuine"):
        return "bonafide"
    return "spoof"

def compute_prob_for_sequence(score_tensors, target_token_ids):
    """
    For each generation step i, return P(target_token_ids[i % len(target_token_ids)]).
    When target_token_ids has a single element (single-token label), this yields
    P(that token) at every generation step as a (num_steps,) tensor.
    Use the first element for a single-step label prediction.

    Previously this averaged across all target_token_ids per step, which produced
    (P(bona) + P(spoof)) / 2 when both ids were passed — a meaningless value.
    Now each step is matched to its own positional target id.
    """
    n_ids = len(target_token_ids)
    probs = []
    for i, logits in enumerate(score_tensors[0]):
        p = F.softmax(logits, dim=-1)
        tid = target_token_ids[i % n_ids]
        probs.append(p[tid])
    return torch.stack(probs)


def pad_mel_spec(mel, target_len=3000):
    """
    Pad or trim mel spectrogram to target_len along time axis.
    - Trims tail if longer; zero-pads at end if shorter.
    - Preserves mel bin dimension.
    """
    n_mels, orig_len = mel.shape
    if orig_len > target_len:
        mel = mel[:, :target_len]
    elif orig_len < target_len:
        pad_width = target_len - orig_len
        mel = torch.nn.functional.pad(mel, (0, pad_width))
    return mel


def append_metrics_to_file(step, eer, accuracy, filename):
    """
    Append step/eer/accuracy metrics to a TSV-like log file.
    - Writes tab-separated line with percent formatting for eer/accuracy.
    - Opens file in append mode; caller manages path.
    """
    with open(filename, "a") as f:
        f.write(f"{step}\t{eer:.5f}%\t{accuracy:.5f}%\n")


# ====== (Optional) Plot EER ======
def plot_single_eer_file(filename, label, out_png):
    """
    Plot EER vs step from a metrics file and save PNG.
    - Expects lines of step \t EER% in filename; strips '%' before plotting.
    - Saves figure to out_png and prints path; closes figure after saving.
    """
    # Load as strings so we can clean them
    raw_data = np.loadtxt(filename, dtype=str)
    
    # Remove '%' and convert to float
    steps = raw_data[:, 0].astype(float)
    eers = np.char.replace(raw_data[:, 1], '%', '').astype(float)
    
    plt.figure()
    plt.plot(steps, eers, label=label)
    plt.xlabel("Step")
    plt.ylabel("EER (%)")
    plt.legend()
    plt.title(f"{label} EER by Step")
    plt.grid()
    plt.savefig(out_png)
    plt.close()
    print(f"Saved: {out_png}")
    
    


def get_resume_checkpoint_info(cur_ckpts, configs):
    """
    Determine the checkpoint to resume from.
    Accepts a list of full checkpoint paths. Ignores when only_eval is set.
    Parses checkpoint-epoch*.pt / checkpoint-step*.pt names and selects the one
    with the highest mtime (not the highest numeric value, to avoid mixing
    epoch numbers and step numbers from different run modes).
    Falls back to mtime selection when no epoch/step pattern matches.
    Returns (resume_value, checkpoint_path) or (None, None) when nothing found.
    """
    if cur_ckpts and not configs.only_eval:
        print(
            "Warning: found previous run and gonna resume from that. the inputted `resume` argument is ignored!"
        )

        # normalize to list
        if isinstance(cur_ckpts, str):
            filenames = [os.path.basename(cur_ckpts)]
            ckpt_dir = os.path.dirname(cur_ckpts)
        else:
            filenames = [
                os.path.basename(f)
                for f in cur_ckpts
                if os.path.basename(f).startswith("checkpoint")
            ]
            ckpt_dir = os.path.join(configs.save_path, configs.name)
        
        # Regex patterns
        epoch_pattern = re.compile(r"checkpoint-epoch(\d+)\.pt$")
        step_pattern  = re.compile(r"checkpoint-step(\d+)\.pt$")

        parsed = []

        for name in filenames:
            m_epoch = epoch_pattern.search(name)
            m_step = step_pattern.search(name)
            if m_epoch:
                parsed.append(("epoch", int(m_epoch.group(1)), name))
            elif m_step:
                parsed.append(("step", int(m_step.group(1)), name))

        if not parsed:
            # Fallback: pick latest checkpoint file by modification time
            ckpt_files = [
                os.path.join(ckpt_dir, name)
                for name in os.listdir(ckpt_dir)
                if name.endswith((".bin", ".pt", ".ckpt", ".pth"))
            ]
            if not ckpt_files:
                return None, None
            latest_path = max(ckpt_files, key=os.path.getmtime)
            print(f"Resuming from latest checkpoint by mtime: {latest_path}")
            return 0, latest_path

        # Sort by mtime of the actual file so epoch/step numbers from different
        # run modes (epoch-based vs step-based) don't mislead the selection.
        # checkpoint-last.pt is preferred only when it is the newest file.
        def _ckpt_mtime(item):
            _kind, _val, _name = item
            _path = os.path.join(ckpt_dir, _name)
            try:
                return os.path.getmtime(_path)
            except OSError:
                return 0.0

        parsed.sort(key=_ckpt_mtime)
        latest_type, latest_value, latest_name = parsed[-1]

        ckpt_path = os.path.join(ckpt_dir, latest_name)
        print(f"Resuming from {latest_type} {latest_value}: {ckpt_path}")

        return latest_value, ckpt_path

    return None, None


def log_metrics(log_file, epoch=None, step=None, metrics=None, examples=None):
    """
    Append metrics to a text log file (epoch- or step-based).
    - Formats header from epoch or step; defaults to generic 'Metrics:' header.
    - Writes metric key/value pairs and optional examples (dicts or strings), separated by blank lines.
    """
    header = None
    if epoch is not None:
        header = f"Epoch {epoch + 1}:"
    elif step is not None:
        header = f"Step {step}:"
    else:
        header = "Metrics:"

    lines = [f"{header}\n"]
    if metrics:
        lines.extend([f"  {k}: {v}\n" for k, v in metrics.items()])
    if examples:
        # Normalize to a list for consistent formatting
        if not isinstance(examples, (list, tuple)):
            examples_list = [examples]
        else:
            examples_list = list(examples)

        for idx, ex in enumerate(examples_list, start=1):
            prefix = "example" if len(examples_list) == 1 else f"example_{idx}"
            if isinstance(ex, dict):
                payload = ", ".join(f"{k}={v}" for k, v in ex.items())
            else:
                payload = str(ex)
            lines.append(f"  {prefix}: {payload}\n")
    lines.append("\n")  # blank line between entries

    with open(log_file, "a") as f:
        f.writelines(lines)


_EPOCH_HEADER_PAT = re.compile(r"^Epoch\s+(\d+):\s*$")


def max_epoch_in_log(log_path):
    """
    Return the highest N appearing as "Epoch N:" in log_path, or None.

    log_metrics() writes a full eval entry (header + all metric lines) via a
    single writelines() call, so the presence of "Epoch N:" implies eval N
    completed. Used by resume logic to detect already-finished evals when the
    checkpoint's counters are stale (e.g. saved before eval finished).
    """
    if not log_path or not os.path.isfile(log_path):
        return None
    max_n = None
    try:
        with open(log_path, "r") as f:
            for line in f:
                m = _EPOCH_HEADER_PAT.match(line)
                if m:
                    n = int(m.group(1))
                    if max_n is None or n > max_n:
                        max_n = n
    except OSError:
        return None
    return max_n


def bump_last_eval_epoch_from_log(counter, log_path, split_label=""):
    """
    Reconcile a ckpt-loaded last_*_eval_epoch counter against the eval log.

    Returns max(counter, max_epoch_in_log(log_path)). Prints a notice when the
    log shows a higher epoch than the checkpoint (indicating the eval finished
    but its updated counter wasn't persisted before the crash/checkpoint write).
    """
    log_max = max_epoch_in_log(log_path)
    if log_max is None:
        return counter
    if counter is None or log_max > counter:
        prefix = f"[Resume{(' ' + split_label) if split_label else ''}]"
        if counter is not None:
            print(f"{prefix} Log {log_path} shows eval completed up to epoch {log_max} (ckpt had {counter}); bumping counter.")
        else:
            print(f"{prefix} Log {log_path} shows eval completed up to epoch {log_max}; setting counter.")
        return log_max
    return counter


_EER_LINE_PAT = re.compile(r"^\s{2}eer:\s+([0-9.eE+\-]+)\s*$")


def get_best_eer_from_log(log_path):
    """
    Scan a metrics log file and return the minimum EER value ever recorded.

    Guards against a resume where the checkpoint's best_eer is stale (e.g.
    the checkpoint was saved mid-epoch before the epoch's eval updated
    best_eer, or the checkpoint was from an earlier run that saw worse EER).

    Returns the minimum finite EER float, or None if the file does not exist
    or contains no parseable EER lines.
    """
    if not log_path or not os.path.isfile(log_path):
        return None
    best = None
    try:
        with open(log_path, "r") as f:
            for line in f:
                m = _EER_LINE_PAT.match(line)
                if m:
                    try:
                        val = float(m.group(1))
                    except ValueError:
                        continue
                    if math.isfinite(val) and (best is None or val < best):
                        best = val
    except OSError:
        return None
    return best


def log_eval_stats_table(log_file, header, threshold, attack_stats=None, codec_stats=None):
    """
    Append evaluation stats (per attack/codec) to a log file.
    - attack_stats/codec_stats expect mapping of name -> {'correct': int, 'total': int}.
    - Threshold is logged when provided; skips logging when no stats are available.
    """
    attack_stats = attack_stats or {}
    codec_stats = codec_stats or {}
    if not attack_stats and not codec_stats:
        return

    lines = [header]
    if threshold is not None:
        lines.append(f"Threshold@EER: {threshold:.6f}")

    if attack_stats:
        lines.append("Attack  | Correct / Total")
        for name in sorted(attack_stats.keys()):
            counts = attack_stats[name]
            lines.append(f"{name} | {counts.get('correct', 0)} / {counts.get('total', 0)}")

    if codec_stats:
        if attack_stats:
            lines.append("")  # blank line between tables
        lines.append("Codec | Correct / Total")
        for name in sorted(codec_stats.keys()):
            counts = codec_stats[name]
            lines.append(f"{name} | {counts.get('correct', 0)} / {counts.get('total', 0)}")

    lines.append("")  # trailing blank line for readability
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def maybe_log_eval_stats(enabled, log_file, header, eval_stats):
    """
    Guarded wrapper to log eval stats when enabled and available.
    - Avoids file writes when disabled, missing stats, or no log file path.
    """
    if not (enabled and eval_stats and log_file):
        return
    log_eval_stats_table(
        log_file=log_file,
        header=header,
        threshold=eval_stats.get("threshold_at_eer"),
        attack_stats=eval_stats.get("attack"),
        codec_stats=eval_stats.get("codec"),
    )


# ---------- AMP / grad accumulation ----------
def autocast_ctx(use_fp16: bool = False, use_bf16: bool = False, device: str = "cuda"):
    """
    Mixed-precision context helper.
    - Returns torch.cuda.amp.autocast for fp16, torch.autocast for bf16, otherwise nullcontext.
    - Keeps call sites concise when toggling precision modes.
    """
    if use_fp16:
        return torch.cuda.amp.autocast(dtype=torch.float16)
    if use_bf16:
        return torch.autocast(device_type=device, dtype=torch.bfloat16)
    return contextlib.nullcontext()



def load_prompts_json(filepath="train_cot_labeling.json"):
    """
    Load prompts JSON file and return parsed object.
    - Uses utf-8 encoding; caller handles structure expectations.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts


# ========== Audio helpers for data generation / inference ==========
def get_audio_files(directory: str, extensions: Optional[Sequence[str]] = None) -> List[str]:
    """
    Recursively list audio files under a directory.
    - Filters by common audio extensions; returns sorted absolute paths.
    """
    exts = extensions or (
        ".wav",
        ".flac",
        ".mp3",
        ".m4a",
        ".ogg",
        ".opus",
        ".aac",
        ".wma",
        ".ac3",
        ".alac",
        ".wv",
        ".g726",
        ".g722",
        ".gsm",
        ".amr",
        ".awb",
    )
    exts = tuple(e.lower() for e in exts)
    results: List[str] = []
    for root, _, files in os.walk(directory):
        for name in files:
            if name.lower().endswith(exts):
                results.append(os.path.join(root, name))
    return sorted(results)


# Process-wide cache so repeated calls reuse an already-loaded model/processor
# instead of reloading from disk each time.
_VOXTRAL_CACHE = {
    "model": None,
    "processor": None,
    "model_id": None,
    "device": None,
    "n_audio_tokens": None,
}


def _resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_voxtral_components(
    model_id: str,
    device: str,
    n_audio_tokens: int,
):
    cache = _VOXTRAL_CACHE
    # Only reload from disk when nothing is cached yet or a different model_id is
    # requested.
    if cache["model"] is None or cache["model_id"] != model_id:
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        model = VoxtralForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)
        model.eval()

        cache.update(
            {
                "model": model,
                "processor": processor,
                "model_id": model_id,
                "device": device,
                "n_audio_tokens": n_audio_tokens,
            }
        )
    else:
        # Model for this model_id is already cached; just move it to the newly
        # requested device if it changed.
        if cache["device"] != device:
            cache["model"] = cache["model"].to(device)
            cache["device"] = device
        cache["n_audio_tokens"] = n_audio_tokens
    return cache["processor"], cache["model"], cache["n_audio_tokens"]


def _build_voxtral_input_ids(processor, prompt: str, n_audio_tokens: int) -> torch.Tensor:
    vocab = processor.tokenizer.get_vocab()
    bos_id = vocab.get("<s>", processor.tokenizer.bos_token_id)
    inst_id = vocab.get("[INST]")
    endinst_id = vocab.get("[/INST]")
    aud_id = vocab.get("[AUDIO]")
    begin_aud_id = vocab.get("[BEGIN_AUDIO]")
    if inst_id is None or endinst_id is None or aud_id is None:
        raise ValueError("Required Voxtral chat tokens ([INST],[/INST],[AUDIO]) are missing from the tokenizer.")

    # Manually assembles Voxtral's chat-format token sequence — <s>[INST][BEGIN_AUDIO]
    # [AUDIO]*n_audio_tokens <prompt> [/INST] — for direct control over audio-token count.
    pieces = [torch.tensor([bos_id, inst_id], dtype=torch.long)]
    if begin_aud_id is not None:
        pieces.append(torch.tensor([begin_aud_id], dtype=torch.long))
    pieces.append(torch.full((n_audio_tokens,), aud_id, dtype=torch.long))
    prompt_ids = processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    pieces.append(prompt_ids)
    pieces.append(torch.tensor([endinst_id], dtype=torch.long))
    return torch.cat(pieces, dim=0)


def _decode_voxtral_output(processor, generated_ids: torch.Tensor, prompt_len: int) -> str:
    gen_only = generated_ids[prompt_len:]
    return processor.tokenizer.decode(gen_only, skip_special_tokens=True).strip()


def original_voxtral_inference(
    audio,
    prompt: str,
    *,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    n_audio_tokens: Optional[int] = None,
    max_new_tokens: int = 128,
):
    """
    Run the base Voxtral model on an audio tensor with a text prompt.
    - Accepts (waveform, sample_rate) tuple or waveform only (defaults to 16 kHz).
    - Caches processor/model for repeated calls.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    # Normalizes the flexible audio argument into (waveform, sample_rate) — accepts a
    # raw tensor (assumed 16 kHz), a (waveform, sample_rate) tuple, or a file path.
    waveform = audio
    sample_rate = 16_000
    if isinstance(audio, (list, tuple)) and len(audio) >= 2:
        waveform, sample_rate = audio[0], int(audio[1])
    elif isinstance(audio, str):
        from utils.codec import load_audio  # local import to avoid circular deps

        waveform, sample_rate = load_audio(audio)

    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16_000:
        waveform = transforms.Resample(orig_freq=sample_rate, new_freq=16_000)(waveform)
        sample_rate = 16_000

    model_id = model_id or os.getenv("VOXTRAL_MODEL", "mistralai/Voxtral-Small-24B-2507")
    n_audio_tokens = int(n_audio_tokens or os.getenv("N_AUDIO_TOKENS", 375))
    device = _resolve_device(device)

    processor, model, n_audio_tokens = _load_voxtral_components(
        model_id=model_id,
        device=device,
        n_audio_tokens=n_audio_tokens,
    )

    features = processor.feature_extractor(
        waveform, sampling_rate=sample_rate, return_tensors="pt"
    )["input_features"][0]

    input_ids = _build_voxtral_input_ids(processor, prompt, n_audio_tokens)
    attn = torch.ones_like(input_ids)

    input_features = features.unsqueeze(0).to(device)
    input_ids_b = input_ids.unsqueeze(0).to(device)
    attn_b = attn.unsqueeze(0).to(device)

    with torch.inference_mode():
        generated = model.generate(
            input_features=input_features,
            input_ids=input_ids_b,
            attention_mask=attn_b,
            max_new_tokens=max_new_tokens,
            num_beams=1,
        )
    return _decode_voxtral_output(processor, generated[0].cpu(), prompt_len=input_ids.shape[0])
