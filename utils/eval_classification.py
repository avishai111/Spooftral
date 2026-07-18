"""
eval_classification.py
======================
Evaluation pipeline for Classification-as-Generation (CAG) in Voxtral.

The single public entry point is evaluate_epoch_classification_as_generation,
which iterates over an eval DataLoader, scores logP("bonafide") and
logP("spoof") for every sample, and computes EER and related metrics.

Supporting private helpers:
  _get_tokenizer         : (imported from generation_text…)
  _extract_prompt_text   : Decode the human-readable prompt from a batch.
  _summarize_group       : Aggregate per-attack / per-codec accuracy counts.
"""

import gc
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.helper_function import compute_eer_with_threshold
from utils.training_helpers import eprint
from utils.generation_text_bonafide_spoof_classification import (
    build_cag_candidate,
    build_cag_candidates_batched,  # vectorised batch candidate builder
    make_lm_labels,
    score_label_logprob,
    pad_right,  # shared right-padding utility
)
from utils.config_helper_functions import resolve_label_texts


# ---------------------------------------------------------------------------
# _extract_prompt_text
# ---------------------------------------------------------------------------

def _extract_prompt_text(batch, processor, eval_loader, sample_idx: int = 0):
    """
    Decode the human-readable prompt string from a batch element.

    Used exclusively for debug logging and example_prediction reporting.
    The function skips BOS, instruction tokens, audio tokens, and ENDINST
    markers using metadata stored on the dataset object when available.

    Parameters
    ----------
    batch       : dict
        A collated batch from the DataLoader.  Must contain "input_ids".
    processor   : HF Processor
        Provides the underlying tokenizer used for decoding.
    eval_loader : DataLoader
        The DataLoader whose dataset may carry metadata attributes
        (_bos_inst_tensor, _begin_aud_tensor, _T_rapped_audio_tokens,
        _endinst_id) that define token-span boundaries.
    sample_idx  : int
        Index into the batch to decode (default 0 = first sample).

    Local variables
    ---------------
    input_ids    : torch.Tensor, (L,) — token ids for the selected sample.
    attn         : torch.Tensor or None — batch attention mask.
    seq_len      : int — non-padded sequence length for the selected sample.
    ids          : torch.Tensor, (seq_len,) on CPU — real tokens.
    ds           : Dataset or None — underlying dataset (may be None for Subset).
    prompt_start : int — index at which the human-readable prompt begins,
                   incremented by special prefix lengths.
    end_token_id : int or None — ENDINST token id; marks the end of the prompt.
    end_idx      : int — index one past the last prompt token.
    end_positions: torch.Tensor — positions of end_token_id in ids.
    prompt_tokens: torch.Tensor — the prompt token slice.
    prompt_text  : str — decoded prompt string.

    Returns
    -------
    str or None
        Decoded prompt text, stripped of leading/trailing whitespace.
        Returns None on any failure or when the resulting string is empty.
    """
    try:
        input_ids = batch["input_ids"][sample_idx]
        attn = batch.get("attention_mask")
        # Determine actual (non-padded) sequence length for this sample.
        seq_len = int(attn[sample_idx].sum().item()) if attn is not None else input_ids.size(0)
        ids = input_ids[:seq_len].detach().cpu()

        ds = getattr(eval_loader, "dataset", None)
        prompt_start = 0
        end_token_id = None

        # Walk the dataset metadata to skip non-prompt prefix tokens.
        if ds is not None:
            try:
                # Skip BOS + instruction prefix tokens if recorded on the dataset.
                prompt_start += len(getattr(ds, "_bos_inst_tensor", []))
            except Exception:
                pass
            try:
                # Skip the audio-begin sentinel token if present.
                if getattr(ds, "_begin_aud_tensor", None) is not None:
                    prompt_start += len(getattr(ds, "_begin_aud_tensor"))
            except Exception:
                pass
            try:
                # Skip the N audio feature tokens that represent the audio signal.
                prompt_start += int(getattr(ds, "_T_rapped_audio_tokens", 0))
            except Exception:
                pass
            # ENDINST token marks the end of the instruction/prompt span.
            end_token_id = getattr(ds, "_endinst_id", None)

        # Find the end of the prompt span (defaults to seq_len if no ENDINST found).
        end_idx = seq_len
        if end_token_id is not None:
            try:
                end_positions = (ids == end_token_id).nonzero(as_tuple=False)
                if end_positions.numel() > 0:
                    end_idx = int(end_positions[0].item())
            except Exception:
                pass

        # Slice the human-readable prompt tokens.
        if prompt_start >= end_idx:
            prompt_tokens = ids[:end_idx]
        else:
            prompt_tokens = ids[prompt_start:end_idx]

        prompt_text = processor.tokenizer.decode(prompt_tokens.tolist(), skip_special_tokens=True).strip()
        return prompt_text if prompt_text else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# _summarize_group
# ---------------------------------------------------------------------------

def _summarize_group(correct_mask, groups):
    """
    Aggregate classification correctness counts per group name.

    Used to break down accuracy by attack type (e.g. A01–A19) or codec
    (e.g. mp3, opus) for ASVspoof5 evaluation.

    Parameters
    ----------
    correct_mask : array-like of bool, length N
        True where the model predicted correctly for each sample.
    groups       : list of str or None, length N
        Group label for each sample.  None entries are excluded.

    Local variables
    ---------------
    summary      : dict — output mapping group name → {correct, total}.
    groups_arr   : np.ndarray — groups cast to object array.
    correct_mask : np.ndarray, bool — cast to numpy bool array.
    min_len      : int — minimum of the two array lengths (for safety trimming).
    valid_mask   : np.ndarray, bool — True where group entry is not None.
    unique_groups: np.ndarray — sorted unique group labels.
    grp_mask     : np.ndarray, bool — True for samples belonging to one group.
    total        : int — sample count for this group.
    correct      : int — correctly-classified sample count for this group.

    Returns
    -------
    dict
        Keys are group name strings; values are dicts with keys "correct"
        (int) and "total" (int).  Returns an empty dict if groups is empty.
    """
    summary = {}
    if not groups:
        return summary

    groups_arr = np.asarray(groups, dtype=object)
    correct_mask = np.asarray(correct_mask, dtype=bool)

    if groups_arr.shape[0] != correct_mask.shape[0]:
        raise RuntimeError(
            f"_summarize_group: length mismatch groups={groups_arr.shape[0]} "
            f"correct_mask={correct_mask.shape[0]} — data pipeline bug"
        )

    # Exclude samples whose group is None (metadata unavailable).
    valid_mask = np.array([g is not None for g in groups_arr], dtype=bool)
    if not np.any(valid_mask):
        return summary

    groups_arr = groups_arr[valid_mask]
    correct_mask = correct_mask[valid_mask]

    # Aggregate correct/total counts for each unique group label.
    unique_groups = np.unique(groups_arr)
    for group in unique_groups:
        grp_mask = groups_arr == group
        total = int(np.count_nonzero(grp_mask))
        correct = int(np.count_nonzero(correct_mask[grp_mask]))
        summary[str(group)] = {"correct": correct, "total": total}

    return summary


# ---------------------------------------------------------------------------
# evaluate_epoch_classification_as_generation
# ---------------------------------------------------------------------------

def evaluate_epoch_classification_as_generation(
    model,
    eval_loader,
    processor,
    device,
    generation_cfg=None,
    use_amp=False,
    amp_dtype="fp16",
    debug_log_path=None,
    eval_step=None,
    enable_debug_logging=True,
    return_example=False,
    return_eval_stats=False,
    return_per_sample=False,
    include_speaker_ids: bool = False,
    **kwargs,
) -> dict:
    """
    Run one full evaluation epoch using Classification-as-Generation scoring.

    For every batch the function:
      1. Builds combined (bonafide + spoof) candidate tensors in a single
         vectorised operation via build_cag_candidates_batched.
      2. Runs the model ONCE per batch (both candidates in a single forward).
      3. Extracts per-sample log-probabilities via score_label_logprob.
      4. Applies softmax over [logP_bona, logP_spoof] for EER computation.

    Parameters
    ----------
    model              : nn.Module
        The Voxtral model wrapper.
    eval_loader        : DataLoader
        Yields batches with keys: input_ids, attention_mask,
        classification_labels, and optionally input_features / raw_audio_input
        / raw_audio_lengths / attack_types / codecs / file_ids / speaker_ids.
    processor          : HF Processor
        Used for tokenization and prompt decoding.
    device             : torch.device
        Target compute device.
    generation_cfg     : dict or None
        Sub-configuration with keys:
          - label_texts (dict, {0: str, 1: str})
          - max_answer_tokens (int or None)
          - temperature (float, default 1.0)
          - normalize_logprob_by_tokens (bool, default True)
    use_amp            : bool
        Whether to wrap the forward pass in torch.autocast.
    amp_dtype          : str
        "fp16" or "bf16" — selects the autocast precision.
    debug_log_path     : str or None
        Path to a debug log file; forwarded to eprint.
    eval_step          : int or None
        Current global step number; shown in the tqdm progress bar.
    enable_debug_logging: bool
        If False, debug prints and tqdm outputs are suppressed.
    return_example     : bool
        If True, include the first decoded sample's prediction in the output
        dict under key "example_prediction".
    return_eval_stats  : bool
        If True, include per-attack / per-codec accuracy counts in the output
        dict under key "eval_stats".
    return_per_sample  : bool
        If True, include per-sample scores, labels, file_ids, and optionally
        attack types and speaker_ids in the output dict.
    include_speaker_ids: bool
        If True (and return_per_sample=True), include speaker_id column.
    **kwargs
        Silently absorbed; not forwarded.

    Local variables
    ---------------
    gen_cfg                  : dict — resolved generation config (defaults filled).
    label_texts              : dict {0: str, 1: str} — class label strings.
    prompt_template          : str — empty; prompt is encoded in input_ids already.
    max_answer_tokens        : int or None — label token truncation limit.
    temperature              : float — logit temperature for scoring.
    normalize_logprob_by_tokens: bool — whether to divide logprob by token count.
    all_scores_batches       : list of np.ndarray — batch-wise spoof probabilities.
    all_labels_batches       : list of np.ndarray — batch-wise ground-truth labels.
    score_diffs_batches      : list of np.ndarray — batch-wise logP_spoof - logP_bona.
    total_loss_sum           : float — accumulated CE loss numerator.
    total_eval_samples       : int — total number of evaluated samples.
    attack_types_all         : list of str — per-sample attack type strings.
    codecs_all               : list of str — per-sample codec strings.
    file_ids_all             : list — per-sample file identifiers.
    speaker_ids_all          : list — per-sample speaker identifiers.
    speaker_ids_present      : bool — True if any non-trivial speaker id was seen.
    collect_speakers         : bool — whether to collect speaker ids this run.
    collect_eval_stats       : bool — whether to collect attack/codec stats.
    collect_attacks          : bool — whether to collect attack/codec metadata.
    _tok_eval                : Tokenizer — unwrapped from processor.
    _eval_label0_ids         : torch.Tensor, (L_label0,) on device — bonafide ids.
    _eval_label1_ids         : torch.Tensor, (L_label1,) on device — spoof ids.
    ac                       : context manager — autocast or nullcontext.
    tqdm_desc                : str — progress bar description.
    example_info             : dict or None — decoded first-sample example.
    batch                    : dict — one collated batch from eval_loader.
    bs                       : int — batch size for this batch.
    pad_id                   : int — tokenizer pad token id.
    cand_b / cand_s          : (B, L_c) — bonafide / spoof candidate input_ids.
    attn_b / attn_s          : (B, L_c) — corresponding attention masks.
    mask_b / mask_s          : (B, L_c), bool — label position masks.
    labels_b / labels_s      : (B, L_c) — LM label tensors.
    L_max_ev                 : int — max sequence length for combined batch.
    cand_all_ev              : (2B, L_max_ev) — stacked candidates.
    attn_all_ev              : (2B, L_max_ev) — stacked attention masks.
    mask_all_ev              : (2B, L_max_ev), bool — stacked label masks.
    labels_all_ev            : (2B, L_max_ev) — stacked LM labels.
    pos_all_ev               : (2B, L_max_ev) — position ids.
    _model_unwrapped         : base model (strips DDP wrapper).
    audio_embeds_cache       : tensor or None — shared audio embeddings.
    fwd_kwargs_ev            : dict — extra forward arguments (audio embeddings).
    logp_all_ev              : (2B,) — combined log-probs.
    logp_b / logp_s          : (B,) — bonafide / spoof log-probs split from above.
    logps                    : (B, 2) — stacked [logP_bona, logP_spoof].
    probs                    : (B, 2) — softmax probabilities.
    prob_spoof_batch         : (B,) float32 CPU — spoof probability per sample.
    score_diff_batch         : (B,) float32 CPU — logP_spoof - logP_bona.
    cls_labels_batch         : (B,) long — ground-truth classification labels.
    batch_loss_sum           : scalar — sum of CE loss for this batch.
    scores_np                : (N,) float32 — all spoof probabilities concatenated.
    labels_np                : (N,) int32 — all ground-truth labels concatenated.
    score_diffs_np           : (N,) float32 — all score diffs concatenated.
    eer_softmax              : float — EER from softmax scores.
    threshold_softmax        : float — threshold at EER.
    eer_score_diff           : float — EER from raw score-diff scores.
    mean_diff                : float — mean score diff across eval set.
    mean_prob_spoof          : float — mean spoof probability across eval set.
    avg_loss                 : float — sample-weighted mean CE loss.
    preds                    : (N,) int32 — predicted labels at threshold_softmax.
    correct_mask             : (N,) bool — True where prediction matches label.
    attack_summary           : dict — per-attack correct/total counts.
    codec_summary            : dict — per-codec correct/total counts.
    eval_stats               : dict — combined stats dict.

    Returns
    -------
    dict with keys:
        "loss"              : float — sample-weighted mean CE loss over eval set.
        "eer"               : float — EER (%) from softmax scores (same as eer_softmax).
        "eer_softmax"       : float — EER (%) from softmax probabilities.
        "eer_score_diff"    : float — EER (%) from raw logP_spoof - logP_bona.
        "mean_score_diff"   : float — mean(logP_spoof - logP_bona).
        "mean_prob_spoof"   : float — mean softmax spoof probability.
        "example_prediction": dict  [only when return_example=True]
            Decoded first-sample info with keys: file_id, label, pred,
            llm_output, prompt, prob_spoof, score_diff.
        "eval_stats"        : dict  [only when return_eval_stats=True]
            Keys: threshold_at_eer, attack (per-attack summary),
            optionally codec (per-codec summary).
        "scores"            : np.ndarray, float64  [only when return_per_sample=True]
        "labels"            : np.ndarray, int64    [only when return_per_sample=True]
        "file_id"           : np.ndarray, object   [only when return_per_sample=True]
        "attack"            : np.ndarray, object   [only when return_per_sample=True]
        "speaker_id"        : np.ndarray, object   [only when return_per_sample=True
                                                    and include_speaker_ids=True]
    """
    _was_training = model.training  # save caller's mode to restore in finally
    model.eval()
    gc.collect()

    # Flush VRAM cache before evaluation to get a clean memory baseline.
    if enable_debug_logging and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Resolve generation config with safe defaults.
    gen_cfg = generation_cfg or {}
    label_texts = resolve_label_texts(gen_cfg.get("label_texts", {}), context_label="Eval")
    if not label_texts.get(0) or not label_texts.get(1):
        raise ValueError(
            f"generation_classification.label_texts must define entries for keys 0 and 1. "
            f"Got: {dict(label_texts)}"
        )

    # Warn if label_texts[1] does not look like a spoof/fake class.
    # EER and score direction assume pos_label=1 is the spoof/fake class.
    _SPOOF_SYNONYMS = {"spoof", "spoofed", "fake", "synthetic", "generated"}
    _spoof_is_label1 = any(w in str(label_texts.get(1, "")).lower() for w in _SPOOF_SYNONYMS)
    if not _spoof_is_label1:
        import warnings
        warnings.warn(
            f"label_texts[1]={label_texts.get(1)!r} does not look like a spoof class. "
            "EER and score direction assume pos_label=1 is spoof. Check your label_texts config."
        )
    prompt_template = ""  # prompt is already encoded in each sample's input_ids
    max_answer_tokens = gen_cfg.get("max_answer_tokens", None)
    temperature = gen_cfg.get("temperature", 1.0)
    normalize_logprob_by_tokens = bool(gen_cfg.get("normalize_logprob_by_tokens", True))

    # Per-batch accumulators — collected as lists and concatenated post-loop
    # to avoid repeated numpy array growth.
    all_scores_batches, all_labels_batches = [], []
    score_diffs_batches = []

    # Use separate sum + count accumulators so that the final mean is correctly
    # sample-weighted (averaging batch means over-weights the smaller last batch
    # when drop_last=False).
    total_loss_sum = 0.0
    total_eval_samples = 0

    attack_types_all, codecs_all = [], []
    file_ids_all = []
    speaker_ids_all = []
    speaker_ids_present = False
    collect_speakers = bool(include_speaker_ids) and bool(return_per_sample)
    collect_eval_stats = bool(return_eval_stats)
    collect_attacks = collect_eval_stats or bool(return_per_sample)

    # Pre-tokenise both label strings ONCE before the batch loop.  Moving to
    # device here (rather than per-batch) eliminates repeated Python dispatcher
    # overhead for tensors that never change.
    _tok_eval = getattr(processor, "tokenizer", processor)
    _eval_label0_ids = _tok_eval(label_texts[0], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    _eval_label1_ids = _tok_eval(label_texts[1], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if max_answer_tokens is not None and max_answer_tokens > 0:
        _eval_label0_ids = _eval_label0_ids[:max_answer_tokens]
        _eval_label1_ids = _eval_label1_ids[:max_answer_tokens]
    # Move to device once outside the loop — they are constant across all batches.
    _eval_label0_ids = _eval_label0_ids.to(device)
    _eval_label1_ids = _eval_label1_ids.to(device)

    # Invariant #8: without per-token normalization, the shorter label wins by raw
    # log-prob sum unconditionally, silently biasing every score and destroying EER.
    if not normalize_logprob_by_tokens and _eval_label0_ids.shape[0] != _eval_label1_ids.shape[0]:
        raise ValueError(
            f"normalize_logprob_by_tokens=False with labels of different token lengths is "
            f"mathematically unsound: '{label_texts[0]}' tokenizes to "
            f"{_eval_label0_ids.shape[0]} token(s) but '{label_texts[1]}' tokenizes to "
            f"{_eval_label1_ids.shape[0]} token(s). The shorter label accumulates a higher "
            f"total log-prob on every sample regardless of content, silently biasing EER. "
            f"Fix: set normalize_logprob_by_tokens: true in your YAML (CLAUDE.md invariant #8)."
        )

    # Set up the autocast context for mixed-precision evaluation.
    _eval_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if use_amp and amp_dtype == "bf16":
        ac = torch.autocast(device_type=_eval_device_type, dtype=torch.bfloat16)
    elif use_amp and amp_dtype == "fp16":
        ac = torch.autocast(device_type=_eval_device_type, dtype=torch.float16)
    else:
        ac = nullcontext()  # no-op context: runs at default dtype

    tqdm_desc = f"Evaluating (step {eval_step})" if eval_step is not None else "Evaluating"
    example_info = None

    with torch.inference_mode():  # disable gradient tracking for all eval ops
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc=tqdm_desc, leave=False)):
            # Move tensor-valued batch entries to the target device non-blockingly.
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            bs = batch["input_ids"].size(0)

            # Label token tensors are already on device (moved once before this loop).
            label0_ids_dev = _eval_label0_ids
            label1_ids_dev = _eval_label1_ids
            pad_id = processor.tokenizer.pad_token_id

            # Build candidates for BOTH labels using the vectorised helper.
            # This replaces an O(bs) per-sample Python loop with two tensor ops.
            cand_b, attn_b, mask_b = build_cag_candidates_batched(
                batch["input_ids"], batch["attention_mask"], label0_ids_dev, pad_id
            )
            labels_b = make_lm_labels(cand_b, mask_b)

            cand_s, attn_s, mask_s = build_cag_candidates_batched(
                batch["input_ids"], batch["attention_mask"], label1_ids_dev, pad_id
            )
            labels_s = make_lm_labels(cand_s, mask_s)

            # Pad both candidate batches to the same length and concatenate them
            # into a single 2B-row batch so the model runs once per eval batch.
            L_max_ev = max(cand_b.size(1), cand_s.size(1))
            cand_all_ev   = torch.cat([pad_right(cand_b,        L_max_ev, pad_id),
                                       pad_right(cand_s,        L_max_ev, pad_id)], dim=0)
            del cand_b, cand_s  # free before creating the next combined tensor
            attn_all_ev   = torch.cat([pad_right(attn_b,        L_max_ev, 0),
                                       pad_right(attn_s,        L_max_ev, 0)],      dim=0)
            del attn_b, attn_s
            mask_all_ev   = torch.cat([pad_right(mask_b.long(), L_max_ev, 0).bool(),
                                       pad_right(mask_s.long(), L_max_ev, 0).bool()], dim=0)
            del mask_b, mask_s  # not used after folding into the combined mask
            labels_all_ev = torch.cat([pad_right(labels_b,      L_max_ev, -100),
                                       pad_right(labels_s,      L_max_ev, -100)],   dim=0)
            del labels_b, labels_s

            # Explicit position ids prevent the model from using stale KV-cache
            # positions when audio embeddings are injected at the beginning.
            pos_all_ev = (
                torch.arange(L_max_ev, device=device, dtype=torch.long)
                .unsqueeze(0).expand(2 * bs, -1)
            )

            with ac:
                # Compute audio embeddings from the first B rows (one set per
                # sample).  Tile to 2B by repeating so both bonafide and spoof
                # candidate rows share the same audio context.
                _model_unwrapped = model.module if hasattr(model, "module") else model
                audio_embeds_cache = _model_unwrapped.compute_audio_embeds(
                    input_features=batch.get("input_features"),
                    input_ids=batch["input_ids"],
                    raw_audio_input=batch.get("raw_audio_input"),
                    raw_audio_lengths=batch.get("raw_audio_lengths"),
                )
                fwd_kwargs_ev = {}
                if audio_embeds_cache is not None:
                    # Tile: first B rows for bonafide, next B rows for spoof.
                    fwd_kwargs_ev["precomputed_audio_embeds"] = torch.cat(
                        [audio_embeds_cache, audio_embeds_cache], dim=0
                    )
                del audio_embeds_cache  # original (non-tiled) copy no longer needed

                # Single model forward for both label candidates in one pass.
                logp_all_ev = score_label_logprob(
                    model,
                    cand_all_ev,
                    attn_all_ev,
                    mask_all_ev,
                    labels=labels_all_ev,
                    temperature=temperature,
                    normalize_by_length=normalize_logprob_by_tokens,
                    position_ids=pos_all_ev,
                    **fwd_kwargs_ev,
                )
                # Split the 2B-row output into bonafide (first B) and spoof (last B).
                logp_b = logp_all_ev[:bs]
                logp_s = logp_all_ev[bs:]
                del logp_all_ev   # combined tensor freed now that splits are held
                del fwd_kwargs_ev  # tiled audio embeddings no longer needed

            # Stack log-probs and compute softmax classification probabilities.
            # score_label_logprob is the single authority on normalisation;
            # no additional post-hoc division is applied here.
            logps = torch.stack([logp_b, logp_s], dim=1)            # (B, 2): column 0 = bonafide, column 1 = spoof
            probs = torch.softmax(logps, dim=1)                      # (B, 2)
            prob_spoof_batch = probs[:, 1].detach().float().cpu()    # (B,)
            score_diff_batch = (logps[:, 1] - logps[:, 0]).detach().float().cpu()  # (B,)

            # Accumulate CE loss using sum + count to enable sample-weighted averaging.
            # Note: logps contains length-normalised log-probs; cross_entropy is applied
            # on top of them as discriminative scores, which is consistent with how the
            # training loss is computed.
            cls_labels_batch = batch["classification_labels"].long()
            batch_loss_sum = F.cross_entropy(logps, cls_labels_batch, reduction="sum")
            total_loss_sum += float(batch_loss_sum.detach().cpu().item())
            total_eval_samples += bs

            # Append batch results to the accumulators (staying on CPU).
            all_scores_batches.append(prob_spoof_batch.numpy())
            all_labels_batches.append(cls_labels_batch.detach().cpu().numpy().astype("int32"))
            score_diffs_batches.append(score_diff_batch.numpy())

            if collect_attacks:
                attack_types_all.extend(batch.get("attack_types", []))
                codecs_all.extend(batch.get("codecs", []))

            if return_per_sample:
                file_ids_batch = batch.get("file_ids", None)
                if isinstance(file_ids_batch, (list, tuple)):
                    file_ids_all.extend(file_ids_batch)
                else:
                    file_ids_all.extend([None] * bs)

                if collect_speakers:
                    speaker_ids_batch = batch.get("speaker_ids", None)
                    if isinstance(speaker_ids_batch, (list, tuple)) and len(speaker_ids_batch) == bs:
                        speaker_ids_all.extend(speaker_ids_batch)
                        # Mark speaker_ids as present if any non-trivial value is found.
                        if any(
                            (s is not None and str(s).strip() and str(s).strip().lower() != "unknown")
                            for s in speaker_ids_batch
                        ):
                            speaker_ids_present = True
                    else:
                        speaker_ids_all.extend([None] * bs)

            # Capture the decoded prompt and prediction for the first sample seen.
            if return_example and example_info is None and bs > 0:
                try:
                    file_ids = batch.get("file_ids", None)
                    file_id = file_ids[0] if isinstance(file_ids, (list, tuple)) and len(file_ids) > 0 else None
                except Exception:
                    file_id = None
                prob0 = float(prob_spoof_batch[0].item())
                prompt_text = _extract_prompt_text(batch, processor, eval_loader, sample_idx=0)
                pred_label_text = label_texts[1 if prob0 > 0.5 else 0]
                example_info = {
                    "file_id": file_id,
                    "label": int(batch["classification_labels"][0].item()),
                    "pred": int(prob0 > 0.5),
                    "llm_output": pred_label_text,
                    "prompt": prompt_text,
                    "prob_spoof": prob0,
                    "score_diff": float(score_diff_batch[0].item()),
                }

            # Debug print for the very first batch to verify scoring sanity.
            if bs > 0 and batch_idx == 0:
                score_prob0 = prob_spoof_batch[0]
                score_diff0 = score_diff_batch[0]
                label0 = int(batch["classification_labels"][0].item())
                pred0 = 1 if score_prob0.item() > 0.5 else 0
                eprint(
                    debug_log_path,
                    enable_debug_logging,
                    "[DEBUG][EVAL CAG] logP(bona), logP(spoof), diff, prob_spoof, label, pred:",
                    float(logps[0, 0].item()),
                    float(logps[0, 1].item()),
                    float(score_diff0.item()),
                    float(score_prob0.item()),
                    label0,
                    pred0,
                )

    # --- Post-loop: compute aggregate metrics ---
    try:
        # Guard against empty eval set (no batches yielded by the loader).
        if not all_scores_batches or not all_labels_batches:
            return {
                "loss": float("nan"),
                "eer": float("nan"),
                "eer_softmax": float("nan"),
                "eer_score_diff": float("nan"),
                "mean_score_diff": float("nan"),
                "mean_prob_spoof": float("nan"),
            }

        # Concatenate all per-batch accumulators into single arrays.
        scores_np = np.concatenate(all_scores_batches).astype("float32")
        labels_np = np.concatenate(all_labels_batches).astype("int32")
        score_diffs_np = (
            np.concatenate(score_diffs_batches).astype("float32")
            if score_diffs_batches
            else np.array([], dtype="float32")
        )

        # Compute EER from softmax probabilities.
        eer_softmax, threshold_softmax = compute_eer_with_threshold(y_true=labels_np, y_score=scores_np)

        # Guard against NaN EER (e.g. empty eval set or single-class batch).
        if np.isnan(eer_softmax):
            import logging as _logging
            _logger = _logging.getLogger(__name__)
            _logger.warning(
                "EER computation returned nan (empty or single-class batch); skipping metrics."
            )
            return {
                "loss": float("nan"),
                "eer": float("nan"),
                "eer_softmax": float("nan"),
                "eer_score_diff": float("nan"),
                "mean_score_diff": float("nan"),
                "mean_prob_spoof": float("nan"),
            }

        # Guard against length mismatch between score diffs and labels.
        if score_diffs_np.size != labels_np.size:
            min_len = min(score_diffs_np.size, labels_np.size)
            print(
                f"[Warning] score_diffs size ({score_diffs_np.size}) != labels size "
                f"({labels_np.size}); truncating to {min_len}. This should not happen "
                "— check for empty batches or batch-size mismatches."
            )
            score_diffs_np = score_diffs_np[:min_len]
            labels_score_np = labels_np[:min_len]
        else:
            labels_score_np = labels_np

        # Compute EER from raw score-difference metric (logP_spoof - logP_bona).
        eer_score_diff, _ = compute_eer_with_threshold(y_true=labels_score_np, y_score=score_diffs_np)
        mean_diff = float(score_diffs_np.mean()) if score_diffs_np.size > 0 else 0.0
        mean_prob_spoof = float(scores_np.mean()) if scores_np.size > 0 else 0.0

        # Sample-weighted mean loss: sum(batch_loss) / total_samples.
        avg_loss = total_loss_sum / max(1, total_eval_samples)

        # Guard * 100.0 against non-finite EER values.
        eer_softmax_pct = eer_softmax * 100.0 if np.isfinite(eer_softmax) else float("nan")
        eer_score_diff_pct = eer_score_diff * 100.0 if np.isfinite(eer_score_diff) else float("nan")

        eprint(
            debug_log_path,
            enable_debug_logging,
            "  EER_softmax:",
            eer_softmax_pct,
            "EER_score_diff:",
            eer_score_diff_pct,
            "mean_diff:",
            mean_diff,
            "mean_prob_spoof:",
            mean_prob_spoof,
            "avg_loss:",
            avg_loss,
        )

        # Build the primary metrics dict.
        metrics = {
            "loss": avg_loss,
            "eer": eer_softmax_pct,         # main headline metric
            "eer_softmax": eer_softmax_pct,
            "eer_score_diff": eer_score_diff_pct,
            "mean_score_diff": mean_diff,
            "mean_prob_spoof": mean_prob_spoof,
        }

        # Per-sample arrays for detailed analysis (e.g. DET curves, error analysis).
        if return_per_sample:
            metrics["scores"] = scores_np.astype("float64")
            metrics["labels"] = labels_np.astype("int64")
            # Always include file_id when return_per_sample=True so callers
            # can unconditionally index metrics["file_id"].  Fill missing IDs with None.
            metrics["file_id"] = np.asarray(
                file_ids_all if file_ids_all else [None] * len(scores_np), dtype=object
            )
            if collect_speakers and speaker_ids_present:
                metrics["speaker_id"] = np.asarray(speaker_ids_all, dtype=object)
            if attack_types_all:
                metrics["attack"] = np.asarray(attack_types_all, dtype=object)

        # Per-attack / per-codec correctness breakdown.
        if collect_eval_stats:
            preds = (scores_np >= float(threshold_softmax)).astype("int32")
            correct_mask = preds == labels_np[: preds.shape[0]]
            attack_summary = _summarize_group(correct_mask, attack_types_all)
            codec_summary = (
                _summarize_group(correct_mask, codecs_all)
                if any(c is not None for c in codecs_all)
                else {}
            )
            eval_stats = {
                "threshold_at_eer": float(threshold_softmax),
                "attack": attack_summary,
            }
            if codec_summary:
                eval_stats["codec"] = codec_summary
            metrics["eval_stats"] = eval_stats

        if return_example and example_info is not None:
            # Re-label example_info with the calibrated EER threshold instead of 0.5.
            if np.isfinite(threshold_softmax):
                example_info["pred"] = int(example_info["prob_spoof"] >= float(threshold_softmax))
                example_info["llm_output"] = label_texts[example_info["pred"]]
            metrics["example_prediction"] = example_info

        return metrics

    finally:
        # Restore the model's original train/eval mode so callers that set
        # eval mode before this function (e.g. inference scripts) are not
        # silently switched back to train mode with active dropout.
        model.train(_was_training)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
