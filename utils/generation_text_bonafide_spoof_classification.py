"""
generation_text_bonafide_spoof_classification.py
================================================
Classification-as-Generation (CAG) scoring utilities for Voxtral.

The core idea: instead of a classification head, we score how probable each
class label string is under the LM given an audio-conditioned prompt.  The
class with the highest log-probability wins.  All public helpers here build
or consume those candidate sequences and compute their log-probabilities.

Public API
----------
  pad_right                     : Right-pad a 2-D tensor along dim=1.
  build_cag_candidate           : Build one [prompt | label] token sequence.
  build_cag_candidates_batched  : Vectorised batch version (no Python loop).
  make_lm_labels                : Mask out prompt tokens for teacher-forcing.
  score_label_logprob           : Run model forward and compute label logprob.
"""

import torch
import torch.nn.functional as F
from utils.losses import for_causal_lm_loss


# ---------------------------------------------------------------------------
# pad_right
# ---------------------------------------------------------------------------

def pad_right(t: torch.Tensor, target_len: int, pad_value) -> torch.Tensor:
    """
    Right-pad a 2-D tensor along dim=1 to *target_len*.

    Used whenever two candidate tensors (bonafide/spoof) need to be stacked
    into a single batch of uniform sequence length before a model forward pass.

    Parameters
    ----------
    t          : torch.Tensor, shape (B, L)
        The tensor to pad.  B = batch size, L = current sequence length.
    target_len : int
        Desired final length along dim=1.  If L >= target_len the tensor is
        returned unchanged (no truncation is performed).
    pad_value  : scalar (int or float)
        Value to fill the appended columns with.  Typically the tokenizer's
        pad_token_id for input_ids or -100 for label tensors.

    Local variables
    ---------------
    extra : int
        Number of columns to append (target_len - t.size(1)).  Non-positive
        means no padding is needed.

    Returns
    -------
    torch.Tensor, shape (B, target_len)
        Original tensor with *extra* columns of *pad_value* appended on the
        right.  Returns *t* unmodified when no padding is needed.
    """
    extra = target_len - t.size(1)
    if extra <= 0:
        # Already at or beyond target length — nothing to do.
        return t
    # Allocate a new_full block of shape (B, extra) filled with pad_value and
    # concatenate it on the right (dim=1).
    return torch.cat([t, t.new_full((t.size(0), extra), pad_value)], dim=1)


# ---------------------------------------------------------------------------
# _get_tokenizer
# ---------------------------------------------------------------------------

def _get_tokenizer(proc_or_tok):
    """
    Unwrap a HuggingFace Processor to a bare tokenizer.

    HuggingFace multimodal processors wrap an underlying tokenizer as the
    `.tokenizer` attribute.  This helper accepts either a Processor or a raw
    tokenizer and returns whichever is the actual tokenizer object.  Avoids
    isinstance checks scattered throughout the codebase.

    Parameters
    ----------
    proc_or_tok : HuggingFace Processor or Tokenizer
        Either a processor (which has a `.tokenizer` attribute) or a tokenizer
        directly.

    Returns
    -------
    HuggingFace Tokenizer
        The underlying tokenizer object.
    """
    # getattr returns the .tokenizer sub-object if it exists, otherwise
    # falls back to the object itself (already a tokenizer).
    return getattr(proc_or_tok, "tokenizer", proc_or_tok)


# ---------------------------------------------------------------------------
# build_cag_candidate
# ---------------------------------------------------------------------------

def build_cag_candidate(
    processor_or_tokenizer,
    prompt_template,  # ignored; base prompt already included in input_ids
    label_text,
    base_input_ids,
    attention_mask,
    max_answer_tokens=None,
    precomputed_label_tokens=None,
):
    """
    Build a single candidate sequence for Classification-as-Generation scoring.

    The base_input_ids already contain the full audio-conditioned prompt
    (audio tokens + instruction text).  This function tokenizes *label_text*
    and appends it to produce the full candidate: [prompt | label_tokens].
    The resulting tensors are ready to be fed into score_label_logprob.

    Parameters
    ----------
    processor_or_tokenizer : HF Processor or Tokenizer
        Source for tokenizing label_text.
    prompt_template : str
        Unused placeholder kept for API compatibility.  The prompt is already
        encoded inside base_input_ids; appending it again would double-count.
    label_text : str
        The class label string to append (e.g. "bonafide" or "spoof").
    base_input_ids : torch.Tensor, shape (L_prompt,)
        1-D integer token ids of the prompt.  Only the first *base_len* tokens
        (determined by attention_mask) are used; trailing padding is stripped.
    attention_mask : torch.Tensor or None, shape (L_prompt,)
        1-D binary mask.  The number of 1s gives *base_len*, the real prompt
        length excluding padding.  Pass None to treat the entire sequence as
        non-padding.
    max_answer_tokens : int or None
        If set, truncate the tokenised label sequence to this many tokens.
        Useful to prevent pathologically long label strings from dominating.
    precomputed_label_tokens : torch.Tensor or None, shape (L_label,)
        Optional pre-tokenized label IDs.  When provided, tokenization of
        label_text is skipped entirely.  Callers should precompute this once
        per label per run to avoid redundant tokenizer calls inside per-sample
        loops.

    Local variables
    ---------------
    tok              : Tokenizer — unwrapped from processor_or_tokenizer.
    device           : torch.device — device of base_input_ids.
    base_len         : int — real prompt length (sum of attention_mask ones).
    base_ids         : torch.Tensor, (base_len,) — prompt tokens without padding.
    label_tokens     : torch.Tensor, (L_label,) — tokenized label on device.
    candidate_ids    : torch.Tensor, (base_len + L_label,) — full candidate.
    attn_base        : torch.Tensor — prompt portion of the attention mask.
    extra_len        : int — number of label tokens appended after the base mask.
    attn_tail        : torch.Tensor — ones for the label portion of the mask.
    attention_mask_cand  : torch.Tensor, (base_len + L_label,) — full mask.
    label_positions_mask : torch.Tensor, (base_len + L_label,) bool.
    label_start      : int — index where the label span begins.

    Returns
    -------
    candidate_ids       : 1-D tensor, shape (L_prompt + L_label,)
        Full candidate token sequence [prompt | label].
    attention_mask_cand : 1-D tensor, shape (L_prompt + L_label,)
        Attention mask of ones covering the entire candidate.
    label_positions_mask : 1-D bool tensor, shape (L_prompt + L_label,)
        True at label-span positions, False at prompt positions.
        Used by make_lm_labels and score_label_logprob to restrict the loss.
    """
    tok = _get_tokenizer(processor_or_tokenizer)
    device = base_input_ids.device

    # Compute the real (non-padded) prompt length.
    base_len = int(attention_mask.sum().item()) if attention_mask is not None else base_input_ids.shape[-1]
    # Strip trailing padding from the prompt token sequence.
    base_ids = base_input_ids[:base_len]

    if precomputed_label_tokens is not None:
        # Fast path: caller already tokenized the label — just move to device.
        label_tokens = precomputed_label_tokens.to(device)
    else:
        # Tokenize the label string.  add_special_tokens=False prevents BOS/EOS
        # from being prepended/appended to the label word.
        label_tokens = tok(label_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if max_answer_tokens is not None and max_answer_tokens > 0:
            label_tokens = label_tokens[:max_answer_tokens]
        label_tokens = label_tokens.to(device)

    # Concatenate: [prompt tokens | label tokens] → full candidate sequence.
    # prompt_template is intentionally ignored — the base prompt is fully
    # encoded inside base_ids already.
    candidate_ids = torch.cat([base_ids, label_tokens], dim=0)

    # Build the attention mask for the full candidate.
    if attention_mask is not None:
        # Use the original prompt mask (up to base_ids length) …
        attn_base = attention_mask[: base_ids.shape[0]].to(device)
        extra_len = candidate_ids.shape[0] - attn_base.shape[0]
        if extra_len > 0:
            # … and extend with 1s for all appended label tokens.
            attn_tail = torch.ones(extra_len, device=device, dtype=attn_base.dtype)
            attention_mask_cand = torch.cat([attn_base, attn_tail], dim=0)
        else:
            attention_mask_cand = attn_base
    else:
        # No input mask provided — every position is attended to.
        attention_mask_cand = torch.ones_like(candidate_ids, device=device)

    # Construct the label-position boolean mask.  Positions corresponding to the
    # appended label tokens are True; prompt positions are False.
    label_positions_mask = torch.zeros_like(candidate_ids, dtype=torch.bool, device=device)
    label_start = candidate_ids.shape[0] - label_tokens.shape[0]
    label_positions_mask[label_start:] = True

    return candidate_ids, attention_mask_cand, label_positions_mask


# ---------------------------------------------------------------------------
# build_cag_candidates_batched
# ---------------------------------------------------------------------------

def build_cag_candidates_batched(
    base_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    label_ids: torch.Tensor,
    pad_id: int,
) -> tuple:
    """
    Vectorised batch-level version of build_cag_candidate.

    Strips per-sample padding before appending the label tokens, mirroring the
    single-sample build_cag_candidate behaviour exactly.  This ensures the
    logit used to predict the first label token is the hidden state of the last
    *real* prompt token, not of a padding token, and that position IDs are
    consistent regardless of batch composition.

    Parameters
    ----------
    base_input_ids : torch.Tensor, shape (B, L_prompt)
        Padded prompt token ids for the whole batch, already on device.
    attention_mask : torch.Tensor, shape (B, L_prompt)
        Binary attention mask corresponding to base_input_ids.  Used to compute
        each sample's real (non-padded) prompt length.
    label_ids      : torch.Tensor, shape (L_label,)
        Precomputed label token ids, already on device.  The same sequence is
        appended to every sample.
    pad_id         : int
        Padding token id used to fill trailing positions in the output.

    Local variables
    ---------------
    B         : int — batch size.
    L_p       : int — padded prompt width.
    L_l       : int — label sequence length.
    device    : torch.device — device of base_input_ids.
    real_lens : torch.Tensor, (B,) long — actual (non-padded) prompt length per sample.
    L_max     : int — max real prompt length + L_l; output width.
    col       : torch.Tensor, (1, L_max) — column index broadcast helper.
    rlen      : torch.Tensor, (B, 1) — real_lens reshaped for broadcasting.
    prompt_mask: torch.Tensor, (B, L_max) bool — True at real prompt positions.
    label_mask : torch.Tensor, (B, L_max) bool — True at appended label positions.
    base_ext  : torch.Tensor, (B, L_max) — base_input_ids padded/clipped to L_max.

    Returns
    -------
    candidate_ids        : torch.Tensor, shape (B, L_max)
        Real prompt tokens followed by label tokens, with padding at the tail.
        Layout per sample i: [real_tokens(real_lens[i]) | label_tokens(L_l) | pad...].
    attention_mask_cand  : torch.Tensor, shape (B, L_max)
        1 at prompt + label positions, 0 at trailing padding.
    label_positions_mask : torch.Tensor, shape (B, L_max), dtype=bool
        True only at the L_l label token positions per sample.
    """
    B, L_p = base_input_ids.shape
    L_l = label_ids.shape[0]
    device = base_input_ids.device

    # Per-sample real (non-padded) prompt length derived from the attention mask.
    real_lens = attention_mask.sum(dim=1).long()          # (B,)
    # Sized from the batch's longest real prompt (not the already-padded L_p), so the
    # candidate sequences get a minimal, consistent width before label tokens are appended.
    L_max = int(real_lens.max().item()) + L_l             # output sequence width

    # Column index for vectorised mask construction.
    col  = torch.arange(L_max, device=device).unsqueeze(0)  # (1, L_max)
    rlen = real_lens.unsqueeze(1)                            # (B, 1)

    # Boolean masks: which columns belong to prompt tokens / label tokens.
    prompt_mask = col < rlen                                 # (B, L_max)
    label_mask  = (col >= rlen) & (col < rlen + L_l)        # (B, L_max)

    # Allocate output (default fill = pad / 0 / False).
    candidate_ids = base_input_ids.new_full((B, L_max), pad_id)
    attn_cand     = attention_mask.new_zeros(B, L_max)

    # Fill real prompt tokens.  Real tokens occupy columns 0..real_lens[i]-1 in
    # both base_input_ids and candidate_ids, so a masked copy works directly
    # after extending base_input_ids to L_max width.
    if L_p >= L_max:
        base_ext = base_input_ids[:, :L_max]
    else:
        base_ext = F.pad(base_input_ids, (0, L_max - L_p), value=pad_id)
    candidate_ids[prompt_mask] = base_ext[prompt_mask]

    # Fill label tokens.  Every row has exactly L_l True entries in label_mask
    # (by construction), so flattening in row-major order aligns with label_ids
    # repeated B times.
    candidate_ids[label_mask] = label_ids.unsqueeze(0).expand(B, L_l).reshape(-1)

    # Attention mask: attend to prompt + label positions; ignore trailing padding.
    attn_cand[prompt_mask | label_mask] = 1

    return candidate_ids, attn_cand, label_mask


# ---------------------------------------------------------------------------
# make_lm_labels
# ---------------------------------------------------------------------------

def make_lm_labels(input_ids_cand, label_positions_mask):
    """
    Build teacher-forcing label tensor for a candidate sequence.

    Clones input_ids_cand and masks out every non-label position with -100
    (PyTorch's standard ignore index for cross-entropy loss).  The resulting
    tensor, when passed to a causal LM loss function, computes loss ONLY over
    the label span — the model is not penalised for its predictions on the
    prompt portion.

    Parameters
    ----------
    input_ids_cand      : torch.Tensor, shape (B, L) or (L,)
        Full candidate token ids (prompt + label), on any device.
    label_positions_mask: torch.Tensor, shape (B, L) or (L,), dtype=bool
        True at positions belonging to the label span, False at prompt positions.

    Local variables
    ---------------
    labels : torch.Tensor — clone of input_ids_cand with -100 at prompt positions.

    Returns
    -------
    torch.Tensor, same shape as input_ids_cand
        Label tensor: label-span positions contain the real token ids; all other
        positions contain -100 and are excluded from the CE loss.
    """
    labels = input_ids_cand.clone()
    # Set prompt positions to -100 so cross-entropy ignores them.
    labels[~label_positions_mask] = -100
    return labels


# ---------------------------------------------------------------------------
# score_label_logprob
# ---------------------------------------------------------------------------

def score_label_logprob(
    model,
    input_ids_cand,
    attention_mask_cand,
    label_positions_mask,
    temperature=1.0,
    normalize_by_length=True,
    labels=None,
    return_outputs=False,
    return_lm_loss=False,
    **forward_kwargs,
):
    """
    Score a candidate sequence by computing the log-probability of its label span.

    Design notes
    ------------
    * The model is called WITHOUT labels so that HuggingFace's internal loss
      (which force-upcasts logits to float32 before CE inside ForCausalLMLoss)
      is never triggered.  Logits are extracted directly and NLL is computed
      manually via for_causal_lm_loss, keeping dtype consistent with AMP.

    * The full ``outputs`` object is deleted immediately after logits are
      extracted (non-return_outputs path), releasing past_key_values, hidden
      states, and any other non-logit fields from VRAM before NLL computation.

    * VRAM management: logits are upcasted to fp32 HERE in the caller before
      passing to for_causal_lm_loss, and the lower-precision original is
      immediately deleted.  This ensures only ONE fp32 copy of the logit
      tensor lives in VRAM at a time.  The transient 2× spike during ``.float()``
      is unavoidable in Python but is extremely short-lived.

    * The label-span mask is shifted left by one position (shifted_mask) because
      in an autoregressive model logit[t] predicts token[t+1].  After the shift,
      shifted_mask[t] is True when we want to include logit t in the loss.

    * logprob = -NLL.  When normalize_by_length=True the sum is divided by the
      number of label tokens, giving average log-prob per token.  This
      normalisation is required to eliminate length bias when class labels have
      different token counts (e.g. "bonafide" = 2 tokens, "spoof" = 1 token).

    Parameters
    ----------
    model               : nn.Module
        The wrapped Voxtral model.  Must expose a callable forward that accepts
        input_ids + attention_mask and returns an object with a .logits field.
    input_ids_cand      : torch.Tensor, shape (B, L)
        Full candidate token ids (prompt + label).
    attention_mask_cand : torch.Tensor, shape (B, L)
        Binary attention mask for input_ids_cand.
    label_positions_mask: torch.Tensor, shape (B, L), dtype=bool
        True at label-span positions.  Used to compute shifted_mask and to
        build the LM label tensor.
    temperature         : float
        Softmax temperature applied to raw logits before NLL.  Values > 1
        flatten the distribution; values < 1 sharpen it.  Default 1.0 (no-op).
    normalize_by_length : bool
        Divide logprob (and lm_loss_per_sample) by the label token count.
        Set True (default) to avoid length bias between class labels.
    labels              : torch.Tensor or None
        Pre-built LM label tensor.  Derived from make_lm_labels if None.
        Pass precomputed labels to avoid the clone inside the loop.
    return_outputs      : bool
        If True, attach lm_loss_per_sample to the output object and return
        (logprob, outputs) instead of just logprob.
    return_lm_loss      : bool
        If True, return (logprob, lm_loss_per_sample) without keeping the
        full output object alive in VRAM.
    **forward_kwargs
        Forwarded verbatim to model() (e.g. position_ids,
        precomputed_audio_embeds).

    Local variables
    ---------------
    device              : torch.device — device of model parameters.
    batch_size          : int — B, number of samples in the batch.
    outputs             : model return value — freed after logits are extracted.
    logits              : torch.Tensor, (B, L, V) — raw vocabulary logits.
    temperature         : float — validated/clamped temperature scalar.
    shifted_mask        : torch.Tensor, (B, L), bool — label_positions_mask
                          shifted left by one (aligns logit t with token t+1).
    tokens_per_sample   : torch.Tensor, (B,) — number of label tokens per sample.
    _vocab_size         : int — vocabulary size (logits.size(-1)).
    logits_fp32         : torch.Tensor, (B, L, V), float32 — upcasted logits.
    token_nll           : torch.Tensor, (B, L-1) — per-token NLL from for_causal_lm_loss.
    mask_f              : torch.Tensor, (B, L-1), float — shifted_mask cast to float.
    summed              : torch.Tensor, (B,) — total NLL summed over label tokens.
    denom               : torch.Tensor, (B,) — token count for normalisation.
    lm_loss_per_sample  : torch.Tensor, (B,) — per-sample loss (optionally returned).
    logprob             : torch.Tensor, (B,) — log-probability of label span.

    Returns
    -------
    logprob             : torch.Tensor, shape (B,)
        Log-probability of the label span.  Total if normalize_by_length=False,
        per-token average if normalize_by_length=True.
    lm_loss_per_sample  : torch.Tensor, shape (B,)  [only when return_lm_loss=True]
        Per-sample NLL on the label span (same scale as logprob).
    outputs             : model output object  [only when return_outputs=True]
        Full model output with lm_loss_per_sample attached.
    """
    # Derive device from the input tensor (already moved to the target device by
    # the caller).  Avoid next(model.parameters()).device which is fragile when
    # model parameters span multiple devices (e.g. QLoRA with device_map="auto").
    device = input_ids_cand.device

    # Move remaining inputs to the same device in case they are still on CPU.
    input_ids_cand = input_ids_cand.to(device)
    attention_mask_cand = attention_mask_cand.to(device)
    label_positions_mask = label_positions_mask.to(device)
    batch_size = input_ids_cand.size(0)

    # Build LM labels if the caller did not precompute them.
    if labels is None:
        labels = make_lm_labels(input_ids_cand, label_positions_mask)
    else:
        labels = labels.to(device)

    # Run the model forward WITHOUT passing labels to suppress HuggingFace's
    # internal ForCausalLMLoss upcast, which would transiently hold both the
    # AMP-precision logits and a float32 copy simultaneously.
    outputs = model(
        input_ids=input_ids_cand,
        attention_mask=attention_mask_cand,
        **forward_kwargs,
    )

    # Extract logits from the output object.  Different wrappers return either
    # a dict-like or an attribute-based output container.
    if isinstance(outputs, dict):
        logits = outputs.get("lm_logits", None)
        if logits is None:
            logits = outputs.get("logits", None)
    elif hasattr(outputs, "logits"):
        logits = outputs.logits
    else:
        logits = None

    # Free the full output container (past_key_values, hidden_states, …) now
    # that we have the logits reference.  This reclaims VRAM before the
    # expensive NLL computation that follows.
    if not return_outputs:
        del outputs
        outputs = None

    if logits is None:
        raise ValueError("Model forward did not return logits for logprob scoring.")

    # Apply temperature scaling.  Validated to be a positive float.
    temperature = float(temperature) if temperature is not None else 1.0
    if temperature <= 0:
        temperature = 1.0
    logits = logits / temperature

    # Compute the shifted label-position mask.
    # shifted_mask[b, t] = label_positions_mask[b, t+1]:
    #   logit at position t predicts token t+1, so the logit for a label token
    #   at position j is located at position j-1 (= j shifted left by 1).
    shifted_mask = F.pad(label_positions_mask, (0, 1), value=False)[..., 1:]
    # Count the number of label tokens per sample for length normalisation.
    tokens_per_sample = shifted_mask.sum(dim=1)

    # --- VRAM-safe upcast to fp32 ---
    # Upcast HERE so that when for_causal_lm_loss runs there is only ONE copy
    # of the logit tensor (the fp32 one) in VRAM.  If we passed the AMP-
    # precision tensor directly, for_causal_lm_loss would create its own fp32
    # copy while this scope still held a reference — causing a 3× spike on the
    # largest tensor in the forward pass (B × L × vocab_size).
    _vocab_size = logits.size(-1)
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits_fp32 = logits.float()  # transient 2× spike; unavoidable in Python
        del logits                    # remove the AMP-precision reference; logits_fp32 owns the data now
    else:
        logits_fp32 = logits
        del logits                    # remove the alias; logits_fp32 is the sole reference

    # Compute per-token NLL over the full label span using causal LM loss.
    # _skip_upcast=True prevents an internal fp32 copy since we already upcasted.
    token_nll = for_causal_lm_loss(
        logits_fp32,
        labels,
        vocab_size=_vocab_size,
        ignore_index=-100,
        reduction="none",
        _skip_upcast=True,   # already fp32 — prevent a redundant copy
    )
    # Release the fp32 logits now that the NLL tensor has been computed.
    del logits_fp32

    # Sum the per-token NLL values over the label span for each sample.
    mask_f = shifted_mask.to(dtype=token_nll.dtype)
    summed = (token_nll * mask_f).sum(dim=-1)                            # (B,)
    denom = tokens_per_sample.clamp_min(1).to(dtype=token_nll.dtype, device=token_nll.device)

    # Optionally divide by the label length so short and long labels are on
    # the same scale when they are compared for classification.
    if normalize_by_length:
        lm_loss_per_sample = summed / denom    # per-token average NLL  (B,)
        logprob = -(summed / denom)            # per-token average logprob  (B,)
    else:
        lm_loss_per_sample = summed            # total NLL (unnormalized)  (B,)
        logprob = -summed                      # total logprob (unnormalized)  (B,)

    # Handle the various return-mode combinations.
    if return_outputs:
        # Attach the per-sample loss to the output object for downstream use.
        if isinstance(outputs, dict):
            outputs["lm_loss_per_sample"] = lm_loss_per_sample
        else:
            try:
                setattr(outputs, "lm_loss_per_sample", lm_loss_per_sample)
            except Exception:
                pass  # output object is immutable; loss is still returned below
        if return_lm_loss:
            return logprob, lm_loss_per_sample, outputs
        return logprob, outputs

    if return_lm_loss:
        return logprob, lm_loss_per_sample

    return logprob
