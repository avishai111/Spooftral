# Spooftral: Can Voxtral Audio-Language Model Detect Speech Spoofing?

This project fine-tunes [Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) (`mistralai/Voxtral-Mini-3B-2507`, built on the Ministral-3B language model) for automatic speaker verification (ASV) spoofing detection using a **Classification-as-Generation (CAG)** paradigm. Rather than adding a classification head on top of the model, the system scores the log-probability of each class label string (e.g. `"bonafide"` / `"spoof"`) conditioned on the audio input, then selects the label with the higher score. This lets the full generative LLM be the discriminator with no architectural changes and no extra parameters.

Label texts are fully dynamic and YAML-driven — nothing is hardcoded. The golden prompt, class names, and all loss hyperparameters come from config.

This is the reference implementation for the paper *"Spooftral: Can Voxtral Audio-Language Model Detect Speech Spoofing?"*. The paper's **Spooftral** model corresponds to the CAG paradigm below (full ALM adapted with DoRA), and **Spooftral-Enc** corresponds to the Audio-Adapter + MLP classification-head paradigm (LM excluded). Using lightweight DoRA adaptation alone — no model fusion, no additional spoofing-specific training data — Spooftral reaches an EER of **4.25%** on the ASVspoof5 evaluation set. See [Results](#results) below for the full numbers.

- **Primary evaluation metric:** Equal Error Rate (EER, lower = better)
- **Datasets:** ASVspoof5 (primary), ASVspoof2021 LA/DF, ASVspoof2019 LA

## Contents

- [Architecture](#architecture)
- [Two Training Paradigms](#two-training-paradigms)
- [Classification-as-Generation (CAG) Detail](#classification-as-generation-cag-detail)
- [Fine-Tuning Strategy](#fine-tuning-strategy)
- [Projector PEFT](#projector-peft-catastrophic-forgetting-mitigation)
- [Loss Functions](#loss-functions)
- [Forward Pass Optimisation](#forward-pass-optimisation-batched-2b-forward)
- [Resume Behaviour](#resume-behaviour)
- [Checkpoint Format](#checkpoint-format)
- [Training Scripts](#training-scripts)
- [Inference Scripts](#inference-scripts)
- [Embedding & Projector Analysis](#embedding--projector-analysis)
- [Configuration](#configuration-configsyaml)
- [Key Source Files](#key-source-files)
- [How to Run](#how-to-run)
- [Evaluation](#evaluation)
- [Datasets](#datasets)
- [Results](#results)
- [Dependencies](#dependencies)
- [Notes](#notes)

## Architecture

Voxtral is a multimodal speech-language model composed of three components:

**1. Audio Encoder**
Voxtral's built-in Whisper-based audio tower. The encoder can be frozen, fully fine-tuned, or adapted via LoRA/DoRA independently of the LM (see `trainable_audio_encoder`).

**2. Multi-Modal Projector**
Linear bridge (`linear_1 → GELU → linear_2`) that maps audio-encoder hidden states into the LM hidden size (3072 for Ministral-3B). It can be:
- trained end-to-end (default, `projector_peft=False`), or
- PEFT-wrapped: base linear weights frozen, LoRA/DoRA adapters only (`projector_peft=True`; see [Projector PEFT](#projector-peft-catastrophic-forgetting-mitigation))

When `k_downsample != 4` the built-in projector is replaced by the custom `VoxtralModalProjector_Encoder_Audio` sized to `k_downsample*1280 → 3072`.

**3. Language Model (Ministral-3B)**
Autoregressive LLM receiving audio embeddings prepended to the token sequence. Adapted via LoRA / DoRA / AdaLoRA / full FT / frozen — chosen via `finetune_mode`.

During CAG inference/training the model receives:

```text
[audio_embeddings | instruction_tokens | label_tokens]
```

and the log-probability of the label tokens is used as the class score.

## Two Training Paradigms

This repository supports two complementary architectures:

### A. Classification-as-Generation (CAG) — primary paradigm, paper name **Spooftral**

No classification head. The full Voxtral stack scores logprobs of label strings (`"bonafide"` / `"spoof"`). Built by `VoxtralStageWrapper_classification`.
Training scripts: [`training_voxtral_generation_tokens05_21.py`](training_voxtral_generation_tokens05_21.py) (root) and [`more_training_voxtral_scripts/training_voxtral_generation_tokens*.py`](more_training_voxtral_scripts/)

With DoRA applied to all attention projection layers (rank 32, α=64 on ASVspoof5), Spooftral trains only **~55.1M parameters — 1.17% of the 4.71B total**.

### B. Audio-Adapter + MLP Classification Head — paper name **Spooftral-Enc**

Lightweight alternative that bypasses the LM entirely:

```text
audio encoder → multi_modal_projector → pool → MLP head → logits
```

The LM is frozen (and optionally offloaded to CPU via `drop_language_model`) so all compute goes into the encoder + adapter + head. Built by `AudioAdapterHeadClassifier` in [`utils/build_model_for_classification_head.py`](utils/build_model_for_classification_head.py). The classification head is a linear projection (3072 → 256) → GELU → dropout (p=0.1) → linear (256 → 2).
Training scripts: [`training_audio_adapter_head_5_margin.py`](training_audio_adapter_head_5_margin.py) (root) and [`more_training_audio_adapter_scripts/training_audio_adapter_head*.py`](more_training_audio_adapter_scripts/)
YAMLs: `configs/hyperparameters_*_classification_head.yaml`

With DoRA applied to the audio encoder's attention layers, Spooftral-Enc trains only **~37.4M parameters — 5.55% of its 674M total**.

Both paradigms share the same protocol parsers, augmentation pipeline, EER computation, checkpoint format, and resume logic. The choice is dataset-agnostic — every supported dataset works with either.

There is also [`scripts/train.py`](scripts/train.py), a unified entry point that takes any CAG config via `--config` instead of hardcoding it, as an alternative to launching the per-experiment scripts above directly.

## Classification-as-Generation (CAG) Detail

For each audio sample, one candidate sequence per class label is built:

```text
candidate_0: [prompt_tokens | label_texts[0]]   (e.g. "bonafide")
candidate_1: [prompt_tokens | label_texts[1]]   (e.g. "spoof")
```

The class texts are read exclusively from the YAML field `generation_classification.label_texts` — never hardcoded.

Each candidate is scored by `score_label_logprob()`, which:
- Runs a single LM forward pass (no teacher-forcing labels passed to the model, avoiding HuggingFace's internal fp32 upcast inside `ForCausalLMLoss`)
- Extracts raw logits, then releases the full output object from VRAM before any NLL computation
- Releases the logits tensor itself before calling `for_causal_lm_loss` so that the fp32 upcast inside that function does not create a double-live copy (the previous design held fp16 original + fp32 copy simultaneously, causing a 3× VRAM spike on the largest tensor in the forward pass)
- Computes NLL only over the label-token span using a shifted boolean mask (shifted because `logit[t]` predicts `token[t+1]`)
- Optionally length-normalises by dividing total NLL by the label token count, which is required to eliminate bias when label strings have different lengths

**Decision rule:**

```text
softmax([logP(label_0 | audio), logP(label_1 | audio)])
prediction = argmax of above
```

This is equivalent to a generative classifier without any extra parameters.

## Fine-Tuning Strategy

Parameter-efficient fine-tuning via **DoRA** (Weight-Decomposed Low-Rank Adaptation).

- `finetune_mode: 'dora'`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj` (all attention projections)
- Default rank/alpha: 16 / 32 (`configs/hyperparameters_2019LA.yaml`, `hyperparameters_2021_LA_2.yaml`, `hyperparameters_2021_DF_2.yaml`). For ASVspoof5 the rank/alpha is increased to 32 / 64 — this is the "Attention DoRA" configuration (`configs/hyperparameters_05_*.yaml`).
- The audio encoder can independently be frozen or DoRA-adapted via `trainable_audio_encoder` in the YAML config.
- The `multi_modal_projector` (audio→LM bridge) can independently be wrapped in LoRA/DoRA adapters via `projector_peft` (see below).

### Projector PEFT (catastrophic-forgetting mitigation)

The `multi_modal_projector` is the single bottleneck between the audio encoder and the LM, so unconstrained end-to-end updates here can shift the LM input distribution in ways the LM-side LoRA cannot compensate for. The YAML flag `projector_peft` exposes a parameter-efficient alternative:

- `projector_peft: False` — default; legacy end-to-end training of projector
- `projector_peft: True` — freeze base projector weights, train LoRA/DoRA adapters on `linear_1` / `linear_2` only

When `projector_peft: True`:
- Requires `finetune_mode` in `('lora', 'dora')`. Other modes raise `ValueError` at wrapper init so the misconfiguration fails fast.
- Adapter hyperparameters reuse `lora_rank` / `lora_alpha` / `lora_dropout` from the LM PEFT — no new YAML keys.
- Target modules (`'linear_1'`, `'linear_2'`) cover both Voxtral's built-in projector (`k_downsample=4`) and the custom `VoxtralModalProjector_Encoder_Audio` (`k_downsample != 4`). Same Linear submodule names in both implementations.
- The PEFT-wrapped projector's state_dict keys change to `base_model.model.linear_*.{base_layer,lora_A,lora_B[,lora_magnitude_vector]}`. These roundtrip cleanly through `wrapper_state_dict` + `strict=True`. Toggling the flag across resume is intentionally rejected (key mismatch — same policy as changing `lora_rank`).
- Backward compatibility: default `False` is bit-for-bit identical to the pre-flag codebase; old checkpoints load unchanged.

## Loss Functions

All loss computations operate on length-normalised log-probs to eliminate label length bias. For example, `"bonafide"` tokenises to 2 tokens and `"spoof"` to 1; without per-token normalisation the model would systematically assign higher raw log-prob to the shorter label even with equal per-token confidence.

**1. Cross-Entropy Classification Loss** (always active)
CE over a 2-class softmax of the length-normalised logprobs:

```text
loss = F.cross_entropy([logP(label_0), logP(label_1)], true_class)
```

Optional per-class weighting via `class_weights` in the YAML.

**2. Margin Loss** (optional, `use_margin_loss: True`)
Hinge loss that enforces a minimum score gap between the correct and incorrect class. Uses the same length-normalised logprobs as CE so the margin threshold is meaningful regardless of label token count.

Formula (per sample):

```text
diff  = logP(correct_label) - logP(wrong_label)
loss  = relu(margin - diff)
```

Separate margins for each semantic class (`margin_bonafide`, `margin_spoof`) are assigned by inspecting `label_texts_map` values at startup, not by hardcoding index 0 = bonafide. This preserves correctness when the YAML label order is swapped or custom label names are used.

**3. Teacher-Forcing Regularisation** (optional, `teacher_forcing_loss: True`)
Auxiliary NLL encouraging the model to fluently generate the correct label. Only the ground-truth label's NLL is used — the wrong candidate's loss is never included. Scaled by `teacher_forcing_loss_lambda` (default 0.1). The per-sample NLL is a scalar extracted via `return_lm_loss=True`, so the full logit tensor is never kept alive during this computation.

```text
Total loss = cag_loss_weight * ce_loss
           + lambda_margin   * margin_loss   (if use_margin_loss)
           + tf_lambda       * tf_loss       (if teacher_forcing_loss)
```

## Forward Pass Optimisation (Batched 2B Forward)

Both candidates for every sample are processed in a single LLM forward pass:

1. Audio encoder runs once per batch → embeddings cached as `audio_embeds_cache`.
2. Bonafide and spoof candidate sequences are right-padded to a common length and concatenated into a `(2*B, L_max)` batch.
3. Audio embeds are tiled: `torch.cat([embeds, embeds], dim=0)`.
4. One LM forward pass returns logits for all `2*B` sequences.
5. Logits are released from VRAM immediately after NLL extraction.
6. Results are split: `logp_b = logp_all[:B]`, `logp_s = logp_all[B:]`.

This halves LLM forward passes compared to scoring each candidate separately.

**VRAM management during NLL extraction:**
- `score_label_logprob` deletes its local logits reference before calling `for_causal_lm_loss`, so that function's fp32 upcast does not overlap with the original AMP-precision tensor.
- `for_causal_lm_loss` deletes the fp32 tensor immediately after `view(-1, vocab_size)` is created, before `fixed_cross_entropy` allocates loss buffers.
- The AMP autocast context is created once before the epoch loop, not recreated on every batch step.

## Resume Behaviour

All training state is saved in checkpoints and fully restored on resume:

- `start_epoch` / `start_step`
- `best_eer` (best dev EER — controls which dev checkpoint is saved)
- `best_eval_eer` (best eval EER — controls which eval checkpoint is saved)
- optimizer / scheduler / scaler state
- RNG states (torch / cuda / numpy / python)
- W&B run ID (resumes the same run rather than opening a new one)
- Last eval timestamps (epoch / step) for each split (8 keys)

Auto-resume is gated by `configs.resume=True` in the YAML. It does **not** trigger automatically on any checkpoint found in `output_dir`. This prevents accidental resumption when the user intentionally starts a fresh run.

Progressive-training `requires_grad` flags are also explicitly restored because PyTorch `state_dict` does not persist `requires_grad`. `load_state_dict` uses `strict=True` everywhere — mismatched architectures (e.g. changed `lora_rank`, toggled `projector_peft`) raise immediately rather than silently training with wrong weights. To recover, delete the checkpoint and start fresh.

## Checkpoint Format

Every checkpoint is a dict with these keys:

| Key | Description |
|---|---|
| `wrapper_state_dict` | model weights (only LoRA deltas for PEFT runs) |
| `optimizer` | optimizer state |
| `scheduler` | LR scheduler state |
| `scaler` | GradScaler state (only for fp16 AMP) |
| `epoch` | 0-indexed loop variable at save time |
| `epoch_completed` | `True` = end-of-epoch save, `False` = mid-epoch step save |
| `global_step` | global optimizer step count |
| `best_eer` | best dev EER seen so far |
| `best_dev_eer` | best dev EER (independent tracker) |
| `best_eval_eer` | best eval/test EER (independent tracker) |
| `rng_states` | `{torch, cuda, numpy, python}` RNG states |
| `last_eval_epoch`/`step` | per-split last-eval trackers (8 keys total) |
| `wandb_id` | W&B run ID for reconnecting |
| `configs` | config dict snapshot for mismatch detection |

Saves are atomic (`.pt` via tmp→`os.replace` with fsync, adapters via tmp→`os.rename`, JSON via tmp→`os.replace`) for power-loss safety.

On resume:
- `epoch_completed=True` → `start_epoch = epoch + 1` (completed; continue next)
- `epoch_completed=False` → `start_epoch = epoch` (interrupted; replay epoch)

Cleanup pins the metric-best files (referenced by `best_dev_metrics.json` / `best_eval_metrics.json`), `best_dev/` and `best_eval/` subdirectories, and `checkpoint-last.pt`. Those are never deleted by `cleanup_old_checkpoints`.

## Training Scripts

Each script corresponds to a dataset / paradigm / experiment variant.

### CAG scripts (no classification head — score label logprobs)

| Script | Dataset / Variant | YAML |
|---|---|---|
| [`training_voxtral_generation_tokens05_21.py`](training_voxtral_generation_tokens05_21.py) (root) | ASVspoof5, ASVspoof2021-derived margin schedule | `hyperparameters_05_dif_margins21.yaml` |
| `more_training_voxtral_scripts/training_voxtral_generation_tokens05_21_bigmargin.py` | Same as above, larger margin | `hyperparameters_05_dif_margins21bigmargin.yaml` |
| `more_training_voxtral_scripts/training_voxtral_generation_tokens05_21_LLM.py` | ASVspoof5, LLM-only PEFT (no audio-encoder adapters) | `hyperparameters_05_dif_margins21_LLM.yaml` |
| `more_training_voxtral_scripts/training_voxtral_generation_tokens05_no_margin.py` | ASVspoof5, no margin loss (primary clean baseline) | `hyperparameters_05_no_margin.yaml` |
| `more_training_voxtral_scripts/training_voxtral_generation_tokens05_no_margin_realfake.py` | ASVspoof5, no margin, `"real"`/`"fake"` label texts | `hyperparameters_05_no_margin_realfake.yaml` |
| `more_training_voxtral_scripts/training_voxtral_generation_tokens2019LA.py` | ASVspoof2019 Logical Access | `hyperparameters_2019LA.yaml` |
| `more_training_voxtral_scripts/training_voxtral_generation_tokens2021_LA.py` | ASVspoof2021 Logical Access | `hyperparameters_2021_LA_2.yaml` |
| `more_training_voxtral_scripts/training_voxtral_generation_tokens2021_DF.py` | ASVspoof2021 Deepfake | `hyperparameters_2021_DF_2.yaml` |

All CAG scripts share [`utils/training_evaluate_classification.py`](utils/training_evaluate_classification.py) and differ only in which YAML config file they load. [`scripts/train.py`](scripts/train.py) provides the same training loop as a single entry point parameterised by `--config` instead of a dedicated script per variant.

### Audio-Adapter + MLP-head scripts (LM frozen / optionally offloaded)

| Script | Dataset / Variant | YAML |
|---|---|---|
| [`training_audio_adapter_head_5_margin.py`](training_audio_adapter_head_5_margin.py) (root) | ASVspoof5, with margin loss | `hyperparameters_5_classification_head_margin.yaml` |
| `more_training_audio_adapter_scripts/training_audio_adapter_head.py` | ASVspoof5 base trainer | `hyperparameters_5_classification_head.yaml` |
| `more_training_audio_adapter_scripts/training_audio_adapter_head_2019LA.py` | ASVspoof2019 LA | `hyperparameters_2019LA_classification_head.yaml` |
| `more_training_audio_adapter_scripts/training_audio_adapter_head_2021LA.py` | ASVspoof2021 LA | `hyperparameters_2021LA_classification_head.yaml` |
| `more_training_audio_adapter_scripts/training_audio_adapter_head_2021DF.py` | ASVspoof2021 Deepfake | `hyperparameters_2021DF_classification_head.yaml` |

These share [`utils/training_helpers_audio_adapter_head.py`](utils/training_helpers_audio_adapter_head.py) and load `configs/hyperparameters_*_classification_head.yaml`.

## Inference Scripts

Standalone evaluation entry-points for a saved checkpoint (no training loop):

| Script | Purpose |
|---|---|
| [`inference_checkpoint_new.py`](inference_checkpoint_new.py) (root) | Generic CAG inference, ASVspoof5 |
| [`inference_checkpoint_no_margin.py`](inference_checkpoint_no_margin.py) (root) | Companion to the no-margin CAG training variant |
| `more_inference_scripts/inference_checkpoint_2021_2019.py` | CAG inference across ASVspoof2021 DF/LA and ASVspoof2019 LA |
| `more_inference_scripts/inference_checkpoint_LLM.py` | CAG inference for the LLM-only PEFT variant |
| `more_inference_scripts/inference_checkpoint_real_fake.py` | CAG inference for the `"real"`/`"fake"` label-text variant |
| `more_inference_scripts/inference_checkpoint_5_encoder_2.py` | Audio-Adapter + MLP-head inference, ASVspoof5 |
| `more_inference_scripts/inference_checkpoint_2021_2019_encoder.py` | Audio-Adapter + MLP-head inference across ASVspoof2021 DF/LA and ASVspoof2019 LA |

Common behavior: load checkpoint via `load_checkpoint_into_model` (`strict=True`), rebuild the wrapper with matching `projector_peft` / `finetune_mode` / `lora_rank` values, and emit per-utterance scores + EER.

## Embedding & Projector Analysis

[`calc_embeddings_projector_codes/`](calc_embeddings_projector_codes/) is a standalone representation-analysis pipeline, separate from the CAG / classification-head training loops above. It extracts frozen Voxtral embeddings, packages them into dataset-level NPZ bundles, and trains/evaluates lightweight linear classifiers directly on top of those embeddings — used for the linear-probing numbers in [Results](#results) (LLM-decoder vs. audio-adapter embeddings). See [`calc_embeddings_projector_codes/README.md`](calc_embeddings_projector_codes/README.md) for full details; these scripts are configured by editing constants at the top of each file rather than via the `configs/*.yaml` system used elsewhere in this repo.

Pipeline (each stage optional after the first two):

| Script | Purpose |
|---|---|
| [`calc_embeddings.py`](calc_embeddings_projector_codes/calc_embeddings.py) | Extracts per-file embeddings from a selected Voxtral stage (`audio_projector`, `llama`, or `lm_head`) and writes a metadata CSV |
| [`aggregate_embeddings_to_npz.py`](calc_embeddings_projector_codes/aggregate_embeddings_to_npz.py) | Merges per-file embeddings + metadata CSV into a single `total_{train,dev,eval}_embeddings.npz` bundle |
| [`verify_npz_embeddings.py`](calc_embeddings_projector_codes/verify_npz_embeddings.py) | Sanity-checks an NPZ bundle (shape, NaN/Inf, metadata-length consistency) before training |
| [`training_on_embeds_one_net_5.py`](calc_embeddings_projector_codes/training_on_embeds_one_net_5.py) | Trains a linear PyTorch Lightning classifier on the frozen embeddings, tracking EER |
| [`score_embeddings_pipeline.py`](calc_embeddings_projector_codes/score_embeddings_pipeline.py) | Scores saved checkpoints across train/dev/eval splits, with optional bootstrap confidence intervals |
| [`compare_embeddings.py`](calc_embeddings_projector_codes/compare_embeddings.py) | Compares two embedding spaces via cosine similarity, L2 distance, CKA, and pairwise-distance correlation |
| [`plot_pacmap_embeddings.py`](calc_embeddings_projector_codes/plot_pacmap_embeddings.py) | Produces the paper's Figure 2: PaCMAP 2D projections of audio-adapter vs. LLM-decoder embeddings across ASVspoof5 train/dev/eval, fit on train and transformed onto dev/eval |

## Configuration (`configs/*.yaml`)

All hyperparameters are YAML-driven. Nothing is hardcoded in training scripts. Key fields in `configs/hyperparameters_05_no_margin.yaml` (primary config):

```yaml
model_id: mistralai/Voxtral-Mini-3B-2507
DATASET: asvspoof5
BATCH_SIZE: 24
NUM_EPOCHS: 20
gradient_accumulation_steps: 8
lr: 5e-5
lr_scheduler_type: cosine
scheduler.warmup_ratio: 0.05
max_grad_norm: 1.0

finetune_mode: dora           # lora | dora | adalora | frozen | full
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj]

# Audio encoder: full FT (True), frozen (False), or DoRA-adapted ("dora")
trainable_audio_encoder: dora
# External encoder selector: None | wav2vec2 | hubert | wavlm
encoder_type: None
# Audio temporal downsample before the projector. 4 = built-in projector;
# any other value swaps in the custom VoxtralModalProjector_Encoder_Audio.
k_downsample: 4

# Wrap multi_modal_projector in LoRA/DoRA adapters (freezes base projector
# weights). Requires finetune_mode in ('lora','dora'). Default False = legacy
# end-to-end training of the projector. Adapter r/alpha/dropout shared with LM.
projector_peft: False

# Precision (mutually exclusive with use_amp / use_fp16):
precision: bfloat16          # bfloat16 | float16 | float32 | amp
use_4bit: False              # QLoRA quantisation
resume: False                # set True to auto-resume from latest ckpt

generation_classification:
  enable: True
  label_texts: {0: "bonafide", 1: "spoof"}   # fully dynamic; no hardcoding
  normalize_logprob_by_tokens: True          # length-normalisation ON (required)
  temperature: 1.0
  teacher_forcing_loss: False
  loss_type: cross_entropy

use_margin_loss: False
margin_bonafide: 0.1   # margin applied to samples whose true class is "bonafide"
margin_spoof: 0.05     # margin applied to samples whose true class is "spoof"
lambda_margin: 0.05    # weight of the margin loss term (values above match the
                        # paper's Spooftral (Attention DoRA + Margin) config,
                        # used in all margin-loss experiments)

augmentation_on: True
CODEC_PERTURB: True   # mp3, aac, opus, flac, wav, g722, amr, …
AUG_ADD_NOISE: True
AUG_TIME_MASK: True
AUG_FREQ_MASK: True
GAIN_PERTURB: True
TIMESHIFT_PERTURB: True
SELECTIVE_AUGMENTATION: True
MAX_AUGMENTATIONS_PER_SAMPLE: 4
```

> **Note on margin semantics:** `margin_bonafide` is applied to samples whose true class label text contains `"bonafide"` (case-insensitive substring match against `label_texts_map` values). The semantic assignment is YAML-driven — if you use custom label names, the `margin_*` keys still match by text content, not by index. If no match is found the fallback is index 0 → bonafide, index 1 → spoof.

## Key Source Files

**[`utils/training_evaluate_classification.py`](utils/training_evaluate_classification.py)**
Core training loop. `train_classification_model()` orchestrates batched forward passes, loss computation, evaluation, checkpointing, and logging. Supports resume from checkpoint, EER-based model selection, W&B logging.

**[`utils/generation_text_bonafide_spoof_classification.py`](utils/generation_text_bonafide_spoof_classification.py)**
CAG primitives: `build_cag_candidate()` (tokenises and appends label text to prompt), `make_lm_labels()` (creates teacher-forcing labels with `-100` mask), `score_label_logprob()` (full scoring function with shifted mask trick).

**[`utils/eval_classification.py`](utils/eval_classification.py)**
EER evaluation loop for dev/eval sets. Runs under `torch.inference_mode()`, restores train mode in a `finally` block.

**[`utils/build_model.py`](utils/build_model.py)**
Model construction: loads Voxtral, applies DoRA/LoRA to LM and (optionally) audio encoder + projector. Key classes/functions: `build_voxtral_model()`, `VoxtralStageWrapper_classification`, `VoxtralModalProjector_Encoder_Audio`, `_adapt_state_keys()`, `compute_audio_embeds()`.

**[`utils/build_model_for_classification_head.py`](utils/build_model_for_classification_head.py)**
`AudioAdapterHeadClassifier` — alternative architecture that replaces the LM with a small MLP head. Modes: `audio_encoder_tuning` (frozen/finetune/lora/dora), `pooling_strategy`, `drop_language_model`.

**[`utils/training_helpers_audio_adapter_head.py`](utils/training_helpers_audio_adapter_head.py)**
Training loop for the classification-head paradigm (mirrors the CAG loop interface but skips all generation-scoring code paths).

**[`utils/checkpoint_utils.py`](utils/checkpoint_utils.py)**
Checkpoint save/load with full resume state: weights, optimizer, scheduler, scaler, RNG states, per-split last-eval trackers, W&B run ID. Config-mismatch warnings on resume. Atomic via `.tmp` → `os.replace` + fsync.

**[`utils/cleanup_checkpoints.py`](utils/cleanup_checkpoints.py)**
`cleanup_old_checkpoints()` — prunes old `.pt` files in `output_dir` while pinning the metric-best files.

**[`utils/freezing.py`](utils/freezing.py)**
`verify_model_frozen()` — read-only diagnostic that prints a `requires_grad` summary, used before/after inference to confirm the model is in the expected frozen state.

**[`utils/losses.py`](utils/losses.py)**
`for_causal_lm_loss()` — token-level NLL with configurable reduction. Upcasts to float32 internally for numerical stability, deletes the fp32 tensor immediately after `.view()` to minimise peak VRAM.

**[`utils/dataset.py`](utils/dataset.py)**
ASVspoof dataset loaders (5 / 2021 LA / 2021 DF / 2019 LA) with on-the-fly augmentation. ASVspoof5 protocol is parsed with `split("\t")` (tab-separated).

**[`utils/augmentation.py`](utils/augmentation.py) / [`utils/codec.py`](utils/codec.py) / [`utils/RawBoost.py`](utils/RawBoost.py) / [`utils/data_utils_rawboost.py`](utils/data_utils_rawboost.py)**
Augmentation stack: `codec.py` (mp3/aac/opus/flac/wav/g722/amr sims), `augmentation.py` (add-noise, time/freq mask, gain, time-shift), `RawBoost.py` + `data_utils_rawboost.py` (RawBoost LnL/ISD/SSI variants).

**[`utils/config.py`](utils/config.py)**
YAML loader. Sets `cudnn.deterministic=True` / `benchmark=False` at startup. Forces `processor.feature_extractor.do_normalize=True`. Exposes `PROJECTOR_PEFT`, `K_DOWNSAMPLE`, `NUM_CLASSES`, and all PEFT hyperparameters as module-level constants for training scripts.

**[`configs/`](configs/)**
One YAML file per experiment variant:
- CAG configs: `hyperparameters_05*.yaml` (ASVspoof5 variants), `hyperparameters_2019LA.yaml`, `hyperparameters_2021_LA_2.yaml`, `hyperparameters_2021_DF_2.yaml`
- Classification-head configs: `hyperparameters_*_classification_head.yaml`

**`inference_checkpoint*.py`** (root and `more_inference_scripts/`)
Standalone inference for trained checkpoints (multiple dataset variants).

**[`calc_embeddings_projector_codes/`](calc_embeddings_projector_codes/)**
Standalone embedding-extraction and linear-probing pipeline (see [Embedding & Projector Analysis](#embedding--projector-analysis)); configured by editing constants in each script rather than YAML.

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Edit or copy the relevant YAML config** in `configs/`:
   ```bash
   cp configs/hyperparameters_05_no_margin.yaml configs/my_experiment.yaml
   ```
   Update the placeholder paths (`train_protocol`, `dev_protocol`, `train_audio_dir`, `save_path`, `wandb.dir`, `wandb.entity`, etc.) for your own environment.

3. **Set W&B credentials** if using online logging:
   ```bash
   export WANDB_API_KEY=<your_key>
   ```

4. **Launch training.** The scripts do not initialise `torch.distributed` (no `init_process_group`, no `DistributedSampler`, no per-rank device placement — `DEVICE` in [`utils/config.py`](utils/config.py) is just `"cuda"` if available). This means there is currently no real multi-process data-parallel path: use a single `python` invocation on one GPU.

   ```bash
   python more_training_voxtral_scripts/training_voxtral_generation_tokens05_no_margin.py
   ```

   Or via the unified entry point:

   ```bash
   python scripts/train.py --config configs/hyperparameters_05_no_margin.yaml
   ```

   To pin the run to a specific GPU on a multi-GPU machine, set `CUDA_VISIBLE_DEVICES` before launching:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/hyperparameters_05_no_margin.yaml
   ```

   `torchrun --nproc_per_node=N ...` will *run*, but without the distributed setup above it just starts `N` independent, non-coordinated copies of the same script contending for the same GPU and writing to the same `save_path` — not real distributed training. Treat `torchrun`/`accelerate launch` as future multi-GPU hooks rather than something to use as-is today.

5. **Checkpoints are saved to** `<save_path>/<project_name>/` (defined in the YAML). The best checkpoint (lowest EER on the dev set) is retained as `checkpoint_best_dev.pt`. A separate `checkpoint_best_eval.pt` tracks the lowest EER on the eval (test) set independently.

6. **To resume**, re-launch the same command. The training loop detects the latest checkpoint automatically and restores all state, including `best_eval_eer`, so no spurious "new best" saves occur after resume.

7. **Evaluation only** (no training): set `only_eval: True` in the YAML, then launch as above.

## Evaluation

Primary metric: **EER** (Equal Error Rate) — the FAR/FRR crossover point. Lower EER = better discrimination between bonafide and spoof speech.

Two EER values are tracked independently:
- `best_eer` — best EER on the dev (validation) set; governs model saving
- `best_eval_eer` — best EER on the eval (test) set; saved to a separate dir

Both are saved in every checkpoint and fully restored on resume. Both are logged to W&B summary and printed to stdout/log files.

## Datasets

| Dataset | Notes |
|---|---|
| **ASVspoof5** | Track 1 (closed condition). Protocol files: `ASVspoof5.train.tsv`, `ASVspoof5.dev.track_1.tsv`, `ASVspoof5.eval.track_1.tsv`. Audio format: FLAC. Protocol is TAB-separated and must be parsed with `split("\t")` — using whitespace split collapses multi-space fields. |
| **ASVspoof2021 LA** | Logical Access evaluation track. Protocol: `trial_metadata.txt` with key fields: `file_id`, `label`, `system_id`. |
| **ASVspoof2021 DF** | Deepfake track. Larger, more diverse spoofing systems than LA. Used by the `*_DF` training scripts and `*_DF` YAMLs. |
| **ASVspoof2019 LA** | Used for cross-dataset evaluation and the classification-head variant. |

All datasets use binary labels: bonafide (genuine speech) / spoof (generated or replayed speech). The label strings fed to the model are YAML-defined, not hardcoded — the protocol parser maps protocol labels to YAML `label_texts` entries.

## Results

Numbers below are reported EER (%) from the paper.

**Linear probing — audio-adapter vs. LLM-decoder embeddings (frozen Voxtral, no adaptation).** Confirms that spoof-discriminative cues are more separable *before* the LLM decoder than after it, motivating task-specific adaptation:

| Database | LLM Dev. | LLM Eval. | Audio Adapter Dev. | Audio Adapter Eval. |
|---|---|---|---|---|
| ASVspoof2019 LA | 6.60 | 7.17 | 2.79 | 5.48 |
| ASVspoof2021 LA | 9.26 | 17.34 | 2.79 | 13.57 |
| ASVspoof2021 DF | 7.02 | 19.20 | 2.79 | 9.64 |
| ASVspoof5 | 20.73 | 19.09 | 11.58 | 9.75 |

**Spooftral vs. Spooftral-Enc after DoRA adaptation** (Attention DoRA, + margin regularisation on ASVspoof5):

| Database | Spooftral Dev. | Spooftral Eval. | Spooftral-Enc Dev. | Spooftral-Enc Eval. |
|---|---|---|---|---|
| ASVspoof2019 LA | 0.02 | 0.41 | 0.03 | 0.40 |
| ASVspoof2021 LA | 0.02 | 6.46 | 0.03 | 6.70 |
| ASVspoof2021 DF | 0.05 | 3.88 | 0.05 | 4.76 |
| ASVspoof5 | 4.50 | **4.25** | 5.29 | 5.08 |

**ASVspoof5 configuration ablation** — margin regularisation improves the Attention DoRA configuration on both splits:

| Configuration | Dev. EER | Eval. EER |
|---|---|---|
| Spooftral (LLM Attention DoRA only, audio encoder frozen) | 7.89 | 7.01 |
| Spooftral (Attention DoRA, incl. audio encoder) | 4.77 | 4.46 |
| Spooftral (Attention DoRA + Margin) | **4.50** | **4.25** |
| Spooftral-Enc (Attention DoRA) | 5.29 | 5.08 |

The `bonafide`/`spoof` instruction interface also outperforms a `real`/`fake` interface under identical DoRA settings (4.77/4.46 vs. 5.67/4.47 Dev./Eval. EER) — see [`hyperparameters_05_no_margin_realfake.yaml`](configs/hyperparameters_05_no_margin_realfake.yaml).

## Dependencies

See [`requirements.txt`](requirements.txt) (`pip install -r requirements.txt`). Summary:

| Package | Purpose |
|---|---|
| `torch >= 2.1` | bfloat16 AMP, gradient checkpointing |
| `transformers >= 4.45` | `VoxtralForConditionalGeneration` |
| `peft >= 0.12` | DoRA support |
| `torchaudio` | audio I/O and augmentation |
| `bitsandbytes` | 4-bit/8-bit quantisation, QLoRA |
| `wandb` | experiment tracking |
| `numpy`, `scipy` | EER computation, RawBoost augmentation |
| `scikit-learn` | EER/ROC computation, classification metrics |
| `matplotlib` | plotting scripts, debug spectrogram dumps |
| `tqdm` | progress bars |
| `evaluate` | HuggingFace evaluate library |
| `huggingface_hub` | model/config downloads |
| `python-dotenv` | `.env` loading in `utils/config.py` |
| `omegaconf` | optional config-container support |
| `pyyaml` | config loading |
| `soundfile` | FLAC reading |
| `ffmpeg` | codec augmentation; **system binary**, must be on `PATH` (not pip-installable) |

## Notes

- **Mixed precision:** bfloat16 AMP throughout. The LM loss is computed in float32 internally (via `for_causal_lm_loss`) for numerical stability, but the fp32 copy is freed immediately after the flat view is taken.
- Gradient checkpointing is enabled by default (`use_reentrant=False`) to reduce activation memory during the combined 2B forward pass.
- The Whisper audio encoder processes at 16 kHz. All audio is resampled before feature extraction.
- `normalize_logprob_by_tokens` must be `True` (YAML default) to eliminate length bias in CE, margin, and teacher-forcing losses when label strings tokenise to different numbers of tokens.

## License

Licensed under the [Apache License 2.0](LICENSE).

## Contact

If you have questions, issues, or want to collaborate, feel free to reach out:

- 📧 Email: [Avishai Weizman](mailto:avishai11900@gmail.com)
- 🔗 GitHub: [github.com/avishai111](https://github.com/avishai111)
- 💼 LinkedIn: [linkedin.com/in/avishai-weizman/](https://linkedin.com/in/avishai-weizman/)
