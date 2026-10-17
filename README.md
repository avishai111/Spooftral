# Spooftral: Can Voxtral Audio-Language Model Detect Speech Spoofing?

[![arXiv](https://img.shields.io/badge/arXiv-2609.28713-b31b1b.svg)](https://doi.org/10.48550/arXiv.2609.28713)
[![DOI](https://zenodo.org/badge/1305091860.svg)](https://doi.org/10.5281/zenodo.23442861)

Spooftral adapts [Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) (`mistralai/Voxtral-Mini-3B-2507`), an audio-language model built on Ministral-3B, for speech spoofing detection. Instead of adding a conventional classification head, we use **Classification-as-Generation (CAG)**: given an audio recording, the model scores candidate labels such as `"bonafide"` and `"spoof"` by their conditional log-probabilities and chooses the higher-scoring label.

The approach uses the existing language-model output to make predictions. It does not require a separate classification head, although fine-tuning with DoRA introduces trainable adapter parameters. Prompts, class labels, and training settings are configured through YAML files.

This repository accompanies our paper, *“Spooftral: Can Voxtral Audio-Language Model Detect Speech Spoofing?”* We study two approaches:

1. **Spooftral:** CAG with DoRA adaptation of the audio-language model.
2. **Spooftral-Enc:** an audio-encoder-based model with a small MLP classification head, without running the language-model decoder.

With attention DoRA and margin regularisation, Spooftral achieves **4.25% EER** on the ASVspoof5 evaluation set, without model fusion or additional spoofing-specific training data. The complete paper results are listed [below](#results).

**Primary metric:** Equal Error Rate (EER; lower is better).  
**Datasets:** ASVspoof5 (primary), ASVspoof2021 LA/DF, and ASVspoof2019 LA.

Trained models are available on [Google Drive](https://drive.google.com/drive/folders/1MrBPnjTPT7poTqeRx954piV-oY76brQS?usp=sharing).

## Contents

- [Architecture](#architecture)
- [Two Training Paradigms](#two-training-paradigms)
- [Classification-as-Generation (CAG)](#classification-as-generation-cag)
- [Fine-Tuning](#fine-tuning)
- [Projector PEFT](#projector-peft)
- [Loss Functions](#loss-functions)
- [Batched CAG Forward Pass](#batched-cag-forward-pass)
- [Checkpoints and Resume](#checkpoints-and-resume)
- [Training Scripts](#training-scripts)
- [Inference Scripts](#inference-scripts)
- [Embedding and Projector Analysis](#embedding-and-projector-analysis)
- [Configuration](#configuration)
- [Key Source Files](#key-source-files)
- [How to Run](#how-to-run)
- [Evaluation](#evaluation)
- [Datasets](#datasets)
- [Results](#results)
- [Dependencies](#dependencies)
- [License](#license)
- [Contact](#contact)

## Architecture

Voxtral has three components that matter for this project.

### 1. Audio encoder

Voxtral uses a Whisper-based audio encoder. Depending on the experiment, we freeze it, fine-tune it, or adapt it using LoRA/DoRA. The `trainable_audio_encoder` setting controls this independently of the language model.

### 2. Multimodal projector

The projector connects the audio encoder to the language model:

```text
linear_1 → GELU → linear_2
```

It maps audio features to the language-model hidden size (3072 for Ministral-3B). By default, the projector is trained end to end (`projector_peft: False`). Setting `projector_peft: True` freezes its base linear weights and trains LoRA/DoRA adapters instead; see [Projector PEFT](#projector-peft).

For `k_downsample != 4`, the built-in projector is replaced by `VoxtralModalProjector_Encoder_Audio`, which maps `k_downsample * 1280` features to 3072 dimensions.

### 3. Language model

The Ministral-3B decoder receives the audio embeddings together with the text tokens. It can be fine-tuned with LoRA, DoRA, AdaLoRA, or full fine-tuning, or kept frozen, according to `finetune_mode`.

The prediction is based on the log-probability assigned to the label tokens.

## Two Training Paradigms

### A. Classification-as-Generation (Spooftral)

The primary approach uses the Voxtral decoder itself as the classifier. There is no additional classification head: the model compares the conditional log-probabilities of candidate label strings.

The implementation is `VoxtralStageWrapper_classification`. Training entry points include [`training_voxtral_generation_tokens05_21.py`](training_voxtral_generation_tokens05_21.py) and the scripts in [`more_training_voxtral_scripts/`](more_training_voxtral_scripts/).

For the ASVspoof5 attention-DoRA configuration (rank 32, α = 64), approximately **55.1 million parameters (1.17% of 4.71 billion)** are trainable.

### B. Audio-Adapter + MLP Head (Spooftral-Enc)

This alternative uses the audio side of Voxtral and skips the decoder:

```text
audio encoder → multi_modal_projector → pooling → MLP head → logits
```

The MLP head is:

```text
Linear(3072, 256) → GELU → Dropout(0.1) → Linear(256, 2)
```

The language model is not used for predictions and can optionally be offloaded to CPU via `drop_language_model`. The implementation is `AudioAdapterHeadClassifier` in [`utils/build_model_for_classification_head.py`](utils/build_model_for_classification_head.py).

The two approaches use the same dataset and evaluation infrastructure, with separate model wrappers and training loops. They both support the datasets listed in this README. For CAG experiments, [`scripts/train.py`](scripts/train.py) also provides a unified entry point that accepts `--config`.

## Classification-as-Generation (CAG)

For each recording, the code constructs one candidate sequence per class:

```text
candidate_0: [prompt_tokens | label_texts[0]]   # e.g. "bonafide"
candidate_1: [prompt_tokens | label_texts[1]]   # e.g. "spoof"
```

The candidate texts come from `generation_classification.label_texts` in the YAML configuration.

`score_label_logprob()` obtains the logits from the language model and computes the negative log-likelihood only over the label tokens. It uses a shifted mask because the logits at position `t` predict the token at `t + 1`.

To reduce peak GPU memory use, the implementation releases references to intermediate output and logits tensors before the loss computation's float32 conversion. It can also normalise each candidate score by the number of label tokens; this is important when the labels have different token lengths.

The decision rule is:

```text
scores = [logP(label_0 | audio), logP(label_1 | audio)]
prediction = argmax(scores)
```

A softmax over the two scores gives normalised class probabilities without changing the predicted class.

## Fine-Tuning

Our main parameter-efficient fine-tuning method is **DoRA** (Weight-Decomposed Low-Rank Adaptation).

For the language model, the typical configuration is:

```yaml
finetune_mode: dora
target_modules: [q_proj, k_proj, v_proj, o_proj]
```

The 2019 LA and 2021 LA/DF configurations use rank 16 and α = 32. For the ASVspoof5 attention-DoRA experiments, these are increased to rank 32 and α = 64. The corresponding values are defined in each experiment's YAML file.

The audio encoder is configured separately through `trainable_audio_encoder`. The multimodal projector can also use LoRA/DoRA through `projector_peft`.

### Projector PEFT

The multimodal projector is the connection between the encoder and the decoder. Training it without constraints can change the representations seen by the language model. We therefore added an option to adapt the projector while keeping its original linear weights frozen.

```yaml
projector_peft: False  # train the projector normally (default)
projector_peft: True   # freeze base weights; train LoRA/DoRA adapters
```

When enabled, this option:

- Requires `finetune_mode` to be `lora` or `dora`. Other modes raise `ValueError` during model construction.
- Reuses `lora_rank`, `lora_alpha`, and `lora_dropout` from the language-model configuration.
- Targets `linear_1` and `linear_2` in either the built-in or custom projector.
- Changes the projector's checkpoint keys, so switching `projector_peft` between training and resume is not supported.

The default remains `False` for compatibility with older runs and checkpoints.

## Loss Functions

CAG training compares the candidate-label scores. The configuration can combine cross-entropy with a margin term and an optional teacher-forcing regulariser.

### 1. Classification cross-entropy

Cross-entropy is applied to the two class scores:

```python
loss_ce = F.cross_entropy(class_scores, true_class)
```

`class_scores` contains token-length-normalised log-probabilities when `normalize_logprob_by_tokens: True`. This reduces the bias toward labels that happen to tokenize into fewer tokens. Optional class weights can be set with `class_weights`.

### 2. Margin loss (optional)

With `use_margin_loss: True`, a hinge loss encourages a gap between the correct and incorrect label scores:

```text
diff = logP(correct_label) - logP(wrong_label)
loss_margin = relu(margin - diff)
```

`margin_bonafide` and `margin_spoof` allow different margins for the two classes. The code uses the configured label texts to determine which margin belongs to which class, so changing the label order does not by itself swap their meanings. See the configuration note below for custom label names.

## Batched CAG Forward Pass

Rather than running the decoder separately for both labels, the CAG implementation scores both candidates in one forward pass:

1. The audio encoder runs once per input batch and its embeddings are cached.
2. The two sets of candidate sequences are padded to the same length.
3. The cached audio embeddings are duplicated to match a batch of size `2 * B`.
4. One language-model forward pass produces the logits for both candidates.
5. The label scores are extracted and split back into the two classes.

This reduces the number of decoder forward passes compared with evaluating each candidate separately.


### Checkpoint fields

| Key | Description |
| --- | --- |
| `wrapper_state_dict` | Model state (contents depend on the fine-tuning setup) |
| `optimizer` | Optimizer state |
| `scheduler` | Learning-rate scheduler state |
| `scaler` | GradScaler state for fp16 AMP, when used |
| `epoch` | Zero-indexed epoch at save time |
| `epoch_completed` | Whether the epoch finished before the checkpoint was saved |
| `global_step` | Optimizer step counter |
| `best_eer` | Best dev EER recorded |
| `best_dev_eer` | Independent best dev EER tracker |
| `best_eval_eer` | Independent best eval EER tracker |
| `rng_states` | Torch, CUDA, NumPy, and Python RNG states |
| `last_eval_epoch` / `last_eval_step` | Per-split evaluation trackers (eight keys in total) |
| `wandb_id` | W&B run ID |
| `configs` | Saved configuration snapshot |

Resume behaviour depends on whether the last epoch finished:

```text
epoch_completed = True  → continue with epoch + 1
epoch_completed = False → restart that epoch
```

Checkpoint saves use temporary files and atomic replacement/rename operations. Cleanup retains `checkpoint-last.pt`, the files referenced by `best_dev_metrics.json` and `best_eval_metrics.json`, and the `best_dev/` and `best_eval/` directories.

## Training Scripts

Each entry point selects a dataset, training paradigm, or experiment configuration.

### Classification-as-Generation

| Script | Dataset / variant | YAML |
| --- | --- | --- |
| [`training_voxtral_generation_tokens05_21.py`](training_voxtral_generation_tokens05_21.py) | ASVspoof5, ASVspoof2021 DoRA + Margin | `hyperparameters_05_dif_margins21.yaml` |
| `more_training_voxtral_scripts/training_voxtral_generation_tokens05_21_bigmargin.py` | ASVspoof5, larger Margin | `hyperparameters_05_dif_margins21bigmargin.yaml` |
| `more_training_voxtral_scripts/training_voxtral_generation_tokens05_21_LLM.py` | ASVspoof5, LLM-only DoRA | `hyperparameters_05_dif_margins21_LLM.yaml` |
| `more_training_voxtral_scripts/training_voxtral_generation_tokens05_no_margin.py` | ASVspoof5, no-margin baseline | `hyperparameters_05_no_margin.yaml` |
| `more_training_voxtral_scripts/training_voxtral_generation_tokens05_no_margin_realfake.py` | ASVspoof5, `"real"` / `"fake"` labels | `hyperparameters_05_no_margin_realfake.yaml` |
| `more_training_voxtral_scripts/training_voxtral_generation_tokens2019LA.py` | ASVspoof2019 LA | `hyperparameters_2019LA.yaml` |
| `more_training_voxtral_scripts/training_voxtral_generation_tokens2021_LA.py` | ASVspoof2021 LA | `hyperparameters_2021_LA_2.yaml` |
| `more_training_voxtral_scripts/training_voxtral_generation_tokens2021_DF.py` | ASVspoof2021 DF | `hyperparameters_2021_DF_2.yaml` |

These scripts use [`utils/training_evaluate_classification.py`](utils/training_evaluate_classification.py). You can also run CAG experiments through [`scripts/train.py`](scripts/train.py) by specifying a YAML file with `--config`.

### Audio-Adapter + MLP head

| Script | Dataset / variant | YAML |
| --- | --- | --- |
| [`training_audio_adapter_head_5_margin.py`](training_audio_adapter_head_5_margin.py) | ASVspoof5, margin loss | `hyperparameters_5_classification_head_margin.yaml` |
| `more_training_audio_adapter_scripts/training_audio_adapter_head.py` | ASVspoof5 baseline | `hyperparameters_5_classification_head.yaml` |
| `more_training_audio_adapter_scripts/training_audio_adapter_head_2019LA.py` | ASVspoof2019 LA | `hyperparameters_2019LA_classification_head.yaml` |
| `more_training_audio_adapter_scripts/training_audio_adapter_head_2021LA.py` | ASVspoof2021 LA | `hyperparameters_2021LA_classification_head.yaml` |
| `more_training_audio_adapter_scripts/training_audio_adapter_head_2021DF.py` | ASVspoof2021 DF | `hyperparameters_2021DF_classification_head.yaml` |

These scripts use [`utils/training_helpers_audio_adapter_head.py`](utils/training_helpers_audio_adapter_head.py).

## Inference Scripts

Use the following scripts to evaluate saved checkpoints without running the training loop.

| Script | Purpose |
| --- | --- |
| [`inference_checkpoint_new.py`](inference_checkpoint_new.py) | General CAG inference for ASVspoof5 |
| [`inference_checkpoint_no_margin.py`](inference_checkpoint_no_margin.py) | No-margin CAG variant |
| `more_inference_scripts/inference_checkpoint_2021_2019.py` | CAG inference for ASVspoof2021 LA/DF and ASVspoof2019 LA |
| `more_inference_scripts/inference_checkpoint_LLM.py` | LM-only PEFT variant |
| `more_inference_scripts/inference_checkpoint_real_fake.py` | `"real"` / `"fake"` label variant |
| `more_inference_scripts/inference_checkpoint_5_encoder_2.py` | Audio-Adapter + MLP-head inference for ASVspoof5 |
| `more_inference_scripts/inference_checkpoint_2021_2019_encoder.py` | Audio-Adapter + MLP-head inference for ASVspoof2021 LA/DF and ASVspoof2019 LA |

Inference rebuilds the model with the matching fine-tuning configuration, loads the checkpoint using `load_checkpoint_into_model` with `strict=True`, and produces utterance-level scores and EER.

## Embedding and Projector Analysis

[`calc_embeddings_projector_codes/`](calc_embeddings_projector_codes/) contains our representation-analysis scripts. This pipeline is separate from the main training code: it extracts Voxtral embeddings, groups them into NPZ datasets, trains lightweight classifiers, and compares representations from different model stages.

We used this analysis for the frozen-model linear-probing results in the paper. See [`calc_embeddings_projector_codes/README.md`](calc_embeddings_projector_codes/README.md) for usage details. Unlike the main training scripts, these utilities are configured through constants in the scripts themselves.

| Script | Purpose |
| --- | --- |
| [`calc_embeddings.py`](calc_embeddings_projector_codes/calc_embeddings.py) | Extracts embeddings at the `audio_projector`, `llama`, or `lm_head` stage and writes metadata |
| [`aggregate_embeddings_to_npz.py`](calc_embeddings_projector_codes/aggregate_embeddings_to_npz.py) | Combines embeddings and metadata into train/dev/eval NPZ files |
| [`verify_npz_embeddings.py`](calc_embeddings_projector_codes/verify_npz_embeddings.py) | Checks shapes, NaN/Inf values, and metadata consistency |
| [`training_on_embeds_one_net_5.py`](calc_embeddings_projector_codes/training_on_embeds_one_net_5.py) | Trains a linear classifier on saved embeddings |
| [`score_embeddings_pipeline.py`](calc_embeddings_projector_codes/score_embeddings_pipeline.py) | Evaluates embedding classifiers, optionally with bootstrap confidence intervals |
| [`compare_embeddings.py`](calc_embeddings_projector_codes/compare_embeddings.py) | Compares embeddings using cosine similarity, L2 distance, CKA, and pairwise-distance correlation |
| [`plot_pacmap_embeddings.py`](calc_embeddings_projector_codes/plot_pacmap_embeddings.py) | Produces PaCMAP visualisations for train/dev/eval embeddings |

## Configuration

Most training settings are defined in `configs/*.yaml`. The following example is based on `configs/hyperparameters_05_no_margin.yaml` and shows the main options; refer to the actual configuration files for experiment-specific values.

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

finetune_mode: dora   # lora | dora | adalora | frozen | full
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj]

# Audio encoder: full fine-tuning (True), frozen (False), or DoRA ("dora")
trainable_audio_encoder: dora

# Optional external encoder: None | wav2vec2 | hubert | wavlm
encoder_type: None

# 4 uses Voxtral's built-in projector; other values select a custom one
k_downsample: 4

# Adapt the projector with LoRA/DoRA instead of updating its base weights
projector_peft: False

precision: bfloat16  # bfloat16 | float16 | float32 | amp
use_4bit: False
resume: False      # set True to resume from an existing checkpoint

generation_classification:
  enable: True
  label_texts: {0: "bonafide", 1: "spoof"}
  normalize_logprob_by_tokens: True
  temperature: 1.0
  teacher_forcing_loss: False
  loss_type: cross_entropy

use_margin_loss: False
margin_bonafide: 0.1
margin_spoof: 0.05
lambda_margin: 0.05

augmentation_on: True
CODEC_PERTURB: True
AUG_ADD_NOISE: True
AUG_TIME_MASK: True
AUG_FREQ_MASK: True
GAIN_PERTURB: True
TIMESHIFT_PERTURB: True
SELECTIVE_AUGMENTATION: True
MAX_AUGMENTATIONS_PER_SAMPLE: 4
```

**Margin labels:** The code assigns `margin_bonafide` by checking whether a configured label contains `"bonafide"`. If you introduce new class names, check this mapping rather than assuming arbitrary names will be recognised automatically.

## Key Source Files

| File | Role |
| --- | --- |
| [`utils/training_evaluate_classification.py`](utils/training_evaluate_classification.py) | CAG training, losses, evaluation, checkpointing, and W&B logging |
| [`utils/generation_text_bonafide_spoof_classification.py`](utils/generation_text_bonafide_spoof_classification.py) | Candidate construction, label masks, and log-probability scoring |
| [`utils/eval_classification.py`](utils/eval_classification.py) | Dev/eval EER loop with inference mode and training-mode restoration |
| [`utils/build_model.py`](utils/build_model.py) | Voxtral loading, fine-tuning setup, projector, and CAG wrapper |
| [`utils/build_model_for_classification_head.py`](utils/build_model_for_classification_head.py) | `AudioAdapterHeadClassifier` and its tuning/pooling settings |
| [`utils/training_helpers_audio_adapter_head.py`](utils/training_helpers_audio_adapter_head.py) | Training for the MLP-head paradigm |
| [`utils/checkpoint_utils.py`](utils/checkpoint_utils.py) | Checkpoint save/load and resume state |
| [`utils/cleanup_checkpoints.py`](utils/cleanup_checkpoints.py) | Old-checkpoint cleanup while preserving selected checkpoints |
| [`utils/freezing.py`](utils/freezing.py) | Diagnostic summary of frozen and trainable parameters |
| [`utils/losses.py`](utils/losses.py) | Token-level causal LM loss with float32 computation |
| [`utils/dataset.py`](utils/dataset.py) | Dataset protocols, audio loading, and on-the-fly augmentation |
| [`utils/augmentation.py`](utils/augmentation.py), [`utils/codec.py`](utils/codec.py) | Audio perturbation and codec simulations |
| [`utils/RawBoost.py`](utils/RawBoost.py), [`utils/data_utils_rawboost.py`](utils/data_utils_rawboost.py) | RawBoost augmentations |
| [`utils/config.py`](utils/config.py) | YAML loading and experiment settings |
| [`configs/`](configs/) | Configurations for CAG and classification-head experiments |
| [`calc_embeddings_projector_codes/`](calc_embeddings_projector_codes/) | Frozen-embedding extraction and linear probing |


## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Some augmentations also require the `ffmpeg` system executable to be installed and available on `PATH`.

### 2. Set up a configuration

Copy an existing YAML file and adjust the paths and settings for your environment:

```bash
cp configs/hyperparameters_05_no_margin.yaml configs/my_experiment.yaml
```

In particular, check the protocol paths, audio directories, `save_path`, and W&B settings.

### 3. Configure W&B (if enabled)

```bash
export WANDB_API_KEY=<your_key>
```

### 4. Start training

For example, to train the ASVspoof5 CAG no-margin configuration:

```bash
python more_training_voxtral_scripts/training_voxtral_generation_tokens05_no_margin.py
```

Or use the unified entry point with a configuration file:

```bash
python scripts/train.py --config configs/hyperparameters_05_no_margin.yaml
```

To select a GPU on a machine with several devices:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/hyperparameters_05_no_margin.yaml
```

### 5. Run evaluation only

Set `only_eval: True` in the relevant YAML configuration and launch the same entry point.

## Evaluation

We report **Equal Error Rate (EER)**, where the false acceptance and false rejection rates are equal. Lower values indicate better separation between bonafide and spoofed speech.

The training code tracks best dev and eval EER independently:

- `best_eer` / `best_dev_eer`: best validation EER, used for dev-based checkpoint selection.
- `best_eval_eer`: best evaluation EER, retained separately for analysis.

These values are logged and stored in checkpoints so that they can be restored after a resume.

**For fair model comparison, use the dev set for selecting checkpoints and tuning hyperparameters.** Tracking the best result on the eval set is useful for diagnostics, but choosing a final model based on eval performance introduces test-set selection bias.

## Datasets

| Dataset | Use in this repository |
| --- | --- |
| **ASVspoof5** | Main benchmark, Track 1 (closed condition). Uses `ASVspoof5.train.tsv`, `ASVspoof5.dev.track_1.tsv`, and `ASVspoof5.eval.track_1.tsv`; audio files are FLAC. The protocol parser reads whitespace-delimited fields despite the `.tsv` file extension. |
| **ASVspoof2021 LA** | Logical Access evaluation data; `trial_metadata.txt` includes `file_id`, `label`, and `system_id`. |
| **ASVspoof2021 DF** | Deepfake evaluation data with a wider range of synthesis/conversion conditions; used by the DF-specific scripts. |
| **ASVspoof2019 LA** | Logical Access data used for experiments and cross-dataset analysis. |

The task is binary: **bonafide** (genuine speech) versus **spoof** (manipulated or generated speech). Candidate output labels for CAG are supplied through the YAML configuration.

## Results

The following EER values (%) are reported in the paper.

### Linear probing with frozen Voxtral

We trained linear classifiers on frozen representations from the audio adapter and the LLM decoder. The audio-side representations gave lower EER across these evaluations, motivating the adaptation experiments.

| Database | LLM Dev. | LLM Eval. | Audio Adapter Dev. | Audio Adapter Eval. |
| --- | ---: | ---: | ---: | ---: |
| ASVspoof2019 LA | 6.60 | 7.17 | 2.79 | 5.48 |
| ASVspoof2021 LA | 9.26 | 17.34 | 2.79 | 13.57 |
| ASVspoof2021 DF | 7.02 | 19.20 | 2.79 | 9.64 |
| ASVspoof5 | 20.73 | 19.09 | 11.58 | 9.75 |

### Spooftral and Spooftral-Enc

DoRA-adapted results (with attention DoRA and margin regularisation for the ASVspoof5 Spooftral result):

| Database | Spooftral Dev. | Spooftral Eval. | Spooftral-Enc Dev. | Spooftral-Enc Eval. |
| --- | ---: | ---: | ---: | ---: |
| ASVspoof2019 LA | 0.02 | 0.41 | 0.03 | 0.40 |
| ASVspoof2021 LA | 0.02 | 6.46 | 0.03 | 6.70 |
| ASVspoof2021 DF | 0.05 | 3.88 | 0.05 | 4.76 |
| ASVspoof5 | 4.50 | **4.25** | 5.29 | 5.08 |

### ASVspoof5 ablation

| Configuration | Dev. EER | Eval. EER |
| --- | ---: | ---: |
| Spooftral (LLM attention DoRA only, frozen audio encoder) | 7.89 | 7.01 |
| Spooftral (attention DoRA, including audio encoder) | 4.77 | 4.46 |
| Spooftral (attention DoRA + margin) | **4.50** | **4.25** |
| Spooftral-Enc (attention DoRA) | 5.29 | 5.08 |

We also compared label wording under the same DoRA setup. `"bonafide"` / `"spoof"` produced 4.77% dev and 4.46% eval EER, while `"real"` / `"fake"` produced 5.67% dev and 4.47% eval EER. The corresponding configuration is [`hyperparameters_05_no_margin_realfake.yaml`](configs/hyperparameters_05_no_margin_realfake.yaml).

## Dependencies

Install the Python packages from [`requirements.txt`](requirements.txt). The main dependencies and their roles are:

| Package | Purpose |
| --- | --- |
| `torch >= 2.1` | Model training, AMP, gradient checkpointing |
| `transformers >= 4.45` | Voxtral model support |
| `peft >= 0.12` | LoRA/DoRA adaptation |
| `torchaudio` | Audio I/O and transformations |
| `bitsandbytes` | Quantisation and QLoRA |
| `wandb` | Experiment tracking |
| `numpy`, `scipy` | Metrics and signal processing |
| `scikit-learn` | ROC, EER, and classification metrics |
| `matplotlib` | Visualisations |
| `tqdm` | Progress bars |
| `evaluate` | Hugging Face evaluation utilities |
| `huggingface_hub` | Model and config downloads |
| `python-dotenv` | Environment variable loading |
| `omegaconf` | Optional config-container support |
| `pyyaml` | YAML configuration parsing |
| `soundfile` | FLAC/audio reading |
| `ffmpeg` | Codec augmentation; install as a **system executable** and add to `PATH` |

## License

The Spooftral source code is distributed under the [Apache License 2.0](LICENSE), subject to the copyright and license terms of any incorporated third-party components. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

The associated IEEE publication, including its text, figures, and tables, is governed by the applicable IEEE copyright terms and is not covered by the repository's Apache 2.0 license.

### Third-party software

This repository includes adapted code from [RawBoost](https://github.com/TakHemlata/RawBoost-antispoofing) and [ConfidenceIntervals](https://github.com/luferrer/ConfidenceIntervals). Their original MIT license texts and the locations of the relevant code are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The same file also documents the Mistral AI Voxtral base-model license.

## Contact

We recently refactored the code to make it cleaner, and easier to understand.

If you have questions, issues, or want to collaborate, please feel free to reach out:

- 📧 Email: [Avishai Weizman](mailto:avishai11900@gmail.com)
- 🔗 GitHub: [github.com/avishai111](https://github.com/avishai111)
- 💼 LinkedIn: [linkedin.com/in/avishai-weizman/](https://linkedin.com/in/avishai-weizman/)
