# Embedding and Projector Analysis

This folder contains the representation-analysis pipeline used to extract Voxtral embeddings, convert them into dataset-level NPZ bundles, train lightweight classifiers on top of those embeddings, and score or compare the resulting representations.

The code here is aimed at research workflows, not packaged CLI tools. Most scripts are configured by editing constants inside the files before running them.

## What this folder is for

Typical use cases:

- extract embeddings from different Voxtral stages such as `audio_projector`, `llama`, or `lm_head`
- build compact NPZ datasets from per-file embedding dumps
- train a simple linear classifier on top of frozen embeddings
- score saved projector checkpoints on train/dev/eval splits
- compare two embedding spaces directly
- sanity-check generated NPZ bundles before training

## Main workflow

The usual pipeline is:

1. Run `calc_embeddings.py`
   Extract per-file embeddings and save a metadata CSV.

2. Run `aggregate_embeddings_to_npz.py`
   Merge those per-file embeddings into `total_train_embeddings.npz`, `total_dev_embeddings.npz`, or `total_eval_embeddings.npz`.

3. Optionally run `verify_npz_embeddings.py`
   Check that the NPZ bundle is structurally valid before training.

4. Run `training_on_embeds_one_net_5.py`
   Train a lightweight classifier on the frozen embeddings.

5. Optionally run `score_embeddings_pipeline.py`
   Score saved checkpoints and write per-split outputs for later analysis.

6. Optionally run `compare_embeddings.py`
   Compare two embedding sets with similarity and distance metrics.

## Files in this folder

### `calc_embeddings.py`

Extracts per-file embeddings from Voxtral for ASVspoof datasets.

What it does:

- loads `mistralai/Voxtral-Mini-3B-2507`
- builds Voxtral-style audio-plus-prompt inputs
- extracts embeddings from a selected stage
- saves one embedding file per audio sample
- writes a metadata CSV that points to those files

Important parameters to edit:

- `DATASET_TYPE`
- `ASVSPOOF2021_TASK`
- `SETS`
- `DEVICE`
- `REPO_ID`
- `BATCH_SIZE`
- `OUTPUT_STAGE`
- `EMBEDDING_QUESTION`
- `RESUME`
- `N_AUDIO_TOKENS`
- hardcoded dataset root paths inside the script

Supported output stages:

- `audio_projector`
- `llama`
- `lm_head`

Outputs:

- per-file embedding `.npy` files
- metadata CSV such as `audio_projector_dev_voxtral_all_embeddings_metadata.csv`

### `aggregate_embeddings_to_npz.py`

Builds a single NPZ dataset from the per-file outputs produced by `calc_embeddings.py`.

What it does:

- reads the metadata CSV
- loads all referenced embedding files
- merges metadata fields
- optionally fills missing fields from official ASVspoof protocol files
- writes a single NPZ bundle

The output NPZ can contain:

- `embeddings`
- `labels`
- `attacks`
- `speaker_ids`
- `codecs`
- `file_ids`

Important parameters to edit:

- `METADATA_PATH`
- `OUTPUT_PATH`
- `PROTOCOL_PATH`

### `verify_npz_embeddings.py`

Validates a generated NPZ bundle before downstream use.

Checks include:

- `embeddings` exists and is 2D
- no NaN or Inf values
- metadata lengths match the number of embeddings
- optional expected dimension and expected count

Important parameters to edit:

- `NPZ_PATH`
- `EXPECTED_DIM`
- `EXPECTED_COUNT`
- `REQUIRE_FILE_IDS`
- `STRICT`

### `training_on_embeds_one_net_5.py`

Trains a simple PyTorch Lightning classifier on frozen embedding datasets.

What it does:

- loads train/dev/eval NPZ files
- trains a linear projection head
- monitors EER during training
- saves checkpoints and plots
- can save eval score snapshots during training

Current training entry point is configured in the `__main__` block.

Important parameters to edit:

- `DB1_NAME`
- `DB1_DIR`
- `TRAIN_FILE`
- `DEV_FILE`
- `EVAL_FILE`
- `INPUT_DIM`
- `PROJECT_OUT`
- `BATCH_SIZE`
- `NUM_EPOCHS`
- `NUM_WORKERS`
- `SAVE_EVAL_NPZ_EVERY`
- `LEARNING_RATES`
- `CLASS_WEIGHTS_LIST`

### `score_embeddings_pipeline.py`

Runs inference with trained embedding classifiers and writes score outputs for analysis.

What it does:

- loads NPZ datasets
- restores a trained checkpoint
- computes spoof probabilities
- evaluates EER and accuracy
- optionally computes bootstrap confidence intervals
- writes per-split score files

This script is driven by a list of jobs in `PARAMETER_RUN_CONFIGS`.

Important parameters to edit:

- `PARAMETER_RUN_CONFIGS`
- `SPLITS_TO_SCORE`
- `BATCH_SIZE`
- `NUM_WORKERS`

Relevant environment variables:

- `CONFIDENCE_INTERVALS_DIR`
- `EER_NUM_BOOTSTRAPS`
- `EER_ALPHA`
- `EER_MAX_RESAMPLE_RETRIES`
- `EER_USE_BOOTSTRAP`

### `compare_embeddings.py`

Compares two NPZ embedding files and reports representation-similarity metrics.

Metrics include:

- average cosine similarity
- average L2 distance
- CKA
- pairwise distance correlation

Important parameters to edit:

- `NPZ_PATH_1`
- `NPZ_PATH_2`
- `OUTPUT_FILE`
- `USE_FLOAT16`

### `README.md`

This file.

## Quick usage

### 1) Extract embeddings

```bash
python calc_embeddings.py
```

### 2) Aggregate per-file outputs into NPZ

```bash
python aggregate_embeddings_to_npz.py
```

### 3) Verify the NPZ bundle

```bash
python verify_npz_embeddings.py
```

### 4) Train a projector / linear classifier

```bash
python training_on_embeds_one_net_5.py
```

### 5) Score trained checkpoints

```bash
python score_embeddings_pipeline.py
```

### 6) Compare two embedding sets

```bash
python compare_embeddings.py
```

## Expected dataset format

The downstream scripts expect dataset-level NPZ bundles such as:

- `total_train_embeddings.npz`
- `total_dev_embeddings.npz`
- `total_eval_embeddings.npz`

At minimum, those files should contain:

- `embeddings`
- `labels`

The aggregation script can also add:

- `attacks`
- `speaker_ids`
- `codecs`
- `file_ids`

## Outputs

Depending on the stage of the pipeline, this folder can produce:

- per-file embedding `.npy` files
- metadata CSV files
- merged NPZ datasets
- classifier checkpoints
- EER plots
- saved score snapshots
- per-split scoring outputs
- embedding comparison reports
