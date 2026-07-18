"""
Score embedding datasets with a trained classifier checkpoint.

The pipeline loads train/dev/eval NPZ embeddings, restores a checkpoint,
computes spoof probabilities, evaluates EER/accuracy, and writes per-split
score outputs for analysis.
"""

import csv
import gc
import glob
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


_TQDM_DISABLED = str(os.environ.get("TQDM_DISABLE", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}


def _progress(iterable, **kwargs):
    """
    Wrap iterables with tqdm when available and not disabled.
    """
    if tqdm is None or _TQDM_DISABLED:
        return iterable
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(iterable, **kwargs)


@dataclass
class parameter_run:
    """
    Job configuration for one pipeline run.

    Required per job:
    - db_name, db_dir
    - checkpoint
    - output_dir

    Optional per job:
    - train_file/dev_file/eval_file (defaults point to the standard NPZ names)
    - name (for output filename prefix)
    """
    db_name: str = ""
    db_dir: str = ""
    checkpoint: str = ""
    output_dir: str = ""
    train_file: str = "total_train_embeddings.npz"
    dev_file: str = "total_dev_embeddings.npz"
    eval_file: str = "total_eval_embeddings.npz"
    name: str = ""

    def split_path(self, split_name: str) -> str:
        """
        Resolve the full NPZ path for a requested split.
        Accepts: train/dev/eval.
        """
        # Step 1: Choose the configured file name for the requested split.
        if split_name == "train":
            split_file = self.train_file
        elif split_name == "dev":
            split_file = self.dev_file
        elif split_name == "eval":
            split_file = self.eval_file
        else:
            raise ValueError(f"Unsupported split '{split_name}'")

        # Step 2: Expand user/home markers and detect absolute vs relative path.
        candidate = os.path.expanduser(split_file)
        if os.path.isabs(candidate):
            # Step 3a: Absolute path -> normalize and return as is.
            return os.path.abspath(candidate)
        # Step 3b: Relative path -> resolve against dataset base directory.
        return os.path.abspath(os.path.join(self.db_dir, split_file))


def make_parameter_run_jobs_for_dataset(
    db_name: str,
    db_dir: str,
    checkpoints: Sequence[str],
    output_dir: str,
    train_file: str = "total_train_embeddings.npz",
    dev_file: str = "total_dev_embeddings.npz",
    eval_file: str = "total_eval_embeddings.npz",
) -> List[parameter_run]:
    """
    Helper: build multiple parameter_run jobs for the same dataset directory and output directory,
    each with a different checkpoint.
    """
    # Step 1: Initialize the output jobs list.
    jobs = []
    # Step 2: Build one job object per checkpoint.
    for idx, checkpoint in enumerate(checkpoints, start=1):
        jobs.append(
            parameter_run(
                db_name=db_name,
                db_dir=db_dir,
                checkpoint=checkpoint,
                output_dir=output_dir,
                train_file=train_file,
                dev_file=dev_file,
                eval_file=eval_file,
                name=f"{db_name}_ckpt{idx:03d}",
            )
        )
    # Step 3: Return the complete job list.
    return jobs


# Jobs list:
# Each item is one `parameter_run(...)` object with its own relevant params.
# The pipeline will iterate this list and run one full scoring job per item.
#
# `checkpoint` value supports:
# 1) direct .ckpt path
# 2) run directory that contains .ckpt files and saved eval EER metadata
# 3) wildcard pattern (e.g. /path/to/runs/run_*), which expands to many jobs

#Audio projector checkpoints for ASVspoof2019 LA, ASVspoof2021 LA/DF and ASVspoof5
PARAMETER_RUN_CONFIGS = [
    #ASVspoof2019 LA
    parameter_run(
        name="db1_audio_projector_2019_LA",
        db_name="audio_projector_2019",
        db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/audio_projector/",
        checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/audio_projector/checkpoints/audio_projector_2019/lr0p01_bs2048_ep1000_in3072_out2_cw1p0_1p0/run_20260203_190648/best-epoch=492-val_loss=0.0573.ckpt",
        output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/audio_projector/checkpoints/audio_projector_2019/lr0p01_bs2048_ep1000_in3072_out2_cw1p0_1p0/run_20260203_190648/scores_text_outputs/",
    ),
    parameter_run(
        name="db2_audio_projector_2019_LA",
        db_name="audio_projector_2019",
        db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/audio_projector/",
        checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/audio_projector/checkpoints/audio_projector_2019/lr0p01_bs2048_ep1000_in3072_out2_cw1_3/run_20260203_203335/best-epoch=443-val_loss=0.0695.ckpt",
        output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/audio_projector/checkpoints/audio_projector_2019/lr0p01_bs2048_ep1000_in3072_out2_cw1_3/run_20260203_203335/scores_text_outputs/",
    ),
    parameter_run(
        name="db3_audio_projector_2019_LA",
        db_name="audio_projector_2019",
        db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/audio_projector/",
        checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/audio_projector/checkpoints/audio_projector_2019/lr0p01_bs2048_ep1000_in3072_out2_cw1_5/run_20260203_221302/best-epoch=393-val_loss=0.0814.ckpt",
        output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/audio_projector/checkpoints/audio_projector_2019/lr0p01_bs2048_ep1000_in3072_out2_cw1_5/run_20260203_221302/scores_text_outputs/",
    ),
    
    #ASVspoof2021 LA
    parameter_run(
        name="db1_audio_projector_2021_LA",
        db_name="audio_projector_2021",
        db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/audio_projector/",
        checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/audio_projector/checkpoints/audio_projector_2021/lr0p01_bs2048_ep1000_in3072_out2_cw1p0_1p0/run_20260204_064350/best-epoch=492-val_loss=0.0573.ckpt",
        output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/audio_projector/checkpoints/audio_projector_2021/lr0p01_bs2048_ep1000_in3072_out2_cw1p0_1p0/run_20260204_064350/scores_text_outputs/",
    ),
    parameter_run(
        name="db2_audio_projector_2021_LA",
        db_name="audio_projector_2021",
        db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/audio_projector/",
        checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/audio_projector/checkpoints/audio_projector_2021/lr0p01_bs2048_ep1000_in3072_out2_cw1_3/run_20260204_083912/best-epoch=434-val_loss=0.0684.ckpt",
        output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/audio_projector/checkpoints/audio_projector_2021/lr0p01_bs2048_ep1000_in3072_out2_cw1_3/run_20260204_083912/scores_text_outputs/",
    ),
    parameter_run(
        name="db3_audio_projector_2021_LA",
        db_name="audio_projector_2021",
        db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/audio_projector/",
        checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/audio_projector/checkpoints/audio_projector_2021/lr0p01_bs2048_ep1000_in3072_out2_cw1_5/run_20260204_103512/best-epoch=447-val_loss=0.0843.ckpt",
        output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/audio_projector/checkpoints/audio_projector_2021/lr0p01_bs2048_ep1000_in3072_out2_cw1_5/run_20260204_103512/scores_text_outputs/",
    ),
    
    #ASVspoof2021 DF
    parameter_run(
        name="db1_audio_projector_2021_DF",
        db_name="audio_projector_2021",
        db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/audio_projector/",
        checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/audio_projector/checkpoints/audio_projector_2021/lr0p01_bs2048_ep1000_in3072_out2_cw1p0_1p0/run_20260204_204943/best-epoch=492-val_loss=0.0573.ckpt",
        output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/audio_projector/checkpoints/audio_projector_2021/lr0p01_bs2048_ep1000_in3072_out2_cw1p0_1p0/run_20260204_204943/scores_text_outputs/",
    ),
    parameter_run(
        name="db2_audio_projector_2021_DF",
        db_name="audio_projector_2021",
        db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/audio_projector/",
        checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/audio_projector/checkpoints/audio_projector_2021/lr0p01_bs2048_ep1000_in3072_out2_cw1_3/run_20260205_015136/best-epoch=434-val_loss=0.0684.ckpt",
        output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/audio_projector/checkpoints/audio_projector_2021/lr0p01_bs2048_ep1000_in3072_out2_cw1_3/run_20260205_015136/scores_text_outputs/",
    ),
    parameter_run(
        name="db3_audio_projector_2021_DF",
        db_name="audio_projector_2021",
        db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/audio_projector/",
        checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/audio_projector/checkpoints/audio_projector_2021/lr0p01_bs2048_ep1000_in3072_out2_cw1_5/run_20260205_063958/best-epoch=447-val_loss=0.0843.ckpt",
        output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/audio_projector/checkpoints/audio_projector_2021/lr0p01_bs2048_ep1000_in3072_out2_cw1_5/run_20260205_063958/scores_text_outputs/",
    ),
    
    
    #ASVspoof5
    parameter_run(
        name="db1_audio_projector_5",
        db_name="audio_projector_5",
        db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/audio_projector/",
        checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/audio_projector/checkpoints/audio_projector_5/lr0p001_bs2048_ep1000_in3072_out2_cw1p0_1p0/run_20260205_111744/best-epoch=82-val_loss=0.4400.ckpt",
        output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/audio_projector/checkpoints/audio_projector_5/lr0p001_bs2048_ep1000_in3072_out2_cw1p0_1p0/run_20260205_111744/scores_text_outputs/",
    ),
    parameter_run(
        name="db2_audio_projector_5",
        db_name="audio_projector_5",
        db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/audio_projector/",
        checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/audio_projector/checkpoints/audio_projector_5/lr0p001_bs2048_ep1000_in3072_out2_cw1_3/run_20260205_204030/best-epoch=148-val_loss=0.2254.ckpt",
        output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/audio_projector/checkpoints/audio_projector_5/lr0p001_bs2048_ep1000_in3072_out2_cw1_3/run_20260205_204030/scores_text_outputs/",
    ),
    parameter_run(
        name="db3_audio_projector_5",
        db_name="audio_projector_5",
        db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/audio_projector/",
        checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/audio_projector/checkpoints/audio_projector_5/lr0p001_bs2048_ep1000_in3072_out2_cw1_5/run_20260206_060029/best-epoch=139-val_loss=0.1576.ckpt",
        output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/audio_projector/checkpoints/audio_projector_5/lr0p001_bs2048_ep1000_in3072_out2_cw1_5/run_20260206_060029/scores_text_outputs/",
    ),
]

# #Llama checkpoints for ASVspoof2019 LA, ASVspoof2021 LA/DF and ASVspoof5
# PARAMETER_RUN_CONFIGS = [
#     #ASVspoof2019 LA
#     parameter_run(
#         name="db1_llama_2019_LA",
#         db_name="llama_2019",
#         db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama/",
#         checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama/checkpoints/llama_2019/lr0p001_bs2048_ep400_in3072_out2_cw1p0_1p0/run_20260206_201102/best-epoch=223-val_loss=0.0863.ckpt",
#         output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama/checkpoints/llama_2019/lr0p001_bs2048_ep400_in3072_out2_cw1p0_1p0/run_20260206_201102/scores_text_outputs/",
#     ),
#     parameter_run(
#         name="db2_llama_2019_LA",
#         db_name="llama_2019",
#         db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama/",
#         checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama/checkpoints/llama_2019/lr0p001_bs2048_ep400_in3072_out2_cw1_3/run_20260206_204853/best-epoch=135-val_loss=0.0906.ckpt",
#         output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama/checkpoints/llama_2019/lr0p001_bs2048_ep400_in3072_out2_cw1_3/run_20260206_204853/scores_text_outputs/",
#     ),
#     parameter_run(
#         name="db3_llama_2019_LA",
#         db_name="llama_2019",
#         db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama/",
#         checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama/checkpoints/llama_2019/lr0p001_bs2048_ep400_in3072_out2_cw1_5/run_20260206_212636/best-epoch=183-val_loss=0.1020.ckpt",
#         output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2019_embeds/llama/checkpoints/llama_2019/lr0p001_bs2048_ep400_in3072_out2_cw1_5/run_20260206_212636/scores_text_outputs/",
#     ),
    
#     #ASVspoof2021 LA
#     parameter_run(
#         name="db1_llama_2021_LA",
#         db_name="llama_2021",
#         db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/llama/",
#         checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/llama/checkpoints/llama_2021/lr0p0005_bs2048_ep400_in3072_out2_cw1p0_1p0/run_20260206_230446/best-epoch=342-val_loss=0.0916.ckpt",
#         output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/llama/checkpoints/llama_2021/lr0p0005_bs2048_ep400_in3072_out2_cw1p0_1p0/run_20260206_230446/scores_text_outputs/",
#     ),
#     parameter_run(
#         name="db2_llama_2021_LA",
#         db_name="llama_2021",
#         db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/llama/",
#         checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/llama/checkpoints/llama_2021/lr0p0005_bs2048_ep400_in3072_out2_cw1_3/run_20260206_235533/best-epoch=248-val_loss=0.1045.ckpt",
#         output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/llama/checkpoints/llama_2021/lr0p0005_bs2048_ep400_in3072_out2_cw1_3/run_20260206_235533/scores_text_outputs/",
#     ),
#     parameter_run(
#         name="db3_llama_2021_LA",
#         db_name="llama_2021",
#         db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/llama/",
#         checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/llama/checkpoints/llama_2021/lr0p0005_bs2048_ep400_in3072_out2_cw1_5/run_20260207_004631/best-epoch=371-val_loss=0.1184.ckpt",
#         output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_LA_embeds/llama/checkpoints/llama_2021/lr0p0005_bs2048_ep400_in3072_out2_cw1_5/run_20260207_004631/scores_text_outputs/",
#     ),
    
#     #ASVspoof2021 DF
#     parameter_run(
#         name="db1_llama_2021_DF",
#         db_name="llama_2021",
#         db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/llama/",
#         checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/llama/checkpoints/llama_2021/lr0p0005_bs2048_ep400_in3072_out2_cw1p0_1p0/run_20260207_023226/best-epoch=342-val_loss=0.0916.ckpt",
#         output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/llama/checkpoints/llama_2021/lr0p0005_bs2048_ep400_in3072_out2_cw1p0_1p0/run_20260207_023226/scores_text_outputs/",
#     ),
#     parameter_run(
#         name="db2_llama_2021_DF",
#         db_name="llama_2021",
#         db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/llama/",
#         checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/llama/checkpoints/llama_2021/lr0p0005_bs2048_ep400_in3072_out2_cw1_3/run_20260207_035607/best-epoch=248-val_loss=0.1045.ckpt",
#         output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/llama/checkpoints/llama_2021/lr0p0005_bs2048_ep400_in3072_out2_cw1_3/run_20260207_035607/scores_text_outputs/",
#     ),
#     parameter_run(
#         name="db3_llama_2021_DF",
#         db_name="llama_2021",
#         db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/llama/",
#         checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/llama/checkpoints/llama_2021/lr0p0005_bs2048_ep400_in3072_out2_cw1_5/run_20260207_051839/best-epoch=371-val_loss=0.1184.ckpt",
#         output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof2021_DF_embeds/llama/checkpoints/llama_2021/lr0p0005_bs2048_ep400_in3072_out2_cw1_5/run_20260207_051839/scores_text_outputs/",
#     ),
    
    
#     #ASVspoof5
#     parameter_run(
#         name="db1_llama_5",
#         db_name="llama_5",
#         db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/llama/",
#         checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/llama/checkpoints/llama_5/lr0p0005_bs2048_ep400_in3072_out2_cw1p0_1p0/run_20260207_064424/best-epoch=95-val_loss=0.8006.ckpt",
#         output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/llama/checkpoints/llama_5/lr0p0005_bs2048_ep400_in3072_out2_cw1p0_1p0/run_20260207_064424/scores_text_outputs/",
#     ),
#     parameter_run(
#         name="db2_llama_5",
#         db_name="llama_5",
#         db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/llama/",
#         checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/llama/checkpoints/llama_5/lr0p0005_bs2048_ep400_in3072_out2_cw1_3/run_20260207_083912/best-epoch=04-val_loss=0.4194.ckpt",
#         output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/llama/checkpoints/llama_5/lr0p0005_bs2048_ep400_in3072_out2_cw1_3/run_20260207_083912/scores_text_outputs/",
#     ),
#     parameter_run(
#         name="db3_llama_5",
#         db_name="llama_5",
#         db_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/llama/",
#         checkpoint="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/llama/checkpoints/llama_5/lr0p0005_bs2048_ep400_in3072_out2_cw1_5/run_20260207_104714/best-epoch=00-val_loss=0.2790.ckpt",
#         output_dir="/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/llama/checkpoints/llama_5/lr0p0005_bs2048_ep400_in3072_out2_cw1_5/run_20260207_104714/scores_text_outputs/",
#     ),
# ]

# Which splits to score: any subset of ["train", "dev", "eval"] or ["all"]
SPLITS_TO_SCORE = ["dev", "eval"]

# Inference settings
BATCH_SIZE = 2048
NUM_WORKERS = max(0, int(os.environ.get("NUM_WORKERS", "8")))
if os.cpu_count():
    NUM_WORKERS = min(NUM_WORKERS, int(os.cpu_count()))
DATALOADER_PREFETCH_FACTOR = max(1, int(os.environ.get("DATALOADER_PREFETCH_FACTOR", "2")))
WRITE_CHUNK_SIZE = max(1, int(os.environ.get("WRITE_CHUNK_SIZE", "4096")))

# confidence_intervals settings:
# - CONFIDENCE_INTERVALS_DIR: optional local folder for import fallback.
CONFIDENCE_INTERVALS_DIR = os.environ.get("CONFIDENCE_INTERVALS_DIR", "").strip()
EER_NUM_BOOTSTRAPS = int(os.environ.get("EER_NUM_BOOTSTRAPS", "1000"))
EER_ALPHA = float(os.environ.get("EER_ALPHA", "5"))
EER_MAX_RESAMPLE_RETRIES = int(os.environ.get("EER_MAX_RESAMPLE_RETRIES", "2"))
EER_USE_BOOTSTRAP = str(os.environ.get("EER_USE_BOOTSTRAP", "1")).strip().lower() in {"1", "true", "yes", "y", "on"}
AUTO_FLIP_SCORE_POLARITY = str(os.environ.get("AUTO_FLIP_SCORE_POLARITY", "1")).strip().lower() in {"1", "true", "yes", "y", "on"}
SCORE_POLARITY_EER_MARGIN = max(0.0, float(os.environ.get("SCORE_POLARITY_EER_MARGIN", "1e-4")))

# Simple direct import from the configured folder (if provided).
if CONFIDENCE_INTERVALS_DIR:
    _ci_dir = os.path.abspath(os.path.expanduser(CONFIDENCE_INTERVALS_DIR))
    # If path points to ".../confidence_intervals", Python needs its parent in sys.path.
    if os.path.basename(_ci_dir).lower() == "confidence_intervals":
        _ci_dir = os.path.dirname(_ci_dir)
    if _ci_dir not in sys.path:
        sys.path.insert(0, _ci_dir)

_CONF_INT_IMPORT_ERROR = None
try:
    from confidence_intervals import evaluate_with_conf_int, get_bootstrap_indices
except Exception as _import_error:
    evaluate_with_conf_int = None
    get_bootstrap_indices = None
    _CONF_INT_IMPORT_ERROR = _import_error


def normalize_label_to_int(label_value):
    """
    Normalize common textual/numeric label representations to int class IDs.

    Mapping:
    - bonafide/genuine/0 -> 0
    - spoof/1 -> 1
    """
    # Step 1: Decode bytes-like inputs to text.
    if isinstance(label_value, (bytes, bytearray)):
        label_value = label_value.decode("utf-8", errors="ignore")
    # Step 2: Fast-path for integer inputs.
    if isinstance(label_value, (int, np.integer)):
        v = int(label_value)
        if v in (0, 1):
            return v
    # Step 3: Accept float inputs only when exactly 0.0/1.0.
    if isinstance(label_value, (float, np.floating)):
        if float(label_value) in (0.0, 1.0):
            return int(label_value)
    # Step 4: Normalize textual forms.
    txt = str(label_value).strip().lower()
    if txt in {"bonafide", "genuine", "0", "0.0"}:
        return 0
    if txt in {"spoof", "1", "1.0"}:
        return 1
    # Step 5: Fail loudly on unsupported label values.
    raise ValueError(f"Unsupported label value: {label_value}")


def normalize_label_to_text(label_value):
    """
    Normalize common textual/numeric label representations to canonical text labels.

    Mapping:
    - bonafide/genuine/0 -> "bonafide"
    - spoof/1 -> "spoof"
    """
    # Step 1: Decode bytes-like inputs to text.
    if isinstance(label_value, (bytes, bytearray)):
        label_value = label_value.decode("utf-8", errors="ignore")
    # Step 2: Fast-path for integer inputs.
    if isinstance(label_value, (int, np.integer)):
        return "spoof" if int(label_value) == 1 else "bonafide"
    # Step 3: Fast-path for float inputs.
    if isinstance(label_value, (float, np.floating)):
        if float(label_value) == 1.0:
            return "spoof"
        if float(label_value) == 0.0:
            return "bonafide"
    # Step 4: Normalize textual forms.
    txt = str(label_value).strip().lower()
    if txt in {"bonafide", "genuine", "0", "0.0"}:
        return "bonafide"
    if txt in {"spoof", "1", "1.0"}:
        return "spoof"
    # Step 5: Preserve unknown values as string for visibility.
    return str(label_value)


def sanitize_text_field(value, default_value):
    """
    Convert metadata values to clean text and replace empty/missing-like values.
    """
    # Step 1: Decode bytes-like inputs.
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    # Step 2: Trim and normalize empty/NaN/None-like strings.
    txt = str(value).strip()
    if txt == "" or txt.lower() in {"nan", "none"}:
        return default_value
    # Step 3: Return sanitized text value.
    return txt


def _as_finite_float_or_none(value):
    """
    Convert scalar-like values to finite float when possible.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _append_text_log_line(log_path: Optional[str], message: str) -> None:
    """
    Append one UTF-8 text line to log_path when provided.
    """
    # Step 1: Skip when no log path is configured.
    if not log_path:
        return
    # Step 2: Ensure parent directory exists.
    log_dir = os.path.dirname(os.path.abspath(log_path))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    # Step 3: Append one line.
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(str(message).rstrip("\n") + "\n")


def _extract_conf_int_bounds(ci_payload):
    """
    Extract (lower, upper) CI bounds from common payload shapes.
    Returns (None, None) when bounds cannot be resolved.
    """
    # Step 1: Handle sequence payloads.
    if isinstance(ci_payload, (tuple, list, np.ndarray)):
        seq = list(ci_payload)
        if len(seq) == 2:
            lower = _as_finite_float_or_none(seq[0])
            upper = _as_finite_float_or_none(seq[1])
            if lower is not None and upper is not None:
                return lower, upper
        elif len(seq) > 2:
            # Treat long numeric sequences as bootstrap samples and derive CI quantiles.
            finite_vals = np.asarray(
                [v for v in (_as_finite_float_or_none(x) for x in seq) if v is not None],
                dtype=np.float64,
            )
            if finite_vals.size >= 2:
                alpha_frac = float(max(0.0, min(99.999, float(EER_ALPHA)))) / 100.0
                lo_q = alpha_frac / 2.0
                hi_q = 1.0 - lo_q
                lower = float(np.quantile(finite_vals, lo_q))
                upper = float(np.quantile(finite_vals, hi_q))
                return lower, upper

    # Step 2: Handle dictionary payloads with common key names.
    if isinstance(ci_payload, dict):
        key_pairs = (
            ("lower", "upper"),
            ("low", "high"),
            ("ci_lower", "ci_upper"),
            ("lower_bound", "upper_bound"),
            ("lower_ci", "upper_ci"),
            ("ci_low", "ci_high"),
            ("min", "max"),
        )
        for lower_key, upper_key in key_pairs:
            if lower_key in ci_payload and upper_key in ci_payload:
                lower = _as_finite_float_or_none(ci_payload[lower_key])
                upper = _as_finite_float_or_none(ci_payload[upper_key])
                if lower is not None and upper is not None:
                    return lower, upper
        for nested_key in (
            "ci",
            "confidence_interval",
            "conf_int",
            "interval",
            "bootstrap_values",
            "bootstrap_metrics",
            "samples",
            "values",
        ):
            if nested_key in ci_payload:
                nested_bounds = _extract_conf_int_bounds(ci_payload[nested_key])
                if nested_bounds != (None, None):
                    return nested_bounds

    # Step 3: Handle object payloads with common attribute names.
    attr_pairs = (
        ("lower", "upper"),
        ("low", "high"),
        ("ci_lower", "ci_upper"),
        ("lower_bound", "upper_bound"),
    )
    for lower_attr, upper_attr in attr_pairs:
        if hasattr(ci_payload, lower_attr) and hasattr(ci_payload, upper_attr):
            lower = _as_finite_float_or_none(getattr(ci_payload, lower_attr))
            upper = _as_finite_float_or_none(getattr(ci_payload, upper_attr))
            if lower is not None and upper is not None:
                return lower, upper

    # Step 4: CI bounds are unavailable/unknown shape.
    return None, None


def compute_eer(
    y_true,
    y_score,
    conditions=None,
    return_threshold=False,
    return_conf_int=False,
    fallback_log_path: Optional[str] = None,
    fallback_context: str = "",
):
    """
    Compute EER point-estimate on full data and threshold from full-set ROC crossing.
    When confidence intervals are requested, return the bootstrap EER center.
    For single-class bootstrap draws, fallback uses the full-set EER point-estimate.
    If a bootstrap draw has a single class, resample again (up to EER_MAX_RESAMPLE_RETRIES).
    Optionally return confidence-interval bounds when available.
    Optionally writes fallback notices to a text file when one-class fallback is used.
    """
    import warnings

    # Step 1: Convert to NumPy arrays.
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if conditions is not None:
        conditions = np.asarray(conditions)
    if len(y_true) != len(y_score):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} elements, y_score has {len(y_score)}")

    def _raw_eer(labels, scores, with_threshold=False):
        """
        Raw EER on one labels/scores array pair.
        """
        if len(labels) == 0 or len(scores) == 0:
            warnings.warn("Cannot compute EER on empty arrays, returning 1.0 (worst case)")
            return (1.0, 0.5) if with_threshold else 1.0
        if np.unique(labels).size < 2:
            warnings.warn("Cannot compute EER with only one unique label, returning 0.5")
            return (0.5, 0.5) if with_threshold else 0.5

        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        diff = np.abs(fpr - fnr)
        if diff.size == 0:
            warnings.warn("Could not compute EER (no threshold points), returning 0.5")
            return (0.5, 0.5) if with_threshold else 0.5

        eer_idx = int(np.argmin(diff))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        threshold = thresholds[eer_idx] if eer_idx < len(thresholds) else 0.5
        if not np.isfinite(threshold):
            threshold = 0.5
        return (eer, float(threshold)) if with_threshold else eer

    def _pack_result(eer_center, threshold, conf_bounds):
        # Keep backward-compatible return shapes while supporting optional CI output.
        if return_threshold and return_conf_int:
            return float(eer_center), float(threshold), conf_bounds
        if return_threshold:
            return float(eer_center), float(threshold)
        if return_conf_int:
            return float(eer_center), conf_bounds
        return float(eer_center)

    # Step 2: Compute EER point-estimate + threshold on the full dataset.
    eer_point, threshold = _raw_eer(y_true, y_score, with_threshold=True)

    # Fast path: no CI requested, so return the exact point estimate directly.
    if not return_conf_int:
        return _pack_result(eer_point, threshold, (None, None))

    use_bootstrap = bool(EER_USE_BOOTSTRAP and int(EER_NUM_BOOTSTRAPS) > 1)
    if not use_bootstrap:
        return _pack_result(eer_point, threshold, (None, None))

    if evaluate_with_conf_int is None or get_bootstrap_indices is None:
        raise ImportError(
            "Could not import confidence_intervals utilities. "
            "Set CONFIDENCE_INTERVALS_DIR to your local confidence_intervals folder."
        ) from _CONF_INT_IMPORT_ERROR

    full_has_two_classes = np.unique(y_true).size >= 2
    fallback_stats = {
        "full_single_class": 0,
        "retry_exhausted": 0,
    }

    # Step 3: Compute CI payload using confidence_intervals API with retry-on-single-class.
    retry_seed_state = {"value": 0}

    def _eer_metric(labels, scores):
        # Keep CI bootstrap aligned to the already-computed full-set center.
        if not full_has_two_classes:
            fallback_stats["full_single_class"] += 1
            return float(eer_point)

        # Main path: if the provided bootstrap has both classes, evaluate directly.
        if np.unique(labels).size >= 2:
            return _raw_eer(labels, scores, with_threshold=False)

        # Retry path: sample new bootstrap indices until both classes appear.
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
            return _raw_eer(retry_labels, retry_scores, with_threshold=False)

        # Last fallback: if retries are exhausted on one-class data, keep center unchanged.
        fallback_stats["retry_exhausted"] += 1
        return float(eer_point)

    eer_center, ci_payload = evaluate_with_conf_int(
        y_score,
        _eer_metric,
        y_true,
        conditions,
        num_bootstraps=max(1, int(EER_NUM_BOOTSTRAPS)),
        alpha=float(EER_ALPHA),
    )
    # Log one-class fallback events to an optional text file.
    if fallback_stats["full_single_class"] > 0 or fallback_stats["retry_exhausted"] > 0:
        context_prefix = f"[{fallback_context}] " if str(fallback_context).strip() else ""
        _append_text_log_line(
            fallback_log_path,
            (
                f"{context_prefix}"
                "לא הצליח לחשב על מחלקה אחת ולכן מוחזר ה-center_eer "
                f"(center_eer={float(eer_point):.10f}, "
                f"full_single_class={int(fallback_stats['full_single_class'])}, "
                f"retry_exhausted={int(fallback_stats['retry_exhausted'])})"
            ),
        )
    eer_from_ci = _as_finite_float_or_none(eer_center)
    if eer_from_ci is None:
        context_prefix = f"[{fallback_context}] " if str(fallback_context).strip() else ""
        _append_text_log_line(
            fallback_log_path,
            (
                f"{context_prefix}"
                "חישוב ה-CI החזיר center לא תקין ולכן מוחזר ה-center_eer "
                f"(center_eer={float(eer_point):.10f})"
            ),
        )
        eer_from_ci = float(eer_point)
    conf_bounds = _extract_conf_int_bounds(ci_payload)

    # Step 4: Return EER (+threshold/+CI when requested).
    return _pack_result(eer_from_ci, threshold, conf_bounds)


def _infer_labels_path(embeddings_path):
    """
    Infer the paired labels file path for a .npy embeddings file.
    """
    # Step 1: Handle *_embeddings.npy naming convention.
    if embeddings_path.endswith("_embeddings.npy"):
        return embeddings_path.replace("_embeddings.npy", "_labels.npy")
    # Step 2: Handle generic .npy naming by appending _labels suffix.
    if embeddings_path.endswith(".npy"):
        return embeddings_path[:-4] + "_labels.npy"
    # Step 3: Return None when no known pattern matches.
    return None


class EmbeddingDataset(Dataset):
    """
    Dataset wrapper for loading embedding files plus protocol metadata fields.
    Supports:
    - .npz bundles with embeddings/labels/(optional protocol fields)
    - .npy embeddings with paired *_labels.npy file
    """
    _FILE_ID_KEYS = ("file_id", "file_ids", "files")
    _SPEAKER_ID_KEYS = ("speaker_id", "speaker_ids", "speakers")
    _ATTACK_KEYS = ("attack", "attacks")

    def __init__(self, npz_path):
        """
        Load embeddings/labels from .npz or .npy(+labels) and prepare protocol fields.
        """
        # Step 1: Initialize runtime flags and optional metadata containers.
        self._use_memmap = False
        self.file_ids = None
        self.speaker_ids = None
        self.attacks = None
        self.label_texts = None

        # Step 2: Load NPZ input (full arrays).
        if npz_path.endswith(".npz"):
            with np.load(npz_path, allow_pickle=True) as d:
                # Step 2.1: Load numeric embeddings and labels.
                raw_embeddings = np.asarray(d["embeddings"], dtype=np.float32)
                raw_labels = np.asarray(d["labels"])
                self.embeddings = torch.tensor(raw_embeddings, dtype=torch.float32)
                self.labels = torch.tensor(
                    [normalize_label_to_int(v) for v in raw_labels],
                    dtype=torch.long,
                )
                # Step 2.2: Keep canonical text labels.
                n = len(raw_labels)
                self.label_texts = np.array(
                    [normalize_label_to_text(v) for v in raw_labels],
                    dtype=object,
                )
                # Step 2.3: Read optional protocol arrays if present.
                file_ids = self._extract_npz_optional_array(d, self._FILE_ID_KEYS, n)
                speaker_ids = self._extract_npz_optional_array(d, self._SPEAKER_ID_KEYS, n)
                attacks = self._extract_npz_optional_array(d, self._ATTACK_KEYS, n)
                # Step 2.4: Sanitize optional arrays for stable export later.
                if file_ids is not None:
                    self.file_ids = np.array(
                        [sanitize_text_field(v, f"sample_{i:07d}") for i, v in enumerate(file_ids)],
                        dtype=object,
                    )
                if speaker_ids is not None:
                    self.speaker_ids = np.array(
                        [sanitize_text_field(v, "unknown") for v in speaker_ids],
                        dtype=object,
                    )
                if attacks is not None:
                    self.attacks = np.array(
                        [sanitize_text_field(v, "-") for v in attacks],
                        dtype=object,
                    )
        # Step 3: Load NPY input via memory-mapping and infer paired labels file.
        elif npz_path.endswith(".npy"):
            labels_path = _infer_labels_path(npz_path)
            if not labels_path or not os.path.exists(labels_path):
                raise FileNotFoundError(f"Labels file not found for embeddings: {labels_path}")
            # Step 3.1: Memmap embeddings for lower RAM usage.
            emb = np.load(npz_path, mmap_mode="r")
            if emb.dtype != np.float32:
                emb = emb.astype(np.float32, copy=False)
            self.embeddings = emb
            # Step 3.2: Memmap labels and normalize dtype.
            labels = np.load(labels_path, mmap_mode="r")
            if labels.dtype != np.int64:
                labels = labels.astype(np.int64, copy=False)
            self.labels = labels
            # Step 3.3: Build text labels for export.
            self.label_texts = np.array(
                [normalize_label_to_text(v) for v in self.labels],
                dtype=object,
            )
            self._use_memmap = True
        # Step 4: Reject unsupported file types.
        else:
            raise ValueError(f"Unsupported embeddings file type: {npz_path}")

    @staticmethod
    def _extract_npz_optional_array(npz_obj, key_candidates, expected_len):
        """
        Return the first matching optional NPZ array from key_candidates.
        """
        # Step 1: Try each candidate key in order.
        for key in key_candidates:
            if key in npz_obj.files:
                # Step 2: Flatten and validate length alignment.
                arr = np.asarray(npz_obj[key]).reshape(-1)
                if len(arr) != expected_len:
                    raise ValueError(
                        f"Field '{key}' length mismatch in npz: expected {expected_len}, got {len(arr)}"
                    )
                # Step 3: Return the first valid match.
                return arr
        # Step 4: Return None when no candidate keys exist.
        return None

    def _ensure_protocol_fields(self):
        """
        Ensure file_id/speaker_id/attack/label arrays exist for every sample.
        """
        # Step 1: Compute dataset length once.
        n = len(self.labels)
        # Step 2: Fill missing metadata arrays with stable defaults.
        if self.file_ids is None:
            self.file_ids = np.array([f"sample_{i:07d}" for i in range(n)], dtype=object)
        if self.speaker_ids is None:
            self.speaker_ids = np.array(["unknown"] * n, dtype=object)
        if self.attacks is None:
            self.attacks = np.array(["-"] * n, dtype=object)
        # Step 3: Create text labels when missing.
        if self.label_texts is None:
            if isinstance(self.labels, torch.Tensor):
                labels_np = self.labels.cpu().numpy()
            else:
                labels_np = np.asarray(self.labels)
            self.label_texts = np.array(
                [normalize_label_to_text(v) for v in labels_np],
                dtype=object,
            )

    def get_protocol_fields(self):
        """
        Return protocol-style metadata aligned with dataset sample order.
        """
        # Step 1: Guarantee all fields exist.
        self._ensure_protocol_fields()
        # Step 2: Return a dictionary used by text-score export.
        return {
            "file_id": self.file_ids,
            "speaker_id": self.speaker_ids,
            "attack": self.attacks,
            "label": self.label_texts,
        }

    def __len__(self):
        """
        Return number of samples in this dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Fetch one sample as (embedding_tensor, label_tensor/int).
        """
        # Step 1: If memmap mode is active, convert numpy rows on demand.
        if self._use_memmap:
            x = torch.from_numpy(self.embeddings[idx])
            y = torch.as_tensor(self.labels[idx], dtype=torch.long)
            return x, y
        # Step 2: Otherwise return preloaded tensors.
        return self.embeddings[idx], self.labels[idx]


class LinearProjector(nn.Module):
    """
    Lightweight linear head used for binary spoof scoring inference.
    """
    def __init__(self, input_dim, output_dim):
        """
        Minimal linear classifier used for checkpoint reconstruction and inference.
        """
        super().__init__()
        # Step 1: Build one linear layer that maps embedding -> 2-class logits.
        self.projector = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass: logits = linear(embeddings).
        """
        # Step 1: Apply the linear projection.
        return self.projector(x)


@dataclass
class CandidateInfo:
    """
    Resolved checkpoint candidate plus saved eval-EER metadata source.
    """
    run_dir: str
    checkpoint_path: str
    eval_eer_percent: Optional[float]
    eval_eer_source: str


def _select_names(available_names: Sequence[str], selected_names: Sequence[str], field_name: str) -> List[str]:
    """
    Validate a user selection list against available names.
    Supports special value ['all'].
    """
    # Step 1: Guard against empty selection.
    if not selected_names:
        raise ValueError(f"{field_name} cannot be empty.")
    # Step 2: Expand 'all' to all available names.
    if len(selected_names) == 1 and str(selected_names[0]).lower() == "all":
        return list(available_names)
    # Step 3: Validate explicit selections.
    selected = [str(name) for name in selected_names]
    missing = [name for name in selected if name not in available_names]
    if missing:
        raise ValueError(f"Unknown values in {field_name}: {missing}")
    # Step 4: Return validated list.
    return selected


def _ensure_file_exists(path: str, kind: str) -> None:
    """
    Raise a clear error if a required file does not exist.
    """
    # Step 1: Validate path points to a file.
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{kind} not found: {path}")


def _read_scalar_float(np_obj, key: str) -> Optional[float]:
    """
    Read a scalar float from an NPZ field. Return None when unavailable/invalid.
    """
    # Step 1: Ensure key exists in NPZ bundle.
    if key not in np_obj.files:
        return None
    # Step 2: Flatten to handle scalar/array forms uniformly.
    arr = np.asarray(np_obj[key]).reshape(-1)
    if arr.size == 0:
        return None
    # Step 3: Convert first value to float and validate finiteness.
    value = float(arr[0])
    if not np.isfinite(value):
        return None
    # Step 4: Return parsed scalar value.
    return value


def _read_best_checkpoint_from_best_npz(run_dir: str) -> Optional[str]:
    """
    Read preferred checkpoint path from run_dir/best_eval_scores.npz, if available.
    """
    # Step 1: Locate the expected NPZ file.
    npz_path = os.path.join(run_dir, "best_eval_scores.npz")
    if not os.path.isfile(npz_path):
        return None

    # Step 2: Parse best_checkpoint field.
    with np.load(npz_path, allow_pickle=True) as data:
        if "best_checkpoint" not in data.files:
            return None
        value_arr = np.asarray(data["best_checkpoint"]).reshape(-1)
        if value_arr.size == 0:
            return None
        value = value_arr[0]

    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    checkpoint_hint = str(value).strip()
    if not checkpoint_hint:
        return None

    # Step 3: Resolve absolute/relative checkpoint hint and return if it exists.
    abs_hint = os.path.abspath(os.path.expanduser(checkpoint_hint))
    if os.path.isfile(abs_hint):
        return abs_hint
    run_relative_hint = os.path.abspath(os.path.join(run_dir, checkpoint_hint))
    if os.path.isfile(run_relative_hint):
        return run_relative_hint
    return None


def _read_eval_eer_from_best_npz(run_dir: str) -> Optional[tuple]:
    """
    Read eval EER metadata from run_dir/best_eval_scores.npz.
    Returns (eer_percent, source_tag) or None.
    """
    # Step 1: Locate the expected NPZ file.
    npz_path = os.path.join(run_dir, "best_eval_scores.npz")
    if not os.path.isfile(npz_path):
        return None
    # Step 2: Prefer eer_percent; fallback to eer ratio converted to percent.
    with np.load(npz_path, allow_pickle=True) as data:
        eer_percent = _read_scalar_float(data, "eer_percent")
        if eer_percent is not None:
            return float(eer_percent), f"{npz_path}:eer_percent"
        eer_raw = _read_scalar_float(data, "eer")
        if eer_raw is not None:
            eer_as_percent = float(eer_raw * 100.0) if eer_raw <= 1.0 else float(eer_raw)
            return eer_as_percent, f"{npz_path}:eer"
    # Step 3: Return None when both fields are missing/invalid.
    return None


def _read_eval_eer_from_log(run_dir: str) -> Optional[tuple]:
    """
    Read eval EER from run_dir/eer_eval/eer_eval_log.txt.
    Priority: explicit 'min_eval' row, otherwise min over epoch rows.
    """
    # Step 1: Locate log file.
    log_path = os.path.join(run_dir, "eer_eval", "eer_eval_log.txt")
    if not os.path.isfile(log_path):
        return None

    # Step 2: Parse rows and collect epoch EER values.
    epoch_eers = []
    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            tag = parts[0].strip().lower()
            try:
                eer_val = float(parts[1])
            except ValueError:
                continue
            if not np.isfinite(eer_val):
                continue
            if tag == "min_eval":
                return float(eer_val), f"{log_path}:min_eval"
            if parts[0].isdigit():
                epoch_eers.append(float(eer_val))

    # Step 3: Fallback to minimum epoch EER if min_eval row is absent.
    if epoch_eers:
        return float(min(epoch_eers)), f"{log_path}:epoch_min"
    # Step 4: Return None if no usable values were found.
    return None


def _read_saved_eval_eer_percent(run_dir: str) -> Optional[tuple]:
    """
    Resolve saved eval EER from NPZ first, then from text log.
    """
    # Step 1: Try NPZ source.
    from_npz = _read_eval_eer_from_best_npz(run_dir)
    if from_npz is not None:
        return from_npz
    # Step 2: Fallback to log source.
    from_log = _read_eval_eer_from_log(run_dir)
    if from_log is not None:
        return from_log
    # Step 3: Return None when no saved EER source exists.
    return None


def _pick_latest_file(paths: Sequence[str]) -> Optional[str]:
    """
    Pick newest file by modification time from a list.
    """
    # Step 1: Guard empty input.
    if not paths:
        return None
    # Step 2: Return path with max mtime.
    return max(paths, key=lambda p: os.path.getmtime(p))


def _resolve_checkpoint_path_from_dir(run_dir: str) -> str:
    """
    Resolve a checkpoint file inside a run directory using prioritized rules.
    Priority:
    best_eval_scores.npz:best_checkpoint -> best_eval_eer.ckpt ->
    best.ckpt -> best-*.ckpt(latest) -> last.ckpt -> *.ckpt(latest).
    """
    # Step 1: Try best checkpoint hint from best_eval_scores.npz.
    best_eval_ckpt_from_npz = _read_best_checkpoint_from_best_npz(run_dir)
    if best_eval_ckpt_from_npz is not None:
        return best_eval_ckpt_from_npz

    # Step 2: Try explicit best_eval_eer.ckpt.
    best_eval_ckpt = os.path.join(run_dir, "best_eval_eer.ckpt")
    if os.path.isfile(best_eval_ckpt):
        return best_eval_ckpt

    # Step 3: Try explicit best.ckpt.
    best_ckpt = os.path.join(run_dir, "best.ckpt")
    if os.path.isfile(best_ckpt):
        return best_ckpt

    # Step 4: Try best-*.ckpt naming pattern.
    best_pattern = _pick_latest_file(glob.glob(os.path.join(run_dir, "best-*.ckpt")))
    if best_pattern is not None:
        return best_pattern

    # Step 5: Fallback to last.ckpt.
    last_ckpt = os.path.join(run_dir, "last.ckpt")
    if os.path.isfile(last_ckpt):
        return last_ckpt

    # Step 6: Final fallback to any checkpoint file.
    any_ckpt = _pick_latest_file(glob.glob(os.path.join(run_dir, "*.ckpt")))
    if any_ckpt is not None:
        return any_ckpt

    # Step 7: Fail if no checkpoint exists.
    raise FileNotFoundError(f"No .ckpt files found under run directory: {run_dir}")


def _resolve_candidate(candidate: str) -> CandidateInfo:
    """
    Resolve an input candidate (file/dir) into a concrete checkpoint and EER metadata.
    """
    # Step 1: Normalize candidate path.
    abs_candidate = os.path.abspath(os.path.expanduser(candidate))
    # Step 2: Resolve checkpoint from file or directory.
    explicit_ckpt_selected = False
    if os.path.isfile(abs_candidate):
        if not abs_candidate.lower().endswith(".ckpt"):
            raise ValueError(f"Candidate file must be a .ckpt file: {abs_candidate}")
        run_dir = os.path.dirname(abs_candidate)
        ckpt_path = abs_candidate
        explicit_ckpt_selected = True
    elif os.path.isdir(abs_candidate):
        run_dir = abs_candidate
        ckpt_path = _resolve_checkpoint_path_from_dir(run_dir)
    else:
        raise FileNotFoundError(f"Candidate path does not exist: {abs_candidate}")

    # Step 2.1: Warn when an explicit checkpoint is used but best_eval_eer.ckpt exists.
    if explicit_ckpt_selected:
        best_eval_ckpt = os.path.join(run_dir, "best_eval_eer.ckpt")
        if os.path.isfile(best_eval_ckpt):
            selected_abs = os.path.abspath(ckpt_path)
            best_abs = os.path.abspath(best_eval_ckpt)
            if selected_abs != best_abs:
                print(
                    "Warning: using explicit checkpoint that is not best_eval_eer.ckpt. "
                    f"selected={selected_abs}, best_eval={best_abs}"
                )

    # Step 3: Load saved eval EER metadata if available.
    eer_info = _read_saved_eval_eer_percent(run_dir)
    if eer_info is None:
        eer_percent = None
        eer_source = "missing(best_eval_scores.npz/eer_eval_log.txt)"
    else:
        eer_percent, eer_source = eer_info

    # Step 4: Build and return structured candidate info.
    return CandidateInfo(
        run_dir=run_dir,
        checkpoint_path=ckpt_path,
        eval_eer_percent=eer_percent,
        eval_eer_source=eer_source,
    )


def _find_state_key(state_dict, suffix):
    """
    Find an exact or unique suffix-matching key in a checkpoint state dict.
    """
    # Step 1: Prefer exact match.
    if suffix in state_dict:
        return suffix
    # Step 2: Fallback to suffix matching for Lightning-style key prefixes.
    matches = [k for k in state_dict.keys() if str(k).endswith(suffix)]
    if not matches:
        raise KeyError(f"Could not find key ending with '{suffix}' in checkpoint state_dict.")
    if len(matches) > 1:
        raise KeyError(f"Multiple keys end with '{suffix}', cannot disambiguate: {matches}")
    # Step 3: Return unique matched key.
    return matches[0]


def _load_model(ckpt_path: str, device: torch.device):
    """
    Load LinearProjector weights from checkpoint and return eval-ready model on device.
    """
    # Step 1: Load checkpoint structure.
    checkpoint = torch.load(ckpt_path, map_location=device)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format in: {ckpt_path}")

    # Step 2: Resolve projector parameter keys.
    weight_key = _find_state_key(state_dict, "projector.weight")
    bias_key = _find_state_key(state_dict, "projector.bias")
    weight = state_dict[weight_key]
    bias = state_dict[bias_key]
    if weight.ndim != 2:
        raise ValueError(f"Unexpected projector.weight shape: {tuple(weight.shape)}")

    # Step 3: Reconstruct model with inferred dimensions.
    output_dim, input_dim = int(weight.shape[0]), int(weight.shape[1])
    model = LinearProjector(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(
        {
            "projector.weight": weight,
            "projector.bias": bias,
        }
    )
    # Step 4: Move model to device and switch to eval mode.
    model.to(device)
    model.eval()
    return model


def _compute_scores(
    model,
    device: torch.device,
    dataset: EmbeddingDataset,
    batch_size: int,
    num_workers: int,
    progress_desc: str = "",
) -> np.ndarray:
    """
    Run batched inference and return spoof-class scores aligned to dataset order.
    """
    # Step 1: Build deterministic (shuffle=False) inference loader.
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = DATALOADER_PREFETCH_FACTOR
    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )
    # Step 2: Iterate batches, compute logits, convert to spoof probabilities.
    chunks = []
    batch_desc = (str(progress_desc).strip() or "inference") + " [batches]"
    with torch.inference_mode():
        for x, _ in _progress(
            loader,
            total=len(loader),
            desc=batch_desc,
            unit="batch",
            leave=False,
        ):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            scores = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            chunks.append(scores.astype(np.float32, copy=False))
            del x, logits, scores

    # Step 3: Return concatenated scores (or empty array for empty dataset).
    if not chunks:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _compute_raw_eer_point(labels, scores) -> float:
    """
    Lightweight point-estimate EER (no bootstrap), used for score-polarity diagnostics.
    """
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if len(labels) == 0 or len(scores) == 0:
        return 1.0
    if len(labels) != len(scores):
        raise ValueError(
            f"Length mismatch in raw EER point estimate: labels={len(labels)}, scores={len(scores)}"
        )
    if np.unique(labels).size < 2:
        return 0.5

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    if fpr.size == 0:
        return 0.5
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def _maybe_flip_score_polarity(labels, scores, split_desc: str = "") -> np.ndarray:
    """
    Detect inverted score polarity and optionally flip with (1 - score).
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not AUTO_FLIP_SCORE_POLARITY or scores.size == 0:
        return scores

    direct_eer = _compute_raw_eer_point(labels, scores)
    inverse_eer = _compute_raw_eer_point(labels, -scores)
    if np.isfinite(direct_eer) and np.isfinite(inverse_eer):
        if inverse_eer + float(SCORE_POLARITY_EER_MARGIN) < direct_eer:
            prefix = f"[{split_desc}] " if split_desc else ""
            print(
                f"{prefix}Detected inverted score polarity "
                f"(direct_eer={direct_eer * 100.0:.4f}%, inverted_eer={inverse_eer * 100.0:.4f}%). "
                "Flipping scores via (1 - score)."
            )
            return np.clip(1.0 - scores, 0.0, 1.0).astype(np.float32, copy=False)
    return scores


def _export_scores_for_split(
    model,
    device: torch.device,
    npz_path: str,
    output_path: str,
    batch_size: int,
    num_workers: int,
    progress_desc: str = "",
) -> tuple[int, str]:
    """
    Score one split NPZ and export rows to text:
    file_id, speaker_id, attack, label, scores
    Also export EER + confidence intervals to a separate text file.
    """
    # Step 1: Load dataset and run model inference.
    _ensure_file_exists(npz_path, "Embeddings npz")
    dataset = EmbeddingDataset(npz_path)
    split_desc = str(progress_desc).strip() or os.path.basename(npz_path)
    scores = _compute_scores(
        model,
        device,
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        progress_desc=split_desc,
    )
    fields = dataset.get_protocol_fields()

    # Step 2: Convert protocol arrays to flat aligned vectors.
    file_ids = np.asarray(fields["file_id"], dtype=object).reshape(-1)
    speaker_ids = np.asarray(fields["speaker_id"], dtype=object).reshape(-1)
    attacks = np.asarray(fields["attack"], dtype=object).reshape(-1)
    labels = np.asarray(fields["label"], dtype=object).reshape(-1)

    # Step 3: Validate alignment between metadata and computed scores.
    n = len(scores)
    if len(file_ids) != n or len(speaker_ids) != n or len(attacks) != n or len(labels) != n:
        raise ValueError(
            "Mismatch between scores length and protocol fields lengths: "
            f"scores={n}, file_ids={len(file_ids)}, speaker_ids={len(speaker_ids)}, "
            f"attacks={len(attacks)}, labels={len(labels)}"
        )

    # Step 4: Compute split EER from label/score arrays for three condition modes.
    label_ids = np.asarray([normalize_label_to_int(v) for v in labels], dtype=np.int64)
    scores = _maybe_flip_score_polarity(label_ids, scores, split_desc=split_desc)
    speaker_conditions = np.asarray(speaker_ids, dtype=object)
    attack_conditions = np.asarray(attacks, dtype=object)

    # Step 4.1: Resolve one-class fallback log path (used during EER computation loop).
    output_file_name = os.path.basename(output_path)
    if output_file_name.startswith("scores_"):
        fallback_file_name = f"eer_single_class_fallback_{output_file_name[len('scores_'):]}"
    else:
        root, ext = os.path.splitext(output_file_name)
        if not ext:
            ext = ".txt"
        fallback_file_name = f"{root}_eer_single_class_fallback{ext}"
    fallback_log_path = os.path.join(os.path.dirname(output_path), fallback_file_name)
    try:
        fallback_log_size_before = (
            os.path.getsize(fallback_log_path) if os.path.isfile(fallback_log_path) else 0
        )
    except OSError:
        fallback_log_size_before = 0

    eer_values = {}
    eer_conf_int_bounds = {}
    eer_modes = (
        ("speaker", speaker_conditions),
        ("attack", attack_conditions),
        ("none", None),
    )
    for mode_name, mode_conditions in _progress(
        eer_modes,
        total=len(eer_modes),
        desc=f"{split_desc} [eer]",
        unit="mode",
        leave=False,
    ):
        eer_val, _, eer_ci_bounds = compute_eer(
            label_ids,
            scores,
            conditions=mode_conditions,
            return_threshold=True,
            return_conf_int=True,
            fallback_log_path=fallback_log_path,
            fallback_context=f"{split_desc}/{mode_name}",
        )
        eer_values[mode_name] = float(eer_val)
        eer_conf_int_bounds[mode_name] = eer_ci_bounds

    eer_speaker = eer_values["speaker"]
    eer_attack = eer_values["attack"]
    eer_none = eer_values["none"]
    eer_ci_speaker = eer_conf_int_bounds["speaker"]
    eer_ci_attack = eer_conf_int_bounds["attack"]
    eer_ci_none = eer_conf_int_bounds["none"]

    def _ratio_to_percent(value):
        ratio = _as_finite_float_or_none(value)
        if ratio is None:
            return np.nan
        return ratio * 100.0

    def _format_metric(value):
        value = _as_finite_float_or_none(value)
        if value is None:
            return "nan"
        return f"{value:.6f}"

    eer_percent_speaker = eer_speaker * 100.0
    eer_percent_attack = eer_attack * 100.0
    eer_percent_none = eer_none * 100.0
    eer_percent_speaker_ci_lower = _ratio_to_percent(eer_ci_speaker[0])
    eer_percent_speaker_ci_upper = _ratio_to_percent(eer_ci_speaker[1])
    eer_percent_attack_ci_lower = _ratio_to_percent(eer_ci_attack[0])
    eer_percent_attack_ci_upper = _ratio_to_percent(eer_ci_attack[1])
    eer_percent_none_ci_lower = _ratio_to_percent(eer_ci_none[0])
    eer_percent_none_ci_upper = _ratio_to_percent(eer_ci_none[1])

    # Step 5: Resolve separate EER+CI output path in the same folder.
    if output_file_name.startswith("scores_"):
        eer_file_name = f"eer_conf_intervals_{output_file_name[len('scores_'):]}"
    else:
        root, ext = os.path.splitext(output_file_name)
        if not ext:
            ext = ".txt"
        eer_file_name = f"{root}_eer_conf_intervals{ext}"
    eer_output_path = os.path.join(os.path.dirname(output_path), eer_file_name)

    # Step 6: Write CSV text file with required score columns/order.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_id", "speaker_id", "attack", "label", "scores"])
        chunk_size = max(1, int(WRITE_CHUNK_SIZE))
        chunk_starts = range(0, n, chunk_size)
        total_chunks = (n + chunk_size - 1) // chunk_size if n else 0
        for start_idx in _progress(
            chunk_starts,
            total=total_chunks,
            desc=f"{split_desc} [write]",
            unit="chunk",
            leave=False,
        ):
            stop_idx = min(start_idx + chunk_size, n)
            writer.writerows(
                [
                    [
                        sanitize_text_field(file_ids[idx], f"sample_{idx:07d}"),
                        sanitize_text_field(speaker_ids[idx], "unknown"),
                        sanitize_text_field(attacks[idx], "-"),
                        normalize_label_to_text(labels[idx]),
                        f"{float(scores[idx]):.10f}",
                    ]
                    for idx in range(start_idx, stop_idx)
                ]
            )

    # Step 7: Write EER + confidence-interval summary to a separate file.
    with open(eer_output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "eer_percent", "ci_lower_percent", "ci_upper_percent"])
        writer.writerow(
            [
                "speaker",
                _format_metric(eer_percent_speaker),
                _format_metric(eer_percent_speaker_ci_lower),
                _format_metric(eer_percent_speaker_ci_upper),
            ]
        )
        writer.writerow(
            [
                "attack",
                _format_metric(eer_percent_attack),
                _format_metric(eer_percent_attack_ci_lower),
                _format_metric(eer_percent_attack_ci_upper),
            ]
        )
        writer.writerow(
            [
                "none",
                _format_metric(eer_percent_none),
                _format_metric(eer_percent_none_ci_lower),
                _format_metric(eer_percent_none_ci_upper),
            ]
        )

    # Step 8: Emit log path info when one-class fallback notices were written.
    if os.path.isfile(fallback_log_path):
        try:
            if os.path.getsize(fallback_log_path) > fallback_log_size_before:
                print(f"[{split_desc}] wrote one-class fallback notices to: {fallback_log_path}")
        except OSError:
            pass

    # Step 9: Release memory and return number of written rows + EER summary path.
    del dataset, scores
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return n, eer_output_path


def _expand_checkpoint_patterns(jobs: Sequence[parameter_run]) -> List[parameter_run]:
    """
    Expand wildcard checkpoint patterns into concrete parameter_run objects.
    """
    # Step 1: Initialize output list.
    expanded = []
    # Step 2: Expand each job checkpoint if it contains wildcard syntax.
    for job in jobs:
        ckpt_value = str(job.checkpoint)
        pattern = os.path.expanduser(ckpt_value)
        if glob.has_magic(pattern):
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"Checkpoint glob matched nothing: {ckpt_value}")
            for match in matches:
                expanded.append(
                    parameter_run(
                        db_name=job.db_name,
                        db_dir=job.db_dir,
                        checkpoint=match,
                        output_dir=job.output_dir,
                        train_file=job.train_file,
                        dev_file=job.dev_file,
                        eval_file=job.eval_file,
                        name=job.name,
                    )
                )
        else:
            expanded.append(job)
    # Step 3: Return expanded jobs.
    return expanded


def _validate_job(job: parameter_run, idx: int) -> None:
    """
    Validate required fields for one parameter_run object.
    """
    # Step 1: Collect missing required fields.
    missing = []
    if not str(job.db_name).strip():
        missing.append("db_name")
    if not str(job.db_dir).strip():
        missing.append("db_dir")
    if not str(job.checkpoint).strip():
        missing.append("checkpoint")
    if not str(job.output_dir).strip():
        missing.append("output_dir")
    # Step 2: Raise if validation failed.
    if missing:
        raise ValueError(f"parameter_run job #{idx} is missing required fields: {missing}")


def run_score_export_pipeline(jobs: Optional[Sequence[parameter_run]] = None):
    """
    Execute scoring pipeline for a list of parameter_run jobs.

    Flow:
    1) Take input list (`jobs`) or default `PARAMETER_RUN_CONFIGS`.
    2) Expand checkpoint wildcards to concrete jobs.
    3) For each job:
       - load checkpoint
       - score requested splits (SPLITS_TO_SCORE)
       - write output text files
    """
    # Step 1: Validate/resolve active splits configuration.
    split_space = ("train", "dev", "eval")
    active_splits = _select_names(split_space, SPLITS_TO_SCORE, "SPLITS_TO_SCORE")

    # Step 2: Use explicit jobs input, or fallback to global job list.
    if jobs is None:
        jobs = PARAMETER_RUN_CONFIGS
    if not jobs:
        raise ValueError(
            "PARAMETER_RUN_CONFIGS is empty. Add at least one parameter_run(...) config before running."
        )

    # Step 3: Expand wildcard checkpoints and prepare runtime device.
    expanded_jobs = _expand_checkpoint_patterns(jobs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Scoring device: {device}")
    print(f"Total parameter_run jobs: {len(expanded_jobs)}")

    # Step 4: Execute each job independently.
    for idx, job in _progress(
        enumerate(expanded_jobs, start=1),
        total=len(expanded_jobs),
        desc="parameter_run jobs",
        unit="job",
    ):
        _validate_job(job, idx)
        job_name = str(job.name).strip() or f"{job.db_name}_run_{idx:03d}"
        candidate = _resolve_candidate(job.checkpoint)
        ckpt_path = candidate.checkpoint_path

        if candidate.eval_eer_percent is None:
            print(
                f"[{job_name}] checkpoint: {ckpt_path} "
                f"(eval_eer=unavailable, source={candidate.eval_eer_source})"
            )
        else:
            print(
                f"[{job_name}] checkpoint: {ckpt_path} "
                f"(eval_eer={candidate.eval_eer_percent:.6f}%, source={candidate.eval_eer_source})"
            )

        # Step 5: Load model once per job and export all requested splits.
        output_dir = os.path.abspath(os.path.expanduser(job.output_dir))
        os.makedirs(output_dir, exist_ok=True)
        model = _load_model(ckpt_path, device)
        try:
            # One parameter_run job can export multiple split files (train/dev/eval)
            for split_name in _progress(
                active_splits,
                total=len(active_splits),
                desc=f"{job_name} [splits]",
                unit="split",
                leave=False,
            ):
                npz_path = job.split_path(split_name)
                out_path = os.path.join(
                    output_dir,
                    f"scores_{split_name}.txt",
                )
                n_rows, eer_out_path = _export_scores_for_split(
                    model=model,
                    device=device,
                    npz_path=npz_path,
                    output_path=out_path,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    progress_desc=f"{job_name}/{split_name}",
                )
                print(f"[{job_name}/{split_name}] wrote {n_rows} rows to: {out_path}")
                print(f"[{job_name}/{split_name}] wrote EER+CI to: {eer_out_path}")
        finally:
            # Step 6: Cleanup model/resources before moving to next job.
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    run_score_export_pipeline()
