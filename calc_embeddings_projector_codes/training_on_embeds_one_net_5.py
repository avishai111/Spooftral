"""
Train a lightweight classifier on precomputed ASVspoof embeddings.

The script builds data loaders from NPZ/NPY embeddings, trains a PyTorch
Lightning model, tracks EER on validation/eval splits, and saves checkpoints
and diagnostic plots.
"""
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use('Agg')  # Backend ×œ×œ× GUI - ×—×©×•×‘ ×œ×”×¨×¦×” ×¢×œ cluster
import matplotlib.pyplot as plt
import os
import random
import time
import gc
import shutil
from pytorch_lightning.callbacks import ModelCheckpoint , Callback, TQDMProgressBar
try:
    import psutil
except ImportError:
    psutil = None

# ×§×™×‘×•×¢ ×–×¨×¢×™× ×œ×©×—×–×•×¨ ×ª×•×¦××•×ª
def set_random_seeds(seed=42):
    """
    Set all relevant RNG seeds to make runs reproducible.
    Steps:
    1) Seed Python's random module.
    2) Seed NumPy.
    3) Seed PyTorch CPU.
    4) Seed all CUDA devices (if available).
    """
    # Step 1: Python RNG
    random.seed(seed)
    # Step 2: NumPy RNG
    np.random.seed(seed)
    # Step 3: PyTorch CPU RNG
    torch.manual_seed(seed)
    # Step 4: PyTorch CUDA RNG(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_random_seeds(seed=42)

def clear_cuda_cache():
    """
    Clear CUDA cache and synchronize device to reduce memory fragmentation.
    Steps:
    1) Check that a CUDA device exists.
    2) Release cached GPU memory.
    3) Synchronize to complete queued kernels.
    """
    # Step 1: Guard for GPU availability
    if torch.cuda.is_available():
        # Step 2: Release cached blocks
        torch.cuda.empty_cache()
        # Step 3: Ensure all GPU work is finished
        torch.cuda.synchronize()

# === Callbacks ===
class EvalEERCallback(Callback):
    # ×—×™×©×•×‘ EER ×¢×œ ×¡×˜ eval ×‘×¡×•×£ ×›×œ epoch ××™×ž×•×Ÿ
    def __init__(self, eval_set, batch_size, num_workers=0, save_npz_every=0, save_npz_dir=None):
        """
        __init__: Describe the purpose of this function.
        Steps:
        1) Validate inputs or state.
        2) Perform the core logic.
        3) Return results or update state.
        """
        super().__init__()
        self.eval_set = eval_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_npz_every = int(save_npz_every) if save_npz_every else 0
        self.save_npz_dir = save_npz_dir
        self.best_eval_eer = None
        self.best_eval_threshold = None
        self.best_eval_scores = None
        self.best_eval_labels = None
        self.best_eval_epoch = None
        self.last_eval_scores = None
        self.last_eval_labels = None
        # ×™×¦×™×¨×ª DataLoader ×¤×¢× ××—×ª ×‘×œ×‘×“ ×œ×ž× ×™×¢×ª ×“×œ×™×¤×ª ×–×™×›×¨×•×Ÿ
        self._eval_loader = None

    def _get_eval_loader(self):
        """
        Build or return the cached eval DataLoader.
        Steps:
          1) Create the loader on first use.
          2) Return the cached loader thereafter.
        """
        if self._eval_loader is None:
            self._eval_loader = DataLoader(
                self.eval_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available()
            )
        return self._eval_loader

    def _should_save_npz(self, epoch_idx):
        """
        _should_save_npz: Describe the purpose of this function.
        Steps:
        1) Validate inputs or state.
        2) Perform the core logic.
        3) Return results or update state.
        """
        if not self.save_npz_dir or self.save_npz_every < 1:
            return False
        return (epoch_idx + 1) % self.save_npz_every == 0

    def _save_npz_snapshot(self, epoch_idx, scores, labels, eer_raw, eer_threshold):
        """
        _save_npz_snapshot: Describe the purpose of this function.
        Steps:
        1) Validate inputs or state.
        2) Perform the core logic.
        3) Return results or update state.
        """
        os.makedirs(self.save_npz_dir, exist_ok=True)
        epoch_num = epoch_idx + 1
        snapshot_path = os.path.join(self.save_npz_dir, f"eval_scores_epoch_{epoch_num:04d}.npz")
        np.savez(
            snapshot_path,
            scores=scores,
            labels=labels,
            eer=eer_raw,
            eer_percent=eer_raw * 100.0 if np.isfinite(eer_raw) else np.nan,
            threshold=eer_threshold,
            epoch=epoch_num,
        )
        print(f"\n[Epoch {epoch_idx}] Eval scores snapshot saved to: {snapshot_path}")

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        on_validation_epoch_end: Describe the purpose of this function.
        Steps:
        1) Validate inputs or state.
        2) Perform the core logic.
        3) Return results or update state.
        """
        pl_module.eval()
        device = pl_module.device

        # ×ž×¢×§×‘ ××—×¨×™ ×–×™×›×¨×•×Ÿ GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / 1e9
            print(f"\n[Epoch {pl_module.current_epoch}] GPU memory before eval: {mem_before:.3f} GB")

        all_logits, all_labels = [], []
        # ×©×™×ž×•×© ×‘-DataLoader ×§×™×™× ×‘×ž×§×•× ×™×¦×™×¨×ª ×—×“×© ×‘×›×œ epoch
        with torch.no_grad():
            for x, y in self._get_eval_loader():
                x = x.to(device, non_blocking=True)
                out = pl_module(x)
                all_logits.append(out.cpu())
                all_labels.append(y.cpu())
                # ×©×—×¨×•×¨ ×ž×¤×•×¨×© ×©×œ ×˜× ×¡×•×¨×™× ×ž×”-GPU
                del x, out

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels).numpy()

        # ×©×—×¨×•×¨ ×”×¨×©×™×ž×•×ª ×ž×™×“ ××—×¨×™ ×”×©×™×ž×•×©
        del all_logits, all_labels

        # × ×™×§×•×™ GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / 1e9
            print(f"[Epoch {pl_module.current_epoch}] GPU memory after eval cleanup: {mem_after:.3f} GB")
        if len(np.unique(labels)) < 2:
            eer = np.nan
            eer_threshold = np.nan
            print(f"\n[Epoch {pl_module.current_epoch}] Skipping Eval EER calculation: only one class present in eval labels.")
        else:
            probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
            self.last_eval_scores = probs.astype(np.float32)
            self.last_eval_labels = labels.astype(np.int64)
            eer_raw, eer_threshold = compute_eer(labels, probs, return_threshold=True)
            if np.isfinite(eer_raw):
                if self.best_eval_eer is None or eer_raw < self.best_eval_eer:
                    self.best_eval_eer = float(eer_raw)
                    self.best_eval_threshold = float(eer_threshold)
                    self.best_eval_scores = self.last_eval_scores
                    self.best_eval_labels = self.last_eval_labels
                    self.best_eval_epoch = int(pl_module.current_epoch)
            eer = eer_raw * 100.0
            print(f"\n[Epoch {pl_module.current_epoch}] Eval EER: {eer:.2f}%")
            if self._should_save_npz(pl_module.current_epoch):
                self._save_npz_snapshot(
                    pl_module.current_epoch,
                    self.last_eval_scores,
                    self.last_eval_labels,
                    float(eer_raw),
                    float(eer_threshold),
                )
        # Always use the model's own eval_eers list
        if not hasattr(pl_module, "eval_eers"):
            pl_module.eval_eers = []
        if not hasattr(pl_module, "eval_eer_thresholds"):
            pl_module.eval_eer_thresholds = []
        pl_module.eval_eers.append(eer)
        pl_module.eval_eer_thresholds.append(eer_threshold)
        monitor_eer = float(eer) if np.isfinite(eer) else 1e9
        pl_module.log('eval_eer', monitor_eer, prog_bar=True, on_epoch=True)

        # ×©×—×¨×•×¨ logits ×•-labels ×©×›×‘×¨ ×œ× × ×—×•×¦×™×
        del logits, labels

# === Data ===
def _infer_labels_path(embeddings_path):
    """
    _infer_labels_path: Describe the purpose of this function.
    Steps:
    1) Inspect the embeddings filename.
    2) Derive a matching labels filename.
    3) Return the inferred path.
    """
    if embeddings_path.endswith("_embeddings.npy"):
        return embeddings_path.replace("_embeddings.npy", "_labels.npy")
    if embeddings_path.endswith(".npy"):
        return embeddings_path[:-4] + "_labels.npy"
    return None

class EmbeddingDataset(Dataset):
    def __init__(self, npz_path):
        """
        __init__: Describe the purpose of this function.
        Steps:
        1) Load embeddings (NPZ full load or NPY memmap).
        2) Load or map labels.
        3) Store arrays for __getitem__.
        """
        self._use_memmap = False
        if npz_path.endswith(".npz"):
            # NPZ path: load full arrays into RAM.
            d = np.load(npz_path, allow_pickle=True)
            self.embeddings = torch.tensor(d['embeddings'], dtype=torch.float32)
            label_map = {'bonafide': 0, 'genuine': 0, 'spoof': 1}
            self.labels = torch.tensor([label_map[str(l)] for l in d['labels']], dtype=torch.long)
        elif npz_path.endswith(".npy"):
            # NPY path: use memmap for embeddings and numeric labels in a paired labels file.
            labels_path = _infer_labels_path(npz_path)
            if not labels_path or not os.path.exists(labels_path):
                raise FileNotFoundError(f"Labels file not found for embeddings: {labels_path}")
            emb = np.load(npz_path, mmap_mode='r')
            if emb.dtype != np.float32:
                emb = emb.astype(np.float32, copy=False)
            self.embeddings = emb
            labels = np.load(labels_path, mmap_mode='r')
            if labels.dtype != np.int64:
                labels = labels.astype(np.int64, copy=False)
            self.labels = labels
            self._use_memmap = True
        else:
            raise ValueError(f"Unsupported embeddings file type: {npz_path}")



    def __len__(self):
        """
        __len__: Describe the purpose of this function.
        Steps:
        1) Validate inputs or state.
        2) Perform the core logic.
        3) Return results or update state.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        __getitem__: Describe the purpose of this function.
        Steps:
        1) Validate inputs or state.
        2) Perform the core logic.
        3) Return results or update state.
        """
        if self._use_memmap:
            x = torch.from_numpy(self.embeddings[idx])
            y = torch.as_tensor(self.labels[idx], dtype=torch.long)
            return x, y
        return self.embeddings[idx], self.labels[idx]


# === Model ===
class LinearProjector(pl.LightningModule):
    def __init__(self, input_dim=3072, output_dim=2, lr=1e-3, class_weights=None):
        """
        __init__: Describe the purpose of this function.
        Steps:
        1) Validate inputs or state.
        2) Perform the core logic.
        3) Return results or update state.
        """
        super().__init__()
        self.save_hyperparameters()
        # ×ž×§×¨×Ÿ ×œ×™× ××¨×™ ×ž×”×ž×¨×—×‘ ×”×ž×§×•×¨×™ ×œ×©× ×™ ×œ×•×’×™×˜×™× (genuine/spoof)
        self.projector = nn.Linear(input_dim, output_dim)
        weight_tensor = None
        if class_weights is not None:
            if len(class_weights) != output_dim:
                raise ValueError(
                    f"class_weights length must match output_dim ({output_dim}), got {len(class_weights)}"
                )
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        self.loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
        self.lr = lr
        self.train_logits = []
        self.train_labels = []
        self.val_logits = []
        self.val_labels = []
        self.train_eers = []    # For train EERs
        self.val_eers = []      # For dev EERs
        self.eval_eers = []     # For eval EERs
        self.train_eer_thresholds = []  # Threshold at train EER
        self.val_eer_thresholds = []    # Threshold at dev EER
        self.eval_eer_thresholds = []   # Threshold at eval EER

    def forward(self, x):
        """
        forward: Describe the purpose of this function.
        Steps:
        1) Validate inputs or state.
        2) Perform the core logic.
        3) Return results or update state.
        """
        return self.projector(x)

    def on_train_epoch_start(self):
        """
        on_train_epoch_start: Describe the purpose of this function.
        Steps:
        1) Validate inputs or state.
        2) Perform the core logic.
        3) Return results or update state.
        """
        # Reset buffers each epoch
        self.train_logits = []
        self.train_labels = []

    def training_step(self, batch, batch_idx):
        """
        training_step: Describe the purpose of this function.
        Steps:
        1) Validate inputs or state.
        2) Perform the core logic.
        3) Return results or update state.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_logits.append(logits.detach().cpu())
        self.train_labels.append(y.detach().cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        """
        validation_step: Describe the purpose of this function.
        Steps:
        1) Validate inputs or state.
        2) Perform the core logic.
        3) Return results or update state.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.val_logits.append(logits.detach().cpu())
        self.val_labels.append(y.detach().cpu())
        return loss

    def on_train_epoch_end(self):
        """
        on_train_epoch_end: Describe the purpose of this function.
        Steps:
        1) Validate inputs or state.
        2) Perform the core logic.
        3) Return results or update state.
        """
        if len(self.train_logits) > 0:
            all_logits = torch.cat(self.train_logits)
            all_probs = torch.softmax(all_logits, dim=-1)[:, 1].numpy()
            all_labels = torch.cat(self.train_labels).numpy()
            if np.unique(all_labels).size < 2:
                eer = np.nan
                eer_threshold = np.nan
                raise ValueError(
                    f"[Epoch {self.current_epoch}] Skipping Train EER calculation: only one class present in train labels."
                )
            else:
                eer, eer_threshold = compute_eer(all_labels, all_probs, return_threshold=True)
                eer = eer * 100.0
                print(f"\n[Epoch {self.current_epoch}] Train EER: {eer:.2f}%")
            self.log('train_eer', eer if eer==eer else 0, prog_bar=False, on_epoch=True)
            self.train_eers.append(eer)
            self.train_eer_thresholds.append(eer_threshold)
            # ×©×—×¨×•×¨ ×ž×¤×•×¨×© ×©×œ ×¨×©×™×ž×•×ª ×•× ×™×§×•×™ ×–×™×›×¨×•×Ÿ
            del all_logits, all_probs, all_labels
            self.train_logits.clear()
            self.train_labels.clear()
            self.train_logits = []
            self.train_labels = []
            # × ×™×§×•×™ GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        """
        on_validation_epoch_end: Describe the purpose of this function.
        Steps:
        1) Validate inputs or state.
        2) Perform the core logic.
        3) Return results or update state.
        """
        if len(self.val_logits) > 0:
            all_logits = torch.cat(self.val_logits)
            all_probs = torch.softmax(all_logits, dim=-1)[:, 1].numpy()
            all_labels = torch.cat(self.val_labels).numpy()
            # ×—×™×©×•×‘ EER ×¢×œ ×¡×˜ ×”-dev
            if np.unique(all_labels).size < 2:
                eer = np.nan
                eer_threshold = np.nan
                raise ValueError(f"\n[Epoch {self.current_epoch}] Skipping Validation EER calculation: only one class present in dev labels.")
            else:
                eer, eer_threshold = compute_eer(all_labels, all_probs, return_threshold=True)
                eer = eer * 100.0
                print(f"\n[Epoch {self.current_epoch}] Validation EER: {eer:.2f}%")
            self.log('val_eer', eer if eer==eer else 0, prog_bar=True, on_epoch=True)
            self.val_eers.append(eer)
            self.val_eer_thresholds.append(eer_threshold)
            # ×©×—×¨×•×¨ ×ž×¤×•×¨×© ×©×œ ×¨×©×™×ž×•×ª ×•× ×™×§×•×™ ×–×™×›×¨×•×Ÿ
            del all_logits, all_probs, all_labels
            self.val_logits.clear()
            self.val_labels.clear()
            self.val_logits = []
            self.val_labels = []
            # × ×™×§×•×™ GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()



    def configure_optimizers(self):
        """
        configure_optimizers: Describe the purpose of this function.
        Steps:
        1) Validate inputs or state.
        2) Perform the core logic.
        3) Return results or update state.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,
            gamma=0.5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# === Metrics ===
def compute_eer(y_true, y_score, return_threshold=False):
    """
    Compute the equal error rate (EER) and its operating threshold.
    Steps:
    1) Validate inputs and convert to NumPy arrays.
    2) Compute ROC curve (FPR/TPR) and derive FNR.
    3) Find the point where FPR and FNR are closest.
    4) Return EER and the associated threshold (or a safe fallback).
    """
    import warnings

    if len(y_true) == 0 or len(y_score) == 0:
        warnings.warn("Cannot compute EER on empty arrays, returning 1.0 (worst case)")
        return (1.0, 0.5) if return_threshold else 1.0

    if len(y_true) != len(y_score):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} elements, y_score has {len(y_score)}")

    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_score, np.ndarray):
        y_score = np.array(y_score)

    if np.unique(y_true).size < 2:
        warnings.warn("Cannot compute EER with only one unique label, returning 0.5")
        return (0.5, 0.5) if return_threshold else 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr

    diff = np.abs(fpr - fnr)
    if diff.size == 0:
        warnings.warn("Could not compute EER (no threshold points), returning 0.5")
        return (0.5, 0.5) if return_threshold else 0.5

    eer_idx = int(np.argmin(diff))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    threshold = 0.5
    if eer_idx < len(thresholds):
        threshold = thresholds[eer_idx]
    if not np.isfinite(threshold):
        threshold = 0.5
    return (float(eer), float(threshold)) if return_threshold else float(eer)

def compute_eer_with_ci(y_true, y_score, num_bootstraps=1000, alpha=0.05, random_seed=42):
    """
    compute_eer_with_ci: Compute EER center + bootstrap confidence interval.
    Steps:
    1) Compute full-set EER center/threshold.
    2) Bootstrap EER samples.
    3) If a bootstrap sample has one class, fallback to EER center.
    4) Return center, threshold, CI low, CI high.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) != len(y_score):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} elements, y_score has {len(y_score)}")

    eer_center, threshold = compute_eer(y_true, y_score, return_threshold=True)
    eer_center = float(eer_center)
    threshold = float(threshold)

    if len(y_true) == 0:
        return eer_center, threshold, np.nan, np.nan

    # Requested fallback: when only one class is available, return eer_center.
    if np.unique(y_true).size < 2:
        return eer_center, threshold, eer_center, eer_center

    num_bootstraps = int(max(1, num_bootstraps))
    rng = np.random.default_rng(int(random_seed))
    n = len(y_true)
    sampled_eers = np.empty(num_bootstraps, dtype=np.float64)

    for i in range(num_bootstraps):
        indices = rng.integers(0, n, size=n)
        b_labels = y_true[indices]
        b_scores = y_score[indices]
        if np.unique(b_labels).size < 2:
            sampled_eers[i] = eer_center
        else:
            b_eer = compute_eer(b_labels, b_scores, return_threshold=False)
            sampled_eers[i] = float(b_eer) if np.isfinite(b_eer) else eer_center

    q_low = 100.0 * (float(alpha) / 2.0)
    q_high = 100.0 * (1.0 - float(alpha) / 2.0)
    ci_low = float(np.percentile(sampled_eers, q_low))
    ci_high = float(np.percentile(sampled_eers, q_high))
    return eer_center, threshold, ci_low, ci_high

# === Run Helpers ===
def _fmt_tag_value(value):
    """
    _fmt_tag_value: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    return str(value).replace(".", "p")

def build_run_tag(learning_rate, batch_size, num_epochs, input_dim, project_out, class_weights):
    """
    build_run_tag: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    if class_weights is None:
        cw_tag = "cwNone"
    else:
        cw_tag = f"cw{_fmt_tag_value(class_weights[0])}_{_fmt_tag_value(class_weights[1])}"
    return (
        f"lr{_fmt_tag_value(learning_rate)}_bs{batch_size}_ep{num_epochs}"
        f"_in{input_dim}_out{project_out}_{cw_tag}"
    )

def _read_active_run_id(active_run_path):
    """
    _read_active_run_id: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    if not os.path.exists(active_run_path):
        return None
    with open(active_run_path, "r", encoding="utf-8") as f:
        run_id = f.read().strip()
    return run_id if run_id else None

def prepare_run_dirs(train_path, dbname, run_tag, resume_from_last=True):
    """
    prepare_run_dirs: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    base_dir = os.path.join(os.path.dirname(train_path), "checkpoints1", dbname)
    run_root = os.path.join(base_dir, run_tag)
    os.makedirs(run_root, exist_ok=True)
    active_run_path = os.path.join(run_root, "active_run.txt")

    resume_mode = False
    already_completed = False
    run_id = None
    checkpoint_dir = None
    if resume_from_last:
        active_run_id = _read_active_run_id(active_run_path)
        if active_run_id:
            candidate_dir = os.path.join(run_root, active_run_id)
            running_marker = os.path.join(candidate_dir, "RUNNING")
            completed_marker = os.path.join(candidate_dir, "COMPLETED")
            if os.path.isdir(candidate_dir):
                if os.path.exists(completed_marker):
                    already_completed = True
                    run_id = active_run_id
                    checkpoint_dir = candidate_dir
                elif os.path.exists(running_marker):
                    resume_mode = True
                    run_id = active_run_id
                    checkpoint_dir = candidate_dir

    if checkpoint_dir is None:
        run_id = time.strftime("run_%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(run_root, run_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(active_run_path, "w", encoding="utf-8") as f:
            f.write(run_id)

    running_marker = os.path.join(checkpoint_dir, "RUNNING")
    completed_marker = os.path.join(checkpoint_dir, "COMPLETED")
    if not already_completed:
        if not os.path.exists(running_marker):
            with open(running_marker, "w", encoding="utf-8") as f:
                f.write(f"started_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    return {
        "base_dir": base_dir,
        "run_root": run_root,
        "run_id": run_id,
        "checkpoint_dir": checkpoint_dir,
        "resume_mode": resume_mode,
        "already_completed": already_completed,
        "running_marker": running_marker,
        "completed_marker": completed_marker,
    }

def resolve_resume_checkpoint(checkpoint_dir, resume_mode):
    """
    resolve_resume_checkpoint: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    if not resume_mode:
        return None
    last_ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
    if not os.path.exists(last_ckpt_path):
        return None
    print(f"Resuming from checkpoint: {last_ckpt_path}")
    try:
        ckpt = torch.load(last_ckpt_path, map_location="cpu")
        resumed_epoch = ckpt.get("epoch", None)
        if resumed_epoch is not None:
            print(f"Resuming from epoch: {resumed_epoch}")
    except Exception as e:
        print(f"Warning: failed to read epoch from checkpoint: {e}")
    return last_ckpt_path

def get_checkpoint_epoch(ckpt_path):
    """
    get_checkpoint_epoch: Describe the purpose of this function.
    Steps:
    1) Load the checkpoint on CPU.
    2) Extract the saved epoch number.
    3) Return the epoch (or None on failure).
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        return ckpt.get("epoch", None)
    except Exception as e:
        print(f"Warning: failed to read epoch from checkpoint: {e}")
        return None

def create_callbacks(checkpoint_dir, eval_set, batch_size, num_workers, save_eval_npz_every):
    """
    create_callbacks: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    checkpoint_val_loss_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename='best_val_loss-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
        save_last=True,
    )
    checkpoint_eval_eer_callback = ModelCheckpoint(
        monitor='eval_eer',
        dirpath=checkpoint_dir,
        filename='best_eval_eer-{epoch:02d}-{eval_eer:.4f}',
        save_top_k=1,
        mode='min',
        save_last=False,
    )
    eval_npz_dir = None
    if save_eval_npz_every and save_eval_npz_every > 0:
        eval_npz_dir = os.path.join(checkpoint_dir, "eval_scores_snapshots")
    eval_eer_callback = EvalEERCallback(
        eval_set,
        batch_size,
        num_workers=num_workers,
        save_npz_every=save_eval_npz_every,
        save_npz_dir=eval_npz_dir,
    )
    return checkpoint_val_loss_callback, checkpoint_eval_eer_callback, eval_eer_callback

def save_best_eval_scores(eval_eer_callback, checkpoint_dir):
    """
    save_best_eval_scores: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    best_eval_scores_path = os.path.join(checkpoint_dir, "best_eval_scores.npz")
    if getattr(eval_eer_callback, "best_eval_scores", None) is not None:
        np.savez(
            best_eval_scores_path,
            scores=eval_eer_callback.best_eval_scores,
            labels=eval_eer_callback.best_eval_labels,
            eer=eval_eer_callback.best_eval_eer,
            eer_percent=eval_eer_callback.best_eval_eer * 100.0,
            threshold=eval_eer_callback.best_eval_threshold,
            epoch=eval_eer_callback.best_eval_epoch,
        )
        print(f"Best eval scores saved to: {best_eval_scores_path}")
    else:
        print("Warning: best eval scores not available; skipping save.")

    # Release large eval arrays held by callback now that they've been persisted
    eval_eer_callback._eval_loader = None
    eval_eer_callback.eval_set = None
    eval_eer_callback.best_eval_scores = None
    eval_eer_callback.best_eval_labels = None
    eval_eer_callback.last_eval_scores = None
    eval_eer_callback.last_eval_labels = None
    return best_eval_scores_path

def build_hp_header(
    dbname,
    learning_rate,
    batch_size,
    num_epochs,
    input_dim,
    project_out,
    num_workers,
    class_weights,
    save_eval_npz_every,
):
    """
    build_hp_header: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    return (
        f"# db={dbname}\n"
        f"# lr={learning_rate}\n"
        f"# batch_size={batch_size}\n"
        f"# num_epochs={num_epochs}\n"
        f"# input_dim={input_dim}\n"
        f"# project_out={project_out}\n"
        f"# num_workers={num_workers}\n"
        f"# class_weights={class_weights}\n"
        f"# save_eval_npz_every={save_eval_npz_every}\n"
        f"# seed=42\n"
        f"# optimizer=AdamW\n"
        f"# optimizer_params={{lr:{learning_rate}}}\n"
    )

def write_hyperparameters(
    checkpoint_dir,
    run_tag,
    dbname,
    learning_rate,
    batch_size,
    num_epochs,
    input_dim,
    project_out,
    num_workers,
    class_weights,
    save_eval_npz_every,
):
    """
    write_hyperparameters: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    hp_file_path = os.path.join(checkpoint_dir, "hyperparameters.txt")
    with open(hp_file_path, "w", encoding="utf-8") as f:
        f.write(f"db={dbname}\n")
        f.write(f"lr={learning_rate}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"num_epochs={num_epochs}\n")
        f.write(f"input_dim={input_dim}\n")
        f.write(f"project_out={project_out}\n")
        f.write(f"num_workers={num_workers}\n")
        f.write(f"class_weights={class_weights}\n")
        f.write(f"save_eval_npz_every={save_eval_npz_every}\n")
        f.write(f"seed=42\n")
        f.write(f"optimizer=AdamW\n")
        f.write(f"run_tag={run_tag}\n")
        f.write(f"checkpoint_dir={checkpoint_dir}\n")
    print(f"Hyperparameters saved to: {hp_file_path}")
    return hp_file_path

def write_eer_log(log_path, hp_header, eers, thresholds):
    """
    write_eer_log: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(hp_header)
        f.write("epoch\teer\tthreshold\n")
        n = max(len(eers), len(thresholds))
        for i in range(n):
            eer_v = eers[i] if i < len(eers) else np.nan
            thr_v = thresholds[i] if i < len(thresholds) else np.nan
            f.write(f"{i}\t{0.0 if eer_v != eer_v else eer_v:.6f}\t{0.0 if thr_v != thr_v else thr_v:.6f}\n")

def append_final_eer(log_path, label, eer, threshold):
    """
    append_final_eer: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{label}\t{0.0 if eer != eer else eer:.6f}\t{0.0 if threshold != threshold else threshold:.6f}\n")

def summarize_min_eer(eers, thresholds):
    """
    summarize_min_eer: Describe the purpose of this function.
    Steps:
    1) Convert inputs to NumPy and handle empty/all-NaN cases.
    2) Find the minimal EER and its epoch index.
    3) Return (min_eer, threshold_at_min, epoch_index).
    """
    if not eers:
        return np.nan, np.nan, None
    arr = np.array(eers, dtype=float)
    if np.all(np.isnan(arr)):
        return np.nan, np.nan, None
    min_idx = int(np.nanargmin(arr))
    min_eer = float(arr[min_idx])
    min_thr = thresholds[min_idx] if min_idx < len(thresholds) else np.nan
    return min_eer, min_thr, min_idx

def resolve_best_checkpoint(best_model_path, checkpoint_dir):
    """
    resolve_best_checkpoint: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    if best_model_path and os.path.exists(best_model_path):
        return best_model_path
    last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        print("Warning: best_model_path not found, using last.ckpt instead")
        return last_ckpt
    raise FileNotFoundError(
        f"No checkpoint found: best_model_path={best_model_path}, last.ckpt also missing"
    )

def save_best_eval_eer_checkpoint(best_model_path, checkpoint_dir):
    """
    save_best_eval_eer_checkpoint: Copy the lowest-eval_eer checkpoint to a fixed filename.
    Steps:
    1) Resolve the best checkpoint path.
    2) Copy it to best_eval_eer.ckpt.
    3) Return destination path.
    """
    src = resolve_best_checkpoint(best_model_path, checkpoint_dir)
    dst = os.path.join(checkpoint_dir, "best_eval_eer.ckpt")
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
    print(f"Saved best eval_eer checkpoint to: {dst}")
    return dst

def save_best_val_loss_checkpoint(best_model_path, checkpoint_dir):
    """
    save_best_val_loss_checkpoint: Copy the lowest-val_loss checkpoint to a fixed filename.
    Steps:
    1) Resolve the best checkpoint path.
    2) Copy it to best_val_loss.ckpt.
    3) Return destination path.
    """
    src = resolve_best_checkpoint(best_model_path, checkpoint_dir)
    dst = os.path.join(checkpoint_dir, "best_val_loss.ckpt")
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
    print(f"Saved best val_loss checkpoint to: {dst}")
    return dst

def get_scores_and_labels(model, device, dataset, batch_size, num_workers):
    """
    get_scores_and_labels: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    all_scores, all_labels = [], []
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            scores = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()
            all_scores.append(scores)
            all_labels.append(y.numpy())
            del x, out
        clear_cuda_cache()
    del loader
    return np.concatenate(all_scores), np.concatenate(all_labels)

def score_split(model, device, npz_path, batch_size, num_workers):
    """
    score_split: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    dataset = EmbeddingDataset(npz_path)
    scores, labels = get_scores_and_labels(model, device, dataset, batch_size, num_workers)
    eer, thr = compute_eer(labels, scores, return_threshold=True)
    eer *= 100.0
    # Free RAM immediately
    del scores, labels, dataset
    gc.collect()
    clear_cuda_cache()
    return eer, thr

def score_split_with_ci(model, device, npz_path, batch_size, num_workers, num_bootstraps=1000, alpha=0.05):
    """
    score_split_with_ci: Score a split and compute EER + CI (% units).
    Steps:
    1) Load split and compute scores/labels.
    2) Compute EER center + threshold + CI in ratio units.
    3) Convert EER/CI to percent and return.
    """
    dataset = EmbeddingDataset(npz_path)
    scores, labels = get_scores_and_labels(model, device, dataset, batch_size, num_workers)
    eer_center, thr, ci_low, ci_high = compute_eer_with_ci(
        labels,
        scores,
        num_bootstraps=num_bootstraps,
        alpha=alpha,
    )
    eer_percent = eer_center * 100.0
    ci_low_percent = ci_low * 100.0 if np.isfinite(ci_low) else np.nan
    ci_high_percent = ci_high * 100.0 if np.isfinite(ci_high) else np.nan
    del scores, labels, dataset
    gc.collect()
    clear_cuda_cache()
    return eer_percent, thr, ci_low_percent, ci_high_percent

def save_eer_curve_plot(checkpoint_dir, dbname, val_eers, eval_eers):
    """
    save_eer_curve_plot: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    if not val_eers or not eval_eers:
        print("Warning: Skipping EER curve plot - val_eers or eval_eers is empty")
        return None
    plt.figure()
    plt.plot(range(1, len(val_eers) + 1), val_eers, marker='o', label="Dev EER")
    plt.plot(range(1, len(eval_eers) + 1), eval_eers, marker='x', label="Eval EER")
    plt.title(f"{dbname} EER during Training")
    plt.xlabel("Epoch")
    plt.ylabel("EER %")
    plt.grid(True)
    plt.legend()
    eer_curve_path = os.path.join(checkpoint_dir, f"{dbname}_val_eval_eer_curve.png")
    plt.savefig(eer_curve_path, dpi=200)
    plt.close()
    print(f"Dev/Eval EER curve plot saved to: {eer_curve_path}")
    return eer_curve_path

# === Pipeline ===
def run_pipeline(
    dbname,
    train_path,
    dev_path,
    eval_path,
    *,
    input_dim,
    project_out,
    batch_size,
    num_epochs,
    learning_rate,
    num_workers,
    class_weights=None,
    resume_from_last=True,
    save_eval_npz_every=0,
    run_final_scoring=False,
):
    """
    run_pipeline: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    run_tag = build_run_tag(
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        input_dim=input_dim,
        project_out=project_out,
        class_weights=class_weights,
    )
    # --- Data ---
    train_set = EmbeddingDataset(train_path)
    dev_set = EmbeddingDataset(dev_path)
    eval_set = EmbeddingDataset(eval_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # --- Model ---
    model = LinearProjector(
        input_dim=input_dim,
        output_dim=project_out,
        lr=learning_rate,
        class_weights=class_weights,
    )
    
    # ×”×’×“×¨×ª ×©×ž×™×¨×ª checkpoint ×œ×¤×™ val_loss (×ª×™×§×™×™×” ×™×™×¢×•×“×™×ª ×œ×›×œ ×§×•× ×¤×™×’×•×¨×¦×™×”)
    run_info = prepare_run_dirs(train_path, dbname, run_tag, resume_from_last=resume_from_last)
    base_dir = run_info["base_dir"]
    checkpoint_dir = run_info["checkpoint_dir"]
    running_marker = run_info["running_marker"]
    completed_marker = run_info["completed_marker"]
    resume_mode = run_info["resume_mode"]
    already_completed = run_info.get("already_completed", False)

    if already_completed:
        print("[Info] Run already completed; skipping.")
        # Cleanup loaded data/model before exiting
        del train_loader, dev_loader, train_set, dev_set, eval_set, model
        gc.collect()
        clear_cuda_cache()
        if os.path.exists(running_marker):
            os.remove(running_marker)
        return {
            "dbname": dbname,
            "run_tag": run_tag,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "input_dim": input_dim,
            "project_out": project_out,
            "num_workers": num_workers,
            "class_weights": class_weights,
            "checkpoint_dir": checkpoint_dir,
            "base_dir": base_dir,
        }

    # If the last checkpoint already reached max_epochs, skip this run and move on.
    if resume_mode:
        last_ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
        if os.path.exists(last_ckpt_path):
            resumed_epoch = get_checkpoint_epoch(last_ckpt_path)
            if resumed_epoch is not None and resumed_epoch >= num_epochs - 1:
                print(f"[Info] Checkpoint already at epoch {resumed_epoch}; marking run completed and skipping.")
                # Cleanup loaded data/model before exiting
                del train_loader, dev_loader, train_set, dev_set, eval_set, model
                gc.collect()
                clear_cuda_cache()
                if os.path.exists(running_marker):
                    os.remove(running_marker)
                with open(completed_marker, "w", encoding="utf-8") as f:
                    f.write(f"completed_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                return {
                    "dbname": dbname,
                    "run_tag": run_tag,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "input_dim": input_dim,
                    "project_out": project_out,
                    "num_workers": num_workers,
                    "class_weights": class_weights,
                    "checkpoint_dir": checkpoint_dir,
                    "base_dir": base_dir,
                }

    checkpoint_val_loss_callback, checkpoint_eval_eer_callback, eval_eer_callback = create_callbacks(
        checkpoint_dir=checkpoint_dir,
        eval_set=eval_set,
        batch_size=batch_size,
        num_workers=num_workers,
        save_eval_npz_every=save_eval_npz_every,
    )
    
    # ×”×’×“×¨×ª ×ž××ž×Ÿ PyTorch Lightning
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        enable_checkpointing=True,
        logger=False,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[
            checkpoint_val_loss_callback,
            checkpoint_eval_eer_callback,
            eval_eer_callback,
            TQDMProgressBar(refresh_rate=50),
        ],
        num_sanity_val_steps=0,
    )

    # ×‘×“×™×§×” ×•×”×“×¤×¡×” ×œ×¤× ×™ ×”×¨×™×¦×” ×× ×ž×©×ª×ž×©×™× ×‘-GPU
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")

    ckpt_path = resolve_resume_checkpoint(checkpoint_dir, resume_mode)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dev_loader, ckpt_path=ckpt_path)

    # Early cleanup: training objects are no longer needed after fit
    del train_loader, dev_loader, train_set, dev_set
    gc.collect()
    del trainer
    clear_cuda_cache()

    # Save best eval scores (by lowest eval EER) for this run
    save_best_eval_scores(eval_eer_callback, checkpoint_dir)
    if eval_eer_callback is not None:
        del eval_eer_callback
    if eval_set is not None:
        del eval_set
    gc.collect()

    # ×›×ª×™×‘×ª ×œ×•×’ EER ×œ×§×‘×¦×™× ××—×¨×™ ×¡×™×•× ×”××™×ž×•×Ÿ (×›×ª×™×‘×” ×—×“-×¤×¢×ž×™×ª)
    train_eer_dir = os.path.join(checkpoint_dir, "eer_train")
    dev_eer_dir = os.path.join(checkpoint_dir, "eer_dev")
    eval_eer_dir = os.path.join(checkpoint_dir, "eer_eval")
    os.makedirs(train_eer_dir, exist_ok=True)
    os.makedirs(dev_eer_dir, exist_ok=True)
    os.makedirs(eval_eer_dir, exist_ok=True)
    train_eer_log_path = os.path.join(train_eer_dir, "eer_train_log.txt")
    dev_eer_log_path = os.path.join(dev_eer_dir, "eer_dev_log.txt")
    eval_eer_log_path = os.path.join(eval_eer_dir, "eer_eval_log.txt")
    hp_header = build_hp_header(
        dbname=dbname,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        input_dim=input_dim,
        project_out=project_out,
        num_workers=num_workers,
        class_weights=class_weights,
        save_eval_npz_every=save_eval_npz_every,
    )

    # ×©×ž×™×¨×ª ×”×™×¤×¨-×¤×¨×ž×˜×¨×™× ×œ×§×•×‘×¥ × ×¤×¨×“ ×‘×ª×™×§×™×™×ª ×”-checkpoint
    write_hyperparameters(
        checkpoint_dir=checkpoint_dir,
        run_tag=run_tag,
        dbname=dbname,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        input_dim=input_dim,
        project_out=project_out,
        num_workers=num_workers,
        class_weights=class_weights,
        save_eval_npz_every=save_eval_npz_every,
    )
    train_eers_list = list(getattr(model, "train_eers", []))
    train_thresholds_list = list(getattr(model, "train_eer_thresholds", []))
    val_eers = list(getattr(model, "val_eers", []))
    val_thresholds_list = list(getattr(model, "val_eer_thresholds", []))
    eval_eers = list(getattr(model, "eval_eers", []))
    eval_thresholds_list = list(getattr(model, "eval_eer_thresholds", []))
    del model
    clear_cuda_cache()
    gc.collect()

    min_train_eer, min_train_thr, min_train_epoch = summarize_min_eer(
        train_eers_list,
        train_thresholds_list,
    )
    min_val_eer, min_val_thr, min_val_epoch = summarize_min_eer(
        val_eers,
        val_thresholds_list,
    )
    min_eval_eer, min_eval_thr, min_eval_epoch = summarize_min_eer(
        eval_eers,
        eval_thresholds_list,
    )
    write_eer_log(
        train_eer_log_path,
        hp_header,
        train_eers_list,
        train_thresholds_list,
    )
    write_eer_log(
        dev_eer_log_path,
        hp_header,
        val_eers,
        val_thresholds_list,
    )
    write_eer_log(
        eval_eer_log_path,
        hp_header,
        eval_eers,
        eval_thresholds_list,
    )

    # Append min-over-epochs summary to logs
    append_final_eer(train_eer_log_path, "min_train", min_train_eer, min_train_thr)
    append_final_eer(dev_eer_log_path, "min_dev", min_val_eer, min_val_thr)
    append_final_eer(eval_eer_log_path, "min_eval", min_eval_eer, min_eval_thr)

    best_val_loss_model_path = checkpoint_val_loss_callback.best_model_path
    best_eval_eer_model_path = checkpoint_eval_eer_callback.best_model_path
    print(f"\nBest val_loss checkpoint for {dbname}: {best_val_loss_model_path}")
    print(f"Best eval_eer checkpoint for {dbname}: {best_eval_eer_model_path}\n")

    # Save stable copies for both minima.
    try:
        save_best_val_loss_checkpoint(best_val_loss_model_path, checkpoint_dir)
    except Exception as e:
        print(f"Warning: failed to save best val_loss checkpoint copy: {e}")
    try:
        best_eval_eer_model_path = save_best_eval_eer_checkpoint(best_eval_eer_model_path, checkpoint_dir)
    except Exception as e:
        print(f"Warning: failed to save best eval_eer checkpoint copy: {e}")

    # Compute CI only on the min-eval_eer checkpoint.
    try:
        save_best_checkpoint_dev_eval_ci(
            checkpoint_dir,
            best_eval_eer_model_path,
            dev_path,
            eval_path,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    except Exception as e:
        print(f"Warning: failed to compute/save best-checkpoint dev/eval EER+CI: {e}")

    if min_train_epoch is not None:
        print(f"[Min EER] Train: {min_train_eer:.2f}% at epoch {min_train_epoch + 1} (thr={min_train_thr:.6f})")
    else:
        print("[Min EER] Train: unavailable (no valid values)")
    if min_val_epoch is not None:
        print(f"[Min EER] Dev:   {min_val_eer:.2f}% at epoch {min_val_epoch + 1} (thr={min_val_thr:.6f})")
    else:
        print("[Min EER] Dev:   unavailable (no valid values)")
    if min_eval_epoch is not None:
        print(f"[Min EER] Eval:  {min_eval_eer:.2f}% at epoch {min_eval_epoch + 1} (thr={min_eval_thr:.6f})")
    else:
        print("[Min EER] Eval:  unavailable (no valid values)")

    # ×˜×¢×™× ×ª ×”×ž×•×“×œ ×”×˜×•×‘ ×‘×™×•×ª×¨ ×œ×”×¡×§×ª ×”×˜×‘×¢×•×ª ×ž×•×§×¨× ×•×ª
    # ×× ××™×Ÿ best checkpoint, × ×©×ª×ž×© ×‘-last.ckpt
    if run_final_scoring:
        best_model_path = resolve_best_checkpoint(best_eval_eer_model_path, checkpoint_dir)

        # ×˜×¢×™× ×” ×œ-GPU ×¢× × ×™×”×•×œ ×–×™×›×¨×•×Ÿ
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LinearProjector.load_from_checkpoint(best_model_path, map_location=device)
        model.to(device)
        model.eval()

        # === Final scoring (per split) ===
        # Train
        eer_train, thr_train = score_split(model, device, train_path, batch_size, num_workers)

        # Dev
        eer_dev, thr_dev = score_split(model, device, dev_path, batch_size, num_workers)

        # Eval
        eer_eval, thr_eval = score_split(model, device, eval_path, batch_size, num_workers)

        print(f"Database: {dbname}\n  Train EER: {eer_train:.2f}%\n  Dev EER:  {eer_dev:.2f}%\n  Eval EER: {eer_eval:.2f}%")

        # Append final EERs to logs
        append_final_eer(train_eer_log_path, "final_train", eer_train, thr_train)
        append_final_eer(dev_eer_log_path, "final_dev", eer_dev, thr_dev)
        append_final_eer(eval_eer_log_path, "final_eval", eer_eval, thr_eval)
        
        # Save EER curve after training
        save_eer_curve_plot(checkpoint_dir, dbname, val_eers, eval_eers)
    else:
        print("[Info] Skipping final scoring and EER curve plot (run_final_scoring=False).")

    # × ×™×§×•×™ scores ×©×›×‘×¨ ×œ× × ×—×•×¦×™×
    gc.collect()

    if os.path.exists(running_marker):
        os.remove(running_marker)
    with open(completed_marker, "w", encoding="utf-8") as f:
        f.write(f"completed_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # === Memory cleanup to prevent leaks between runs ===
    print(f"\n[Cleanup] Starting aggressive cleanup for {dbname}...")
    
    # 1. Clear callback references explicitly
    if 'eval_eer_callback' in locals():
        eval_eer_callback._eval_loader = None
        eval_eer_callback.eval_set = None
        eval_eer_callback.best_eval_scores = None
        eval_eer_callback.best_eval_labels = None
        eval_eer_callback.last_eval_scores = None
        eval_eer_callback.last_eval_labels = None
    
    # 2. Delete PyTorch objects and loaders
    if 'train_loader' in locals(): del train_loader
    if 'dev_loader' in locals(): del dev_loader
    if 'train_set' in locals(): del train_set
    if 'dev_set' in locals(): del dev_set
    if 'eval_set' in locals(): del eval_set
    if 'model' in locals(): del model
    if 'trainer' in locals(): del trainer
    if 'eval_eer_callback' in locals(): del eval_eer_callback
    if 'checkpoint_val_loss_callback' in locals(): del checkpoint_val_loss_callback
    if 'checkpoint_eval_eer_callback' in locals(): del checkpoint_eval_eer_callback

    # 3. Clear CUDA cache
    clear_cuda_cache()

    # 4. Force Garbage Collection multiple times
    for _ in range(3):
        gc.collect()

    # 5. Additional CUDA clear
    clear_cuda_cache()
    if torch.cuda.is_available():
        print(f"[Cleanup] GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.3f} GB")

    # 6. Report System RAM usage
    if psutil:
        proc = psutil.Process()
        print(f"[Cleanup] System RAM usage: {proc.memory_info().rss / 1e9:.3f} GB")
    
    print("[Cleanup] Done.\n")

    return {
        "dbname": dbname,
        "run_tag": run_tag,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "input_dim": input_dim,
        "project_out": project_out,
        "num_workers": num_workers,
        "class_weights": class_weights,
        "checkpoint_dir": checkpoint_dir,
        "base_dir": base_dir,
    }


# === Reporting ===
def write_runs_report(runs):
    """
    write_runs_report: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    by_base = {}
    for r in runs:
        by_base.setdefault(r["base_dir"], []).append(r)
    for base_dir, entries in by_base.items():
        os.makedirs(base_dir, exist_ok=True)
        report_path = os.path.join(base_dir, "runs_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("run_tag\tlr\tbatch_size\tnum_epochs\tinput_dim\tproject_out\tnum_workers\tclass_weights\tcheckpoint_dir\n")
            for r in entries:
                f.write(
                    f"{r['run_tag']}\t{r['learning_rate']}\t{r['batch_size']}\t{r['num_epochs']}\t"
                f"{r['input_dim']}\t{r['project_out']}\t{r['num_workers']}\t{r['class_weights']}\t{r['checkpoint_dir']}\n"
                )

def run_experiment(
    db_name,
    db_dir,
    train_file,
    dev_file,
    eval_file,
    *,
    input_dim,
    project_out,
    batch_size,
    num_epochs,
    num_workers,
    save_eval_npz_every,
    learning_rates,
    class_weights_list,
    run_final_scoring=False,
):
    """
    run_experiment: Describe the purpose of this function.
    Steps:
    1) Validate inputs or state.
    2) Perform the core logic.
    3) Return results or update state.
    """
    runs = []
    for learning_rate in learning_rates:
        for class_weights in class_weights_list:
            print(f"\n=== [{db_name}] lr={learning_rate}, class_weights={class_weights} ===")
            runs.append(run_pipeline(
                db_name,
                os.path.join(db_dir, train_file),
                os.path.join(db_dir, dev_file),
                os.path.join(db_dir, eval_file),
                input_dim=input_dim,
                project_out=project_out,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                num_workers=num_workers,
                class_weights=class_weights,
                save_eval_npz_every=save_eval_npz_every,
                run_final_scoring=run_final_scoring,
            ))
            gc.collect()
    return runs

def save_best_checkpoint_dev_eval_ci(
    checkpoint_dir,
    best_model_path,
    dev_path,
    eval_path,
    *,
    batch_size,
    num_workers,
    num_bootstraps=1000,
    alpha=0.05,
):
    """
    save_best_checkpoint_dev_eval_ci: Evaluate only best checkpoint on dev/eval with CI.
    Steps:
    1) Resolve and load the best checkpoint.
    2) Compute dev/eval EER + CI once.
    3) Save results to split-specific text files.
    """
    resolved_ckpt = resolve_best_checkpoint(best_model_path, checkpoint_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LinearProjector.load_from_checkpoint(resolved_ckpt, map_location=device)
    model.to(device)
    model.eval()

    dev_out_path = os.path.join(checkpoint_dir, "eer_dev", "min_dev_best_checkpoint_with_ci.txt")
    eval_out_path = os.path.join(checkpoint_dir, "eer_eval", "min_eval_best_checkpoint_with_ci.txt")
    os.makedirs(os.path.dirname(dev_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(eval_out_path), exist_ok=True)

    try:
        dev_eer, dev_thr, dev_ci_low, dev_ci_high = score_split_with_ci(
            model,
            device,
            dev_path,
            batch_size,
            num_workers,
            num_bootstraps=num_bootstraps,
            alpha=alpha,
        )
        eval_eer, eval_thr, eval_ci_low, eval_ci_high = score_split_with_ci(
            model,
            device,
            eval_path,
            batch_size,
            num_workers,
            num_bootstraps=num_bootstraps,
            alpha=alpha,
        )
    finally:
        del model
        clear_cuda_cache()
        gc.collect()

    with open(dev_out_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint={resolved_ckpt}\n")
        f.write(f"eer_percent={dev_eer:.6f}\n")
        f.write(f"ci_low_percent={dev_ci_low:.6f}\n")
        f.write(f"ci_high_percent={dev_ci_high:.6f}\n")
        f.write(f"threshold={dev_thr:.6f}\n")
        f.write(f"num_bootstraps={int(num_bootstraps)}\n")
        f.write(f"alpha={float(alpha):.6f}\n")

    with open(eval_out_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint={resolved_ckpt}\n")
        f.write(f"eer_percent={eval_eer:.6f}\n")
        f.write(f"ci_low_percent={eval_ci_low:.6f}\n")
        f.write(f"ci_high_percent={eval_ci_high:.6f}\n")
        f.write(f"threshold={eval_thr:.6f}\n")
        f.write(f"num_bootstraps={int(num_bootstraps)}\n")
        f.write(f"alpha={float(alpha):.6f}\n")

    print(f"Saved dev EER+CI (best checkpoint) to: {dev_out_path}")
    print(f"Saved eval EER+CI (best checkpoint) to: {eval_out_path}")
    return {
        "best_checkpoint": resolved_ckpt,
        "dev_out_path": dev_out_path,
        "eval_out_path": eval_out_path,
    }

# === Entry Point ===
if __name__ == "__main__":
    # ========== PARAMETERS & HYPERPARAMETERS ==========
    # Data paths, files, and main hyperparameters
    DB1_NAME = "audio_projector_5"
    DB1_DIR = "/gpfs0/bgu-benshimo/users/wavishay/voxtral/ASVspoof5_embeds/audio_projector/"
    TRAIN_FILE = "total_train_embeddings.npz"
    DEV_FILE = "total_dev_embeddings.npz"
    EVAL_FILE = "total_eval_embeddings.npz"
    

    # Model/training configuration
    INPUT_DIM = 3072
    PROJECT_OUT = 2
    BATCH_SIZE = 2048
    NUM_EPOCHS = 400
    NUM_WORKERS = 8
    SAVE_EVAL_NPZ_EVERY = 0

    # Experiment grid
    LEARNING_RATES = [1e-2, 1e-3, 5e-4, 1e-4]
    CLASS_WEIGHTS_LIST = [[1.0, 1.0], [1, 3], [1, 5]]

    all_runs = []
    # Run: DB1 only
    all_runs.extend(run_experiment(
            db_name=DB1_NAME,
            db_dir=DB1_DIR,
            train_file=TRAIN_FILE,
            dev_file=DEV_FILE,
            eval_file=EVAL_FILE,
            input_dim=INPUT_DIM,
            project_out=PROJECT_OUT,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            num_workers=NUM_WORKERS,
            save_eval_npz_every=SAVE_EVAL_NPZ_EVERY,
            learning_rates=LEARNING_RATES,
            class_weights_list=CLASS_WEIGHTS_LIST,
            run_final_scoring=False,
        ))

    write_runs_report(all_runs)

