# ================================================================
# EEG BCI - LOSO Training + Final Retrain
# - Seed reproducible
# - Disk cache of windows to speed up later runs
# - DataLoader safe on Windows and large datasets
# - LOSO on 109 subjects + retrain on all with median epoch
# ================================================================

import os
import json
import random
import numpy as np
from pathlib import Path
from collections import Counter
import platform
import gc
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.datasets import EEGMMIDBDataset32
from src.model import MultiBranchCNNLSTM


import mne
from mne.datasets import eegbci
from tqdm.auto import tqdm


TARGET_CHANNELS = [
    "Fp1","Fp2","AF3","AF4","F7","F3","Fz","F4","F8",
    "FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
    "CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","PO7","PO3","PO4","PO8","Oz"
]
N_CHANNELS          = len(TARGET_CHANNELS)  # 32
SAMPLING_FREQ       = 160
EVENT_DURATION_SEC  = 4.0
ALL_SUBJECTS        = list(range(1, 110))

# Architecture
N_BRANCHES            = 4
DEPTH_PER_BRANCH      = 2
START_KERNEL_SIZE     = 15
KERNEL_INCREMENT      = 2
LSTM_HIDDEN_SIZE      = 768
CLASSIFIER_HIDDEN_DIM = 384

# Training
MAX_EPOCHS_LOSO = 40
PATIENCE_LOSO   = 10
BATCH_SIZE      = 16
LEARNING_RATE   = 2.89e-4
WEIGHT_DECAY    = 5.82e-4

# DataLoader: max limits (tuned dynamically later)
NUM_WORKERS_MAX = max(2, min(8, (os.cpu_count() or 4) // 2))
PREFETCH_MAX    = 2
PERSISTENT_MAX  = True

# Pre-processing
L_FREQ, H_FREQ  = 0.0, 79.0
NOTCH_FREQS     = (50,)      # 50 Hz EU
APPLY_ICA       = False      # Optional ICA, disabled by default

# Run-to-task mapping for EEGBCI
RUNS_LR_REAL = [3, 7, 11]    # mano sinistra/destra — real
RUNS_LR_IMAG = [4, 8, 12]    # mano sinistra/destra — imagery
RUNS_BF_REAL = [5, 9, 13]    # piedi/lingua — real
RUNS_BF_IMAG = [6, 10, 14]   # piedi/lingua — imagery

EVENT_ID   = {"T0": 1, "T1": 2, "T2": 3}
LABELS_LR  = {1:"L", 2:"R", 0:"0"}
LABELS_BF  = {1:"B", 2:"F", 0:"0"}

# ------------------------- logging / warning -------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

os.environ["PYTHONHASHSEED"] = "0"

# TF32 precision on Ampere/Ada and AMP mixed precision
try:
    torch.set_float32_matmul_precision("high")  # enable TF32 fast path
except Exception:
    pass
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

RANDOM_STATE = 60
def set_seeds(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {DEVICE}")

# Disk cache (reused across runs)
CACHE_DIR = Path("./cache_eeg")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Save directory
SAVE_DIR = Path("./models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# # ---------- AMP compat layer (autocast & GradScaler) ----------
def _autocast(enabled: bool):
    """Return an autocast context manager compatible with PyTorch 1.x/2.x."""
    try:
        # PyTorch >= 2.x
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    except Exception:
        # PyTorch 1.x
        from torch.cuda.amp import autocast as cuda_autocast
        return cuda_autocast(enabled=enabled)

def make_grad_scaler(enabled: bool):
    """Create a GradScaler without device_type (works on PyTorch 1.x and 2.x)."""
    try:
        return torch.amp.GradScaler(enabled=enabled)
    except Exception:
        from torch.cuda.amp import GradScaler as CudaGradScaler
        return CudaGradScaler(enabled=enabled)


# ================================================================
# 5) TRAIN/VAL — mixed precision
# ================================================================
def run_epoch(model, loader, optim_, criterion, scaler, train=True):
    model.train() if train else model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    use_amp = (DEVICE.type == "cuda")
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            if train:
                optim_.zero_grad(set_to_none=True)
            with _autocast(enabled=use_amp):
                out = model(xb)  # (B, T, n_cls)
                B, T = out.shape[:2]
                # extend the epoch label across all T steps
                loss = criterion(out.reshape(B*T, -1), yb.repeat_interleave(T))
            if train:
                scaler.scale(loss).backward()
                scaler.step(optim_)
                scaler.update()
            # Accumulate metrics (mainly in eval)
            loss_sum += loss.item() * (B*T)
            correct  += (out.argmax(-1) == yb[:, None]).sum().item()
            total    += B*T
    return (loss_sum/total) if total>0 else 0.0, (correct/total) if total>0 else 0.0

# ================================================================
# 6) NORMALIZATION AND DATALOADER
# ================================================================
def standardize_dataset(train_ds, test_ds=None):
    """Standardize per-channel using training (mu, sigma)."""
    mu, sigma = train_ds.channel_stats()
    sigma = torch.where(sigma < 1e-8, torch.ones_like(sigma), sigma)
    train_ds.X = (train_ds.X - mu[None,:,None]) / (sigma[None,:,None] + 1e-8)
    if test_ds is not None and len(test_ds) > 0:
        test_ds.X  = (test_ds.X  - mu[None,:,None]) / (sigma[None,:,None] + 1e-8)
    return mu, sigma

def _estimate_bytes(tensor):
    try:
        return tensor.element_size() * tensor.nelement()
    except Exception:
        return 0

def _choose_loader_params(train_ds, is_retrain=False):
    """
    Choose num_workers/prefetch/persistent safely on Windows
    and/or dataset > ~1GB to avoid deadlocks/out-of-memory.
    """
    bytes_train = _estimate_bytes(train_ds.X)
    BIG_DS = bytes_train > (1 << 30)  # 1 GiB
    is_windows = (platform.system().lower() == "windows")

    if is_windows or BIG_DS:
        # more conservative: single-process (no workers)
        return 0, None, False
    else:
        return NUM_WORKERS_MAX, PREFETCH_MAX, PERSISTENT_MAX

def _make_generator(seed=RANDOM_STATE):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def _make_dataloader(ds, shuffle, drop_last, use_generator=False):
    """Create DataLoader without prefetch_factor when num_workers=0."""
    num_w, prefetch, persistent = _choose_loader_params(ds)
    kwargs = dict(
        dataset=ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_w,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
    )
    if num_w and prefetch:
        kwargs["prefetch_factor"] = prefetch
        kwargs["worker_init_fn"]  = lambda wid: (random.seed(RANDOM_STATE+wid),
                                                 np.random.seed(RANDOM_STATE+wid),
                                                 torch.manual_seed(RANDOM_STATE+wid))
    if use_generator:
        kwargs["generator"] = _make_generator(RANDOM_STATE)
    return DataLoader(**kwargs)

def build_loaders(train_ids, test_ids, labels, mode, window_sec):
    """Create train/test datasets and normalized DataLoaders."""
    train_ds = EEGMMIDBDataset32(train_ids, labels, mode, window_sec, SAMPLING_FREQ)
    test_ds  = EEGMMIDBDataset32(test_ids,  labels, mode, window_sec, SAMPLING_FREQ)
    if len(train_ds) == 0 or len(test_ds) == 0:
        return None, None, None, None
    standardize_dataset(train_ds, test_ds)
    train_ld = _make_dataloader(train_ds, shuffle=True,  drop_last=True,  use_generator=True)
    test_ld  = _make_dataloader(test_ds,  shuffle=False, drop_last=False, use_generator=False)
    return train_ld, test_ld, train_ds, test_ds

# ================================================================
# 7) TRAINING: LOSO and RETRAIN
# ================================================================
def make_optimizer(params):
    """AdamW (fused if available on CUDA)."""
    if DEVICE.type == "cuda":
        try:
            return optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True)
        except TypeError:
            pass
    return optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

def train_one_loso_fold(train_ld, test_ld, n_classes, win_samples):
    """One LOSO fold: early stopping on validation accuracy."""
    model  = MultiBranchCNNLSTM(n_classes, win_samples).to(DEVICE)
    crit   = nn.CrossEntropyLoss()
    opt    = make_optimizer(model.parameters())
    scaler = make_grad_scaler(enabled=(DEVICE.type == "cuda"))

    best_acc, best_epoch, patience = 0.0, 0, 0
    for ep in range(1, MAX_EPOCHS_LOSO + 1):
        # train
        run_epoch(model, train_ld, opt, crit, scaler, True)
        # val
        _, va = run_epoch(model, test_ld,  opt, crit, scaler, False)
        if va > best_acc + 1e-12:
            best_acc, best_epoch, patience = va, ep, 0
        else:
            patience += 1
        if patience >= PATIENCE_LOSO:
            break

    # cleanup before next fold
    del model, opt, scaler
    torch.cuda.empty_cache()
    gc.collect()
    return float(best_acc), int(best_epoch)

def retrain_final(all_train_ld, n_classes, win_samples, epochs_fixed, save_path):
    """Retrain on all subjects for a fixed number of epochs (LOSO median)."""
    model  = MultiBranchCNNLSTM(n_classes, win_samples).to(DEVICE)
    crit   = nn.CrossEntropyLoss()
    opt    = make_optimizer(model.parameters())
    scaler = make_grad_scaler(enabled=(DEVICE.type == "cuda"))

    for _ in tqdm(range(epochs_fixed), desc="Retrain (epochs)", leave=False):
        run_epoch(model, all_train_ld, opt, crit, scaler, True)

    torch.save(model.state_dict(), save_path)
    del model, opt, scaler
    torch.cuda.empty_cache()
    gc.collect()
    return save_path

def dataset_all_subjects(labels, mode, window_sec):
    """Build a single dataset across all subjects and standardize it on itself."""
    ds = EEGMMIDBDataset32(ALL_SUBJECTS, labels, mode, window_sec, SAMPLING_FREQ)
    if len(ds) == 0:
        raise RuntimeError("Dataset completo vuoto: controlla filtri/labels/mode.")
    mu, sigma = ds.channel_stats()
    sigma = torch.where(sigma < 1e-8, torch.ones_like(sigma), sigma)
    ds.X = (ds.X - mu[None,:,None]) / (sigma[None,:,None] + 1e-8)
    return ds, mu, sigma

# ================================================================
# 8) EXPERIMENTS (LOSO + RETRAIN) — default: window=1s, mode='I'
# ================================================================
MODE = "I"            # imagery
WINDOW_SEC = 1.0
WIN_SAMPLES = int(WINDOW_SEC * SAMPLING_FREQ)

EXPERIMENTS = [
    ("loso_5_class_1s_I", ["L","R","0","F","B"]),
    ("loso_4_class_1s_I", ["L","R","0","F"]),
    ("loso_3_class_1s_I", ["L","R","0"]),
]

def run_full_experiment(exp_name, labels, mode=MODE, window_sec=WINDOW_SEC):
    """
    Runs:
      1) LOSO on 109 subjects with early stopping
      2) Report mean±std accuracy + best-epoch distribution
      3) Save JSON per subject (epoch/acc)
      4) Final retrain on all subjects for the 'median epoch'
      5) Save model weights and normalization stats
    """
    print("\n" + "="*78)
    print(f"{exp_name} | LABELS={labels} | MODE={mode} | WIN={window_sec}s | LOSO 109 fold")
    print("="*78)

    accs = []
    best_epoch_by_subject = {}
    best_acc_by_subject   = {}

    for sid in tqdm(ALL_SUBJECTS, desc=f"LOSO ({len(labels)} classes)"):
        set_seeds()
        test_ids  = [sid]
        train_ids = [s for s in ALL_SUBJECTS if s != sid]

        build = build_loaders(train_ids, test_ids, labels, mode, window_sec)
        if build[0] is None:
            best_epoch_by_subject[sid] = None
            best_acc_by_subject[sid]   = None
            continue

        train_ld, test_ld, train_ds, test_ds = build
        best_acc, best_ep = train_one_loso_fold(
            train_ld, test_ld, n_classes=len(labels), win_samples=WIN_SAMPLES
        )
        accs.append(best_acc)
        best_epoch_by_subject[sid] = int(best_ep)
        best_acc_by_subject[sid]   = float(best_acc)

        # explicitly close loaders to free workers/memory
        del train_ld, test_ld, train_ds, test_ds
        gc.collect()

    if len(accs) == 0:
        raise RuntimeError("No valid folds completed. Check that runs/labels exist.")

    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
    valid_epochs = [e for e in best_epoch_by_subject.values() if isinstance(e, int)]
    median_epoch = int(np.rint(np.median(valid_epochs))) if valid_epochs else 1
    counts = dict(sorted(Counter(valid_epochs).items()))

    print(f"\n>> {exp_name} — LOSO Accuracy: {mean_acc:.4f} ± {std_acc:.4f} (su {len(accs)} fold validi)")
    print(">> Best-epoch distribution:", counts)
    print(f">> Rounded median epoch: {median_epoch}")

    # Save best-epoch and best-acc per subject
    json_epoch_path = SAVE_DIR / f"best_epoch_by_subject_{exp_name}.json"
    with open(json_epoch_path, "w") as f:
        json.dump(best_epoch_by_subject, f, indent=2)
    json_acc_path = SAVE_DIR / f"best_acc_by_subject_{exp_name}.json"
    with open(json_acc_path, "w") as f:
        json.dump(best_acc_by_subject, f, indent=2)
    print(f">> Saved: {json_epoch_path}")
    print(f">> Saved: {json_acc_path}")

    # Final retrain on all subjects for 'median_epoch'
    print("\n[RETRAIN] Final training on all 109 subjects "
          f"for {median_epoch} epochs, then save weights & stats...")
    all_ds, mu, sigma = dataset_all_subjects(labels, mode, window_sec)
    all_ld = _make_dataloader(all_ds, shuffle=True, drop_last=True, use_generator=True)

    save_path = SAVE_DIR / f"model_{exp_name}.pth"
    retrain_final(all_ld, n_classes=len(labels), win_samples=WIN_SAMPLES,
                  epochs_fixed=median_epoch, save_path=save_path)
    print(f">> Model saved to: {save_path}")

    # Save normalization stats (for inference)
    stats_path = SAVE_DIR / f"stats_{exp_name}.json"
    with open(stats_path, "w") as f:
        json.dump({"mu": mu.tolist(), "sigma": sigma.tolist(), "channels": TARGET_CHANNELS}, f, indent=2)
    print(f">> Stats saved to: {stats_path}")

    return {
        "exp": exp_name,
        "labels": labels,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "median_epoch": median_epoch,
        "best_epoch_by_subject_path": str(json_epoch_path),
        "best_acc_by_subject_path": str(json_acc_path),
        "final_model_path": str(save_path),
        "stats_path": str(stats_path),
        "valid_folds": len(accs)
    }



if __name__ == "__main__":
    summaries = []
    for exp_name, lbls in EXPERIMENTS:
        set_seeds()
        summaries.append(run_full_experiment(exp_name, lbls, mode=MODE, window_sec=WINDOW_SEC))

    print("\n===== RIEPILOGO =====")
    for s in summaries:
        print(f"{s['exp']:>20} | classes={len(s['labels'])} | LOSO Acc={s['mean_acc']:.4f}±{s['std_acc']:.4f} "
              f"| E_med={s['median_epoch']} | folds={s['valid_folds']}")
        print(f"  best-epoch JSON: {s['best_epoch_by_subject_path']}")
        print(f"  best-acc   JSON: {s['best_acc_by_subject_path']}")
        print(f"  final model : {s['final_model_path']}")
        print(f"  stats          : {s['stats_path']}")
