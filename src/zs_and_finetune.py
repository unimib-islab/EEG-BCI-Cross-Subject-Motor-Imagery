from pathlib import Path

OUT_DIR    = Path(r"./data_conversion/converted_dataset_EEG_RAW_V1/")   # contains *_EEG_RAW_V1.hdf5
MODELS_DIR = Path(r"./models")           # contains *.pth weights and *.json stats

import os, json, math, random, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)

import h5py, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from src.model import MultiBranchCNNLSTM

# ======= MODEL CONSTANTS (match training) =======
TARGET_CHANNELS = [
    "Fp1","Fp2","AF3","AF4","F7","F3","Fz","F4","F8",
    "FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
    "CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8",
    "PO7","PO3","PO4","PO8","Oz"
]
N_CHANNELS = 32
N_BRANCHES = 4
DEPTH_PER_BRANCH = 2
START_KERNEL_SIZE = 15
KERNEL_INCREMENT  = 2
LSTM_HIDDEN_SIZE  = 768
CLASSIFIER_HIDDEN = 384
SFREQ_TARGET = 160.0
WIN_SEC      = 1.0
T_WINDOWS    = 4
L_SAMPLES    = int(WIN_SEC * SFREQ_TARGET)   # 160
EVENT_SAMPLES= T_WINDOWS * L_SAMPLES         # 640

# ======= AVAILABLE MODELS =======
MODEL_INDEX = {
    5: {"labels": ["L","R","0","F","B"],
        "weights": MODELS_DIR / "model_loso_5_class_1s_I.pth",
        "stats":   MODELS_DIR / "stats_loso_5_class_1s_I.json",
        "name":    "loso_5_class_1s_I"},
    4: {"labels": ["L","R","0","F"],
        "weights": MODELS_DIR / "model_loso_4_class_1s_I.pth",
        "stats":   MODELS_DIR / "stats_loso_4_class_1s_I.json",
        "name":    "loso_4_class_1s_I"},
    3: {"labels": ["L","R","0"],
        "weights": MODELS_DIR / "model_loso_3_class_1s_I.pth",
        "stats":   MODELS_DIR / "stats_loso_3_class_1s_I.json",
        "name":    "loso_3_class_1s_I"},
}

from typing import Tuple

def _to_str_list(a):
    """Converte array di bytes in lista di stringhe (necessario per ch_names)."""
    return [x if isinstance(x, str) else x.decode() for x in a]

def _safe_torch_load(p: Path):
    """Load state_dict on CPU; accepts both 'weights_only' and full checkpoints."""
    try:
        return torch.load(str(p), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(p), map_location="cpu")

def build_wins_from_raw(h5_path: Path, label_set: set) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read an *_EEG_RAW_V1.hdf5 file and build windows:
      X: (N_eventi, T_WINDOWS=4, C=32, L=160) in microvolt
      y: array di label (stringhe)
    Filters events for label_set and requires at least 640 consecutive samples.
    """
    with h5py.File(h5_path, "r") as f:
        if "/meta/version" not in f or (f["/meta/version"][()].decode() if hasattr(f["/meta/version"][()], "decode") else f["/meta/version"][()]) != "EEG_RAW_V1":
            raise RuntimeError("meta/version non EEG_RAW_V1.")
        data = f["/raw/data"][:]                     # (C, S)
        sfreq0 = float(f["/raw/sfreq"][()])
        ch_names = _to_str_list(f["/raw/ch_names"][:])
        units = f["/raw/units"][()]
        units = units if isinstance(units, str) else units.decode()
        ev = f["/events/table"]
        onset = ev["onset_sample"][:].astype(np.int64)
        dur   = ev["duration_samples"][:].astype(np.int64)
        labels= np.array([l if isinstance(l,str) else l.decode() for l in ev["label"][:]], dtype="<U1")

    if units.lower() not in {"uv","microvolt","microvolts"}:
        raise RuntimeError("Atteso /raw/units='uV'")
    if abs(sfreq0 - SFREQ_TARGET) > 1e-6:
        raise RuntimeError(f"Expected sfreq=160 Hz, found {sfreq0}.")

    idx = {n:i for i,n in enumerate(ch_names)}
    miss = [c for c in TARGET_CHANNELS if c not in idx]
    if miss:
        raise RuntimeError(f"Canali mancanti: {miss}")
    data = data[[idx[c] for c in TARGET_CHANNELS], :]
    _, S0 = data.shape

    wins, lbls = [], []
    for o,d,lb in zip(onset,dur,labels):
        if lb not in label_set: 
            continue
        if d < EVENT_SAMPLES or (o + EVENT_SAMPLES) > S0:
            continue
        seg = data[:, o:o+EVENT_SAMPLES]                   # (32,640)
        seg = seg.reshape(32, T_WINDOWS, L_SAMPLES).transpose(1,0,2)  # -> (4,32,160)
        wins.append(seg)
        lbls.append(lb)

    if not wins:
        raise RuntimeError("No valid events.")
    return np.stack(wins).astype(np.float32), np.array(lbls, dtype="<U1")


class EEGWinSet(Dataset):
    """PyTorch wrapper for pre-sliced EEG windows."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N,4,32,160)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): 
        return self.X.shape[0]
    def __getitem__(self, i): 
        return self.X[i], self.y[i]

# class ConvBlock(nn.Module):
#     def __init__(self, k):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv1d(N_CHANNELS, N_CHANNELS, k, padding="same"),
#             nn.GroupNorm(8, N_CHANNELS), nn.ReLU(),
#             nn.Conv1d(N_CHANNELS, N_CHANNELS, k, padding="same"),
#             nn.GroupNorm(8, N_CHANNELS), nn.ReLU(),
#             nn.MaxPool1d(2),
#         )
#     def forward(self, x): 
#         return self.block(x)

# class MultiBranchCNNLSTM(nn.Module):
#     """
#     4 1D CNN branches (increasing kernels) per window; then LSTM over time (T=4) and an MLP classifier.
#     Output: logits shape (B, T, n_classes)
#     """
#     def __init__(self, n_classes: int, win_samples: int = 160):
#         super().__init__()
#         self.branches = nn.ModuleList()
#         self.lstms    = nn.ModuleList()
#         for b in range(N_BRANCHES):
#             k = START_KERNEL_SIZE + b*KERNEL_INCREMENT
#             cnn = nn.Sequential(*[ConvBlock(k) for _ in range(DEPTH_PER_BRANCH)])
#             self.branches.append(cnn)
#             with torch.no_grad():
#                 flat = cnn(torch.randn(1, N_CHANNELS, win_samples)).numel()
#             self.lstms.append(nn.LSTM(flat, LSTM_HIDDEN_SIZE, batch_first=True))

#         self.classifier = nn.Sequential(
#             nn.Linear(N_BRANCHES*LSTM_HIDDEN_SIZE, CLASSIFIER_HIDDEN),
#             nn.BatchNorm1d(CLASSIFIER_HIDDEN), nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(CLASSIFIER_HIDDEN, n_classes),
#         )

#     def forward(self, x):  # x: (B,T,C,L)
#         B,T,_,L = x.shape
#         seqs=[]
#         for cnn,lstm in zip(self.branches,self.lstms):
#             feat = cnn(x.reshape(B*T, N_CHANNELS, L)).reshape(B, T, -1)
#             out,_ = lstm(feat)  # (B,T,hidden)
#             seqs.append(out)
#         h = torch.cat(seqs, dim=2)            # (B,T, 4*hidden)
#         logits = self.classifier(h.reshape(B*T, -1)).view(B, T, -1)
#         return logits                         # (B,T,n_cls)

def apply_stats(X: torch.Tensor, stats_path: Path) -> torch.Tensor:
    st = json.loads(stats_path.read_text(encoding="utf-8"))
    mu = torch.tensor(st["mu"], dtype=torch.float32)
    sd = torch.tensor(st["sigma"], dtype=torch.float32)
    sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
    # import ipdb; ipdb.set_trace()
    # broadcasting: (B,T,C,L) -> subtract per-channel mu and divide by sigma
    return (X - mu[None, None, :, None]) / (sd[None, None, :, None] + 1e-8)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb).mean(dim=1)             # (B, n_cls)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total   += yb.numel()
    return (correct / total) if total > 0 else float("nan")


def plot_learning_curve(history, baseline, title=""):
    """
    Test accuracy per-epoch plot.
    history[0] = baseline (zero-shot); history[i] = test accuracy after epoch i (i>=1).
    """
    xs = list(range(len(history)))  # 0..epochs
    plt.figure()
    plt.plot(xs, history, marker="o", label="Test accuracy per epoch")
    plt.axhline(baseline, linestyle="--", label="Baseline 0-shot")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def finetune_one(
    h5_path: Path, 
    n_classes: int, 
    device: torch.device,
    epochs: int = 10, 
    batch_size: int = 64, 
    lr_body: float = 1e-4, 
    lr_cls: float = 5e-4
):
    """
    1) Build dataset from the hdf5 file (filter labels compatible with the chosen model)
    2) Normalize with the official model stats
    3) Temporal split 20% train / 80% test
    4) Load model + weights
    5) Compute zero-shot baseline, then train for N epochs keeping the best test accuracy
    Ritorna dict con baseline_acc, best_acc, history
    """
    entry = MODEL_INDEX[n_classes]
    labels = entry["labels"]
    weights= entry["weights"]
    stats  = entry["stats"]
    if not weights.exists() or not stats.exists():
        raise FileNotFoundError(f"Missing weights/stats for {n_classes} classes: {weights.name} / {stats.name}")

    # 1) dataset
    X_np, lab_np = build_wins_from_raw(h5_path, set(labels))
    idx_map = {l:i for i,l in enumerate(labels)}              # mappa label->indice
    y_np = np.array([idx_map[l] for l in lab_np], dtype=np.int64)

    # 2) normalization
    X = torch.tensor(X_np, dtype=torch.float32)
    X = apply_stats(X, stats)

    # 3) temporal 20/80 split (first 20% = train)
    N = X.shape[0]
    n_tr = max(1, int(round(0.2 * N)))
    X_tr, y_tr = X[:n_tr], torch.tensor(y_np[:n_tr])
    X_te, y_te = X[n_tr:], torch.tensor(y_np[n_tr:])
    if X_te.shape[0] == 0:
        return {"baseline_acc": float("nan"), "best_acc": float("nan"), "history": []}

    ds_tr = EEGWinSet(X_tr.numpy(), y_tr.numpy())
    ds_te = EEGWinSet(X_te.numpy(), y_te.numpy())
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, drop_last=False)

    # 4) model + weights
    model = MultiBranchCNNLSTM(n_classes=n_classes, win_samples=L_SAMPLES).to(device)
    sd = _safe_torch_load(weights)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] state_dict mismatch — missing={len(missing)}, unexpected={len(unexpected)}")

    # optimizer (higher LR for classifier)
    body_params, cls_params = [], []
    for n,p in model.named_parameters():
        (cls_params if "classifier" in n else body_params).append(p)
    opt = torch.optim.Adam([
        {"params": body_params, "lr": lr_body},
        {"params": cls_params,  "lr": lr_cls},
    ])
    crit = nn.CrossEntropyLoss()

    # 5) baseline + short training
    baseline = evaluate(model, dl_te, device)   # accuracy 0-shot
    best_acc = baseline
    history  = [baseline]                       # index 0 = baseline

    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).mean(dim=1)      # (B, n_cls)
            loss = crit(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        acc_te = evaluate(model, dl_te, device)
        history.append(acc_te)
        if acc_te > best_acc:
            best_acc = acc_te

        print(f"[{h5_path.stem} | {n_classes}-cls] epoch {ep:02d}/{epochs} — test_acc={acc_te:.4f} (best={best_acc:.4f})")

    return {"baseline_acc": float(baseline), "best_acc": float(best_acc), "history": [float(x) for x in history]}


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = sorted([p for p in OUT_DIR.glob("*_EEG_RAW_V1.hdf5")])
    print("[INFO] Found EEG_RAW_V1 files:", ", ".join(p.stem for p in files))

    rows = []
    for p in files:
        row = {"name": p.stem}

        for ncls in (3,4,5):
            try:
                res = finetune_one(
                    p, n_classes=ncls, device=device,
                    epochs=10, batch_size=64, lr_body=1e-4, lr_cls=5e-4
                )
                # plot learning curve (baseline + epochs)
                title = f"{p.stem} — {ncls} classes"
                plot_learning_curve(res["history"], res["baseline_acc"], title=title)

                # fill the table
                row[f"{ncls}_classes_zero_shot"] = res["baseline_acc"]
                row[f"{ncls}_classes_finetune"]  = res["best_acc"]
                if np.isfinite(res["baseline_acc"]) and np.isfinite(res["best_acc"]):
                    row[f"{ncls}_classes_delta_pt"] = 100.0*(res["best_acc"] - res["baseline_acc"])
                else:
                    row[f"{ncls}_classes_delta_pt"] = np.nan

            except Exception as e:
                print(f"[ERR ] {p.name} ({ncls}-classes): {e}")
                row[f"{ncls}_classes_zero_shot"] = np.nan
                row[f"{ncls}_classes_finetune"]  = np.nan
                row[f"{ncls}_classes_delta_pt"]  = np.nan

        rows.append(row)

    df_ft = pd.DataFrame(rows).sort_values("name").reset_index(drop=True)
    df_ft.to_csv('./finetune_results.csv')
