import math, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from math import floor
import mne
import matplotlib.pyplot as plt
import json
from pathlib import Path

import os
from src.model import MultiBranchCNNLSTM

import pandas as pd
from src.datasets import EEGMMIDBDataset as EEGMMIDBDataset
from src.utils import apply_stats

# ---------- reproducibility ----------

def set_seeds(seed=60):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

mne.set_log_level("ERROR")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in uso:", DEVICE)


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

def run_epoch(model, loader, optim_, criterion, train=True):
    model.train() if train else model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if train: optim_.zero_grad()
            out = model(xb)                     # (B,T,n_cls)
            B, T = out.shape[:2]
            loss = criterion(out.reshape(B*T,-1), yb.repeat_interleave(T))
            if train:
                loss.backward(); optim_.step()
            loss_sum += loss.item() * (B*T)
            correct  += (out.argmax(-1) == yb[:,None]).sum().item()
            total    += B*T
            # new v2
            # loss = criterion(out, yb)
            # if train:
            #     loss.backward(); optim_.step()
            # loss_sum += loss.item()
            # correct  += ((out.argmax(-1) == yb).sum().item())/out.shape[0]

            

    return loss_sum/len(loader), correct/len(loader)
    


class Experiment:
    def __init__(self, name, labels, mode, window_sec, n_folds, kernel_list):
        self.name, self.labels, self.mode = name, labels, mode.upper()
        self.win_sec, self.n_folds = float(window_sec), int(n_folds)
        self.seq_len = int(EVENT_DURATION_SEC / self.win_sec)
        self.results = []
        self.tr_loss = []
        self.tr_acc = []
        self.vl_loss = []
        self.vl_acc = []
        self.kernel_list = kernel_list

    def run(self, subjects):
        print("\n" + "="*60)
        print(f" {self.name} | LABELS={self.labels} | MODE={self.mode} | "
              f"WIN={self.win_sec}s | FOLDS={self.n_folds}")
        print("="*60)
        subs = subjects.copy(); random.shuffle(subs)
        fold_size = math.ceil(len(subs)/self.n_folds)
        folds = [subs[i:i+fold_size] for i in range(0,len(subs),fold_size)]

        for k, test_ids in enumerate(folds, 1):
            train_ids = [s for i,f in enumerate(folds) if i!=k-1 for s in f]
            print(f"\n→ Fold {k}/{self.n_folds} | Train {train_ids} | Test {test_ids}")

            log_path = os.path.join(LOG_PATH, f'model_f{k}_{self.name}-b{len(self.kernel_list)}.csv')

            ids = random.sample(train_ids, len(train_ids))
            train_ids, val_ids = ids[:int(floor(len(ids)*0.9))], ids[int(floor(len(ids))*0.9):]

            print(f"Train {len(train_ids)} - Val {len(val_ids)}")
            train_ds = EEGMMIDBDataset(train_ids, self.labels, self.mode,
                                       self.win_sec, SAMPLING_FREQ)
            val_ds = EEGMMIDBDataset(val_ids, self.labels, self.mode,
                                       self.win_sec, SAMPLING_FREQ)
            test_ds  = EEGMMIDBDataset(test_ids,  self.labels, self.mode,
                                       self.win_sec, SAMPLING_FREQ)

            mu,sigma = train_ds.channel_stats()
            data_stats = {
                          'mu': mu.cpu().numpy().tolist(), 
                          'sigma':sigma.cpu().numpy().tolist(),
                          'test_subjects': test_ids
                        }

            stats_path = os.path.join(STATS_PATH, f'stats_f{k}_{self.name}-b{len(self.kernel_list)}.json')
            with open(stats_path, "w") as fp:
                json.dump(data_stats , fp) 

            train_ds.X = (train_ds.X - mu[None,:,None]) / sigma[None,:,None]
            val_ds.X = (val_ds.X - mu[None,:,None]) / sigma[None,:,None]
            test_ds.X  = (test_ds.X  - mu[None,:,None]) / sigma[None,:,None]

            train_ld = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  drop_last=True)
            val_ld = DataLoader(val_ds, BATCH_SIZE, shuffle=True,  drop_last=True)
            test_ld  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, drop_last=False)

            model = MultiBranchCNNLSTM(len(self.labels),
                                       int(self.win_sec*SAMPLING_FREQ),
                                       kernel_list=self.kernel_list).to(DEVICE)
            crit = nn.CrossEntropyLoss()
            optim_ = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

            best_acc, test_best_acc, patience = 0.0, 0.0, 0
            tr_acc, vl_acc, tst_acc = [], [], []
            tr_loss, vl_loss, tst_loss = [], [], []          # <-- for optional plot

            for ep in range(1, MAX_EPOCHS+1):
                tl, ta = run_epoch(model, train_ld, optim_, crit, True)   # train-loss / acc
                vl, va = run_epoch(model, val_ld,  optim_, crit, False)  # val-loss / acc
                tstl, tsta = run_epoch(model, test_ld,  optim_, crit, False)  # val-loss / acc

                tr_loss.append(tl); tr_acc.append(ta)
                vl_loss.append(vl); vl_acc.append(va)
                tst_loss.append(tstl); tst_acc.append(tsta)

                improved = va > best_acc + 1e-4
                # best_acc = max(best_acc, va)
                if best_acc < va:
                    test_best_acc = tsta
                    best_acc = va
                # best_acc = va # using the accuracy of the last epoch for evaluation.
                patience  = 0 if improved else patience + 1

                print(f"Ep {ep:02d}/{MAX_EPOCHS} | "
                      f"TrainL {tl:.4f} | TrainA {ta:.3f} | "
                      f"ValL {vl:.4f} | ValA {va:.3f} | "
                      f"TestL {tstl:.4f} | TestA {tsta:.3f} | "
                      f"BestA {test_best_acc:.3f} | "
                      f"{'↑' if improved else ' '} patience {patience}/{PATIENCE}")


                # logging
                if os.path.exists(log_path):
                    df = pd.read_csv(log_path)
                else:
                    df = pd.DataFrame()

                new_df = pd.DataFrame({
                    'Train loss' : [tl],
                    'Train accuracy' : [ta],
                    'Validation loss' : [vl],
                    'Validation accuracy' : [va],
                    'Test loss': [tstl],
                    'Test accuracy': [tsta],
                })
                df = pd.concat([df, new_df], ignore_index=True)

                df.to_csv(log_path, index=False)
                if patience >= PATIENCE:
                    print("Early-stop."); break


            self.results.append(test_best_acc)

            self.tr_loss.extend(tr_loss)
            self.tr_acc.extend(tr_acc)
            self.vl_loss.extend(vl_loss)
            self.vl_acc.extend(vl_acc)

            torch.save(model.state_dict(), f"{SAVE_PATH}/model_f{k}_{self.name}-b{len(self.kernel_list)}.pth" )
        
        
        mean, std = np.mean(self.results), np.std(self.results)
        print("\n" + "-"*60)
        print(f"{self.name} done | Best per fold {self.results}")
        print(f"Media {mean:.4f} ± {std:.4f}")
        print("-"*60)
        with open('./results.txt', 'a') as f:
            f.write("\n" + "-"*60)
            f.write(f'\nmodel_f{k}_{self.name}-b{len(self.kernel_list)}')
            f.write(f"\n{self.name} done | Best per fold {self.results}")
            f.write(f"\nMedia {mean:.4f} ± {std:.4f}\n")
            # f.write("-"*60)

    def fine_tune_per_subject(self):
        """
        Per ogni modello preaddestrato (k-fold del training precedente):
        - per ogni soggetto in subjects:
            - esegue k-fold CV con quel modello come punto di partenza
            - traccia best test acc (al miglior val) per fold
        Restituisce un dict model -> lista best acc per soggetto, + media globale finale.
        """
        print("\n" + "="*60)
        print(f" {self.name} | LABELS={self.labels} | MODE={self.mode} | "
            f"WIN={self.win_sec}s | FOLDS={self.n_folds}")
        print("="*60)

        all_results = {}

        for idx in range(5):
            m_idx = idx+1
            pretrained_model = f"model_f{m_idx}_{self.name}-b{len(self.kernel_list)}.pth"
            stats_path = Path(os.path.join(STATS_PATH, f"stats_f{m_idx}_{self.name}-b{len(self.kernel_list)}.json"))
            data_stats = json.loads(stats_path.read_text(encoding="utf-8"))
            
            subjects = data_stats["test_subjects"]

            print(f"\n{'#'*60}")
            print(f" MODEL {pretrained_model}")
            print(f"{'#'*60}")

            model_subject_accs = []

            model = MultiBranchCNNLSTM(len(self.labels),
                                    int(self.win_sec * SAMPLING_FREQ),
                                    kernel_list=self.kernel_list).to(DEVICE)
            for subj in subjects:
                print(f"\n{'='*60}")
                print(f" MODEL Fold {m_idx} | SUBJECT {subj}")
                print(f"{'='*60}")

                full_ds = EEGMMIDBDataset([subj], self.labels, self.mode,
                                        self.win_sec, SAMPLING_FREQ)

                # normalize data based on fold stats
                full_ds.X = apply_stats(full_ds.X, data_stats)

                n_samples = len(full_ds)
                indices   = list(range(n_samples))
                random.shuffle(indices)

                fold_size = math.ceil(n_samples / self.n_folds)
                folds     = [indices[i:i+fold_size]
                            for i in range(0, n_samples, fold_size)]

                fold_best_accs = []

                for k, test_idx in enumerate(folds, 1):
                    train_val_idx = [idx for i, f in enumerate(folds)
                                    if i != k-1 for idx in f]

                    random.shuffle(train_val_idx)
                    split     = int(floor(len(train_val_idx) * 0.9))
                    train_idx = train_val_idx[:split]
                    val_idx   = train_val_idx[split:]

                    print(f"\n→ Fold {k}/{self.n_folds} | "
                        f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

                    train_ld = DataLoader(Subset(full_ds, train_idx),
                                        BATCH_SIZE, shuffle=True,  drop_last=True)
                    val_ld   = DataLoader(Subset(full_ds, val_idx),
                                        BATCH_SIZE, shuffle=False, drop_last=False)
                    test_ld  = DataLoader(Subset(full_ds, test_idx),
                                        BATCH_SIZE, shuffle=False, drop_last=False)

                    # Fresh load for each fold
                    model.load_state_dict(torch.load(os.path.join(MODEL_PATH,
                                             pretrained_model), map_location=DEVICE))

                    crit   = nn.CrossEntropyLoss()
                    optim_ = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                        weight_decay=WEIGHT_DECAY)

                    log_path = os.path.join(
                        LOG_PATH,
                        f'ft_m{m_idx}_subj{subj}_f{k}_{self.name}'
                        f'-b{len(self.kernel_list)}.csv'
                    )

                    best_val_acc  = 0.0
                    best_test_acc = 0.0
                    patience      = 0

                    for ep in range(1, MAX_EPOCHS + 1):
                        tl, ta     = run_epoch(model, train_ld, optim_, crit, True)
                        vl, va     = run_epoch(model, val_ld,   optim_, crit, False)
                        tstl, tsta = run_epoch(model, test_ld,  optim_, crit, False)

                        improved = va > best_val_acc + 1e-4
                        if improved:
                            best_val_acc  = va
                            best_test_acc = tsta
                            patience      = 0
                            # torch.save(
                            #     model.state_dict(),
                            #     f"{SAVE_PATH}/ft_m{m_idx}_subj{subj}_f{k}_"
                            #     f"{self.name}-b{len(self.kernel_list)}.pth"
                            # )
                        else:
                            patience += 1

                        print(f"  Ep {ep:02d}/{MAX_EPOCHS} | "
                            f"TrL {tl:.4f} TrA {ta:.3f} | "
                            f"VlL {vl:.4f} VlA {va:.3f} | "
                            f"TstL {tstl:.4f} TstA {tsta:.3f} | "
                            f"BestTestA {best_test_acc:.3f} | "
                            f"{'↑' if improved else ' '} pat {patience}/{PATIENCE}")

                        new_df = pd.DataFrame({
                            'epoch':               [ep],
                            'Train loss':          [tl],
                            'Train accuracy':      [ta],
                            'Validation loss':     [vl],
                            'Validation accuracy': [va],
                            'Test loss':           [tstl],
                            'Test accuracy':       [tsta],
                            'Best test accuracy':  [best_test_acc],
                        })
                        df = pd.read_csv(log_path) if os.path.exists(log_path) \
                            else pd.DataFrame()
                        pd.concat([df, new_df], ignore_index=True).to_csv(
                            log_path, index=False
                        )

                        if patience >= PATIENCE:
                            print("  Early-stop.")
                            break

                    print(f"  → Fold {k} best test acc: {best_test_acc:.4f}")
                    fold_best_accs.append(best_test_acc)

                subj_mean = float(np.mean(fold_best_accs))
                model_subject_accs.append(subj_mean)
                print(f"\n  Modello {m_idx} | Subject {subj} | "
                    f"fold accs: {fold_best_accs} | mean: {subj_mean:.4f}")

            all_results[m_idx] = model_subject_accs

            model_mean = float(np.mean(model_subject_accs))
            model_std  = float(np.std(model_subject_accs))
            print(f"\n  Model {m_idx} | acc per subject: {model_subject_accs}")
            print(f"  Model {m_idx} | average: {model_mean:.4f} ± {model_std:.4f}")

        # Media globale su tutti i modelli e tutti i soggetti
        all_accs    = [acc for accs in all_results.values() for acc in accs]
        global_mean = float(np.mean(all_accs))
        global_std  = float(np.std(all_accs))

        print("\n" + "="*60)
        print(f" FINE-TUNING COMPLETED")
        for m_idx, accs in all_results.items():
            print(f" Model {m_idx} | {accs} | average {np.mean(accs):.4f}")
        print(f" Gloabl mean: {global_mean:.4f} ± {global_std:.4f}")
        print("="*60)

        with open('./results_ft.txt', 'a') as f:
            f.write("\n" + "="*60 + "\n")
            f.write(f"fine_tune_per_subject | {self.name}-b{len(self.kernel_list)}\n")
            for m_idx, accs in all_results.items():
                f.write(f"  Model {m_idx} | {accs} | media {np.mean(accs):.4f}\n")
            f.write(f"Global mean: {global_mean:.4f} ± {global_std:.4f}\n")
            f.write("="*60 + "\n")

        return all_results, global_mean, global_std

if __name__ == "__main__":
    
    from src.experiments import *

    kernel_lists = [
        # [31],
        # [31,25],
        # [31,25,13],
        [31,25,13,7]
    ]
    sstl = False

    for name, labels, mode, win, folds in EXPERIMENTS_ABLATION:

        set_seeds()                     # reset RNG


        if sstl:
            from src.config_sstl import *
            Experiment(name, labels, mode, win, folds, [31,25,13,7]).fine_tune_per_subject()
        else:
            from configs.config import *
            # Experiment(name, labels, mode, win, folds, [31,25,13,7]).run(ALL_SUBJECTS)
            for klist in kernel_lists:
                Experiment(name, labels, mode, win, folds, klist).run(ALL_SUBJECTS)
