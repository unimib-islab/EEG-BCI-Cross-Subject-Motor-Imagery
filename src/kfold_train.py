import math, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from math import floor
import mne
import json
from pathlib import Path

from torchsummary import summary

from config import resolve

import os
import pandas as pd
from src.utils import apply_stats

mne.set_log_level("ERROR")




class Experiment:
    def __init__(self, name, seed, labels, mode, window_sec, n_folds, cfg):
        
        self.seed = seed
        self.name, self.labels, self.mode = name, labels, mode.upper()
        self.win_sec, self.n_folds = float(window_sec), int(n_folds)

        self.exp_config = cfg

        self.subjects = cfg.protocol.subjects

        self.results = []
        self.tr_loss = []
        self.tr_acc = []
        self.vl_loss = []
        self.vl_acc = []
        
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        print("Device in use:", self.device)

    def run_epoch(self, model, loader, optim_, criterion, train=True):
        model.train() if train else model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
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
                # return loss_sum/len(loader), correct/len(loader)

        return loss_sum/total, correct/total
    

    def run(self):
        
        print("\n" + "="*60)
        print(f" {self.name} | LABELS={self.labels} | MODE={self.mode} | "
              f"WIN={self.win_sec}s | FOLDS={self.n_folds}")
        print("="*60)
        
        subs = self.subjects.copy(); random.shuffle(subs)
        fold_size = math.ceil(len(subs)/self.n_folds)
        folds = [subs[i:i+fold_size] for i in range(0,len(subs),fold_size)]

        for k, test_ids in enumerate(folds, 1):

            save_name = f'stats_f{k}_{self.name}-b{len(self.exp_config.net_config.scales_kernels)}'

            train_ids = [s for i,f in enumerate(folds) if i!=k-1 for s in f]
            print(f"\n→ Fold {k}/{self.n_folds} | Train {train_ids} | Test {test_ids}")

            log_path = os.path.join(self.exp_config.save_config.save_path, self.exp_config.save_config.log_path, f'{save_name}.csv')

            ids = random.sample(train_ids, len(train_ids))
            train_ids, val_ids = ids[:int(floor(len(ids)*0.9))], ids[int(floor(len(ids))*0.9):]

            print(f"Train {len(train_ids)} - Val {len(val_ids)}")
            
            train_ds = resolve(self.exp_config.data)(train_ids,
                                                     self.labels,
                                                     self.mode,
                                                     self.win_sec,
                                                     self.exp_config,
                                                     self.seed)
            
            val_ds = resolve(self.exp_config.data)(val_ids,
                                                     self.labels,
                                                     self.mode,
                                                     self.win_sec,
                                                     self.exp_config,
                                                     self.seed)
            
            test_ds  = resolve(self.exp_config.data)(test_ids,
                                                     self.labels,
                                                     self.mode,
                                                     self.win_sec,
                                                     self.exp_config,
                                                     self.seed)

            mu,sigma = train_ds.channel_stats()
            data_stats = {
                          'mu': mu.cpu().numpy().tolist(), 
                          'sigma':sigma.cpu().numpy().tolist(),
                          'test_subjects': test_ids
                        }

            stats_path = os.path.join(self.exp_config.save_config.save_path, self.exp_config.save_config.stats_path, f'{save_name}.json')
            with open(stats_path, "w") as fp:
                json.dump(data_stats , fp) 

            train_ds.X = (train_ds.X - mu[None,:,None]) / sigma[None,:,None]
            val_ds.X = (val_ds.X - mu[None,:,None]) / sigma[None,:,None]
            test_ds.X  = (test_ds.X  - mu[None,:,None]) / sigma[None,:,None]

            train_ld = DataLoader(train_ds, self.exp_config.batch_size, shuffle=True,  drop_last=True)
            val_ld = DataLoader(val_ds, self.exp_config.batch_size, shuffle=True,  drop_last=True)
            test_ld  = DataLoader(test_ds,  self.exp_config.batch_size, shuffle=False, drop_last=False)

            model_class = resolve(self.exp_config.model)
            model = model_class(self.exp_config.protocol.n_channels,
                                 len(self.labels),
                                 int(self.win_sec*self.exp_config.protocol.sampling_frequency),
                                 self.exp_config.net_config,
                                 ).to(self.device)
            
            crit = resolve(self.exp_config.criterion)()


            optim_ = resolve(self.exp_config.optimizer)(model.parameters(), 
                                                        lr=self.exp_config.optimizer_config.lr,
                                                        weight_decay=self.exp_config.optimizer_config.w_decay)
            
            # ==================================================================
            # ==================================================================
            # ==================================================================

            print('=' * 15)
            print(f"Model: {model.__class__}")
            
            summary(model, train_ds.X[0].shape)
            
            print(f"Optimizer: {optim_}")
            print('=' * 15)

            # ==================================================================
            # ==================================================================
            # ==================================================================


            best_acc, test_best_acc, patience = 0.0, 0.0, 0
            tr_acc, vl_acc, tst_acc = [], [], []
            tr_loss, vl_loss, tst_loss = [], [], []          # <-- for optional plot

            for ep in range(1, self.exp_config.epochs+1):
                tl, ta = self.run_epoch(model, train_ld, optim_, crit, True)   # train-loss / acc
                vl, va = self.run_epoch(model, val_ld,  optim_, crit, False)  # val-loss / acc
                tstl, tsta = self.run_epoch(model, test_ld,  optim_, crit, False)  # val-loss / acc

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

                print(f"Ep {ep:02d}/{self.exp_config.epochs} | "
                      f"TrainL {tl:.4f} | TrainA {ta:.3f} | "
                      f"ValL {vl:.4f} | ValA {va:.3f} | "
                      f"TestL {tstl:.4f} | TestA {tsta:.3f} | "
                      f"BestA {test_best_acc:.3f} | "
                      f"{'↑' if improved else ' '} patience {patience}/{self.exp_config.early_stopping_config.patience}")


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
                if self.exp_config.early_stopping_config.enabled and patience >= self.exp_config.early_stopping_config.patience:
                    print("Early-stop."); break


            self.results.append(test_best_acc)

            self.tr_loss.extend(tr_loss)
            self.tr_acc.extend(tr_acc)
            self.vl_loss.extend(vl_loss)
            self.vl_acc.extend(vl_acc)

            torch.save(model.state_dict(), f"{self.exp_config}/{save_name}.pth" )
        
        
        mean, std = np.mean(self.results), np.std(self.results)
        print("\n" + "-"*60)
        print(f"{self.name} done | Best per fold {self.results}")
        print(f"Media {mean:.4f} ± {std:.4f}")
        print("-"*60)
        with open('./results.txt', 'a') as f:
            f.write("\n" + "-"*60)
            f.write(f'\n{save_name}')
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

        model_class = resolve(self.exp_config.model)
        dataset_class = resolve(self.exp_config.data)

        for idx in range(5):
            m_idx = idx+1
            save_name = f"stats_f{m_idx}_{self.name}-b{len(self.exp_config.net_config.scales_kernels)}"
            pretrained_model = f"model_f{m_idx}_{self.name}-b{len(self.exp_config.net_config.scales_kernels)}.pth"
            stats_path = Path(os.path.join(self.exp_config.save_config.save_path, self.exp_config.save_config.stats_path, f"{save_name}.json"))
            data_stats = json.loads(stats_path.read_text(encoding="utf-8"))
            
            subjects = data_stats["test_subjects"]

            print(f"\n{'#'*60}")
            print(f" MODEL {pretrained_model}")
            print(f"{'#'*60}")

            model_subject_accs = []

            model = model_class(self.exp_config.protocol.n_channels,
                                 len(self.labels),
                                 int(self.win_sec * self.exp_config.protocol.sampling_frequency),
                                 self.exp_config.net_config
                                 ).to(self.device)
            for subj in subjects:
                print(f"\n{'='*60}")
                print(f" MODEL Fold {m_idx} | SUBJECT {subj}")
                print(f"{'='*60}")

                full_ds = dataset_class([subj],
                                        self.labels,
                                        self.mode,
                                        self.win_sec,
                                        self.exp_config,
                                        self.seed)

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
                                        self.exp_config.batch_size, shuffle=True,  drop_last=True)
                    val_ld   = DataLoader(Subset(full_ds, val_idx),
                                        self.exp_config.batch_size, shuffle=False, drop_last=False)
                    test_ld  = DataLoader(Subset(full_ds, test_idx),
                                        self.exp_config.batch_size, shuffle=False, drop_last=False)

                    # Fresh load for each fold
                    model.load_state_dict(torch.load(os.path.join(
                                            self.exp_config.save_config.save_path,
                                            self.exp_config.save_config.model_save_path,
                                             pretrained_model), map_location=self.device))

                    crit   = resolve(self.exp_config.criterion)()
                    optim_ = resolve(self.exp_config.optimizer)(model.parameters(), 
                                         lr=self.exp_config.optimizer_config.lr,
                                         weight_decay=self.exp_config.optimizer_config.w_decay)

                    log_path = os.path.join(
                        self.exp_config.save_config.save_path, self.exp_config.save_config.log_path,
                        f'{save_name}.csv'
                    )

                    best_val_acc  = 0.0
                    best_test_acc = 0.0
                    patience      = 0

                    for ep in range(1, self.exp_config.epochs + 1):
                        tl, ta     = self.run_epoch(model, train_ld, optim_, crit, True)
                        vl, va     = self.run_epoch(model, val_ld,   optim_, crit, False)
                        tstl, tsta = self.run_epoch(model, test_ld,  optim_, crit, False)

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

                        print(f"  Ep {ep:02d}/{self.exp_config.epochs} | "
                            f"TrL {tl:.4f} TrA {ta:.3f} | "
                            f"VlL {vl:.4f} VlA {va:.3f} | "
                            f"TstL {tstl:.4f} TstA {tsta:.3f} | "
                            f"BestTestA {best_test_acc:.3f} | "
                            f"{'↑' if improved else ' '} pat {patience}/{self.exp_config.early_stopping_config.patience}")

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

                        if patience >= self.exp_config.early_stopping_config.patience:
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
            f.write(f"fine_tune_per_subject | {self.name}-b{len(self.exp_config.net_config.scales_kernels)}\n")
            for m_idx, accs in all_results.items():
                f.write(f"  Model {m_idx} | {accs} | media {np.mean(accs):.4f}\n")
            f.write(f"Global mean: {global_mean:.4f} ± {global_std:.4f}\n")
            f.write("="*60 + "\n")

        return all_results, global_mean, global_std
