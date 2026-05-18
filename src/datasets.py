import numpy as np
import torch
from torch.utils.data import Dataset
import mne
from mne.datasets import eegbci
import tqdm
from pathlib import Path

# Disk cache (reused across runs)
CACHE_DIR = Path("./cache_eeg")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Save directory
SAVE_DIR = Path("./models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_raw(raw: mne.io.BaseRaw, target_sfreq: int, cfg, seed):
    """
    Pre-processing pipeline:
    1.  Notch (60 Hz)                 – rimuove rumore di rete
    2.  Resample a 160 Hz             – unifica tutti i run (alcuni sono a 128 Hz)
    3.  Band-pass 0-79 Hz
    4.  (opt.) ICA                    ? reproducible with RANDOM_STATE
    """
    r = raw.copy()

    # notch first: robust to any sampling frequency
    if cfg.notch_frequency:
        r.notch_filter(list(cfg.notch_frequency), verbose=False)

    # resample BEFORE filtering, if needed
    if r.info["sfreq"] != target_sfreq:
        r.resample(target_sfreq, verbose=False)

    # band-pass 0-79 Hz
    r.filter(cfg.low_frequency, cfg.high_frequency, verbose=False)

    # optional ICA
    if cfg.apply_ica:
        ica = mne.preprocessing.ICA(
            n_components=0.99,
            random_state=seed,
            max_iter="auto",
        )
        ica.fit(r, verbose=False)
        r = ica.apply(r.copy(), verbose=False)

    return r


def select_runs(desired_labels, mode):
    mode = mode.upper()
    inc_r, inc_i = mode in {"R","B"}, mode in {"I","B"}
    runs = []
    if any(l in ("L","R") for l in desired_labels):
        runs += [3, 7, 11] if inc_r else []
        runs += [4, 8, 12] if inc_i else []
    if any(l in ("B","F") for l in desired_labels):
        runs += [5, 9, 13] if inc_r else []
        runs += [6, 10, 14] if inc_i else []
    return sorted(set(runs))

def label_from_event(run, trig):
    if trig == 1: return "0"
    mapping = {1:"L", 2:"R", 0:"0"} if run in [3, 7, 11] + [4, 8, 12] else {1:"B", 2:"F", 0:"0"}
    return mapping[trig-1]

class EEGMMIDBDataset(Dataset):
    """
    Restituisce tensor (Macro, C, L) → poi (B, T, C, L) nel DataLoader.
    DEBUG: count macro and sub-windows.
    """
    def __init__(self, subjects, labels, mode, window_sec, cfg, seed):
        super().__init__()

        self.exp_config = cfg
        self.seed = seed
        self.label_list   = [l.upper() for l in labels]
        self.label_to_idx = {l:i for i,l in enumerate(self.label_list)}
        self.seq_len      = int(self.exp_config.protocol.event_duration_sec
                                 / window_sec)
        
        self.sfreq = self.exp_config.protocol.sampling_frequency

        self.n_channels = self.exp_config.protocol.n_channels

        self.win_samples  = int(window_sec * self.sfreq)
        
        self.exp_samples  = self.seq_len * self.win_samples

        self.debug = {"subjects": list(subjects)}

        self.X, self.y = self._build(subjects, mode, self.sfreq)

        self._collect_debug()

    def _build(self, subjects, mode, sfreq):
        wins, lbls = [], []
        runs = select_runs(self.label_list, mode)
        self.debug["runs"] = runs
        for sid in subjects:
            for run in runs:
                try:
                    paths = eegbci.load_data(sid, [run], path='~/Data/EEG/eegmmidb/1.0.0', verbose=False)
                except Exception:
                    continue
                raw = mne.io.read_raw_edf(paths[0], preload=True,
                                          stim_channel="auto", verbose=False)
                raw.pick_types(eeg=True).set_montage("standard_1005",
                                                     on_missing="ignore")
                raw = preprocess_raw(raw, 
                                     self.sfreq,
                                     self.exp_config.data_preprocessing,
                                     self.seed
                                     )
                events, _ = mne.events_from_annotations(raw,
                                                         event_id={"T0": 1, "T1": 2, "T2": 3},
                                                         verbose=False)
                if events.size == 0: continue

                epochs = mne.Epochs(raw, 
                                    events, 
                                    tmin=0.0, 
                                    tmax=self.exp_config.protocol.event_duration_sec,
                                    baseline=None, preload=True, verbose=False)
                data = epochs.get_data(units="uV")               # (E,C,T)

                for ep, ev in zip(data, epochs.events):
                    lbl = label_from_event(run, ev[-1])
                    if lbl not in self.label_to_idx: continue
                    if ep.shape[-1] < self.exp_samples: continue

                    ep = ep[..., :self.exp_samples]
                    ep = ep.reshape(self.n_channels, self.seq_len, self.win_samples)
                    wins.append(ep.transpose(1,0,2))             # (T,C,L)
                    lbls.append(self.label_to_idx[lbl])

        X = torch.tensor(np.stack(wins), dtype=torch.float32)
        y = torch.tensor(lbls,         dtype=torch.long)
        return X, y

    def _collect_debug(self):
        macro = len(self.y)
        sub   = macro * self.seq_len
        counts = {l:int((self.y==idx).sum()) for l,idx in self.label_to_idx.items()}
        self.debug.update({"n_macro_windows": macro,
                           "n_sub_windows"  : sub,
                           "label_counts"   : counts})
        print(f"[DEBUG] Runs {self.debug['runs']} | Macro {macro} | Sub {sub} | per class {counts}")

    def __len__(self):  return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]
    def channel_stats(self):
        d = self.X.numpy().reshape(-1, self.n_channels, self.win_samples)
        return torch.tensor(d.mean((0,2))), torch.tensor(d.std((0,2)))

class EEGMMIDBDatasetRetrain(Dataset):
    """
    Extract EEGBCI epochs, convert each epoch into window sequences (T, C, L).
    Optional disk cache to avoid redoing parsing/filtering each run.
    """
    def __init__(self, subjects, labels, mode, window_sec, cfg):
        super().__init__()

        self.exp_config = cfg

        self.label_list   = [l.upper() for l in labels]
        self.label_to_idx = {l:i for i,l in enumerate(self.label_list)}
        self.seq_len      = int(self.exp_config.protocol.event_duration_sec
                                 / float(window_sec))  # windows per epoch
        self.sfreq        = self.exp_config.protocol.sampling_frequency
        
        self.win_samples  = int(float(window_sec) * self.sfreq)               # samples per window
        self.exp_samples  = self.seq_len * self.win_samples
        self.mode         = mode
        self.window_sec   = window_sec
        self.X, self.y    = self._build(subjects)

    def _pick_32(self, raw):
        """Align and select the 32 target channels in the required order."""
        eegbci.standardize(raw)
        raw.set_montage("standard_1005", on_missing="ignore", verbose=False)
        raw.pick_types(eeg=True, exclude=[])
        present = raw.info["ch_names"]
        picks = mne.pick_channels(present,
                                  include=self.exp_config.target_channels,
                                  ordered=True)
        if len(picks) != len(self.exp_config.target_channels):
            missing = [c for c in self.exp_config.target_channels if c not in present]
            raise RuntimeError(f"Missing channels: {missing}")
        raw.pick(picks)
        raw.reorder_channels(self.exp_config.target_channels)
        return raw

    def _cache_path(self, sid, run):
        name = f"s{sid:03d}_r{run:02d}_m{self.mode.upper()}_w{self.window_sec:.2f}_sf{self.sfreq}_T{self.seq_len}_L{self.win_samples}.npz"
        return CACHE_DIR / name

    def _load_or_make_windows(self, sid, run):
        """Load from cache or build (and save) windows for (subject, run)."""
        cp = self._cache_path(sid, run)
        if cp.exists():
            try:
                dat = np.load(cp, allow_pickle=False)
                return dat["wins"], dat["lbls"]  # wins: (N,T,C,L), lbls: array di str
            except Exception:
                pass

        # Download/read EDF, align channels, preprocess
        try:
            paths = eegbci.load_data(sid, [run], verbose=False)
            raw = mne.io.read_raw_edf(paths[0], preload=True, stim_channel="auto", verbose=False)
            raw = self._pick_32(raw)
            raw = preprocess_raw(raw, 
                                  self.sfreq,
                                  self.exp_config.data_preprocessing,
                                  self.seed
                                  )
        except Exception:
            return None, None

        # Epochs with T0/T1/T2 events
        events, _ = mne.events_from_annotations(raw,
                                                event_id={"T0": 1, "T1": 2, "T2": 3},
                                                verbose=False)
        if events.size == 0:
            return None, None

        epochs = mne.Epochs(raw, events,
                            tmin=0.0,
                            tmax=self.exp_config.protocol.event_duration_sec,
                            baseline=None, preload=True, verbose=False)
        data = epochs.get_data(units="uV")  # (E, 32, T)

        wins, lbls = [], []
        for ep, ev in zip(data, epochs.events):
            trig = ev[-1]
            lbl = label_from_event(run, trig)  # "L","R","B","F","0"
            if ep.shape[-1] < self.exp_samples:
                continue
            ep = ep[..., :self.exp_samples]
            # (32, T*L) -> (T, C, L)
            ep = ep.reshape(self.n_channels, self.seq_len, self.win_samples).transpose(1,0,2)
            wins.append(ep)
            lbls.append(lbl)

        if not wins:
            return None, None

        wins = np.stack(wins)               # (N,T,C,L)
        lbls = np.array(lbls, dtype="U1")   # single-character strings

        try:
            np.savez_compressed(cp, wins=wins, lbls=lbls)
        except Exception:
            pass
        return wins, lbls

    def _build(self, subjects):
        """Build tensors X,y by concatenating all subject runs/epochs."""
        all_wins, all_lbls = [], []
        runs = select_runs(self.label_list, self.mode)
        for sid in subjects:
            for run in runs:
                wins, lbls = self._load_or_make_windows(sid, run)
                if wins is None:
                    continue
                mask = np.isin(lbls, self.label_list)
                if not mask.any():
                    continue
                all_wins.append(wins[mask])
                mapped = np.vectorize({l:i for i,l in enumerate(self.label_list)}.get)(lbls[mask])
                all_lbls.append(mapped.astype(np.int64))

        if not all_wins:
            # empty dataset (return empty compatible tensors)
            return (torch.empty((0, self.seq_len, self.n_channels, self.win_samples), dtype=torch.float32),
                    torch.empty((0,), dtype=torch.long))

        X = torch.tensor(np.concatenate(all_wins, axis=0), dtype=torch.float32)
        y = torch.tensor(np.concatenate(all_lbls, axis=0), dtype=torch.long)
        return X, y

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

    def channel_stats(self):
        """Per-channel mean and std (used for standardization)."""
        if len(self) == 0:
            return torch.zeros(self.n_channels), torch.ones(self.n_channels)
        d = self.X.numpy().reshape(-1, self.n_channels, self.win_samples)
        mu = d.mean((0,2))
        sd = d.std((0,2))
        sd = np.where(sd < 1e-8, 1.0, sd)
        return torch.tensor(mu, dtype=torch.float32), torch.tensor(sd, dtype=torch.float32)


class EEGMMIDBDataset32(Dataset):
    """
    Extract EEGBCI epochs, convert each epoch into window sequences (T, C, L).
    Optional disk cache to avoid redoing parsing/filtering each run.
    """
    def __init__(self, subjects, labels, mode, window_sec, cfg):
        super().__init__()

        self.exp_config = cfg
        self.label_list   = [l.upper() for l in labels]
        self.label_to_idx = {l:i for i,l in enumerate(self.label_list)}
        self.seq_len      = int(self.exp_config.protocol.event_duration_sec
                                / float(window_sec))  # windows per epoch
        self.sfreq        = cfg.protocol.sampling_frequency
        self.win_samples  = int(float(window_sec) * self.sfreq)               # samples per window
        self.exp_samples  = self.seq_len * self.win_samples
        self.mode         = mode
        self.window_sec   = window_sec
        self.n_channels   = self.exp_config.protocol.n_channels
        self.X, self.y    = self._build(subjects)

    def _pick_32(self, raw):
        """Align and select the 32 target channels in the required order."""
        eegbci.standardize(raw)
        raw.set_montage("standard_1005", on_missing="ignore", verbose=False)
        raw.pick_types(eeg=True, exclude=[])
        present = raw.info["ch_names"]
        picks = mne.pick_channels(present, include=self.exp_config.target_channels, ordered=True)
        if len(picks) != len(self.exp_config.target_channels):
            missing = [c for c in self.exp_config.target_channels if c not in present]
            raise RuntimeError(f"Missing channels: {missing}")
        raw.pick(picks)
        raw.reorder_channels(self.exp_config.target_channels)
        return raw

    def _cache_path(self, sid, run):
        name = f"s{sid:03d}_r{run:02d}_m{self.mode.upper()}_w{self.window_sec:.2f}_sf{self.sfreq}_T{self.seq_len}_L{self.win_samples}.npz"
        return CACHE_DIR / name

    def _load_or_make_windows(self, sid, run):
        """Load from cache or build (and save) windows for (subject, run)."""
        cp = self._cache_path(sid, run)
        if cp.exists():
            try:
                dat = np.load(cp, allow_pickle=False)
                return dat["wins"], dat["lbls"]  # wins: (N,T,C,L), lbls: array di str
            except Exception:
                pass

        # Download/read EDF, align channels, preprocess
        try:
            paths = eegbci.load_data(sid, [run], verbose=False)
            raw = mne.io.read_raw_edf(paths[0], preload=True, stim_channel="auto", verbose=False)
            raw = self._pick_32(raw)
            raw = preprocess_raw(raw, 
                                     self.sfreq,
                                     self.exp_config.data_preprocessing,
                                     self.seed
                                     )
        except Exception:
            return None, None

        # Epochs with T0/T1/T2 events
        events, _ = mne.events_from_annotations(raw, 
                                                event_id={"T0": 1, "T1": 2, "T2": 3}, 
                                                verbose=False)
        if events.size == 0:
            return None, None

        epochs = mne.Epochs(raw, 
                            events, 
                            tmin=0.0,
                            tmax=self.exp_config.protocol.event_duration_sec,
                            baseline=None, preload=True, verbose=False)
        data = epochs.get_data(units="uV")  # (E, 32, T)

        wins, lbls = [], []
        for ep, ev in zip(data, epochs.events):
            trig = ev[-1]
            lbl = label_from_event(run, trig)  # "L","R","B","F","0"
            if ep.shape[-1] < self.exp_samples:
                continue
            ep = ep[..., :self.exp_samples]
            # (32, T*L) -> (T, C, L)
            ep = ep.reshape(self.n_channels, self.seq_len, self.win_samples).transpose(1,0,2)
            wins.append(ep)
            lbls.append(lbl)

        if not wins:
            return None, None

        wins = np.stack(wins)               # (N,T,C,L)
        lbls = np.array(lbls, dtype="U1")   # single-character strings

        try:
            np.savez_compressed(cp, wins=wins, lbls=lbls)
        except Exception:
            pass
        return wins, lbls

    def _build(self, subjects):
        """Build tensors X,y by concatenating all subject runs/epochs."""
        all_wins, all_lbls = [], []
        runs = select_runs(self.label_list, self.mode)
        for sid in subjects:
            for run in runs:
                wins, lbls = self._load_or_make_windows(sid, run)
                if wins is None:
                    continue
                mask = np.isin(lbls, self.label_list)
                if not mask.any():
                    continue
                all_wins.append(wins[mask])
                mapped = np.vectorize({l:i for i,l in enumerate(self.label_list)}.get)(lbls[mask])
                all_lbls.append(mapped.astype(np.int64))

        if not all_wins:
            # empty dataset (return empty compatible tensors)
            return (torch.empty((0, self.seq_len, self.n_channels, self.win_samples), dtype=torch.float32),
                    torch.empty((0,), dtype=torch.long))

        X = torch.tensor(np.concatenate(all_wins, axis=0), dtype=torch.float32)
        y = torch.tensor(np.concatenate(all_lbls, axis=0), dtype=torch.long)
        return X, y

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

    def channel_stats(self):
        """Per-channel mean and std (used for standardization)."""
        if len(self) == 0:
            return torch.zeros(self.n_channels), torch.ones(self.n_channels)
        d = self.X.numpy().reshape(-1, self.n_channels, self.win_samples)
        mu = d.mean((0,2))
        sd = d.std((0,2))
        sd = np.where(sd < 1e-8, 1.0, sd)
        return torch.tensor(mu, dtype=torch.float32), torch.tensor(sd, dtype=torch.float32)