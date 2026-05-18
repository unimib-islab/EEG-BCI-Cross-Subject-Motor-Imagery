# EEG-BCI Cross-Subject Motor Imagery

Notebooks and pretrained models for cross-subject motor imagery classification on EEG (3, 4, and 5 classes). The models are retrained from 64 to 32 channels and are used for zero-shot inference and short fine-tuning on datasets converted to the `EEG_RAW_V1` schema.
## Repository layout

```
.
├─ data_conversion/
│  ├─ conversion-github.ipynb
│  ├─ conversion_manifest.csv
│  └─ converted_dataset_EEG_RAW_V1/
│     ├─ *.hdf5
│     ├─ batch_finetune_results.csv
│     └─ batch_inference_results.csv
├─ zero_shot_inference_and_fine_tuning/
│  └─ inference-github.ipynb
├─ retrain_32_channel_models/
│  └─ retrain.ipynb
├─ Model.ipynb
└─ models/
   ├─ model_loso_3_class_1s_I.rar
   ├─ model_loso_4_class_1s_I.rar
   ├─ model_loso_5_class_1s_I.rar
   ├─ stats_loso_3_class_1s_I.json
   ├─ stats_loso_4_class_1s_I.json
   └─ stats_loso_5_class_1s_I.json
```

## Requirements

- Python 3.10 or later
- NVIDIA GPU recommended
- Key libraries: `torch`, `numpy`, `pandas`, `scikit-learn`, `mne`, `tqdm`, `matplotlib`, `h5py`

## Included models

- `model_loso_3_class_1s_I.rar`
- `model_loso_4_class_1s_I.rar`
- `model_loso_5_class_1s_I.rar`
- Aggregate metrics and normalization stats are in `stats_loso_*.json`.

## Code order:

`main.py` Training with k-fold coss validation + ablation

`loso_and_retrain32.py` Trainging with special LOSO config for ISLab dataset.

## EEG_RAW_V1 data format (HDF5)

- `/raw/data`: `float32` matrix shaped `(channels, samples)`
- `/raw/ch_names`: ordered channel names
- `/info/sample_rate`: integer in Hz
- `/events/table`: columns `onset_sample`, `duration_samples`, `label`

The conversion notebook standardizes all recordings to 32 channels at 160 Hz with harmonized event labels so they are compatible with the provided LOSO models.

## ISLab-MI Dataset
The recordings are available in the repository folder named `data/ISLab-MI/`
