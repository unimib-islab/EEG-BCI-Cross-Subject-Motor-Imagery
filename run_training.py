from config import load_config
from src.kfold_train import Experiment
from src.utils import set_seeds

if __name__ == "__main__":
    
    from src.experiments import * # import list of experiments from py file.

    cfg = load_config('configs/kfold_train_105subjects.yaml')

    for name, labels, mode, win, folds in EXPERIMENTS_ABLATION:

        for seed in cfg.seeds:
            set_seeds(seed)

            if cfg.sstl:
                Experiment(name, seed, labels, mode, win, folds, cfg).fine_tune_per_subject()
            else:
                Experiment(name, seed, labels, mode, win, folds, cfg).run()