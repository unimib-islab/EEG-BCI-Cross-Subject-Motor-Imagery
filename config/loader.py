import yaml
import importlib
from pathlib import Path
from .schema import Config

import sys
from pathlib import Path

# To add the root folder of the project source code
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_config(path: str | Path) -> Config:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return Config(raw)


def resolve(dotted_path: str):
    """Resolve e.g. 'torch.optim.AdamW' to class AdamW."""
    module_path, class_name = dotted_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_net(cfg: Config):
    NetClass = resolve(cfg.net)
    return NetClass(**cfg.net_config.to_dict())


def build_optimizer(cfg: Config, net):
    OptClass = resolve(cfg.optimizer)
    return OptClass(net.parameters(), **cfg.optimizer_config.to_dict())


def build_criterion(cfg: Config):
    return resolve(cfg.criterion)()


def build_dataset(cfg: Config):
    DataClass = resolve(cfg.data)
    return DataClass(apply_ica=cfg.apply_ica)