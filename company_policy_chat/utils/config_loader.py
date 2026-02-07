from pathlib import Path
import os
import yaml


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_config()->dict:

    config_path = (_project_root()/"config"/"config.yaml")
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"config file not found : {path}")
    
    with open(path,"r",encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
