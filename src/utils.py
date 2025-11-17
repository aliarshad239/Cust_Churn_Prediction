from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    random_state: int
    data: dict
    features: dict
    train: dict

def load_config(path: str | Path) -> Config:
    """Load params.yaml config."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)

if __name__ == "__main__":
    cfg = load_config("config/params.yaml")
    print("Config loaded successfully ✅")
