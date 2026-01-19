import json
from dataclasses import asdict
from pathlib import Path
from typing import Union
from .events import RenderConfig


def load_preset(path: Union[str, Path]) -> dict:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_preset(path: Union[str, Path], preset: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(preset, f, indent=2, sort_keys=True)


def default_cfg() -> RenderConfig:
    return RenderConfig()
