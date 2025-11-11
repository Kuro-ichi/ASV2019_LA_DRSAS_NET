from __future__ import annotations
import argparse, yaml
from pathlib import Path
from typing import Any, Dict

def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_config(default_yaml: str, overrides: list[str] | None = None) -> Dict[str, Any]:
    cfg = _read_yaml(Path(default_yaml))
    # parse CLI overrides like: key1.key2=value
    overrides = overrides or []
    for item in overrides:
        key, _, val = item.partition("=")
        keys = key.split(".")
        cursor = cfg
        for k in keys[:-1]:
            cursor = cursor.setdefault(k, {})
        # best-effort typing
        if val.lower() in {"true","false"}: parsed = val.lower()=="true"
        else:
            try:
                parsed = int(val)
            except ValueError:
                try:
                    parsed = float(val)
                except ValueError:
                    parsed = val
        cursor[keys[-1]] = parsed
    return cfg

def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--config", type=str, default="configs/defaults.yaml", help="Path to a YAML config")
    p.add_argument("--override", type=str, nargs="*", default=[], help="Overrides like key.subkey=value")
    return p
