"""
YAML config loader for iterative cross-domain NER experiments.

The schema mirrors experiments/config_*.yaml. Loaded into immutable dataclasses
so downstream code never mutates configuration mid-run.
"""

import hashlib
import os
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import yaml


@dataclass(frozen=True)
class SourceCfg:
    name: str
    train: str
    dev: str
    unit: str = "sentence"


@dataclass(frozen=True)
class TargetCfg:
    name: str
    train: str
    dev: str
    test: str
    unit: str


@dataclass(frozen=True)
class EvalSetCfg:
    name: str
    path: str


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    output_dir: str
    seeds: List[int]
    model_name: str
    tokenizer_name: str
    max_seq_len: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    num_epochs: int
    early_stopping_patience: int
    source: SourceCfg
    target: TargetCfg
    eval_sets: List[EvalSetCfg]
    schedule: List[int] = field(default_factory=list)
    data_dir: str = "data"
    block_size: int = 1
    raw_yaml_path: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["eval_sets"] = {e["name"]: e["path"] for e in d["eval_sets"]}
        return d


def load_config(yaml_path: str) -> ExperimentConfig:
    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    src = SourceCfg(**raw["source"])
    tgt = TargetCfg(**raw["target"])
    eval_sets = [EvalSetCfg(name=k, path=v) for k, v in raw["eval_sets"].items()]

    return ExperimentConfig(
        experiment_name=raw["experiment_name"],
        output_dir=raw["output_dir"],
        seeds=list(raw["seeds"]),
        model_name=raw["model_name"],
        tokenizer_name=raw["tokenizer_name"],
        max_seq_len=int(raw["max_seq_len"]),
        batch_size=int(raw["batch_size"]),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw.get("weight_decay", 0.01)),
        warmup_ratio=float(raw.get("warmup_ratio", 0.1)),
        num_epochs=int(raw["num_epochs"]),
        early_stopping_patience=int(raw["early_stopping_patience"]),
        source=src,
        target=tgt,
        eval_sets=eval_sets,
        schedule=[int(n) for n in raw["schedule"]],
        data_dir=raw.get("data_dir", "data"),
        block_size=int(raw.get("block_size", 1)),
        raw_yaml_path=os.path.abspath(yaml_path),
    )


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_dataset_hashes(cfg: ExperimentConfig) -> Dict[str, str]:
    paths = {
        "source.train": cfg.source.train,
        "source.dev":   cfg.source.dev,
        "target.train": cfg.target.train,
        "target.dev":   cfg.target.dev,
        "target.test":  cfg.target.test,
    }
    for e in cfg.eval_sets:
        paths[f"eval.{e.name}"] = e.path
    return {k: sha256_file(p) for k, p in paths.items() if os.path.exists(p)}


def git_commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def machine_info() -> Dict[str, str]:
    info = {
        "platform": platform.platform(),
        "python":   platform.python_version(),
    }
    try:
        import torch  # local import to avoid hard dep at config-load time
        info["torch"]      = torch.__version__
        info["cuda_avail"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name()
    except Exception:
        pass
    return info
