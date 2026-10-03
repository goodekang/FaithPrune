"""Configuration handling.

A run is described by a stack of YAML files (``configs/default.yaml`` first,
then a model file, then optional benchmark / ablation files) followed by
``key.sub=value`` overrides from the command line. Later entries win.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    name: str = "llava-1.5-7b"
    adapter: str = "llava"               # llava | llava_next | qwen2_5_vl
    path: str = "llava-hf/llava-1.5-7b-hf"
    dtype: str = "float16"
    device: str = "cuda"
    # sdpa: VAMR folded into the additive SDPA mask (default)
    # folded: FlashAttention-2 with the augmented query/key coordinate
    # eager: explicit logits (calibration, analysis, PAI amplification)
    attn_impl: str = "sdpa"
    conv_template: str = "llava_v1"
    # Qwen2.5-VL only: images are resized so that the full model uses 576 tokens
    image_size: Optional[List[int]] = None


@dataclass
class VAMRConfig:
    enabled: bool = True
    gamma_file: Optional[str] = None
    rows: str = "all"                    # all | text  (query rows that receive the bias)
    global_gamma: bool = False           # ablation: one scalar for all layers
    scale: float = 1.0


@dataclass
class AmplifyConfig:
    """PAI-style amplification-only bias z <- z + alpha * |z| on visual keys."""
    enabled: bool = False
    alpha: float = 0.5
    start_layer: int = 2
    end_layer: int = 32
    last_row_only: bool = True


@dataclass
class MethodConfig:
    enabled: bool = True                 # False -> full (unpruned) model
    budget: int = 64                     # K
    # models with a variable number of visual tokens (LLaVA-NeXT): K = round(rho * N_v)
    budget_ratio: Optional[float] = None
    scoring_layer: int = 2               # l_s
    sink_layer: Optional[int] = None     # layer where Sigma is identified (defaults to l_s, or 2 if l_s == 0)
    tau: float = 3.0
    beta: float = 0.3
    lam: float = 0.5
    sigma: float = 0.15
    num_centroids: int = 256
    centroids_file: Optional[str] = None
    # selector: sr2s | topk_relevance | random | uniform_grid | avgpool | external
    selector: str = "sr2s"
    sink_mode: str = "exclude"           # exclude | zero_relevance | keep
    use_relevance: bool = True
    use_alignment: bool = True
    use_coverage: bool = True
    kernel: str = "feature_position"     # feature_position | position_only
    random_from_candidates: bool = False
    external_selection_file: Optional[str] = None
    vamr: VAMRConfig = field(default_factory=VAMRConfig)
    amplify: AmplifyConfig = field(default_factory=AmplifyConfig)
    seed: int = 0


@dataclass
class CalibrationConfig:
    num_images: int = 64
    image_dir: str = "data/coco/train2014"
    image_list: str = "data/splits/calib_coco_train2014_64.txt"
    prompts: List[str] = field(default_factory=lambda: ["Please describe this image in detail."])
    # optional JSONL with {"image", "question"} per line; overrides image_list / prompts
    # (supplementary variant "GQA prompts for calibration")
    prompt_file: Optional[str] = None
    max_new_tokens: int = 512
    statistic: str = "median"            # median | mean_matching
    sequential: bool = True
    target: str = "non_sink"             # non_sink | with_sink
    headwise: bool = False
    output: Optional[str] = None


@dataclass
class DecodingConfig:
    strategy: str = "greedy"             # greedy | sample
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: int = 0
    # decoding-time mitigator: none | vcd | pai | cmac
    mitigator: str = "none"
    vcd_alpha: float = 1.0
    vcd_beta: float = 0.1
    vcd_noise_step: int = 500
    pai_gamma: float = 1.1
    pai_beta: Optional[float] = None     # optional adaptive plausibility cut-off


@dataclass
class EvalConfig:
    benchmark: str = "pope"
    protocol: str = "B"
    data_root: str = "data"
    output_dir: str = "outputs"
    limit: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run_name: str = "faithprune"

    # ------------------------------------------------------------------
    def effective_sink_layer(self) -> int:
        if self.method.sink_layer is not None:
            return int(self.method.sink_layer)
        return self.method.scoring_layer if self.method.scoring_layer > 0 else 2

    def to_dict(self) -> Dict[str, Any]:
        return _to_dict(self)


# ---------------------------------------------------------------------------
# Loading / merging
# ---------------------------------------------------------------------------
def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _from_dict(cls, data: Dict[str, Any]):
    kwargs = {}
    names = {f.name: f for f in fields(cls)}
    for key, value in (data or {}).items():
        if key not in names:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        f = names[key]
        default = f.default_factory() if callable(getattr(f, "default_factory", None)) else None
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _from_dict(type(default), value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def deep_merge(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (other or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(text: str) -> Any:
    low = text.lower()
    if low in ("null", "none"):
        return None
    if low in ("true", "false"):
        return low == "true"
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def apply_overrides(data: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    data = copy.deepcopy(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got '{item}'")
        key, value = item.split("=", 1)
        node = data
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value.strip())
    return data


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # ``_base_`` lets a config inherit from other files (relative paths)
    bases = data.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: Dict[str, Any] = {}
    for b in bases:
        merged = deep_merge(merged, _load_yaml(path.parent / b))
    return deep_merge(merged, data)


def load_config(paths: Sequence[str | Path], overrides: Sequence[str] = ()) -> Config:
    data: Dict[str, Any] = {}
    for p in paths:
        data = deep_merge(data, _load_yaml(p))
    data = apply_overrides(data, overrides)
    return _from_dict(Config, data)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
