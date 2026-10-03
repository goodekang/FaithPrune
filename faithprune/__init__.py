"""FaithPrune: hallucination-aware visual token pruning for MLLMs."""
from .config import Config, load_config, save_config

__version__ = "1.0.0"

__all__ = ["Config", "load_config", "save_config", "FaithPrune"]


def __getattr__(name):
    if name == "FaithPrune":
        from .pipeline import FaithPrune
        return FaithPrune
    raise AttributeError(name)
