"""
===========================================================
Model Module
-----------------------------------------------------------
Bangla LLM Hallucination Detection Challenge

Uses Hugging Face AutoModelForSequenceClassification.

Designed for:
- BanglaBERT
- XLM-R
- ModernBERT
- Future checkpoint resume

===========================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_model(config: Dict) -> torch.nn.Module:
    """
    Build a sequence classification model from the configuration.
    """

    model_name = config["model"]["name"]
    num_labels = config["model"]["num_labels"]

    model_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=model_config,
        ignore_mismatched_sizes=True,
    )

    return model


def prepare_model(model: torch.nn.Module) -> tuple[torch.nn.Module, torch.device]:
    """
    Move model to the correct device.
    """
    device = get_device()
    model.to(device)
    return model, device


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def save_model(model: torch.nn.Module, path: str | Path) -> None:
    """
    Save only model weights.

    Full training checkpoints (optimizer, scheduler, epoch)
    will be handled by trainer.py.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model_weights(
    model: torch.nn.Module,
    path: str | Path,
) -> torch.nn.Module:
    """
    Load model weights.
    """
    device = get_device()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return model


def print_model_summary(model: torch.nn.Module) -> None:
    """Print a compact model summary."""
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Architecture : {model.__class__.__name__}")
    print(f"Device       : {get_device()}")
    print(
        f"Parameters   : {count_trainable_parameters(model):,}"
    )
    print("=" * 60)


if __name__ == "__main__":
    import yaml

    with open("configs/baseline.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg)
    model, _ = prepare_model(model)
    print_model_summary(model)
