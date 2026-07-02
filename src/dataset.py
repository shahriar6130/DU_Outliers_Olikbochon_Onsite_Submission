"""
===========================================================
Dataset Module
-----------------------------------------------------------
Bangla LLM Hallucination Detection Challenge

Supports:
    ✓ BanglaBERT
    ✓ XLM-R
    ✓ ModernBERT
    ✓ Holdout Validation
    ✓ Future K-Fold CV

Author : Team
===========================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import random

import numpy as np
import pandas as pd
import torch
import yaml

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)

# ===========================================================
# Configuration
# ===========================================================

def load_config(config_path: str = "configs/baseline.yaml") -> dict:
    """
    Load YAML configuration.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# ===========================================================
# Random Seed
# ===========================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ===========================================================
# Text Cleaning
# ===========================================================

def clean_text(text: Optional[str], config: dict) -> str:
    """
    Basic preprocessing.
    """
    if text is None:
        text = ""

    text = str(text)

    if text.strip() == "":
        text = config["dataset"]["null_context_token"]

    if text == "[NULL]":
        text = config["dataset"]["null_context_token"]

    if config["dataset"]["remove_extra_whitespace"]:
        text = " ".join(text.split())

    return text

# ===========================================================
# Input Formatting
# ===========================================================

def format_input(
    context: str,
    prompt: str,
    response: str,
    config: dict,
) -> str:
    """
    Build ONE structured input sequence.
    """
    context = clean_text(context, config)
    prompt = clean_text(prompt, config)
    response = clean_text(response, config)

    fmt = config["input_format"]

    if context == config["dataset"]["null_context_token"]:
        context_prefix = fmt["no_context_prefix"]
    else:
        context_prefix = fmt["context_prefix"]

    text = (
        f"{context_prefix}\n"
        f"{context}\n\n"
        f"{fmt['prompt_prefix']}\n"
        f"{prompt}\n\n"
        f"{fmt['response_prefix']}\n"
        f"{response}"
    )

    return text

# ===========================================================
# Load DataFrame
# ===========================================================

def load_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load CSV safely.
    """
    df = pd.read_csv(csv_path)
    return df

def load_test_dataframe(config: dict) -> pd.DataFrame:
    """
    Load the inference/test CSV dataset path safely using configuration parameters.
    """
    return load_dataframe(config["dataset"]["test_csv"])

# ===========================================================
# Dataset
# ===========================================================

class HallucinationDataset(Dataset):
    """
    PyTorch Dataset for Hallucination Detection.
    """
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        config: dict,
        is_test: bool = False,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config
        self.is_test = is_test
        self.max_length = config["dataset"]["max_length"]

        # ------------------------------------------
        # Build formatted text once
        # ------------------------------------------
        self.texts: List[str] = []
        for _, row in self.df.iterrows():
            text = format_input(
                context=row["context"],
                prompt=row["prompt_bn"],
                response=row["response_bn"],
                config=config,
            )
            self.texts.append(text)

        # ------------------------------------------
        # Labels
        # ------------------------------------------
        if not is_test:
            self.labels = self.df["label"].tolist()
        else:
            self.labels = None

        # ------------------------------------------
        # Tokenize ONCE
        # ------------------------------------------
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors=None,
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }

        # --------------------------------------------------
        # Keep original text for error analysis
        # --------------------------------------------------
        item["context"] = self.df.iloc[idx]["context"]
        item["prompt_bn"] = self.df.iloc[idx]["prompt_bn"]
        item["response_bn"] = self.df.iloc[idx]["response_bn"]

        if "id" in self.df.columns:
            item["id"] = self.df.iloc[idx]["id"]

        if not self.is_test:
            item["labels"] = self.labels[idx]

        return item

# ===========================================================
# Custom Data Collator Wrapper
# ===========================================================

class ErrorAnalysisDataCollator:
    """
    Wraps DataCollatorWithPadding to shield non-tensor text strings 
    from tensor conversion while padding numerical input keys cleanly.
    """
    def __init__(self, tokenizer, padding="longest", return_tensors="pt"):
        self.base_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, 
            padding=padding, 
            return_tensors=return_tensors
        )
        self.metadata_keys = ["context", "prompt_bn", "response_bn", "id"]

    def __call__(self, features: List[Dict]) -> Dict:
        # Separate metadata out
        batch_metadata = {
            k: [f[k] for f in features if k in f] 
            for k in self.metadata_keys
        }
        
        # Build tensor-only dicts for the baseline collator
        tensor_features = []
        for f in features:
            tensor_f = {k: v for k, v in f.items() if k not in self.metadata_keys}
            tensor_features.append(tensor_f)
            
        # Perform padding on tensors only
        batch = self.base_collator(tensor_features)
        
        # Reinject metadata strings back into the finalized batch dictionary
        for k, list_values in batch_metadata.items():
            if list_values:
                batch[k] = list_values
                
        return batch

# ===========================================================
# Tokenizer
# ===========================================================

def build_tokenizer(config: dict):
    """
    Load HuggingFace tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        use_fast=True,
    )
    return tokenizer

# ===========================================================
# Train / Validation Split
# ===========================================================

def split_dataframe(
    dataframe: pd.DataFrame,
    config: dict,
):
    """
    Perform stratified train/validation split.
    """
    validation_cfg = config["validation"]

    train_df, val_df = train_test_split(
        dataframe,
        test_size=validation_cfg["test_size"],
        random_state=config["experiment"]["seed"],
        shuffle=validation_cfg["stratify"],
        stratify=dataframe["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    return train_df, val_df

# ===========================================================
# Dataset Factory
# ===========================================================

def create_datasets(config: dict):
    """
    Create training and validation datasets.
    """
    tokenizer = build_tokenizer(config)
    dataframe = load_dataframe(config["dataset"]["train_csv"])
    train_df, val_df = split_dataframe(dataframe, config)

    train_dataset = HallucinationDataset(
        dataframe=train_df,
        tokenizer=tokenizer,
        config=config,
        is_test=False,
    )

    val_dataset = HallucinationDataset(
        dataframe=val_df,
        tokenizer=tokenizer,
        config=config,
        is_test=False,
    )

    return train_dataset, val_dataset

# ===========================================================
# Test Dataset
# ===========================================================

def create_test_dataset(config: dict):
    """
    Create dataset for inference.
    """
    tokenizer = build_tokenizer(config)
    dataframe = load_test_dataframe(config)

    test_dataset = HallucinationDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        config=config,
        is_test=True,
    )

    return test_dataset

# ===========================================================
# DataLoaders
# ===========================================================

def create_dataloaders(config: dict):
    """
    Build PyTorch DataLoaders with custom error analysis collator wrapper.
    """
    train_dataset, val_dataset = create_datasets(config)
    loader_cfg = config["dataloader"]

    # FIXED: Replaced standard DataCollatorWithPadding with metadata-safe wrapper
    data_collator = ErrorAnalysisDataCollator(
        tokenizer=train_dataset.tokenizer,
        padding="longest",
        return_tensors="pt",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=loader_cfg["train_shuffle"],
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        persistent_workers=loader_cfg["persistent_workers"],
        collate_fn=data_collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        persistent_workers=loader_cfg["persistent_workers"],
        collate_fn=data_collator,
    )

    return train_loader, val_loader

# ===========================================================
# Test Loader
# ===========================================================

def create_test_loader(config: dict):
    """
    DataLoader for inference with custom error analysis collator wrapper.
    """
    dataset = create_test_dataset(config)
    loader_cfg = config["dataloader"]

    # FIXED: Replaced standard DataCollatorWithPadding with metadata-safe wrapper
    data_collator = ErrorAnalysisDataCollator(
        tokenizer=dataset.tokenizer,
        padding="longest",
        return_tensors="pt",
    )

    test_loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
        pin_memory=loader_cfg["pin_memory"],
        persistent_workers=loader_cfg["persistent_workers"],
        collate_fn=data_collator,
    )

    return test_loader