"""
===========================================================
Trainer Module

Bangla Hallucination Detection Challenge

Responsibilities
----------------
✓ Build optimizer
✓ Build scheduler
✓ Training loop
✓ Validation loop with detailed error analysis logging
✓ Gradient accumulation
✓ Gradient clipping
✓ Checkpointing
✓ Early stopping
✓ Resume training
✓ WandB logging

Author : Team
===========================================================
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import wandb

from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.metrics import (
    AverageMeter,
    compute_metrics,
    build_confusion_matrix,
    build_classification_report,
)
from src.model import get_device


# ===========================================================
# Trainer
# ===========================================================

class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
    ):
        self.config = config
        self.device = get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ---------------------------------------
        # Training Config
        # ---------------------------------------
        self.epochs = config["training"]["epochs"]
        self.lr = config["training"]["learning_rate"]
        self.weight_decay = config["training"]["weight_decay"]
        self.gradient_accumulation = config["training"]["gradient_accumulation_steps"]
        self.max_grad_norm = config["training"]["max_grad_norm"]
        self.batch_size = config["training"]["batch_size"]

        # ---------------------------------------
        # Optimizer
        # ---------------------------------------
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # ---------------------------------------
        # Scheduler
        # ---------------------------------------
        total_training_steps = math.ceil(
            len(self.train_loader) / self.gradient_accumulation
        ) * self.epochs

        warmup_steps = int(
            total_training_steps
            * config["training"]["warmup_ratio"]
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        # ---------------------------------------
        # Loss
        # ---------------------------------------
        self.criterion = torch.nn.CrossEntropyLoss()

        # ---------------------------------------
        # Automatic Mixed Precision
        # ---------------------------------------
        self.use_amp = (
            torch.cuda.is_available()
            and config["precision"]["use_amp"]
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        # ---------------------------------------
        # Metric Trackers
        # ---------------------------------------
        self.train_loss = AverageMeter()
        self.val_loss = AverageMeter()

        # ---------------------------------------
        # Best Model Tracking
        # ---------------------------------------
        self.best_macro_f1 = -1.0
        self.best_epoch = -1
        self.current_epoch = 0

        # ---------------------------------------
        # Directories
        # ---------------------------------------
        self.checkpoint_dir = Path(config["checkpoint"]["directory"])
        self.output_dir = Path(config["logging"]["output_directory"])

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------
        # Weights & Biases
        # ---------------------------------------
        self.use_wandb = config["wandb"]["enabled"]

        if self.use_wandb:
            wandb.init(
                project=config["wandb"]["project"],
                name=config["wandb"]["run_name"],
                config=config,
            )
            wandb.watch(
                self.model,
                log=None,
            )

        print("=" * 60)
        print("Trainer Initialized")
        print("=" * 60)
        print(f"Device           : {self.device}")
        print(f"Epochs           : {self.epochs}")
        print(f"Batch Size       : {self.batch_size}")
        print(f"Learning Rate    : {self.lr}")
        print(f"Weight Decay     : {self.weight_decay}")
        print(f"Warmup Ratio     : {config['training']['warmup_ratio']}")
        print(f"Mixed Precision  : {self.use_amp}")
        print("=" * 60)

    # ===========================================================
    # Training Loop
    # ===========================================================
    def train_one_epoch(self):
        self.model.train()
        self.train_loss.reset()

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch+1}/{self.epochs}",
            leave=False,
        )

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # ------------------------------------
            # Forward
            # ------------------------------------
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss = loss / self.gradient_accumulation

            # ------------------------------------
            # Backward
            # ------------------------------------
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # ------------------------------------
            # Optimizer Step
            # ------------------------------------
            should_step = (
                (step + 1) % self.gradient_accumulation == 0
                or (step + 1) == len(self.train_loader)
            )

            if should_step:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            # ------------------------------------
            # Metrics
            # ------------------------------------
            batch_loss = loss.item() * self.gradient_accumulation
            self.train_loss.update(batch_loss, input_ids.size(0))

            progress_bar.set_postfix(
                loss=f"{self.train_loss.avg:.4f}",
                lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
            )

            # ------------------------------------
            # WandB
            # ------------------------------------
            if self.use_wandb:
                wandb.log(
                    {
                        "train_loss": self.train_loss.avg,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                    },
                    commit=False,
                )

        return self.train_loss.avg
        
    # ===========================================================
    # Validation
    # ===========================================================
    def validate(self):
        self.model.eval()
        self.val_loss.reset()

        all_predictions = []
        all_probabilities = []
        all_labels = []

        all_contexts = []
        all_prompts = []
        all_responses = []
        all_ids = []

        progress_bar = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False,
        )

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                ):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    logits = outputs.logits

                probabilities = torch.softmax(
                    logits,
                    dim=1,
                )

                predictions = torch.argmax(
                    probabilities,
                    dim=1,
                )

                self.val_loss.update(
                    loss.item(),
                    input_ids.size(0),
                )

                all_predictions.extend(
                    predictions.cpu().numpy().tolist()
                )

                all_probabilities.extend(
                    probabilities[:, 1].cpu().numpy().tolist()
                )

                all_labels.extend(
                    labels.cpu().numpy().tolist()
                )

                all_contexts.extend(batch["context"])
                all_prompts.extend(batch["prompt_bn"])
                all_responses.extend(batch["response_bn"])

                if "id" in batch:
                    all_ids.extend(batch["id"])

                progress_bar.set_postfix(
                    loss=f"{self.val_loss.avg:.4f}"
                )

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------
        metrics = compute_metrics(all_labels, all_predictions)
        confusion = build_confusion_matrix(all_labels, all_predictions)
        report = build_classification_report(all_labels, all_predictions)
        metrics["val_loss"] = self.val_loss.avg

        # --------------------------------------------------
        # Save Validation Predictions
        # --------------------------------------------------
        prediction_dict = {
            "context": all_contexts,
            "prompt_bn": all_prompts,
            "response_bn": all_responses,
            "true_label": all_labels,
            "pred_label": all_predictions,
            "probability": all_probabilities,
        }

        if len(all_ids) > 0:
            prediction_dict["id"] = all_ids

        val_predictions = pd.DataFrame(
            prediction_dict
        )

        save_path = (
            self.output_dir /
            "val_predictions.csv"
        )

        val_predictions.to_csv(
            save_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"\nValidation predictions saved to:\n{save_path}"
        )

        # --------------------------------------------------
        # WandB
        # --------------------------------------------------
        if self.use_wandb:
            wandb.log(metrics, commit=False)

        # --------------------------------------------------
        # Print
        # --------------------------------------------------
        print()
        print("=" * 60)
        print(f"Validation Loss : {self.val_loss.avg:.4f}")
        print(f"Macro F1        : {metrics['macro_f1']:.4f}")
        print(f"Hallucination F1: {metrics['hallucination_f1']:.4f}")
        print(f"Accuracy        : {metrics['accuracy']:.4f}")
        print("=" * 60)
        print()

        print("Confusion Matrix")
        print(confusion)
        print()

        print(report)
        return metrics

    # ===========================================================
    # Save Checkpoint
    # ===========================================================
    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict,
        is_best: bool = False,
    ):
        checkpoint = {
            "epoch": epoch,
            "train_loss": self.train_loss.avg,
            "val_loss": metrics["val_loss"],
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_macro_f1": self.best_macro_f1,
            "config": self.config,
        }

        last_path = self.checkpoint_dir / "last_model.pt"
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

    # ===========================================================
    # Load Checkpoint
    # ===========================================================
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_macro_f1 = checkpoint["best_macro_f1"]
        self.current_epoch = checkpoint["epoch"] + 1

        print(f"Resumed training from epoch {self.current_epoch}")

    # ===========================================================
    # Fit
    # ===========================================================
    def fit(self):
        print("\nStarting Training...\n")

        patience = self.config["training"]["early_stopping_patience"]
        patience_counter = 0

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            print("=" * 70)
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print("=" * 70)

            train_loss = self.train_one_epoch()
            metrics = self.validate()
            macro_f1 = metrics["macro_f1"]

            print()
            print(f"Train Loss : {train_loss:.4f}")
            print(f"Val Loss   : {metrics['val_loss']:.4f}")
            print(f"Macro F1   : {macro_f1:.4f}")
            print(f"Hall F1    : {metrics['hallucination_f1']:.4f}")
            print()

            improved = macro_f1 > self.best_macro_f1

            if improved:
                self.best_macro_f1 = macro_f1
                self.best_epoch = epoch
                self.save_checkpoint(epoch, metrics, is_best=True)
                patience_counter = 0
                print("New Best Model Saved.")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{patience})")

            self.save_checkpoint(epoch, metrics, is_best=False)

            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": metrics["val_loss"],
                        "macro_f1": macro_f1,
                        "hallucination_f1": metrics["hallucination_f1"],
                    }
                )

            if patience_counter >= patience:
                print("\nEarly stopping triggered.")
                break

        print()
        print("=" * 70)
        print("Training Finished")
        print("=" * 70)
        print(f"Best Epoch : {self.best_epoch + 1}")
        print(f"Best Macro F1 : {self.best_macro_f1:.4f}")
        print("=" * 70)

        print()
        print(f"Best checkpoint saved to:\n{self.checkpoint_dir / 'best_model.pt'}")
        print(f"Last checkpoint saved to:\n{self.checkpoint_dir / 'last_model.pt'}")

        if self.use_wandb:
            wandb.finish()