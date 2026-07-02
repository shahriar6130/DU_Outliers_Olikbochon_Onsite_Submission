"""
===========================================================
Training Entry Point

Bangla Hallucination Detection Challenge

Run

    python train.py

===========================================================
"""

from __future__ import annotations

import traceback
from pathlib import Path

import torch

from src.dataset import (
    load_config,
    seed_everything,
    create_dataloaders,
)

from src.model import (
    build_model,
    prepare_model,
    print_model_summary,
)

from src.trainer import Trainer


# ===========================================================
# Main
# ===========================================================


def main():

    print("=" * 70)
    print("Bangla Hallucination Detection")
    print("=" * 70)

    # -------------------------------------------------------
    # Configuration
    # -------------------------------------------------------

    config = load_config()

    # -------------------------------------------------------
    # Seed
    # -------------------------------------------------------

    seed_everything(
        config["experiment"]["seed"]
    )

    # -------------------------------------------------------
    # Dataloader
    # -------------------------------------------------------

    print("\nBuilding DataLoaders...")

    train_loader, val_loader = create_dataloaders(
        config
    )

    print(
        f"Train batches : {len(train_loader)}"
    )

    print(
        f"Validation batches : {len(val_loader)}"
    )

    # -------------------------------------------------------
    # Model
    # -------------------------------------------------------

    print("\nBuilding Model...")

    model = build_model(config)

    model, device = prepare_model(model)

    print_model_summary(model)

    # -------------------------------------------------------
    # Trainer
    # -------------------------------------------------------

    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        config=config,

    )

    # -------------------------------------------------------
    # Resume
    # -------------------------------------------------------

    resume = config["checkpoint"]["resume"]

    checkpoint_path = Path(

        config["checkpoint"]["resume_path"]

    )

    if resume:

        if checkpoint_path.exists():

            print()

            print(

                f"Loading checkpoint:\n"

                f"{checkpoint_path}"

            )

            trainer.load_checkpoint(

                checkpoint_path

            )

        else:

            print()

            print(

                "Checkpoint not found."

            )

            print(

                "Training from scratch."

            )

    # -------------------------------------------------------
    # Train
    # -------------------------------------------------------

    trainer.fit()

    print()

    print("Training Complete.")


# ===========================================================
# Entry
# ===========================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()

        print("=" * 70)

        print(

            "Training Interrupted."

        )

        print("=" * 70)

    except Exception as e:

        print()

        print("=" * 70)

        print("Fatal Error")

        print("=" * 70)

        print(e)

        print()

        traceback.print_exc()

        raise