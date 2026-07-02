"""
===========================================================
Inference Module

Bangla Hallucination Detection Challenge

Responsibilities
----------------
✓ Load best checkpoint
✓ Run inference
✓ Generate predictions & prediction probabilities
✓ Create Kaggle submission.csv
✓ Export test probabilities file for error analysis/ensembling

Author : Team
===========================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.dataset import (
    create_test_loader,
    load_test_dataframe,
)

from src.model import (
    build_model,
    get_device,
)


class InferenceEngine:

    def __init__(self, config: dict):

        self.config = config

        self.device = get_device()

        print("=" * 60)
        print("Loading Model...")
        print("=" * 60)

        self.model = build_model(config)

        checkpoint_path = Path(
            config["checkpoint"]["best_model_path"]
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found:\n{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)

        self.model.eval()

        self.test_loader = create_test_loader(config)

        self.test_df = load_test_dataframe(config)

    # -------------------------------------------------------
    # Predict
    # -------------------------------------------------------
    def predict(self) -> Tuple[List[int], List[float]]:
        """
        Run forward inference over test data splits.
        
        Returns:
            Tuple containing:
            - List of hard prediction class index labels.
            - List of confidence probabilities corresponding to class 1 (hallucination).
        """
        predictions = []
        probabilities = []

        progress = tqdm(
            self.test_loader,
            desc="Inference",
        )

        with torch.inference_mode():

            for batch in progress:

                input_ids = batch["input_ids"].to(self.device)

                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits = outputs.logits
                
                # Compute softmax scores for evaluation/debugging insights
                probs = F.softmax(logits, dim=1)

                pred = torch.argmax(
                    logits,
                    dim=1,
                )

                predictions.extend(
                    pred.cpu().numpy().tolist()
                )
                
                # Capture target class 1 (hallucination) prediction probability column
                probabilities.extend(
                    probs[:, 1].cpu().numpy().tolist()
                )

        return predictions, probabilities

    # -------------------------------------------------------
    # Submission
    # -------------------------------------------------------
    def create_submission(self, save_path: str | None = None):
        """
        Generates clean submission files and exports underlying soft probabilities
        to a separate file for threshold tuning, calibration, and ensembling.
        """
        if save_path is None:
            sub_dir = Path(self.config["submission"]["directory"])
            sub_dir.mkdir(parents=True, exist_ok=True)
            output_filepath = sub_dir / self.config["submission"]["filename"]
            prob_filepath = sub_dir / "test_probabilities.csv"
        else:
            output_filepath = Path(save_path)
            prob_filepath = output_filepath.parent / "test_probabilities.csv"

        predictions, probabilities = self.predict()

        # FIXED: Clean submission DataFrame containing ONLY id and label for Kaggle
        submission = pd.DataFrame(
            {
                "id": self.test_df["id"],
                "label": predictions,
            }
        )

        submission.to_csv(
            output_filepath,
            index=False,
        )

        # FIXED: Save soft probabilities separately for post-processing/ensembling
        prob_df = pd.DataFrame(
            {
                "id": self.test_df["id"],
                "probability": probabilities,
            }
        )

        prob_df.to_csv(
            prob_filepath,
            index=False,
        )

        print("=" * 60)
        print("Inference Pipeline Artifacts Completed!")
        print("=" * 60)
        print(f"Submission saved to          : {output_filepath}  <- upload this to Kaggle")
        print(f"Test probabilities saved to  : {prob_filepath}  <- keep for analysis")
        print(f"Total Rows                   : {len(submission)}")

        return submission