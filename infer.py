"""
===========================================================
Inference Entry Point

Bangla Hallucination Detection Challenge

Run

    python infer.py

This script

✓ Loads configuration
✓ Builds inference engine
✓ Loads best checkpoint
✓ Runs inference
✓ Creates submission.csv

===========================================================
"""

from __future__ import annotations

import traceback
from pathlib import Path

from src.dataset import load_config
from src.inference import InferenceEngine


# ===========================================================
# Main
# ===========================================================

def main():

    print("=" * 70)
    print("Bangla Hallucination Detection")
    print("Inference")
    print("=" * 70)

    # -------------------------------------------------------
    # Configuration
    # -------------------------------------------------------

    config = load_config()

    # -------------------------------------------------------
    # Build Inference Engine
    # -------------------------------------------------------

    engine = InferenceEngine(config)

    # -------------------------------------------------------
    # Submission Path
    # -------------------------------------------------------

    output_dir = Path(
        config["logging"]["output_directory"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission_path = (
        output_dir / "submission.csv"
    )

    # -------------------------------------------------------
    # Run Inference
    # -------------------------------------------------------

    engine.create_submission(
        save_path=submission_path
    )

    print()
    print("=" * 70)
    print("Inference Completed Successfully!")
    print("=" * 70)
    print(f"Submission saved to:\n{submission_path}")


# ===========================================================
# Entry
# ===========================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print("=" * 70)
        print("Inference Interrupted.")
        print("=" * 70)

    except Exception as e:

        print()
        print("=" * 70)
        print("Inference Failed")
        print("=" * 70)

        print(e)
        print()

        traceback.print_exc()

        raise