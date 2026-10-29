#!/usr/bin/env python3
"""
MachineGuard AI: Model Training Pipeline Entry Point
Runs data loading, multi-algorithm training & cross-validation, threshold tuning, and artifact generation.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.train import train_and_compare_all_models
from src.evaluate import generate_evaluation_artifacts

def main():
    parser = argparse.ArgumentParser(description="MachineGuard AI: Train and Compare Models")
    parser.add_argument("--data-path", default="data/ai4i2020.csv", help="Path to AI4I 2020 CSV")
    parser.add_argument("--quick", action="store_true", help="Quick run with smaller parameter exploration")
    args = parser.parse_args()

    results = train_and_compare_all_models(csv_path=args.data_path, quick_run=args.quick)
    generate_evaluation_artifacts()
    print("[SUCCESS] Training pipeline and reporting successfully completed.")

if __name__ == "__main__":
    main()
