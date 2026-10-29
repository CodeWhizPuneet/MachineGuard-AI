#!/usr/bin/env python3
"""
MachineGuard AI: Model Evaluation and Reporting CLI
Generates comprehensive evaluation summaries, reports, and model cards.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import generate_evaluation_artifacts

def main():
    print("[INFO] Generating MachineGuard AI evaluation reports and model cards...")
    generate_evaluation_artifacts()
    print("[SUCCESS] Reports generated in reports/ directory.")

if __name__ == "__main__":
    main()
