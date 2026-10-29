#!/usr/bin/env python3
"""
MachineGuard AI: Dataset Acquisition and Validation Script
Downloads the official AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository.
URL: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
"""

import os
import sys
import argparse
import hashlib
import urllib.request
import zipfile
import io
import csv

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
FALLBACK_CSV_URL = "https://raw.githubusercontent.com/UCI-ML-Repository/ai4i-2020/main/ai4i2020.csv"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CSV_PATH = os.path.join(DATA_DIR, "ai4i2020.csv")

EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

def calculate_checksum(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def normalize_csv(file_path: str) -> bool:
    """Clean UTF-8 BOM if present and verify structure."""
    if not os.path.exists(file_path):
        return False
    with open(file_path, "rb") as f:
        content = f.read()
    # Check for UTF-8 BOM and remove if present
    if content.startswith(b"\xef\xbb\xbf"):
        content = content[3:]
        with open(file_path, "wb") as f:
            f.write(content)
    return True

def verify_dataset(file_path: str = CSV_PATH) -> bool:
    """Verify that the CSV exists, has valid columns, and contains the expected ~10,000 rows."""
    if not os.path.exists(file_path):
        print(f"[ERROR] Dataset file not found at: {file_path}")
        return False

    normalize_csv(file_path)

    try:
        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if not headers:
                print("[ERROR] CSV file is empty.")
                return False

            # Clean headers of BOM or whitespace
            cleaned_headers = [h.strip().lstrip("\ufeff") for h in headers]
            missing_cols = [col for col in EXPECTED_COLUMNS if col not in cleaned_headers]
            if missing_cols:
                print(f"[ERROR] Missing required columns: {missing_cols}")
                print(f"        Found columns: {cleaned_headers}")
                return False

            row_count = sum(1 for _ in reader)

        if row_count < 9500:
            print(f"[WARNING] Row count ({row_count}) is lower than expected 10,000 rows.")
            return False

        checksum = calculate_checksum(file_path)
        print(f"[SUCCESS] Dataset verified successfully:")
        print(f"          Path: {file_path}")
        print(f"          Rows: {row_count}")
        print(f"          Columns: {len(cleaned_headers)}")
        print(f"          SHA-256: {checksum}")
        return True

    except Exception as e:
        print(f"[ERROR] Verification failed with exception: {e}")
        return False

def download_dataset(output_path: str = CSV_PATH, force: bool = False) -> bool:
    """Download AI4I 2020 dataset from UCI repository."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path) and not force:
        print(f"[INFO] Dataset already exists at {output_path}.")
        if verify_dataset(output_path):
            print("[INFO] Use --force to re-download.")
            return True
        print("[WARNING] Existing file verification failed. Proceeding with re-download.")

    print(f"[INFO] Downloading AI4I 2020 Predictive Maintenance Dataset from UCI...")
    headers = {"User-Agent": "Mozilla/5.0 (MachineGuard AI Maintenance System)"}

    # Attempt primary download (official ZIP)
    try:
        req = urllib.request.Request(UCI_ZIP_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            zip_bytes = response.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                # Look for ai4i2020.csv inside archive
                csv_candidates = [f for f in z.namelist() if f.endswith(".csv")]
                if not csv_candidates:
                    raise ValueError("No CSV file found inside UCI zip archive.")
                target_file = csv_candidates[0]
                with z.open(target_file) as source, open(output_path, "wb") as target:
                    target.write(source.read())
        print(f"[INFO] Downloaded and extracted official ZIP to {output_path}")
        return verify_dataset(output_path)
    except Exception as e_primary:
        print(f"[WARNING] Primary download failed: {e_primary}")
        print("[INFO] Attempting fallback source...")

    # Attempt fallback
    try:
        req = urllib.request.Request(FALLBACK_CSV_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response, open(output_path, "wb") as target:
            target.write(response.read())
        print(f"[INFO] Fallback download succeeded.")
        return verify_dataset(output_path)
    except Exception as e_fallback:
        print(f"[ERROR] Both download sources failed.")
        print(f"        Fallback error: {e_fallback}")
        print("\n" + "=" * 65)
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("1. Visit official UCI repository:")
        print("   https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset")
        print("2. Download 'ai4i+2020+predictive+maintenance+dataset.zip'")
        print("3. Extract 'ai4i2020.csv' and place it at:")
        print(f"   {output_path}")
        print("4. Re-run: python scripts/download_data.py --verify-only")
        print("=" * 65 + "\n")
        return False

def main():
    parser = argparse.ArgumentParser(description="MachineGuard AI Dataset Acquisition and Validation")
    parser.add_argument("--output", default=CSV_PATH, help=f"Path to save CSV (default: {CSV_PATH})")
    parser.add_argument("--force", action="store_true", help="Force re-download even if file exists")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing dataset")

    args = parser.parse_args()

    if args.verify_only:
        success = verify_dataset(args.output)
    else:
        success = download_dataset(args.output, args.force)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
