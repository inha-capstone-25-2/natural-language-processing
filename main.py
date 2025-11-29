# main.py

import sys
from pathlib import Path

# ---- PATH SETUP ----
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from app.pipeline import run_sota_pipeline


if __name__ == "__main__":
    run_sota_pipeline(limit=10, top_k=10)
