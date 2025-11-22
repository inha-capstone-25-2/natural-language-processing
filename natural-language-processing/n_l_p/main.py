# test_pipeline_sota.py

import sys
from pathlib import Path

# ---- PATH SETUP ----
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))
print("[DEBUG] sys.path added:", ROOT_DIR)

from app.pipeline import run_sota_pipeline


if __name__ == "__main__":
    # 필요하면 인자 바꿔서 사용
    run_sota_pipeline(limit=10, top_k=10)
