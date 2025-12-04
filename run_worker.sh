#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=$(pwd)

# Run Celery worker
# -A gpu_worker: Application module
# worker: Worker mode
# --loglevel=info: Logging level
# --concurrency=1: Limit to 1 process (due to GPU memory)
# -P solo: Use solo pool (no multiprocessing overhead, good for GPU)

echo "Starting GPU Worker..."
celery -A worker worker --loglevel=info --concurrency=1 -P solo
