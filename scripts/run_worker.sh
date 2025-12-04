#!/bin/bash

# Get the project root directory (one level up from scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Activate virtual environment
source venv/bin/activate

# Set PYTHONPATH to project root
export PYTHONPATH="$PROJECT_ROOT"

# Run Celery worker
echo "Starting GPU Worker from $PROJECT_ROOT..."
celery -A worker worker --loglevel=info --concurrency=1 -P solo
