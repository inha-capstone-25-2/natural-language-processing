#!/bin/bash

# Get the project root directory (one level up from scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"
source venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT"

echo "Starting GPU Summary Server..."
python -m uvicorn server:app --host 0.0.0.0 --port 8000
