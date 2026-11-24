#!/usr/bin/env bash
# Runs the batch evaluation. Stops immediately if any step fails.
set -euo pipefail

cd "$(dirname "$0")/.."

# Activate the environment (Windows path inside Git Bash)
source ../llama_index/.venv/Scripts/activate

if [ -z "${GROQ_API_KEY:-}" ]; then
    echo "ERROR: GROQ_API_KEY is not set" >&2
    exit 1
fi

python scripts/run_batch.py
echo "Done. Log: logs/run.log"