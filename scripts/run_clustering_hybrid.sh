#!/usr/bin/env bash
# Pre-registered SCOP hybrid clustering experiment (report §8, "that stage").
#
# Reads (unchanged):    scripts/scop_clustering.py, SCOP/<tier>/data.csv,
#                       results/clustering/masses_<tier>_g100.npy
# Writes ONLY into:     results/clustering_hybrid/   (new directory)
# Never touches:        fff/**, runs/**, results/clustering/**, paper/**
#
# Override the interpreter with e.g.:
#   PYTHON=/data/python-envs/pytorch/bin/python bash scripts/run_clustering_hybrid.sh
# or, on Windows (Git Bash / WSL):
#   PYTHON="D:/ProgramData/anaconda3/envs/pth/python.exe" bash scripts/run_clustering_hybrid.sh
set -euo pipefail

PYTHON=${PYTHON:-/data/python-envs/pytorch/bin/python}
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "python: $PYTHON"
"$PYTHON" --version

mkdir -p results/clustering_hybrid

echo "============================================"
echo "Step 1: smoke test (envs + caches, no output changes)"
echo "============================================"
"$PYTHON" scripts/check_hybrid_env.py --grid 100

echo "============================================"
echo "Step 2: main pre-registered ablation (R0..R4, full alpha grid)"
echo "============================================"
"$PYTHON" scripts/clustering_hybrid.py --tier all --grid 100

echo "============================================"
echo "Step 3: 5-NN local-signal rows"
echo "============================================"
"$PYTHON" scripts/clustering_hybrid.py --tier all --grid 100 --knn

echo "============================================"
echo "Step 4: strictly-increasing-res_seq sensitivity variant (B2)"
echo "============================================"
"$PYTHON" scripts/clustering_hybrid.py --tier all --grid 100 --strict-run-bound

echo "============================================"
echo "Step 5: nested alpha selection (optional, label-informed)"
echo "============================================"
"$PYTHON" scripts/clustering_hybrid.py --tier all --grid 100 --nested

echo "Done. Output files:"
echo "-------------------------------------------"
ls -la results/clustering_hybrid/
echo
echo "Decision criteria are in the report, §7 (compare against §7's pre-registered"
echo "criteria; do NOT tune anything to improve them post hoc)."
