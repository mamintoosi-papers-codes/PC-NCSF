#!/usr/bin/env bash
set -euo pipefail

PYTHON=/data/python-envs/pytorch/bin/python
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p results/clustering

echo "============================================"
echo "Step 1: reproduce Table 8 (Hellinger + Ward)"
echo "============================================"
$PYTHON scripts/scop_clustering.py --tier all --grid 100

echo "============================================"
echo "Step 2: clustering variants"
echo "============================================"
$PYTHON scripts/clustering_variants.py --tier all --grid 100

echo "============================================"
echo "Step 3: kNN retrieval (local class signal)"
echo "============================================"
$PYTHON scripts/clustering_knn.py --tier all --grid 100

echo "============================================"
echo "Done. Output files:"
echo "-------------------------------------------"
ls -la results/clustering/
