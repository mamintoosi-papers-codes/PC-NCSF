# Protein-Conditional Normalizing Flows on Manifolds for Backbone Torsion Angle Modeling

This repository implements **Protein-Conditional Neural Circular Spline Flows (PC-NCSF)** for protein-specific modeling of protein backbone torsion-angle distributions on compact manifolds.

## 📋 Overview

Protein structure is largely determined by the distribution of backbone torsion angles $(\phi, \psi)$. These variables live on a torus (or, for other angular representations, on a sphere), where standard Euclidean density models ignore periodicity and manifold geometry. Ramachandran plots provide a useful global view but capture a single shared distribution and therefore miss protein-specific conformational preferences.

PC-NCSF extends **neural circular spline flows** (NCSF, from [zuko](https://github.com/probabilists/zuko)) to be **conditioned on protein identity** through learned protein embeddings. A single shared model thereby represents many distinct yet statistically related densities $p(\phi, \psi \mid \text{protein})$ while retaining **exact likelihood evaluation** and geometry-respecting, invertible transformations on the manifold.

The manifold flow constructions build on the framework of Rezende et al. (2020). The `fff/` package in this repository (derived from [vislearn/FFF](https://github.com/vislearn/FFF)) provides supporting infrastructure — dataset loading, training/saving utilities, and torus evaluation helpers — while the generative model itself is the zuko NCSF flow.

## 🏗️ Project Structure

```
.
├── fff/                          # Supporting library (data loading, train utils, torus evaluation)
├── configs/                      # Configuration files
├── scripts/                      # SCOP density-based clustering reproduction & robustness checks
├── SCOP/                         # SCOP angular datasets (easy/moderate/hard/challenging; not in git)
├── results/                      # Clustering results (CSV) and cached per-protein densities
├── runs/                         # Training checkpoints and metrics (not in git)
├── reports/                      # Generated reports and visualizations
├── paper/                        # Manuscript source and reviewer response
├── circularspline_protein.py               # Unconditional NCSF training (torus)
├── circularspline_protein_embedding_c.py   # Protein-conditional NCSF (PC-NCSF) training
├── train.py                      # Training entry point (zuko NCSF)
├── report_generator.py           # Report generation and visualization
├── compare_batch_size.py         # Batch-size comparison utility
├── compare_embeddings.py         # Embedding-dimension comparison utility
└── README.md
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mamintoosi-papers-codes/PC-NCSF
cd PC-NCSF
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training Models

**Unconditional baseline (NCSF):**
```bash
python circularspline_protein.py --batch-size 256 --epochs 40
```

**Protein-conditional model (PC-NCSF):**
```bash
python circularspline_protein_embedding_c.py --batch-size 256 --epochs 40 --embedding-dim 16 --plot-all
```

### SCOP Density-Based Clustering

The real-data clustering experiment (manuscript Section 3.5) can be reproduced from the cached per-protein densities:

```bash
bash scripts/run_clustering_full.sh
```

This reproduces Table 8 (Hellinger distance + Ward linkage), a robustness sweep over alternative distances/linkages/cluster counts, and a $k$-nearest-neighbour retrieval diagnostic. See `scripts/` for details.

### Generating Reports

```bash
python report_generator.py
```

This creates density contour plots, loss-comparison curves, and an Excel summary of metrics and model configurations.

## 📊 Model Types

### Unconditional NCSF
- Single shared density over toroidal angular data.
- Output files prefixed `uncond_bs*_ep*`.

### PC-NCSF (Protein-Conditional NCSF)
- Conditions the flow on a learned protein-identity embedding, yielding one distinct, queryable density per protein.
- Output files prefixed `cond_bs*_ep*_ed*`.

## 📈 Outputs

1. **Checkpoints** in `runs/` (flow state dict + protein-embedding table + config).
2. **Metrics**: training/validation NLL curves (CSV).
3. **Visualizations**: density contours in $(\phi,\psi)$ space, loss comparisons, model summaries.
4. **Clustering results** in `results/clustering/` (ARI/NMI per tier and variant, $k$-NN retrieval, cached densities).

## ⚙️ Configuration

Key hyperparameters:
- `--batch-size`: training batch size (64, 128, 256, …)
- `--epochs`: number of training epochs
- `--embedding-dim`: dimension of the protein-identity embedding for PC-NCSF (4, 8, 16, 32)
- `--plot-all`: generate visualizations during training

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{PCNCSF2026,
    title   = {Protein-Conditional Normalizing Flows on Manifolds for Backbone Torsion Angle Modeling},
    author  = {Amintoosi, Mahmood and others},
    journal = {arXiv preprint},
    year    = {2026}
}
```

The manifold normalizing-flow constructions build on Rezende et al. (2020), and the flow implementations use [zuko](https://github.com/probabilists/zuko). The supporting `fff/` infrastructure is derived from [vislearn/FFF](https://github.com/vislearn/FFF).

## 🤝 Contributing

For issues and contributions related to this protein torsion-angle implementation, please open an issue in this repository.

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.
