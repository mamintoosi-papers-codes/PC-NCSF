# Protein-Conditional Free-form Flows for Backbone Torsion Angle Modeling

This repository implements **Protein-Conditional Free-Form Flows (PC-FFF)** for **protein-specific** modeling of protein backbone torsion angle distributions on the torus manifold.

## 📋 Overview

Protein structure is largely determined by the distribution of backbone torsion angles $(\phi, \psi)$. While traditional Ramachandran plots provide a global view of these distributions, they fail to capture **protein-specific** preferences. Building on recent advances in normalizing flows on manifolds, particularly Free-form Flows (FFF), we implement **Protein-Conditional Free-form Flows (PC-FFF)** that condition the flow transformation on **protein identity**, enabling **protein-specific** modeling of torsion angle distributions.

By embedding **protein identifiers** into continuous representations, our framework learns distinct conditional densities $p(\phi, \psi \mid \text{protein})$. Experiments on the Torus Protein dataset demonstrate that PC-FFF significantly improves validation likelihoods compared to unconditional baselines, yielding more realistic protein-specific distributions.

This work highlights the importance of **protein-aware** conditioning for generative modeling of protein backbone conformations on the torus manifold.

The implementation includes two main approaches:

1. **FFF (Unconditional)**: Standard free-form flow for torus density estimation
2. **PC-FFF (Conditional)**: Conditioned free-form flow that incorporates protein type information via embedding layers

## 🏗️ Project Structure

```
.
├── fff/                     # Core FFF library components
├── runs/                    # Training outputs and checkpoints
├── reports/                 # Generated reports and visualizations
├── configs/                 # Configuration files
├── circularspline_protein.py              # Unconditional model training script
├── circularspline_protein_embedding_c.py  # Conditional model training script
├── report_generator.py      # Report generation and visualization
├── compare_batch_size.py    # Batch size comparison utility
├── compare_embeddings.py    # Embedding dimension comparison utility
└── README.md
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training Models

**Unconditional Model (FFF):**
```bash
python circularspline_protein.py --batch-size 256 --epochs 40
```

**Conditional Model (PC-FFF):**
```bash
python circularspline_protein_embedding_c.py --batch-size 256 --epochs 40 --embedding-dim 16 --plot-all
```

### Generating Reports

After training, generate comprehensive reports and visualizations:
```bash
python report_generator.py
```

This will create:
- Density contour plots for all trained models
- Loss comparison curves
- Excel summary with detailed metrics and model configurations

## 📊 Model Types

### FFF (Free-form Flow)
- Unconditional density estimation on the torus
- Learns the distribution of protein backbone dihedral angles
- Output files prefixed with `uncond_bs*_ep*`

### PC-FFF (Conditioned Free-form Flow)
- Conditional density estimation using residue type embeddings
- Incorporates protein sequence information
- Output files prefixed with `cond_bs*_ep*_ed*`

## 📈 Output Files

The training process generates:

1. **Checkpoints**: Model weights in `runs/` directory
2. **Metrics**: Training/validation loss curves in CSV format
3. **Visualizations**: 
   - Density contour plots (Φ-Ψ space)
   - Loss comparison curves
   - Model performance summaries

Report generator creates:
- `loss_curves_comparison.png`: Comparison of FFF vs PC-FFF performance
- `loss_curves_summary.xlsx`: Detailed Excel report with:
  - All model data
  - Summary statistics
  - Model configurations
  - Separate sheets for each model type

## ⚙️ Configuration

Key hyperparameters:
- `--batch-size`: Training batch size (64, 128, 256, etc.)
- `--epochs`: Number of training epochs
- `--embedding-dim`: Embedding dimension for conditional models (4, 8, 16, 32)
- `--plot-all`: Generate visualizations during training

## 📖 Citation

If you use this code in your research, please cite the original FFF papers:

```bibtex
@inproceedings{PC-FFF2025,
    title = {{Protein-Conditional Free-form Flows for Backbone Torsion Angle Modeling}},
    author = {...},
    booktitle = {arXiv},
    year = {2025}
}
```

## 🤝 Contributing

This implementation is based on the official FFF repository:
https://github.com/vislearn/FFF

For issues and contributions related to this specific protein torsion angle implementation, please open an issue in this repository.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.