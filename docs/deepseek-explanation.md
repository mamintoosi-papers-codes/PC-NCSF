# Protein Torsion Angle Analysis Using Conditional Normalizing Flows

## 📋 Project Overview

This project implements a **conditional normalizing flow framework** to model protein backbone torsion angles (φ, ψ) with residue-level conditioning. The model learns distinct probability distributions for different protein residues using Neural Spline Flows (NSF) with embedding-based conditioning, enabling both density estimation and generation of biologically realistic torsion angles.

## 🧬 Scientific Context

Protein structure is fundamentally determined by torsion angles along the backbone:
- **φ (phi)**: C-N-Cα-C torsion angle
- **ψ (psi)**: N-Cα-C-N torsion angle  
- **Ramachandran plots**: 2D distributions of (φ, ψ) angles that define secondary structure elements

Traditional approaches model these distributions generically, while our conditional approach captures residue-specific preferences.

## 🏗️ Architecture Components

### 1. **Data Processing Pipeline**

```python
# Modified data loading with enhanced residue identification
def _extract_res(name: str) -> str:
    """Extract protein identifier from complex name field"""
    # Input: '119L:ChainA:ASN2' → Output: '119L' (unique protein identifier)
    p_name = str(name).split(":")[0]              
    return p_name.upper()

# Result: 496 unique protein conditions (increased from 20 residue types)
```

### 2. **Mathematical Framework**

The model learns the conditional probability distribution:
```
p(φ, ψ | protein_id)
```

Using a normalizing flow transformation:
```
z = f(φ, ψ; θ_protein_id) where z ∼ N(0, I)
```

### 3. **Model Architecture**

```python
# Embedding layer: Integer protein IDs → continuous representations
embedding = nn.Embedding(num_embeddings=496, embedding_dim=32)

# Conditional Neural Spline Flow
flow = zuko.flows.NCSF(
    features=2,           # (φ, ψ) angles
    context=32,           # embedding dimension
    transforms=8,         # number of flow layers
    hidden_features=[128]*3  # 3 hidden layers with 128 units
)
```

## 🔧 Key Implementation Details

### **Condition Processing Innovation**
The critical modification from un-conditional to protein-specific conditioning.

 496 unique protein identifiers (119L, 1A0R, 2XYZ, etc.)

This enables:
- Protein-specific torsion angle distributions
- Fine-grained structural modeling
- Capture of protein-family specific preferences

### **Training Objective**
Minimize negative log-likelihood for conditional density estimation:
```python
loss = -𝔼[log p(φ, ψ | protein_id)]
```

### **Technical Implementation**
```python
# Integer protein IDs → embedding vectors
protein_embedding = embedding(protein_ids)  # shape: (batch_size, 32)

# Conditional density estimation
log_prob = flow(protein_embedding).log_prob(angles)  # shape: (batch_size)
```

## 📊 Dataset Characteristics

### **Original Data Structure**
- **Source**: Torus protein dataset (TSV format)
- **Columns**: name, phi, psi, subtype  (20 standard amino acid types)

### **Enhanced Conditioning**
- **Protein-based**: 496 unique protein identifiers
- **Sample size**: ~160,000 torsion angle measurements
- **Split**: 80% train, 20% validation

### **Data Transformation**
```python
# Angular embedding: φ, ψ → S¹ × S¹ (torus)
phi_embed = embed_angle_in_2d(φ * 2π / 360)  # Circle embedding
psi_embed = embed_angle_in_2d(ψ * 2π / 360)
data = torch.cat([phi_embed, psi_embed], dim=-1)  # Torus coordinates
```

## 🎯 Scientific Applications

### 1. **Protein-Specific Modeling**
Learn distinct Ramachandran distributions for:
- Different protein families
- Structural motifs
- Functional classes

### 2. **Generative Modeling**
```python
# Sample novel torsion angles for specific proteins
samples = flow(protein_embedding).sample(num_samples=1000)
```

### 3. **Structural Analysis**
- Identify unusual torsion angle preferences
- Detect protein-family signatures
- Validate structural models

## 📈 Model Performance

### **Training Configuration**
```yaml
batch_size: 128
learning_rate: 2e-4
embedding_dim: 32
flow_transforms: 8
hidden_layers: 3
hidden_units: 128
epochs: 20
early_stopping_patience: 10
```

### **Evaluation Metrics**
- Negative log-likelihood (NLL) on test set
- Sample quality metrics
- Structural validity checks

## 🔍 Advanced Features

### **Visualization Suite**
1. **Conditional Density Plots**: Ramachandran-like plots for each protein
2. **Embedding Space Analysis**: PCA visualization of protein representations
3. **Loss Curves**: Training dynamics monitoring
4. **Sample Comparison**: Real vs generated distributions

### **Technical Innovations**
- **Mixed precision training**: Accelerated computation
- **Gradient scaling**: Stable training
- **OneCycle scheduling**: Optimal learning rate progression
- **Early stopping**: Prevent overfitting

## 🚀 Future Directions

### **Immediate Extensions**
1. **Hierarchical conditioning**: Protein family + residue type
2. **Secondary structure conditioning**: α-helix, β-sheet, coil
3. **Environmental factors**: Solvent accessibility, pH dependence

### **Advanced Applications**
1. **Protein design**: Generate novel sequences with desired torsion profiles
2. **Structure prediction**: Priors for torsion angle prediction
3. **Disease modeling**: Mutational effects on structural preferences

## 📋 Output and Analysis

### **Generated Artifacts**
- `loss_curves.png`: Training progression
- `metrics.csv`: Quantitative evaluation
- `contour_cond_*.png`: Protein-specific density plots (496 conditions)
- `embedding_space.png`: Protein representation analysis
- `best_flow.pt`: Trained model checkpoint

### **Analysis Capabilities**
- Compare torsion distributions across protein families
- Identify outlier proteins with unusual angle preferences
- Generate synthetic data for underrepresented proteins

## 🎯 Conclusion

This implementation provides a powerful framework for protein-specific torsion angle modeling, enabling both analytical insights and generative capabilities for structural biology applications. The shift from residue-type to protein-specific conditioning represents a significant advancement in granularity and biological relevance.