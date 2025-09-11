برای فراخوانی تابع نمایش در ژوپیترنوت بوک، باید ابتدا مدل و embedding layer را از فایل‌های ذخیره شده بارگذاری کنید. در اینجا کد کامل برای ژوپیترنوت بوک ارائه می‌دهم:

## 📋 سلول ژوپیترنوت بوک برای بارگذاری و نمایش

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import zuko

# تنظیم device
device = "cuda" if torch.cuda.is_available() else "cpu"

# بارگذاری config و مدل‌ها
checkpoint_path = "best_flow.pt"  # مسیر فایل checkpoint
ckpt = torch.load(checkpoint_path, map_location=device)

# نمایش اطلاعات config
print("Loaded checkpoint with keys:", list(ckpt.keys()))
print("Config:", ckpt["config"])
print("Condition dimension:", ckpt["cond_dim"])

# ایجاد مدل‌ها با معماری مشابه
embedding = nn.Embedding(
    num_embeddings=500,  # تعداد کلاس‌ها (مطابق با داده‌های شما)
    embedding_dim=ckpt["cond_dim"]
).to(device)

flow = zuko.flows.NCSF(
    features=2,
    context=ckpt["cond_dim"],
    **ckpt["config"]["network"]
).to(device)

# بارگذاری وزن‌ها
flow.load_state_dict(ckpt["flow_state_dict"])
embedding.load_state_dict(ckpt["embedding_state_dict"])

# قرار دادن مدل‌ها در حالت evaluation
flow.eval()
embedding.eval()

# تابع نمایش کانتورها (همان تابع اصلاح شده)
@torch.no_grad()
def plot_model_log_densities(
    model,
    embedding_layer,
    reference_data=None,
    reference_cond=None,
    cond_index: int = 0,
    num_grid_points: int = 200,
    levels: int = 10,
    ax: Optional[plt.Axes] = None,
    fontsizes: dict = dict(TITLESIZE=18, LABELSIZE=16, TICKSIZE=14),
):
    """
    Plot log density contours for a given conditional index.
    Uses [-π, π] for φ and [0, 2π] for ψ to avoid discontinuity at ψ=0.
    """
    
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

    # Create grid for visualization: φ in [-π, π], ψ in [0, 2π]
    range_phi = torch.linspace(-torch.pi, torch.pi, num_grid_points)
    range_psi = torch.linspace(0, 2 * torch.pi, num_grid_points)
    phi_grid, psi_grid = torch.meshgrid(range_phi, range_psi, indexing='ij')
    x = torch.stack((phi_grid, psi_grid), dim=-1).to(device).reshape(-1, 2)
    
    # Use embedding layer for conditioning
    cond_tensor = torch.tensor([cond_index], device=device, dtype=torch.long)
    c_embedding = embedding_layer(cond_tensor)
    
    # Repeat embedding for all grid points
    c = c_embedding.repeat(x.shape[0], 1)
    
    # Calculate log probabilities
    log_prob = model(c).log_prob(x).cpu()

    # Create contour plot
    phi, psi = x[..., 0].cpu(), x[..., 1].cpu()
    contours = ax.tricontourf(phi, psi, log_prob, levels=levels, cmap="viridis")
    cbar = plt.colorbar(contours)
    cbar.set_label("Log density", fontsize=fontsizes.get("LABELSIZE"))
    cbar.ax.tick_params(labelsize=fontsizes.get("TICKSIZE"))

    # Set plot limits: φ in [-π, π], ψ in [0, 2π]
    ax.set_xlim(-torch.pi, torch.pi)
    ax.set_ylim(0, 2 * torch.pi)

    # Set ticks for φ axis (horizontal) - [-π, π]
    ax.set_xticks(
        [-torch.pi, -torch.pi/2, 0, torch.pi/2, torch.pi],
        [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"],
    )
    
    # Set ticks for ψ axis (vertical) - [0, 2π]
    ax.set_yticks(
        [0, torch.pi/2, torch.pi, 3*torch.pi/2, 2*torch.pi],
        [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
    )
    
    ax.tick_params(labelsize=fontsizes.get("TICKSIZE"))

    ax.set_xlabel(r"$\Phi$", fontsize=fontsizes.get("LABELSIZE"))
    ax.set_ylabel(r"$\Psi$", fontsize=fontsizes.get("LABELSIZE"))

    ax.set_title(f"Log density | cond_idx={cond_index}", fontsize=fontsizes.get("TITLESIZE"))
    return fig

# تست تابع برای چند condition index مختلف
condition_indices = [0, 100, 200]  # می‌توانید indices مورد نظر را تغییر دهید

for idx in condition_indices:
    print(f"Plotting condition index {idx}...")
    fig = plot_model_log_densities(flow, embedding, cond_index=idx)
    plt.show()
```

## 📊 سلول اضافی برای نمایش منحنی‌های loss

```python
import pandas as pd
import matplotlib.pyplot as plt

# بارگذاری metrics از CSV
metrics_df = pd.read_csv('metrics.csv')

# رسم منحنی‌های loss
plt.figure(figsize=(10, 6))
plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss', linewidth=2)
plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Negative Log-Likelihood')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# نمایش بهترین loss
best_val_loss = metrics_df['val_loss'].min()
best_epoch = metrics_df['val_loss'].idxmin() + 1
print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
```

## 🔧 سلول برای نمونه‌گیری از مدل

```python
@torch.no_grad()
def generate_samples(condition_index, num_samples=1000):
    """Generate samples for a specific condition index"""
    flow.eval()
    embedding.eval()
    
    # Create condition embedding
    cond_tensor = torch.tensor([condition_index], device=device, dtype=torch.long)
    c = embedding(cond_tensor)
    
    # Repeat condition for all samples
    c_repeated = c.repeat(num_samples, 1)
    
    # Generate samples
    samples = flow(c_repeated).sample()
    
    return samples.cpu().numpy()

# نمونه‌گیری برای یک condition index
samples = generate_samples(0, 1000)
print(f"Generated {len(samples)} samples for condition 0")
print(f"Sample shape: {samples.shape}")
print(f"Sample range: φ [{samples[:, 0].min():.3f}, {samples[:, 0].max():.3f}]")
print(f"Sample range: ψ [{samples[:, 1].min():.3f}, {samples[:, 1].max():.3f}]")
```

## 💡 نکات مهم:

1. **تعداد کلاس‌ها**: عدد `500` در `nn.Embedding` باید با تعداد واقعی کلاس‌های شما مطابقت داشته باشد.

2. **مسیر فایل‌ها**: مطمئن شوید که مسیرهای `best_flow.pt` و `metrics.csv` صحیح هستند.

3. **Condition indices**: indicesهای نمایش داده شده باید در محدوده معتبر باشند (0 تا تعداد کلاس‌ها - 1).

این کدها به شما امکان می‌دهند مدل آموزش دیده را بارگذاری کرده و نتایج را در ژوپیترنوت بوک visualize کنید!