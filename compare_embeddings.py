import os
import pandas as pd
import matplotlib.pyplot as plt

# مسیر پوشه runs/var-emb
base_dir = os.path.join("runs", "var-emb")

results = []

for run_folder in os.listdir(base_dir):
    run_path = os.path.join(base_dir, run_folder)
    metrics_path = os.path.join(run_path, "metrics.csv")
    if not os.path.exists(metrics_path):
        continue

    # خواندن CSV
    df = pd.read_csv(metrics_path)

    # کمترین مقدار val_loss
    min_val = df["val_loss"].min()

    # استخراج embedding_dim از نام پوشه
    parts = run_folder.split("_")
    ed = None
    for p in parts:
        if p.startswith("ed"):
            ed = int(p.replace("ed", ""))
            break

    results.append({"embedding_dim": ed, "min_val_loss": min_val})

# ساخت DataFrame نهایی
summary_df = pd.DataFrame(results).sort_values("embedding_dim")
print(summary_df)

# ذخیره در فایل CSV
summary_csv = os.path.join(base_dir, "embedding_comparison.csv")
summary_df.to_csv(summary_csv, index=False)

# رسم نمودار میله‌ای
plt.figure(figsize=(6, 5))
plt.bar(summary_df["embedding_dim"], summary_df["min_val_loss"], color="skyblue")
plt.xlabel("Embedding Dimension")
plt.ylabel("Minimum Validation Loss")
plt.title("Comparison of Validation Loss across Embedding Sizes")
plt.xticks(summary_df["embedding_dim"])
plt.tight_layout()

# ذخیره شکل
plt.savefig(os.path.join(base_dir, "embedding_comparison.png"), dpi=150)
plt.close()
