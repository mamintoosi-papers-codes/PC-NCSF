import os
import pandas as pd
import matplotlib.pyplot as plt

# مسیر پوشه runs/var-batch-size
base_dir = os.path.join("runs", "var-batch-size")

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
    min_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]

    # استخراج batch size از نام پوشه
    parts = run_folder.split("_")
    bs = None
    for p in parts:
        if p.startswith("bs"):
            bs = int(p.replace("bs", ""))
            break

    results.append({"batch_size": bs, "min_val_loss": min_val, "epoch": min_epoch})

# ساخت DataFrame نهایی
summary_df = pd.DataFrame(results).sort_values("batch_size")
print(summary_df)

# ذخیره در فایل CSV
summary_csv = os.path.join(base_dir, "batchsize_comparison.csv")
summary_df.to_csv(summary_csv, index=False)

# رسم نمودار میله‌ای
plt.figure(figsize=(6, 5))
plt.bar(summary_df["batch_size"], summary_df["min_val_loss"], color="lightgreen")
plt.xlabel("Batch Size")
plt.ylabel("Minimum Validation Loss")
plt.title("Comparison of Validation Loss across Batch Sizes")
plt.xticks(summary_df["batch_size"])
plt.tight_layout()

# ذخیره شکل
plt.savefig(os.path.join(base_dir, "batchsize_comparison.png"), dpi=150)
plt.close()
