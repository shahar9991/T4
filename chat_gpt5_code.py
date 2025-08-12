import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# === Path setup ===
csv_path = r"C:\Users\shaha\Downloads\participant_task_matrix.csv"
  # File should be in the same folder as script
out_dir = Path.cwd() / "rdm_outputs"
out_dir.mkdir(exist_ok=True, parents=True)

# === Load CSV, set participant names as index ===
df = pd.read_csv(csv_path, index_col=0)

# === Convert percentage-like strings to numeric values ===
def parse_percentish(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", ".")
    m = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*$", s)
    if m:
        val = float(m.group(1))
        return val / 100.0 if "%" in s else val
    return pd.to_numeric(s, errors="coerce")

X = df.applymap(parse_percentish)

# === If values look like 0–100, convert to 0–1 ===
if X.max(numeric_only=True).max() > 1.5:
    X = X / 100.0

# === Compute correlation between tasks (columns) ===
C = X.corr(method="pearson")
RDM = 1.0 - C
np.fill_diagonal(RDM.values, 0.0)

# === Save cleaned data and results ===
X.to_csv(out_dir / "participant_task_matrix_clean.csv", encoding="utf-8-sig")
C.to_csv(out_dir / "correlation_matrix.csv", encoding="utf-8-sig")
RDM.to_csv(out_dir / "RDM_matrix.csv", encoding="utf-8-sig")

with pd.ExcelWriter(out_dir / "RDM_outputs.xlsx") as writer:
    X.to_excel(writer, sheet_name="InputMatrix")
    C.to_excel(writer, sheet_name="Correlation")
    RDM.to_excel(writer, sheet_name="RDM")

# === Plot heatmap ===
plt.figure(figsize=(7, 6))
plt.imshow(RDM, interpolation="nearest", cmap="viridis")
plt.title("RDM (1 - correlation)")
plt.colorbar()
plt.xticks(ticks=np.arange(RDM.shape[1]), labels=RDM.columns, rotation=45, ha="right")
plt.yticks(ticks=np.arange(RDM.shape[0]), labels=RDM.index)
plt.tight_layout()
plt.savefig(out_dir / "RDM_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()

print("Files saved to:", out_dir.as_posix())
