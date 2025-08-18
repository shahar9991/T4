import pandas as pd
import re
from pathlib import Path

# ===== Input path =====
DATA_PATH = r"C:\Users\shaha\Desktop\book1.xlsx"   # change if needed

# ----- helpers -----
def normalize(s):
    return str(s).strip().lower()

ALIASES = {
    "answer":  {"answer", "resp", "response", "ans"},
    "env":     {"env", "stim", "image", "file", "filename", "path"},
    "rt":      {"rt", "rt_ms", "reaction_time", "latency", "time", "rt (ms)", "rt(ms)"},
    "correct": {"correct", "gt", "ground_truth", "truth", "label"},
}

def auto_load(path: str) -> pd.DataFrame:
    """Try to read Excel/CSV even if header row is not the first row."""
    p = Path(path)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        # try several header rows (0..8); pick the first that contains at least two keywords
        for h in range(0, 9):
            df = pd.read_excel(p, header=h)
            cols = {normalize(c) for c in df.columns}
            if any(k in cols for k in ALIASES["answer"] | ALIASES["env"] | ALIASES["rt"] | ALIASES["correct"]):
                return df
        # fallback: no header
        return pd.read_excel(p, header=None)
    else:
        return pd.read_csv(p)

def find_columns(df: pd.DataFrame):
    """Map real column names -> canonical names (answer/env/rt/correct)."""
    norm_map = {c: normalize(c) for c in df.columns}
    rev = {}
    for canon, options in ALIASES.items():
        for real, norm in norm_map.items():
            if norm in options:
                rev[canon] = real
                break
    return rev

# ===== Load and align columns =====
df0 = auto_load(DATA_PATH)
colmap = find_columns(df0)

# If some names still missing, try to guess by content patterns
if "env" not in colmap:
    # column that looks like a file path with ".jpg" or contains "shifty"
    for c in df0.columns:
        col = df0[c].astype(str).str.lower()
        if col.str.contains(r"\.jpg|shifty_", regex=True).any():
            colmap["env"] = c
            break

if "rt" not in colmap:
    # numeric-looking column
    numcands = [c for c in df0.columns if pd.to_numeric(df0[c], errors="coerce").notna().mean() > 0.9]
    if numcands:
        colmap["rt"] = numcands[0]

# Final sanity check
required = {"answer", "env", "rt", "correct"}
missing = required - set(colmap)
if missing:
    raise RuntimeError(f"Could not find columns for: {', '.join(sorted(missing))}. "
                       f"Found mapping: {colmap}. Available columns: {list(df0.columns)}")

df = df0[[colmap["answer"], colmap["env"], colmap["rt"], colmap["correct"]]].copy()
df.columns = ["answer", "env", "rt", "correct"]

# ===== Clean values =====
df = df.dropna(subset=["answer", "env", "rt", "correct"])
df["answer"]  = df["answer"].astype(str).str.lower().str.strip()
df["correct"] = df["correct"].astype(str).str.lower().str.strip()
df["env"]     = df["env"].astype(str)
df["rt"]      = pd.to_numeric(df["rt"], errors="coerce")

# Extract the signed number after 'shifty_' and take absolute value to collapse +/-N
df["condition"] = (
    df["env"].str.extract(r"[sS]hifty_(-?\d+)", expand=False)
             .astype(float)
             .abs()               # <-- collapse -N and +N
             .astype(int)
)

# Keep only rows with a valid condition
df = df.dropna(subset=["condition"]).copy()
df["condition"] = df["condition"].astype(int)

# Define correctness: only left/right count as correct when equal
df["is_correct"] = ((df["answer"].isin(["left", "right"])) &
                    (df["answer"] == df["correct"])).astype(int)

# ===== Aggregate by condition =====
summary = (
    df.groupby("condition")
      .agg(
          n_trials     = ("is_correct", "size"),
          accuracy_pct = ("is_correct", lambda x: 100.0 * x.mean()),
          mean_rt_ms   = ("rt", "mean"),
          std_rt_ms    = ("rt", "std"),
          median_rt_ms = ("rt", "median")
      )
      .sort_index()
)

print(summary.round(2))

# Save results
out_path = Path.cwd() / "accuracy_rt_by_condition.csv"
summary.to_csv(out_path, encoding="utf-8-sig")
print(f"\nSaved: {out_path}")
