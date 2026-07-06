"""Majority-vote ensemble of submission CSVs (id,label). Ties -> 0 (hallucinated).

Usage: python3 ensemble.py out.csv sub1.csv sub2.csv sub3.csv ...
Use an ODD number of decorrelated submissions when possible.
"""
import sys
import pandas as pd

out, paths = sys.argv[1], sys.argv[2:]
assert len(paths) >= 2, "need at least 2 submissions"
dfs = [pd.read_csv(p).sort_values("id").reset_index(drop=True) for p in paths]
ids = dfs[0]["id"]
for d, p in zip(dfs, paths):
    assert (d["id"].values == ids.values).all(), f"id mismatch in {p}"
votes = sum(d["label"].values for d in dfs)
labels = (votes > len(dfs) / 2).astype(int)  # strict majority for 1; ties -> 0
sub = pd.DataFrame({"id": ids, "label": labels})
sub.to_csv(out, index=False)
agree_all = (sum(d["label"].values == labels for d in dfs) == len(dfs)).mean()
print(f"wrote {out}: {sub.label.value_counts().to_dict()}  unanimous={agree_all:.1%}")
