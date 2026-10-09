# SCOP hybrid clustering stage — how to run it (handoff note; nothing was executed here)

Implements the pre-registered experiment defined in
`reports/scop_hybrid_feasibility_2026-10-09-todo.md` (protocol §4–§5, file plan §8).
**No code was run while creating these files** (per your constraint). The
syntax check was done by reading, not executing.

## New files (this stage only)

| File | Purpose |
|---|---|
| `scripts/clustering_hybrid.py` | The pre-registered ablation: R0 (density Hellinger), R1 (B1 circular stats), R2 (B2 sequential stats), R3 (hybrid, **full α grid**, never tuned), R4 (model-free empirical histogram). Optional `--nested`, `--knn`, `--strict-run-bound` flags. Writes **only** into `results/clustering_hybrid/`. |
| `scripts/run_clustering_hybrid.sh` | Linux/macOS (or Git Bash on Windows) runner: smoke test → main ablation → knn rows → strict-run sensitivity → nested rows. `PYTHON` overridable. |
| `scripts/check_hybrid_env.py` | Read-only smoke test: env versions, schema, cache/checkpoint presence, feature shapes `[n,8]`/`[n,6]`, "every protein has ≥5 valid sequential pairs", Spearman drift ranges re-checked against the measured §9(b) values. Writes nothing. |
| `scripts/run_hybrid_windows.bat` | Optional Windows starter: probes the Anaconda env roots (including `D:\ProgramData\anaconda3\envs`), smoke-tests each candidate python, then runs the same pipeline with the first env that passes. |
| `scripts/README_hybrid.md` | This file. |

Nothing in `fff/**`, `runs/**`, `SCOP/**`, `paper/**`, or `results/clustering/**` was read-modified or touched; `results/clustering_hybrid/` is a brand-new directory so the verified artifacts can't be overwritten.

## Linux box (where the runner will actually be used)

```bash
cd <repo-root>
# default interpreter matches run_clustering_full.sh
bash scripts/run_clustering_hybrid.sh
# or with an env override
PYTHON=/data/python-envs/pytorch/bin/python bash scripts/run_clustering_hybrid.sh
```

The script is self-contained: it creates `results/clustering_hybrid/` on first run and exits nonzero if anything (cache missing, feature loss, env issue) fails the smoke test, so it can't silently write a half-baked ARI table.

## Windows (do a smoke test with the Anaconda envs you mentioned)

You don't need the .bat if you just want to check the env; the smoke test alone is enough:

```bat
D:\ProgramData\anaconda3\envs\<your_env>\python.exe scripts\check_hybrid_env.py --grid 100
```

That confirms `numpy/pandas/scipy/sklearn` versions, both `theta/tau/res_seq` columns, checkpoint + cache presence, protein alignment, and that features build as `[n,8]`/`[n,6]` for **all four tiers**, before the runner does anything else. It should exit 0 with "SMOKE TEST PASSED (no files written)".

Then run the .bat (it picks the first env that passes as `python.exe`), or simply:

```bat
set PYTHON=D:\ProgramData\anaconda3\envs\<env>\python.exe
bash scripts/run_clustering_hybrid.sh   REM if you have Git Bash / WSL
```

## What the outputs mean

For each tier you get one CSV (`hybrid_<tier>_g100.csv`) whose ARI/NMI rows are:

- `R0 density/Hellinger` — the PC-NCSF model distance (Scale-normalised), Ward/average/complete at k = n_categories (fixed, no k-sweep).
- `R1 B1-stats` / `R2 B2-seq` — the handcrafted/sequential-feature-only distances.
- `R3 hybrid` at α ∈ {0, 0.25, 0.5, 0.75, 1} — the full pre-registered grid; **the α values are reported, never "the best" picked**. α=0 is mathematically the "pure auxiliary" case; α=1 is effectively R0 again, so the grid output also gives you the "0" and "1" endpoints to eyeball.
- `R4 empirical-hist` — the model-free comparator that a reviewer will want alongside anything you claim.

Optional extras (disclosed, not "leaky" ways to make the headline number look nicer):
- `--knn` appends 5-NN retrieval accuracy of each representation plus the imbalance-preserving permutation null and majority-class fraction (SAM-style local-signal check, mirrors the final-check report §5).
- `--strict-run-bound` is the B2 sensitivity variant: instead of just requiring `Δres_seq == 1` adjacency, it *drops* any protein whose pdb_id block contains a reset (`Δ res_seq ≤ 0`), i.e. it approximates the 883/975/168/138 strictly-increasing-res_seq protein subsets per tier.
- `--nested` appends nested-kwargs-selected α rows (mean ± sd over 10 fixed half-split seeds, swap-halves), labelled as label-informed per report §5.4.

## What the runner checks/does NOT check for you

- It fails loudly if a density cache for any tier is missing under `results/clustering/`, so it never silently falls back to `protein_densities()` (which would need torch + zuko). **If you actually want a missing cache to be recomputed, run `scripts/scop_clustering.py` for that tier first** — the audit already replaced that step in this pipeline.
- It never writes into the verified `results/clustering/` directory.
- Labels are read for ARI/NMI and the nested half-splitting only; no label-driven feature tuning, no k-sweep, no linkage cherry-picking.

## Cost

- Smoke test: seconds per tier (loads the cache once, builds two 1590-ish distance matrices, no torch, no flow forward pass). Well under 1 GB peak memory.
- Main ablation, `--knn`, `--strict-run-bound`: roughly the <9 s/tier the audit measured for features + distance build, plus a few seconds of `linkage()` per row.
- `--nested` is the slowest step (10 seeds × 2 half-splits × 5 α × 3 linkages of half-size `<n²/4` matrices per tier); expect O(minutes) total for all four tiers, not hours.
