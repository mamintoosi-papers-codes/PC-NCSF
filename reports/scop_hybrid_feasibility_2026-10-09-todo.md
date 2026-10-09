# SCOP Clustering — Feasibility Audit of Hybrid Representations

**Date:** 2026-10-09
**Scope:** Read-only feasibility audit. No source code, model files, cached results, or manuscript files were modified. Lightweight read-only checks were executed and are itemised in §9.
**Evidence tags:** **[V]** = verified fact (tool-confirmed in this repository), **[I]** = inference from verified facts, **[P]** = proposed experiment (not yet run; no results claimed for it).

---

## 1. Executive recommendation

**Recommendation: Proceed with a narrowly defined B1+B2 hybrid** (density + a small fixed set of circular-statistical and sequential features), evaluated as a single pre-registered ablation study that also yields the B1-only, B2-only, model-only, and model-free-comparator rows at negligible extra cost. **B3 (embedding distance) is rejected** on direct repository evidence. If the hybrid criteria in §7 are not met, the honest fallback is stated there.

Why this option, from repository evidence:

| Evidence | Where | Consequence |
|---|---|---|
| Model density distance is strongly correlated with the raw-histogram distance (Spearman 0.62–0.84) and moderately with candidate statistical features (0.50–0.75) **[V]** | §9 check (b) | B1 alone is substantially redundant with what already exists; a hybrid *with ablations* is the only design that can attribute any gain |
| Sequential (adjacent-residue) features are computable for **100% of proteins** in all four tiers and are only moderately correlated with the model distance (0.54–0.76) **[V]** | §9 check (b) | B2 is feasible and carries partially independent information; it is the genuinely new signal |
| B3: the checkpoint exposes only an (n×8) lookup embedding; clustering it gives ARI ≈ 0, indistinguishable from a random control, in every tier **[V]** | §3.3, `variants_*.csv` | B3 must not be investigated further |
| Full experiment costs ≤ 9 s/tier for all feature+distance computation on CPU; no retraining; caches present **[V]** | §6, §9 | The experiment is cheap enough that "try everything first" is achievable |
| Claim 4 of the four-claim split (hybrid representations) is the only one never attempted **[V]** | prior audit, §6 | This is the last open question before an honest conclusion can be reported |

**B1 alone** would mostly re-test what `clustering_variants.py` already covers (marginals, W1, empirical histograms). **B2 alone** is feasible but cannot show whether PC-NCSF contributes anything beyond handcrafted features. Only the narrow hybrid, with its mandatory ablations, answers the actual scientific question: *does the learned density add clustering information beyond what simple angular statistics and adjacent-residue statistics already provide?*

---

## 2. Existing implementation and artifact audit

### 2.1 How protein-level representations and distances are computed **[V]**

All in [scripts/scop_clustering.py](../scripts/scop_clustering.py):

| Function | What it does |
|---|---|
| `load_proteins(tier)` | Reads `SCOP/<tier>/data.csv`, `dropna(subset=["theta","tau"])`, converts degrees→radians, `pd.factorize(df["pdb_id"])` → condition indices, `pd.unique` → protein list (same order as factorize), majority-vote `category` label per protein |
| `build_model(ckpt_path)` | Rebuilds `zuko.flows.NCSF(2, cond_dim, hidden=[128,128,128], transforms=8)` from checkpoint config; loads `flow_state_dict` and `embedding_state_dict` into `nn.Embedding` |
| `protein_densities(...)` | For each protein i: evaluates `flow(emb.weight[i]).log_prob(grid)` on the uniform 100×100 grid over [-π,π)², multiplies by cell area, renormalises to a probability mass matrix → `[n,100,100]` |
| `hellinger_matrix(masses)` | Vectorised pairwise Hellinger via the √-embedding Gram trick: `H² = 0.5·(‖u‖²+‖v‖²−2⟨u,v⟩)` |
| `run_tier(...)` | Loads/creates cache `results/clustering/masses_<tier>_g100.npy`, computes Hellinger matrix, Ward/average/complete/single linkage cut at `k = n_classes`, ARI/NMI, plus a random-label sanity baseline |

Pipeline: **per-protein conditional density on a fixed grid → Hellinger distance → hierarchical clustering → ARI/NMI vs SCOP labels.**

### 2.2 Which representations already exist in the code **[V]**

[scripts/clustering_variants.py](../scripts/clustering_variants.py) already computes, per tier: joint Hellinger/JS/symmetric-KL, Wasserstein-1 on angular marginals (`marginal_w1`), marginal feature vectors, √-joint features, the **empirical per-protein histogram** (`empirical/Hellinger`, the model-free comparator, with 1e-3 smoothing), the **conditional embedding** features (`features/cond-embeddings`) and a random control, plus spectral/k-means/GMM clusterers and a k-sweep 2..10.

[scripts/clustering_knn.py](../scripts/clustering_knn.py) computes k-NN retrieval (k = 1/5/10) with a label-permutation control.

**Not present anywhere in the repository [V]** (grep over `scripts/*.py`): any circular-statistics feature (mean direction, mean resultant length), any sequential/adjacent-residue feature, any use of `res_seq` for feature construction. So B1's circular statistics and all of B2 are genuinely new code; B1's "marginal distribution" flavour is already partially covered.

### 2.3 Available per-protein artifacts **[V]**

- Checkpoints `runs/scop_<tier>/ep20-bs512/cond/best_flow.pt`, keys = `['config','cond_dim','flow_state_dict','embedding_state_dict']`; `cond_dim = 8`; embedding shape `[n, 8]` (verified `[371,8]` for hard). Config: `network = {hidden_features:[128,128,128], transforms:8}`.
- Cached densities `results/clustering/masses_<tier>_g100.npy`, float32, shapes `(1590,100,100)` / `(1594,…)` / `(371,…)` / `(194,…)` (63 MB / 64 MB / 15 MB / 7.8 MB).
- CSV outputs `scop_clustering_*_g100.csv`, `variants_*.csv`, `knn_retrieval.csv` — all present and internally consistent.
- Nothing else per-protein exists beyond these (no intermediate activations, no per-residue latents are saved).

### 2.4 Missing torsion angles **[V]**

Rows with `NaN` in `theta` or `tau` are dropped **identically** in the training loader ([fff/data/scop.py](../fff/data/scop.py) `get_scop_dataset`) and in `load_proteins` / `load_labels`: dropna → `reset_index(drop=True)` → `factorize`. Dropped rows: easy 4,782 (0.65%), moderate 4,782 (0.79%), hard 1,113 (0.43%), challenging 585 (0.52%) **[V, prior audit §2]**.

**New fact from §9 check (a):** *every* protein in *every* tier has at least one dropped row (1590/1590, 1594/1594, 371/371, 194/194). Consequence for B2: after dropping, a missing residue leaves a `res_seq` gap of ≥2 between its neighbours, so gap-based adjacency rules automatically exclude pairs that straddle a missing residue **[I]**.

### 2.5 Protein IDs, domain IDs, and labels **[V]**

- Conditioning unit is **`pdb_id`**, not `domain_id`: `condition_on="protein"` is not a CSV column, so `get_scop_dataset` falls through to `pdb_id`. n_proteins = 1590/1594/371/194 exactly matches embedding rows (alignment verified in prior audit §4).
- 5 multi-domain PDBs (easy: `1jj2,1lt4,1nmu,1qbk`; challenging: `1ta3`) span two SCOP categories; labels resolved by majority vote; majority-vote vs first-residue labels **disagree 0 times** in all tiers (prior audit §3.4); exclusion test already run in `scripts/clustering_final_check_20261009.py`.
- Tier label granularity is exactly one SCOP level each: easy = class, moderate = fold, hard = superfamily, challenging = family (prior audit §8); challenging is imbalanced (21/98/53/22, max class 50.5%).

### 2.6 Reproducibility without retraining **[V]**

Yes. All four density caches and all checkpoints are present; `scripts/run_clustering_full.sh` regenerates every clustering CSV from caches without any training. Reproduction and sample-index alignment were verified earlier today in [reports/scop_clustering_final_check_2026-10-09.md](scop_clustering_final_check_2026-10-09.md) (cache↔checkpoint alignment proven per tier; exclusion and null-baseline checks re-run).

### 2.7 Existing baseline numbers (context, all k=4, Ward) **[V]**

From `results/clustering/variants_*.csv`:

| Tier | PC-NCSF density (Hellinger) | Model-free empirical histogram | cond-embedding k-means | random control k-means |
|---|---|---|---|---|
| easy | 0.0033 | 0.1160 | 0.0006 | 0.0009 |
| moderate | 0.0590 | 0.3206 | 0.0002 | −0.0006 |
| hard | 0.0761 | 0.0742 | 0.0012 | −0.0023 |
| challenging | 0.0995 | 0.0936 | −0.0064 | 0.0039 |

k-NN 5-NN retrieval: 0.359 / 0.563 / 0.574 / 0.567 vs chance 0.25 (challenging: imbalance-preserving permutation null 0.44, majority-class 0.51 — final-check report §5).

---

## 3. Feasibility of B1, B2, B3

### 3.1 B1 — density + statistical features: **FEASIBLE** **[V]**

**Precise candidate features** (all computable from `data.csv` columns `theta, tau` after the standard dropna; no labels used):

- per protein: circular mean direction and mean resultant length `R = |N⁻¹Σ e^{iφ}|` for `theta` and for `tau`;
- circular means and `R` of the joint coordinates `theta+tau` and `theta−tau` (capturing the diagonal structure of the joint density, e.g. α-helix vs β-sheet basins);
- (optionally, but it already exists in code) 1-D Wasserstein distance on the angular marginals.

8 numbers per protein → Euclidean distance on z-scored features.

**Redundancy, measured (no labels):** Spearman correlations between condensed distance vectors, §9 check (b): `D_model ~ D_stats` = 0.504 / 0.525 / 0.575 / 0.745 (easy/moderate/hard/challenging); `D_emp ~ D_stats` = 0.450 / 0.610 / 0.543 / 0.711. **[V]**

**[I]** The statistical features are *partially* redundant with the learned density — unsurprising, since both summarise the same angles — but far from identical (0.50–0.75 leaves substantial independent variance). A hybrid is therefore mathematically non-degenerate, but any gain must be interpreted against the ablation rows, because a hybrid improvement could be entirely driven by the statistical component.

### 3.2 B2 — density + sequential features: **FEASIBLE, with a data caveat** **[V]**

**Data reality check (§9 check (a)):** `data.csv` has columns `pdb_id, protein_name, domain_id, class, category, res_name, res_seq, theta, tau`. There is **no chain identifier and no insertion-code column**. `res_seq` is duplicated within proteins (707/1590 easy, 619/1594 moderate, 203/371 hard, 56/194 challenging proteins contain duplicated `res_seq` — multi-chain/multi-model entries). Adjacent rows are therefore **not** reliably adjacent residues:

| Tier | adjacent-row pairs | `Δres_seq == 1` | `Δres_seq > 1` | `Δres_seq <= 0` (chain/model resets) |
|---|---|---|---|---|
| easy | 725,720 | 373,360 (51.4%) | 1,478 | 350,882 (48.3%) |
| moderate | 598,570 | 353,975 (59.1%) | 1,986 | 242,609 (40.5%) |
| hard | 255,064 | 130,475 (51.2%) | 199 | 124,390 (48.8%) |
| challenging | 111,820 | 73,194 (65.5%) | 27 | 38,599 (34.5%) |

**Adjacency rule that the data supports [P]:** pair two consecutive rows **within the same `pdb_id` block only if `Δres_seq == 1`**. This automatically (i) skips chain/model resets (`Δ≤0`), (ii) skips gaps from dropped residues (`Δ≥2`), (iii) skips unalignable jumps. Under this rule, **100% of proteins in all four tiers** yield ≥5 valid consecutive pairs (§9 check (b)), so every protein gets a feature vector — no protein is lost.

**Residual limitation [I]:** without chain IDs, a rare false adjacency is possible where a next chain happens to start at `last+1`, and insertion codes cannot be verified. This affects a negligible fraction of pairs and must be disclosed as a limitation of any B2 result. A cheap sensitivity check (restrict to strictly-increasing `res_seq` runs: 883/975/168/138 proteins per tier qualify blockwise) can bound the effect.

**Precise candidate features [P]:** over valid pairs, circular mean and `R` of `Δtheta`, circular mean and `R` of `Δtau`, `R` of `Δtheta+Δtau`, and the fraction of valid pairs → 6 numbers per protein.

**Genuinely additional? [I]** Yes in principle: the PC-NCSF density models the *single-residue* joint distribution of `(theta_i, tau_i)` and is unordered by construction; adjacent-pair statistics (helical periodicity, bond-geometry constraints, superfamily-specific loop behaviour) are not determined by that marginal. The measured `D_model ~ D_seq` = 0.542 / 0.542 / 0.592 / 0.761 confirms substantial overlap but clear independent variance **[V]**. Note the paper already states the unordered-density limitation, so B2 also connects directly to the stated limitation.

### 3.3 B3 — protein-specific learned embedding: **NOT FEASIBLE — reject** **[V]**

1. **What exists:** only the `nn.Embedding` lookup `[n, 8]` (`cond_dim=8`), one row per `pdb_id`; checkpoint keys are `config / cond_dim / flow_state_dict / embedding_state_dict`. There is no protein-specific latent produced by the flow itself, no per-protein posterior, no pooled per-residue representation. **[V]**
2. **Identifiability:** the embedding rows are trained *only* through the flow's conditioning path; any invertible reparameterisation of the 8-dim conditioning space that the conditioning layers absorb yields the same likelihood. Euclidean distance between rows is therefore not an invariant, semantically interpretable cross-protein metric **[I]**. Nothing in the checkpoint or code assigns meaning to individual coordinates.
3. **Direct measurement already exists:** `clustering_variants.py` section 6 clustered `features/cond-embeddings` (k-means and GMM) in all four tiers: ARI = 0.0006/0.0002/0.0012/−0.0064 (k-means) and 0.0020/0.0012/−0.0021/−0.0104 (GMM) — indistinguishable from the `features/random (control)` rows (−0.0006…0.0039). **[V]**

**Conclusion: do not invent an embedding analysis; state clearly that no defensible cross-protein embedding representation exists.** (This is also consistent with the response letter's Reviewer-2 Comment-3 wording: the lookup embedding does not generalise.)

---

## 4. Recommended minimal experiment (precise definitions)

**[P]** — nothing below has been run; no ARI/NMI for any hybrid is claimed anywhere in this report.

### 4.1 Representations compared (same tiers, same dropna, same `pdb_id` unit, same majority-vote labels)

| Tag | Representation | Distance |
|---|---|---|
| **R0** | PC-NCSF density (existing cached masses) | `D_H` = Hellinger (`hellinger_matrix`, existing) |
| **R1** | B1 statistics: `f = [meanθ, Rθ, meanτ, Rτ, meanθ+τ, Rθ+τ, meanθ−τ, Rθ−τ]` | Euclidean on z-scored `f` |
| **R2** | B2 sequential: `g = [meanΔθ, RΔθ, meanΔτ, RΔτ, R(Δθ+Δτ), valid-pair fraction]`, pairs with `Δres_seq == 1` inside `pdb_id` blocks | Euclidean on z-scored `g` |
| **R3 (hybrid)** | `D_α = α·D̂_model + (1−α)·D̂_aux`, with `D̂_aux = 0.5·D̂_R1 + 0.5·D̂_R2` (β fixed at 0.5, pre-registered) | convex combination (below) |
| **R4 (model-free comparator)** | per-protein empirical 100×100 histogram + 1e-3 smoothing (existing recipe in `clustering_variants.py`) | Hellinger |

### 4.2 Distance normalisation (why the combination is mathematically meaningful)

1. `D_H` is already bounded in [0,1] and is a metric; via the √-embedding, `H(p,q) = ‖√p−√q‖₂/√2`, so it is Euclidean and Ward's criterion is valid (established in prior audit §5). **[V]**
2. `D_R1`, `D_R2`: z-score each feature column across proteins (unit variance, so no single feature dominates), then Euclidean — a metric on ℝᵈ. **[P]**
3. **Scale alignment:** divide each matrix by its **median off-diagonal entry** (robust, label-free) → `D̂` with median 1. This makes `α` interpretable as a relative weight that is stable across tiers, instead of being an artefact of units. **[P]**
4. A non-negative weighted sum of metrics is a metric (triangle inequality preserved), so `D_α` is a legitimate input to hierarchical clustering; average/complete linkage are valid for any metric, and Ward is justified because each component is Euclidean (a positive combination of Euclidean distances from a concatenated-feature embedding: `D_α = ‖[√α·u ; √(1−α)·w]‖`-type construction holds when components are written in their Euclidean embeddings). **[I]**
5. **Redundancy guard:** because `Spearman(D_model, D_stats) = 0.50–0.75` and `Spearman(D_model, D_emp) = 0.62–0.84` (§9), the experiment must report all ablation rows (R0, R1, R2, R4, and `α ∈ {0, 0.25, 0.5, 0.75, 1}`); a single hybrid number without its components is uninterpretable. **[V/I]**

---

## 5. Evaluation protocol and leakage safeguards

**[P]** for the protocol; baselines it refers to are **[V]**.

1. **Same split/preprocessing everywhere:** dropna on `theta,tau` → `factorize(pdb_id)` → majority-vote labels; n = 1590/1594/371/194. One representation per protein, computed identically for all methods.
2. **Cluster count:** `k = 4 = n_categories` fixed per tier (this is the tier design, not a tuned parameter). Do **not** report the k-sweep (2..10) in any headline table — it is label-informed and already exists only as an exploratory appendix. **[V: it exists in `variants_*.csv`]**
3. **Linkage:** Ward primary (manuscript continuity), average + complete as robustness rows. No per-tier linkage cherry-picking.
4. **Weight selection without leakage:** the pre-registered grid `α ∈ {0, 0.25, 0.5, 0.75, 1}` is reported **in full, without selecting the best**. This is the primary recommendation: with only four tiers and n = 194 on the hardest one, any label-informed selection of α is high-variance. If a single α must be chosen for a manuscript sentence, use **nested selection**: choose α by mean ARI on one half of the proteins (inner), evaluate on the held-out half (outer), swap halves, repeat 10× with fixed seeds; report mean ± sd. **Limitation to state explicitly:** selection still uses labels; the outer estimate is the only honest number; power on `challenging` (n=194, majority class 50.5%) is low.
5. **What ARI/NMI can and cannot support:** they measure agreement between one 4-way partition and one SCOP level per tier (class/fold/superfamily/family respectively). A higher ARI at one level does **not** establish general superiority; ARI is chance-adjusted but sensitive to imbalance (hence the challenging-tier baselines); NMI is not chance-adjusted. Complement with 5-NN retrieval against tier-appropriate baselines (imbalance-preserving permutation null and majority-class baseline, as in the final-check report §5) to distinguish local from global signal.
6. **Leakage checklist:** (i) labels used only for evaluation, never for features; (ii) no feature search — the 8+6 features above are fixed now; (iii) α reported as a full grid, not a tuned scalar; (iv) the model-free comparator R4 is always shown alongside, never called a "ceiling"; (v) `pdb_id`/`domain_id` mismatch disclosed (5 multi-domain PDBs; exclusion test already exists); (vi) all variants written to a **new** output directory so the verified `results/clustering/` artifacts are never overwritten; (vii) fixed seeds (`default_rng(0)` convention already used).
7. **Interpretation rules:** a hybrid > components shows *complementarity*, not PC-NCSF superiority; superiority requires R3 > R4 **and** R3 > (R1+R2 without model) with the model's contribution isolated by the α ablation. If R1 or R2 alone ≥ R0 everywhere, that is evidence *against* a clustering benefit of the density and must be reported as such.

---

## 6. Expected computational cost and implementation effort

**[V]** for measured timings, **[P]** for the estimated implementation effort.

- **Measured (§9 check (b)):** loading `data.csv` + building both feature sets + all four full distance matrices took **8.5 s (easy), 7.4 s (moderate), 1.4 s (hard), 0.6 s (challenging)** on CPU — worst tier under 9 s, dominated by the 100×100 empirical histogram Hellinger. Distance matrices: 1590²×8 B ≈ 20 MB; cached masses 64 MB; total memory **well under 1 GB** (largest object is the 1590×1590 float64 matrix).
- **Expected full experiment:** features + distances ≈ 1 min total for all four tiers; linkage/ARI/NMI for ~20 variant rows per tier is negligible (seconds); spectral/GMM optional rows add little. **No GPU, no retraining, no model forward passes** (densities are cached; the model is only loaded if a cache is missing).
- **Implementation effort:** one new script (~150–200 lines) that imports `TIER_RUNS, ROOT, load_proteins, hellinger_matrix` from `scripts/scop_clustering.py` and reuses `hierarchical`/`evaluate`/empirical-histogram recipes from `scripts/clustering_variants.py`; a runner step; a new output directory. Estimated **half a day** including writing results to `reports/` and a verification pass. Largest risk is bookkeeping (feature ↔ protein index alignment), which can reuse the already-proven alignment pattern.
- **If caches were missing:** the only expensive step is `protein_densities` (n proteins × 10,000 grid points through the flow); not needed here — all four caches exist. **[V]**

---

## 7. Main scientific risks and proceed criteria

### Risks

1. **Statistical/model-free features may beat PC-NCSF by themselves — already partially observed [V]:** empirical histogram vs model at k=4 Ward = 0.116 vs 0.003 (easy) and 0.321 vs 0.059 (moderate), while roughly tied on hard/challenging. If R1/R2-only ≥ R3, the hybrid does not rescue the clustering story and the "why the flow?" question becomes acute. **Mitigation:** the paper's primary claim is *density estimation*, not clustering; frame clustering as a diagnostic, and be prepared to say the representation's class signal is weak regardless of combination.
2. **Hybrid improvement ≠ PC-NCSF superiority [I]:** an improvement could be entirely due to the added features. Only the α-ablation (R3 vs α=0) isolates the model's contribution. Do not write any sentence claiming superiority unless §5.7 holds.
3. **The experiment can backfire for the response letter [I]:** adding stronger baselines gives a reviewer material for "your own histograms cluster better". This is a reason to run the experiment *before* sending anything (which is the current situation), not a reason to skip it.
4. **Label reuse / selection bias:** k, linkage, α, and feature choices are all label-adjacent; §5 constrains each. The k-sweep already in `variants_*.csv` must not migrate into headline tables.
5. **B2 adjacency imperfection [I]:** no chain IDs / insertion codes; rare false adjacencies possible (§3.2). Bound with the strictly-increasing-run sensitivity variant; disclose.
6. **Tier heterogeneity [V]:** each tier is a different SCOP level with different balance; ARI values are not comparable across tiers, and `challenging` (imbalanced) needs its null baselines.
7. **ID mismatch [V]:** conditioning is per `pdb_id`; 5 multi-domain PDBs mix two categories (0.25–0.52% of proteins). Existing exclusion test covers this.

### Pre-registered decision criteria

**Proceed to any hybrid claim in the manuscript only if ALL hold:**
- R3 (nested-selected α, or every α in the grid showing the same sign) improves ARI **and** NMI over R0 on ≥2 of 4 tiers under both Ward and average linkage, with no tier materially degraded;
- R3 > R1-only and R3 > R2-only on those tiers (i.e. the model contributes beyond the added features);
- the result survives the 5-multi-domain-PDB exclusion check;
- the R4 comparator row is reported in the same table regardless of outcome.

**Otherwise:** do not proceed — report honestly that no hybrid of PC-NCSF density with statistical/sequential features yields a defensible clustering improvement, keep the existing conservative wording (Table 8 as reported, kNN as local-signal evidence), and name sequence-aware representations as future work. Either outcome is publishable honesty; neither is fabricated.

---

## 8. Exact code files/functions likely to require modification in a later, separately approved stage

**New files (created in that stage):**
- `scripts/clustering_hybrid.py` — new; features (B1, B2), normalisation, hybrid distance, ablation loop, CSV output.
- `scripts/run_clustering_hybrid.sh` — new runner (mirror of `run_clustering_full.sh`; note the existing runner hardcodes `PYTHON=/data/python-envs/pytorch/bin/python`, with Windows equivalent `C:\programs\anaconda3\envs\pth\python.exe`).
- `results/clustering_hybrid/` — new output directory (never write into the verified `results/clustering/`).
- `reports/scop_hybrid_results_<date>.md` — results report for that stage.

**Existing files read/reused, unchanged:** `scripts/scop_clustering.py` (`TIER_RUNS`, `ROOT`, `load_proteins`, `hellinger_matrix`, `build_model` only if a cache is missing), `scripts/clustering_variants.py` (`hierarchical`, `evaluate`, `marginal_w1`, empirical-histogram recipe), `scripts/clustering_knn.py` (`knn_accuracy` for the local-signal metric).

**Optional one-line addition to an existing file:** `scripts/run_clustering_full.sh` — append a 4th step calling `clustering_hybrid.py` (only if that stage is approved).

**Must NOT be touched in that stage:** `fff/**` (training/loader code), `runs/**` (checkpoints), `results/clustering/masses_*.npy` and all existing result CSVs, `paper/**` (manuscript, response letter, `R1-C5_rewrite_for_Nooshin.md`) — any manuscript edit requires separate approval.

---

## 9. Lightweight read-only checks executed (as required by the constraints)

All commands ran in the project root with `C:\programs\anaconda3\envs\pth\python.exe` (torch 2.3.1, zuko 1.5.0, numpy 2.0.0, pandas 2.3.2, scipy 1.15.3, sklearn 1.5.1). **Every check was read-only: no file in the repository was written, created, or modified except this report; caches and CSVs were opened for reading only.** Two temporary probe scripts were used and **deleted after running**; their logic is reproduced below. Exit codes shown are the Python exit codes.

**(a) Schema / adjacency probe** (`_b2_probe.py`, `EXIT=0`): printed columns, row counts before/after dropna, `pdb_id` block contiguity, strictly-increasing vs duplicated `res_seq` counts per protein, and the `Δres_seq` histogram of adjacent-row pairs. Results: §2.4, §3.2 tables.

**(b) Distance-correlation probe** (`_hybrid_probe.py`, `EXIT=0`): for each tier, built `D_model` (Hellinger on cached masses), `D_emp` (Hellinger on empirical histogram, exact `clustering_variants.py` recipe), `D_stats` (z-scored Euclidean on the 8 circular statistics of §3.1), `D_seq` (z-scored Euclidean on the 6 sequential features of §3.2, `Δres_seq==1` rule); printed Spearman correlations of condensed vectors, the fraction of proteins with usable sequential features, and timings. **No SCOP label was loaded; no ARI/NMI was computed in this check.** Printed output:

| Tier | D_model~D_emp | D_model~D_stats | D_emp~D_stats | D_model~D_seq | D_stats~D_seq | seq-usable | time |
|---|---|---|---|---|---|---|---|
| easy | 0.618 | 0.504 | 0.450 | 0.573 | 0.423 | 1.000 | 8.5 s |
| moderate | 0.709 | 0.525 | 0.610 | 0.542 | 0.452 | 1.000 | 7.4 s |
| hard | 0.844 | 0.575 | 0.543 | 0.592 | 0.420 | 1.000 | 1.4 s |
| challenging | 0.783 | 0.745 | 0.711 | 0.761 | 0.798 | 1.000 | 0.6 s |

**(c) Checkpoint inspection** (one `torch.load(..., weights_only=False)` on `runs/scop_hard/.../best_flow.pt`, `EXIT=0`): printed keys, `cond_dim=8`, embedding shape `[371,8]`, network config. Read-only.

**(d) Greps** over `scripts/*.py` for `res_seq|np.diff|consecutive|circular` (`EXIT=0`): confirmed no sequential/circular feature code exists in the repository (only prose matches in the final-check script's exclusion list).

Core of check (b), for reproduction:

```python
# per protein i (rows of data.csv after dropna(theta,tau)):
#   B1: z = exp(1j*phi).sum(); R = |z|/n; mean = angle(z)   for phi in {th, ta, th+ta, th-ta}
#   B2: pairs = consecutive rows in the same pdb_id with np.diff(res_seq) == 1
#       same circular mean/R on wrapped np.angle(np.exp(1j*np.diff(th))[good]) etc.
#   D = sqrt(((Z[:,None]-Z[None])**2).sum(-1)) after column z-scoring
#   corr = spearmanr(D1[iu], D2[iu]).statistic  with iu = triu_indices(n,1)
```

---

## 10. Conclusion

**Recommendation: Proceed with a narrowly defined B1+B2 hybrid**, executed as the single pre-registered ablation study of §4–§5, with the decision criteria of §7 fixed *before* any ARI/NMI is computed for the hybrid.

Justification from repository evidence: (i) B3 is empirically and conceptually dead — its only artifact clusters at the random-control level in every tier **[V]**; (ii) B1 alone largely re-tests existing marginal/histogram variants and is 0.50–0.75 redundant with the model distance **[V]**; (iii) B2 is fully computable for 100% of proteins with a defensible gap rule, is new in the codebase, and is only moderately redundant with the density **[V]**; (iv) the whole experiment costs minutes on CPU with zero retraining and zero risk to verified artifacts **[V]**; (v) it is the only untested claim in the four-claim split **[V]**.

No manuscript file has been modified, no result for any hybrid has been produced or implied, and if the §7 criteria fail, the honest report is: *combining PC-NCSF densities with statistical or sequential angular features does not yield a defensible clustering improvement on SCOP — and that conclusion will be supported by the same table that would have supported the positive one.*
