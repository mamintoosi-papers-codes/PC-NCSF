# SCOP Clustering — Final Revision Check (Reviewer 1, Comment 5)

**Date:** 2026-10-09
**Scope:** Only the checks needed for the current revision. No new research project. CPU-only. No retraining, no overwrite of existing files, no package changes.
**Script:** `scripts/clustering_final_check_20261009.py`
**New outputs (dated dir):** `results/clustering/final_check_2026-10-09/` → `alignment_check.csv`, `exclusion_test.csv`, `knn_context.csv`, `run.log`
**Runtime:** 3m07s wall-clock, CPU-only, well under 1 GB RAM (Windows: `C:\programs\anaconda3\envs\pth\python.exe`). Estimate given beforehand was < 2 min; actual was 3m07s because of two checkpoint loads for the alignment probes. No stage required the Linux workstation.

---

## 1. Checks completed, and their evidence

### 1.1 Evidence trail of every number quoted to Reviewer 1

| Quantity | Source of truth | File |
|---|---|---|
| Table 8 ARI/NMI (reproduced) | `scripts/scop_clustering.py` + cached densities | `results/clustering/scop_clustering_summary_g100.csv` |
| 67 model-based variants/tier | `scripts/clustering_variants.py` | `results/clustering/variants_{tier}.csv` |
| Model-free histogram comparator | same script, row `empirical/Hellinger` | `results/clustering/variants_{tier}.csv` |
| 5-NN retrieval | `scripts/clustering_knn.py` | `results/clustering/knn_retrieval.csv` |
| **New:** alignment proof, exclusion test, 5-NN permutation null | `scripts/clustering_final_check_20261009.py` | `results/clustering/final_check_2026-10-09/` |

### 1.2 Sample units, filtering, labels, protocol — consistent across both methods

PC-NCSF densities and the empirical histograms use **the same**:
- **Sample unit:** one protein = one `pdb_id` (n = 1590 / 1594 / 371 / 194). The histogram is accumulated per `pdb_id` via the same `order` mapping used for labels.
- **Filtering:** both sides `dropna(subset=["theta","tau"])` on the same CSV — identical to `fff/data/scop.py::get_scop_dataset`.
- **Labels:** the same majority-vote `lab` from `load_proteins`, for every variant in a tier.
- **Grid resolution:** 100×100 on $[-\pi,\pi)^2$.
- **Distance / clustering:** the same `hellinger_matrix` and the same `hierarchical()` with Ward, cut at k = 4.

**Two genuine (minor) methodological differences exist and are disclosed here** — they do not favour the model:
1. **Half-bin offset.** The model grid is `linspace(-π, π, 100, endpoint=False)` (point samples × `dtheta²`); the histogram uses bins `((θ+π)/2π)·g`, i.e. bin **centres** sit half a step from the model grid points. Sub-resolution; does not change the comparison's validity.
2. **Pseudocount.** The histogram adds `+1e-3` per bin before renormalising (≈2% of a typical protein's residue count); the model densities have none. If anything this mild smoothing **helps** the histogram, so the comparison is not biased in the model's favour.

### 1.3 Sample-index alignment — now **proven**, not assumed

Rather than trusting the `factorize` ordering, the check recomputed one excluded protein's density **straight from the checkpoint** and compared it to the cached row:

| Tier / probe | Protein index | max \|recomputed − cached\| | Verdict |
|---|---|---|---|
| easy / `1jj2` | 573 | 2.99e-08 | OK |
| challenging / `1ta3` | 64 | 1.94e-07 | OK |

So `masses[k]` ↔ `proteins[k]` ↔ `emb.weight[k]` holds end-to-end, and it still holds **after** exclusions (the excluded indices are simply dropped from both the mass matrix and the label vector together).

> **Environment note found while doing this:** the checkpoints store **old-zuko** keys `base._0`/`base._1`, and the installed zuko 1.5.0 on this Windows machine expects exactly those. The helper `scripts/scop_clustering.py::build_model` unconditionally renames them to `base.lower`/`base.upper` (a shim written for a *newer* zuko on the earlier Linux box), so on Windows it raises `Missing key(s): base._0, base._1`. The new script therefore carries its own version-agnostic `build_model_compat`, which detects the required direction from the constructed module. **This does not affect any previously reported number** — the clustering results come from cached densities and never called `build_model`. But it means `scripts/scop_clustering.py` cannot regenerate densities on this Windows environment as-is (§3, item 5).

---

## 2. Exclusion test — the five multi-domain PDBs do **not** drive the result

Protocol unchanged: same labels, same Hellinger + Ward, k = n_categories, same ARI/NMI/5-NN. Only the five entries removed. Categories remain 4 in every tier (assertion enforced).

| Tier | Variant | n | ARI | NMI | 5-NN | Ward cluster sizes |
|---|---|---|---|---|---|---|
| easy | original | 1590 | 0.0033 | 0.0096 | 0.3591 | [926, 423, 163, 78] |
| easy | **excluded** | 1586 | **0.0031** | 0.0087 | 0.3588 | [934, 421, 162, 69] |
| moderate | original = excluded | 1594 | 0.0590 | 0.1211 | 0.5634 | [47, 1034, 282, 231] |
| hard | original = excluded | 371 | 0.0761 | 0.0977 | 0.5741 | [138, 167, 56, 10] |
| challenging | original | 194 | 0.0995 | 0.1431 | 0.5670 | [45, 137, 4, 8] |
| challenging | **excluded** | 193 | **0.1040** | 0.1462 | 0.5648 | [44, 137, 4, 8] |

**ΔARI: easy −0.0002, challenging +0.0045; moderate/hard untouched.** Removing the mixed-category protein on Challenging slightly **improves** ARI, which is directionally consistent with the blurring hypothesis, but the magnitude (0.0045) is negligible. **Conclusion: the `pdb_id`/`domain_id` caveat is worth one disclosure sentence; it is not a threat to any reported result.**

### 2.1 A new, useful observation: Ward produces wildly unbalanced clusters

The recovered cluster sizes bear no resemblance to the (near-balanced) SCOP label counts:

| Tier | Ward sizes | True class counts |
|---|---|---|
| easy | [926, 423, 163, 78] | [399, 398, 397, 396] |
| moderate | [47, 1034, 282, 231] | [396, 399, 399, 400] |
| hard | [138, 167, 56, 10] | [98, 100, 96, 77] |
| challenging | [45, 137, 4, 8] | [98, 53, 22, 21] |

This is concrete, checkable evidence for the mechanism-level point: the Hellinger geometry of the densities does **not** contain four balanced natural groups matching the SCOP categories — Ward is forced to cut an unbalanced hierarchy (on `hard`, a cluster of 10 next to one of 167). This strengthens the response far more than a bare "ARI is low".

---

## 3. Discrepancies and unresolved uncertainties

1. **Original Table 8 cannot be regenerated exactly.** Nooshin's original clustering code is not in the repo. Our independent reproduction matches ARI to ≤ 0.003 on every tier and NMI on 3 of 4 tiers; **Challenging NMI differs (0.143 reproduced vs 0.182 reported)** — traced to the NMI normalisation convention (`min`-normalisation yields ≈ 0.178). Recommend reporting the reproduced values with the archived code, and stating the normalisation (sklearn arithmetic mean).
2. **The 5-NN shuffled baseline previously quoted (0.4622 on Challenging) was the mean of only 3 shuffles.** The proper 500-permutation, imbalance-preserving null is **0.4408 ± 0.0292**. Both are mutually consistent; **cite the 500-permutation figure**, not the 3-shuffle one. This is a correction to our own earlier draft.
3. **"Model-free ceiling" is not an accurate term and is retracted** (see §4).
4. **The two §1.2 discretisation differences** (half-bin offset, pseudocount) are undocumented in the manuscript. They are immaterial to the conclusion but should be stated in one clause if the histogram comparison is kept.
5. **`scripts/scop_clustering.py` is not portable to this Windows environment** (zuko key shim, §1.3). Any future density regeneration here needs `build_model_compat` or a newer zuko. Cached densities are unaffected.
6. **Unresolved:** whether the 4 `easy` + 1 `challenging` multi-domain entries would behave differently under a `domain_id`-conditioned retraining. That requires retraining and is **out of revision scope**.

---

## 4. On the empirical histogram — retract "ceiling", state all four tiers

The empirical histogram is **not** a ceiling, for three reasons:
1. It is **one** alternative representation (binned joint histogram + Hellinger), not an upper bound over representations.
2. It is **lightly smoothed** (+1e-3 pseudocount), so it is not "raw" in a strict sense.
3. Decisively, it is **not even an upper bound over clustering results**: it is *above* PC-NCSF on two tiers and *below* on the other two.

| Tier | PC-NCSF (Ward, k=4) | Empirical histogram | Higher |
|---|---|---|---|
| easy | 0.003 | **0.116** | histogram |
| moderate | 0.059 | **0.321** | histogram |
| hard | **0.076** | 0.074 | PC-NCSF |
| challenging | **0.099** | 0.094 | PC-NCSF |

⚠️ **Cherry-picking warning for our own response draft.** The current Nooshin draft says *"On the two most similar tiers the learned densities match or slightly exceed this reference"* — true, but it silently omits Easy and Moderate, where the histogram is 30–40× higher in ARI. A careful reviewer will notice. The honest and still-safe statement is: **the unordered angular observations alone — with or without the flow — yield weak SCOP clustering under this protocol** (histogram ARI ≤ 0.321, and ≤ 0.116 on three of four tiers), and the flow is trained for likelihood, not cluster separation, so its smoothing appears to cost some between-protein contrast on the broad tiers. This **does not touch the paper's primary claim** (density estimation), and disclosing it is far safer than hiding it.

---

## 5. 5-NN retrieval — interpret against tier-appropriate baselines

| Tier | 5-NN observed | Majority-class | Permutation null (500, imbalance-preserving) | p |
|---|---|---|---|---|
| easy | 0.3591 | 0.2509 | 0.2492 ± 0.0120 | 0.002 |
| moderate | 0.5634 | 0.2509 | 0.2489 ± 0.0115 | 0.002 |
| hard | 0.5741 | 0.2695 | 0.2559 ± 0.0257 | 0.002 |
| challenging | 0.5670 | **0.5052** | 0.4408 ± 0.0292 | 0.002 |

- On **easy / moderate / hard**, retrieval is far above every trivial baseline (+0.11 / +0.31 / +0.30 over majority-class) and highly significant.
- On **challenging**, the majority-class baseline is already **0.5052** (one family = 50.5%), so 5-NN 0.5670 is only **+0.062** over "always predict the biggest family". It is still statistically significant (p = 0.002, permutation null 0.4408), but the **effect size is modest** and must be described as such — not as "well above chance".
- The response must therefore say: *local category signal is clear on three tiers and modest on the imbalanced Challenging tier.*

---

## 6. Proposed manuscript sentence (Section 3.3) — **prepared, NOT applied**

> The model conditions on `pdb_id`, whereas SCOP annotations are defined per structural domain; in Easy and Challenging a small number of PDB entries (4 and 1, respectively, i.e. 0.25% and 0.52% of proteins) contain two domains belonging to different categories, and these are assigned the majority-vote category. Recomputing the clustering after excluding these five entries leaves the results unchanged (ARI 0.0033→0.0031 on Easy and 0.0995→0.1040 on Challenging).

Optional trailing clause if the histogram comparator is described in Section 3.5:

> The empirical histograms are evaluated on the same 100×100 discretisation of $[-\pi,\pi)^2$ and clustered with the identical Hellinger–Ward protocol, differing from the model densities only by a half-bin centre offset and a small additive pseudocount.

---

## 7. Recommended edits to the response letter (Reviewer 1, Comment 5) — **prepared, NOT applied**

1. **Replace "model-free ceiling" → "model-free comparator"** everywhere, and report **all four** histogram values (0.116 / 0.321 / 0.074 / 0.094), not just the two favourable tiers.
2. **Add the exclusion result** as robustness: dropping the 5 multi-domain PDBs changes ARI by ≤ 0.0045; labels, protocol and metrics unchanged; alignment verified by recomputing two proteins' densities from the checkpoints (max deviation ~1e-7).
3. **Add the cluster-size imbalance observation** (Ward → [926,423,163,78] etc. vs balanced labels) as concrete mechanism-level evidence that four balanced SCOP groups are absent from the density geometry.
4. **Fix the 5-NN baseline wording:** cite the 500-permutation null (0.249 / 0.249 / 0.256 / 0.441) and the majority-class baselines (0.251 / 0.251 / 0.270 / **0.505**); describe Challenging's local signal as **modest (+0.062 over majority-class)**, significant at p = 0.002. Do not write "well above chance" for Challenging.
5. **Soften the granularity claim to an observation.** The tiers differ **simultaneously** in label depth (class→fold→superfamily→family), n (1590→194), balance, and diversity — this is not a controlled experiment on granularity alone. Write "consistent with", never "because of".
6. **Keep the primary claim on conditional density estimation.** Clustering stays an exploratory probe. Do not add handcrafted/combined feature representations in this revision (they would invite "why the flow, then?").

---

## 8. Final recommendation

**The existing evidence is sufficient for Reviewer 1, Comment 5. No further computation is necessary.**

What was already in place — full reproduction of Table 8, robustness across 67 model-based variants per tier, the kNN local-signal result, and the label-granularity confirmation — was enough. What this check added is **hardening, not new substance**: proven index alignment, a quantified exclusion test showing the multi-domain caveat is immaterial, and correctly-referenced retrieval statistics.

**Before sending to Nooshin, only three things should change** (all textual, no new experiments):
1. Apply the §6 disclosure sentence to Section 3.3.
2. Apply the six §7 response-letter edits — especially retracting "ceiling", disclosing the two tiers where the histogram wins, and fixing the Challenging 5-NN baseline.
3. State the NMI normalisation convention and switch Table 8 to the reproducible values.

**Deliberately not done, and recommended against:** sequence-aware representations (new research project; already named as future work in the paper) and combined model+handcrafted features (undermines the primary claim). The one optional further computation that *would* be cheap and defensible — a `domain_id`-conditioned retraining — is explicitly out of revision scope and requires your approval and the Linux box.
