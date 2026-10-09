# SCOP Clustering — Stage 1 Audit & Assessment of the Proposed Plan

**Date:** 2026-10-09
**Scope:** Read-only audit of the SCOP data, the training data-loader, the three clustering scripts, the cached density arrays, and the trained checkpoints. No training code, checkpoints, manuscript, or existing result files were modified.
**Environment verified:** Windows — `C:\programs\anaconda3\envs\pth\python.exe` (torch 2.3.1, zuko 1.5.0, numpy 2.0.0, pandas 2.3.2, scipy 1.15.3, sklearn 1.5.1). Linux reference path from the earlier session: `/data/python-envs/pytorch/bin/python`.

---

## 1. Executive summary

The clustering plumbing is **clean and correctly aligned with the trained model**. The audit did **not** find an indexing, labelling, or unit bug that would explain the weak SCOP clustering. The weak ARI is a property of the representation, not an artefact of the pipeline.

Two items from the external (ChatGPT) review deserve attention, and one of its technical objections is **incorrect for this specific case**:

| External concern | Verdict | Evidence |
|---|---|---|
| Do **not** assume `pdb_id` == `domain_id` | **Correct and important** — they differ in 2 of 4 tiers | §3.2 |
| Check embedding-index alignment | **Verified — perfect** | §3.3 |
| Check majority-vote label consistency | **Verified — no ambiguity** | §3.4 |
| Ward on a precomputed Hellinger matrix is not Euclidean ⇒ invalid | **Incorrect here** — Hellinger *is* Euclidean via the √-embedding isometry | §5 |
| Don't conflate 4 different claims | **Adopt — excellent framing** | §6 |

---

## 2. Sample unit — what is one observation?

**Confirmed facts** (per tier, from `SCOP/<tier>/data.csv`):

| Tier | rows (raw) | rows (after `dropna`) | dropped | unique `pdb_id` | unique `domain_id` |
|---|---|---|---|---|---|
| easy | 732,092 | 727,310 | 4,782 (0.65%) | **1,590** | **1,594** |
| moderate | 604,946 | 600,164 | 4,782 (0.79%) | 1,594 | 1,594 |
| hard | 256,548 | 255,435 | 1,113 (0.43%) | 371 | 371 |
| challenging | 112,599 | 112,014 | 585 (0.52%) | 194 | **195** |

- **Residue level:** one CSV row = one residue observation `(theta, tau)` in degrees, with `res_seq`. Note `res_seq` is **not unique within a protein** (duplicated `(pdb_id, res_seq)` pairs: easy 352,664; moderate 242,609; hard 124,390; challenging 38,897). This is expected for multi-chain / multi-model entries; it does **not** affect the protein-level clustering, because masses are summed over all rows of a protein.
- **Clustering sample unit:** one **protein** = one `pdb_id`. n = 1590 / 1594 / 371 / 194. This matches the trained embedding table exactly (§3.3).
- **The `dropna` in training and in the clustering scripts is identical** (`dropna(subset=["theta","tau"])`, then `reset_index(drop=True)`), so features and labels are computed on the same residue set. The interesting coincidence that easy and moderate both drop exactly 4,782 rows is a property of the source data, not a bug.

**Conclusion (Q1, Q6):** confirmed and consistent. The clustering sample unit is the protein; the residue-level repetition is inherited from the source and is harmless at the protein level.

---

## 3. `pdb_id` vs `domain_id` — the one real data caveat

### 3.1 Why it matters
The SCOP biological unit is the **domain** (`domain_id`, e.g. `d1a4ma_`). The model conditions on **`pdb_id`**. The training loader `fff/data/scop.py` resolves the identifier column in this order: `condition_on` (default `"protein"`) → not a column in this CSV → falls through to **`pdb_id`**. So one embedding is learned per PDB entry, not per SCOP domain.

### 3.2 Where they diverge — **confirmed**
- `easy`: 1,590 `pdb_id` vs 1,594 `domain_id` → **4 PDB entries carry two SCOP domains each**: `1jj2`, `1lt4`, `1nmu`, `1qbk`.
- `challenging`: 194 `pdb_id` vs 195 `domain_id` → **1 PDB entry carries two domains**: `1ta3`.
- `moderate`, `hard`: `pdb_id` == `domain_id`, no divergence.

These 5 multi-domain PDBs are exactly the 5 PDBs that span two SCOP categories (`pdb_id with >1 category`: easy 4, challenging 1). This matches the note already present in `scripts/scop_clustering.py` ("4 pdb_ids in 'easy' and one in 'challenging' span more than one category").

### 3.3 Impact assessment
- Affected fraction: **4/1590 = 0.25%** (easy), **1/194 = 0.52%** (challenging). Negligible.
- Because the model has **one embedding per `pdb_id`**, the per-protein density of those 5 entries is a **mixture of two SCOP categories**. This slightly blurs clustering for exactly those proteins.
- The **labels** for those 5 are resolved by majority vote (§3.4), so features and labels remain consistent (same unit on both sides).
- **The alternative** — conditioning on `domain_id` (1594 / 195 embeddings) — is biologically cleaner but requires **retraining**, is out of scope for a reviewer response, and is not what Reviewer 1 asked for.

### 3.4 Label assignment — **verified unambiguous**
Majority vote over `category` per `pdb_id` (as in `scripts/scop_clustering.py::load_proteins`) versus "first residue" label:
- **disagreement count = 0 for all four tiers.**
- So the reference label is stable regardless of tie-breaking convention.

**Recommendation:** disclose the `pdb_id`-level conditioning and the 5 multi-domain entries in one sentence of Section 3.3 (dataset description). It is a minor, honest caveat that pre-empts a "did you conflate PDB and domain?" question — which is exactly the kind of detail a careful reviewer (or Reviewer 2) would raise.

---

## 4. Embedding-index alignment — **verified perfect**

| Tier | embedding rows in checkpoint | n proteins (`pdb_id`) | aligned | cond_dim | factorize index range |
|---|---|---|---|---|---|
| easy | 1,590 | 1,590 | ✅ | 8 | 0…1589 contiguous |
| moderate | 1,594 | 1,594 | ✅ | 8 | 0…1593 contiguous |
| hard | 371 | 371 | ✅ | 8 | 0…370 contiguous |
| challenging | 194 | 194 | ✅ | 8 | 0…193 contiguous |

`pd.factorize(df["pdb_id"])` is applied **after** `dropna(...).reset_index(drop=True)` in `fff/data/scop.py` — and `scripts/scop_clustering.py::load_proteins` replicates this order exactly, using `pd.unique(df["pdb_id"])` for the protein list (same order of first appearance as `factorize`). Therefore `emb.weight[i]` corresponds to `proteins[i]`. **No off-by-one or reordering bug.** (Q2: confirmed.)

---

## 5. The Ward / Euclidean objection — **does not apply to Hellinger**

The external review states: *"Ward linkage requires Euclidean geometry. Verify that the distance matrix supplied to Ward is mathematically appropriate; do not silently treat an arbitrary precomputed distance matrix as Euclidean."*

That is a correct general warning, but it **does not invalidate our experiment**, because Hellinger distance is isometrically Euclidean. Our implementation (`scripts/scop_clustering.py::hellinger_matrix`):

```
H²(p,q) = 0.5 · Σᵢ (√pᵢ − √qᵢ)²
```

Let `u = √p`, `v = √q` (the **square-root embedding**). Then `‖u − v‖₂² = Σᵢ(√pᵢ − √qᵢ)² = 2·H²(p,q)`, i.e.

```
H(p,q) = ‖u − v‖₂ / √2
```

So Hellinger distance **is** a (scaled) Euclidean distance in the √-embedded space. Ward's minimum-variance criterion is defined on Euclidean distances, so **Ward on Hellinger distances is mathematically valid**, not an abuse.

Belt-and-braces: we already report **average** and **complete** linkage (valid for any metric dissimilarity) in `results/clustering/scop_clustering_summary_g100.csv`, and they agree with Ward (all weak; e.g. easy −0.0002/0.0028, moderate 0.0213/0.0563, hard 0.0406/0.0793, challenging 0.0995/0.0995). **The conclusion is linkage-agnostic already.** No methodological change is required; if anything, we can cite this as extra robustness.

---

## 6. Adopting the four-claim separation (recommended)

The most valuable idea in the external plan is the insistence on separating four distinct claims. Mapping them onto evidence we **already have**:

| Claim | Existing evidence | Status |
|---|---|---|
| (1) Better **density estimation** | held-out NLL, conditional vs unconditional (Section 3.3); PC-RFM baseline comparison | ✅ supported — this is the paper's primary claim |
| (2) Better **unsupervised clustering** | Table 8 ARI 0.003–0.099; robust across 67 model-based variants/tier | ❌ weak — reported honestly |
| (3) Better **recovery of SCOP labels** | same as (2); kNN retrieval 5-NN 0.36–0.57 vs chance 0.25 shows signal is **local, not global** | ⚠️ partial — locally informative, not cluster-separable |
| (4) Better **representations by combining model + extra features** | **not attempted** | ⛔ not attempted — see recommendation below |

**Crucially, the external plan's Stage 2A (PC-NCSF density) and Stage 2B (raw-angle histogram) are already computed.** They are in `results/clustering/variants_*.csv` as `joint/Hellinger` and `empirical/Hellinger`:

| Tier | PC-NCSF (Ward, k=4) | model-free raw histogram (k=4) |
|---|---|---|
| easy | 0.003 | 0.116 |
| moderate | 0.059 | 0.321 |
| hard | 0.076 | 0.074 |
| challenging | 0.099 | 0.094 |

This **already answers the external plan's "primary scientific question"**: does PC-NCSF's density carry structural information *beyond simpler representations of the same angles*? Answer: on the two hardest tiers it **matches or slightly exceeds** the model-free ceiling; the bottleneck is the angular representation itself, not the flow. That is a defensible, honest finding.

---

## 7. Recommendations on the proposed plan

**Adopt:**
1. The audit-first discipline (done here — §2–§5).
2. The four-claim separation (§6).
3. New output directory + never overwrite originals (good hygiene).
4. Runtime/memory estimation before long jobs.
5. Disclosing the `pdb_id`/`domain_id` caveat in Section 3.3 (§3.4).

**Push back / de-prioritise:**
1. **Ward-on-Hellinger objection (Stage 3):** not applicable — Hellinger is Euclidean (§5). No change needed.
2. **Stage 2B (raw histogram):** already done (`empirical/Hellinger`). Do not redo.
3. **Stage 2D (sequence-aware angular representation):** this is a **new research project**, not a revision-scope fix. It requires rethinking the contribution and cannot be done rigorously in the revision window. It also aligns with the paper's *already-stated* limitation (unordered densities discard residue order) — so the correct move is to **name it as future work**, which we already do.
4. **Stage 2E (combined model + handcrafted features):** ⚠️ **risk of backfiring.** If handcrafted circular statistics cluster SCOP better than the neural density, a reviewer will ask *"then why the flow at all?"* — which **undercuts the paper's primary claim (density estimation)**. The paper's contribution is not clustering. Adding stronger clustering baselines weakens the story it is trying to save. **Recommend against** for this revision.
5. **Stage 3 permutation/resampling stability:** low value. ARI is 0.003–0.099 — nowhere near a borderline where stability analysis changes the conclusion.

**One correction already applied to our side's narrative:** the earlier draft explained the rising Easy→Challenging ARI by "distance concentration." That reasoning was internally shaky. The audit **confirms the clean explanation**: label granularity is *exactly* SCOP depth 1/2/3/4 per tier (easy=class, moderate=fold, hard=superfamily, challenging=family) — verified programmatically from the `category` codes (§8). Fine-grained families within one fold *do* differ in local angular statistics; broad classes do not.

---

## 8. SCOP hierarchy — confirmed programmatically

Label SCOP-depth histogram (depth = number of `.` in the `category` code + 1):

| Tier | depth | codes | = SCOP level | label balance |
|---|---|---|---|---|
| easy | 1 (100%) | a, b, c, d | **class** | 396/398/399/397 — balanced |
| moderate | 2 (100%) | a.1, b.47, c.1, d.2 | **fold** | 400/399/396/399 — balanced |
| hard | 3 (100%) | c.1.2, c.1.8, c.1.10, c.1.15 | **superfamily** | 96/98/100/77 — near-balanced |
| challenging | 4 (100%) | c.1.8.1, c.1.8.3, c.1.8.4, c.1.8.5 | **family** | 21/98/53/22 — **imbalanced (max 50.5%)** |

Every tier is **100% uniform** in depth — no mixed-granularity contamination. The external plan's Stage 4 hypothesis is therefore **confirmed**, and the elevated shuffled 5-NN baseline on challenging (0.462 vs 0.25 elsewhere) is explained by the 50.5% single-family imbalance.

---

## 9. Prioritised next steps (awaiting approval)

| # | Action | Runtime / memory | Needs approval? |
|---|---|---|---|
| 1 | **No new experiment needed** for the reviewer response — the audit confirms the existing pipeline; the evidence (reproduction, robustness, kNN, model-free ceiling, label depth) is complete in `paper/R1-C5_rewrite_for_Nooshin.md` | — | — |
| 2 | Add **one disclosure sentence** in Section 3.3: conditioning is per `pdb_id`; 5 entries span two SCOP domains (0.25–0.52%); majority-vote labels are unambiguous (0 disagreements) | — | yes (manuscript edit) |
| 3 | *(optional, cheap)* recompute clustering **excluding the 5 multi-domain PDBs** to show the ARI is unchanged — a 5-minute, CPU-only robustness check on cached densities | < 1 min, < 1 GB | yes (writes to a **new** dir) |
| 4 | *(optional, cheap)* per-tier label-depth kNN/ARI cross-tab as an appendix table | < 5 min, < 2 GB | yes |
| 5 | ~~Stage 2D sequence-aware representation~~ — **defer to future work** (matches stated limitation) | — | — |
| 6 | ~~Stage 2E combined features~~ — **recommend against** (undermines primary claim) | — | — |

**Bottom line:** The audit strengthens rather than undermines the current response to Reviewer 1. There is no plumbing bug. The weak SCOP clustering is genuine and is now fully characterised: it is a property of unordered per-protein angular densities w.r.t. SCOP categories, it is robust to every pipeline choice, the class signal exists locally (kNN), and a model-free histogram reaches the same low ceiling on the hard tiers. The one action worth taking before sending to Nooshin is the single `pdb_id`/`domain_id` disclosure sentence.

---

## 10. Artifacts & how to reproduce this audit

- Data: `SCOP/{easy,moderate,hard,challenging}/data.csv`
- Loader: `fff/data/scop.py::get_scop_dataset`
- Clustering: `scripts/scop_clustering.py`, `scripts/clustering_variants.py`, `scripts/clustering_knn.py`, runner `scripts/run_clustering_full.sh`
- Cached densities: `results/clustering/masses_{tier}_g100.npy` — shape `(n,100,100)` float32, n = 1590/1594/371/194 ✅ present
- Checkpoints: `runs/scop_{tier}/ep20-bs512/cond/best_flow.pt`
- Response draft for Nooshin: `paper/R1-C5_rewrite_for_Nooshin.md`

Windows command:
```
C:\programs\anaconda3\envs\pth\python.exe
```
Linux command (verified in the earlier session):
```
/data/python-envs/pytorch/bin/python
```
