# R1-C5 Evidence Audit — 2026-10-10

Scope: claim-by-claim audit of the proposed response `paper/R1-C5_rewrite_for_Nooshin.md`
and of every number proposed for Section 3.5, against the latest committed artifacts
(HEAD `4ebd8ad`, "Run scop_pcncsf and scop_clustering", 2026-10-10 12:45) and against
new measurements made for this audit.

Evidence tags: **[V]** verified by inspection, **[M]** measured in this audit,
**[I]** interpretation, **[R]** recommendation. All new measurements are
reproducible via `scripts/r1c5_length_controls.py` (seed 20261010, grid 100,
1000 label shuffles, 200 Mantel permutations) and
`scripts/r1c5_class_length_summary.py`; outputs in
`results/clustering/r1c5_length_controls_g100.csv`,
`results/clustering/r1c5_mantel_g100.csv`,
`results/clustering/r1c5_class_length_summary.csv`.

Repository state: branch `main`, HEAD `4ebd8ad`, working tree clean at start.
Dated backups of the pre-edit paper files: `reports/r1c5_backup_2026-10-10/`.

---

## 1. Primary numbers (Table 8) — VERIFIED, small updates warranted

Latest primary pipeline output (committed, alignment-verified, ambiguous pdb_ids
excluded): `results/clustering/audit2026_pcncsf_g100.csv` [V]

| Tier | m | ward ARI | ward NMI |
|---|---|---|---|
| Easy | 1586 | 0.0030 | 0.0089 |
| Moderate | 1594 | 0.0593 | 0.1244 |
| Hard | 371 | 0.0763 | 0.0965 |
| Challenging | 193 | 0.1032 | 0.1374 |

Independently recomputed in this audit from the committed density caches
(`masses_*_g100.npy`, `scripts/r1c5_length_controls.py`): ARI 0.0031 / 0.0590 /
0.0761 / 0.1040, NMI 0.0087 / 0.1211 / 0.0977 / 0.1462 [M]. Differences vs the
primary pipeline are ≤0.001 ARI (≤0.009 NMI on Challenging) and stem from the
grid-phase convention (cell edges vs cell centres) used when the caches were
originally computed. The conclusion is identical under either set.

Ambiguous labels: Easy contains 4 pdb_ids spanning two categories
(`1jj2`, `1lt4`, `1nmu`, `1qbk`); Challenging contains 1 (`1ta3`, categories
c.1.8.3 and c.1.8.5); Moderate and Hard have none [M]. The committed manuscript
Table 8 (0.001/0.062/0.076/0.097, NMI Challenging 0.182) predates both the
alignment verification and the ambiguous exclusion, and uses the `min`-NMI
normalization for Challenging. **[R]** Replace Table 8 with the primary numbers
above; state the NMI convention (scikit-learn arithmetic mean) in the caption.

## 2. Robustness across variants — VERIFIED with corrected counts

`results/clustering/variants_<tier>.csv` contain **84 variants per tier: 71
model-based + 13 empirical (model-free)** [M, exact count]. The proposed
response's "67 model-based variants per tier" is wrong.

Best model-based ARI per tier [V]:
0.0116 (Easy, marginal-W1 theta+tau, k=10); 0.1117 (Moderate, Hellinger, k=6);
0.0795 (Hard, Hellinger, k=3); 0.1001 (Challenging, marginal-W1 theta, k=5).
These match the rewrite's rounded 0.012/0.112/0.080/0.100.

**[I]** The weak outcome is genuinely robust to distance, linkage, clusterer and
k. Note the best values come from a search over 71 configurations and must be
presented as a robustness envelope, not as headline performance.

## 3. k-NN retrieval — old artifact insufficient; superseded by new measurement

Committed `results/clustering/knn_retrieval.csv` reports 5-NN accuracies
0.359/0.563/0.574/0.567 but its shuffled control averages only **3**
permutations, and its `chance = 0.25` ignores class imbalance [V]. The proposed
response's "500 shuffles, p = 0.002" nulls (0.249/0.249/0.256/0.441) are **not
supported by any committed artifact** [V] — no committed run used 500 shuffles.

New measurement (majority-vote retrieval, self excluded, 1000 shuffles,
ambiguous-excluded subsets) [M]:

| Tier | model 1NN | model 5NN | global null 5NN | length-stratified null 5NN | majority | length 5NN |
|---|---|---|---|---|---|---|
| Easy | 0.335 | 0.359 | 0.249 | 0.265 | 0.250 | 0.446 |
| Moderate | 0.528 | 0.563 | 0.249 | 0.371 | 0.251 | 0.898 |
| Hard | 0.507 | 0.574 | 0.256 | 0.331 | 0.270 | 0.712 |
| Challenging | 0.539 | 0.565 | 0.435 | 0.506 | 0.503 | 0.948 |

Model 5-NN is above the global null on all tiers (p=0.001) and above the
**length-stratified** null on all tiers (p=0.001, 0.001, 0.001, 0.012) [M].
**But length-only retrieval is higher than model retrieval on every tier** [M].
**[I]** The rewrite's "the class signal is local rather than global" survives
only in a weakened form: locally detectable, small, and outperformed by a
one-dimensional length baseline.

## 4. The protein-length confound — CONFIRMED and quantified

Ward k=4 clustering of `log1p(valid_pair_count)` alone (same subsets, same
protocol) [M]:

| Tier | length ARI | length NMI | PC-NCSF ARI | histogram ARI |
|---|---|---|---|---|
| Easy | 0.0467 | 0.0641 | 0.0031 | 0.1170 |
| Moderate | 0.4480 | 0.5229 | 0.0590 | 0.3206 |
| Hard | 0.2198 | 0.2758 | 0.0761 | 0.0742 |
| Challenging | 0.5615 | 0.7215 | 0.1040 | 0.0981 |

Cross-checks: matches the committed DIAGNOSTIC rows of
`results/clustering_hybrid_corrected/hybrid_corrected_all_g100.csv`
(0.0466/0.4480/0.2198/0.5465 on all-protein subsets) to ≤0.001 [V].

Association of length with the labels, η² of log1p(valid_pairs) explained by
category [M]: Easy 0.125, Moderate 0.674, Hard 0.175, Challenging 0.830.
Example medians (valid pairs): Moderate a.1 147 vs c.1 324; Challenging
c.1.8.3 302 vs c.1.8.4 486.

Length-stratified permutation nulls for the model's Ward ARI (null mean,
p-value) [M]: Easy 0.0005 (p=0.020), Moderate 0.0090 (p=0.001),
Hard 0.0188 (p=0.001), Challenging 0.0740 (p=0.015). On Challenging the
observed 0.104 barely exceeds the 0.074 induced by length alignment alone [I].

Mantel correlation of pairwise distances with |Δ log length| [M]:
PC-NCSF −0.014/0.085/0.049/0.051; histogram 0.022/0.006/0.088/0.080
(permutation p ≤ 0.006 except histogram-Moderate p=0.25).

**[I][R]** Protein length is the strongest single label predictor in these
benchmark tiers — stronger than the learned densities and than the model-free
histograms on every tier. Any clustering result on these tiers must be reported
with this control; otherwise a reader can "improve" any representation by
encoding protein length, as the superseded B2 = 0.5155 result accidentally did
(see `reports/scop_clustering_audit_2026-10-10.md` §7).

## 5. Model-free comparators — VERIFIED, but weaker than claimed in the rewrite

Empirical per-protein histogram, same pipeline, k=4 [V from variants CSVs,
reproduced M at 0.117/0.321/0.074/0.098]: 0.116/0.321/0.074/0.094 (committed).
Best-over-k histogram ARIs reach 0.133/0.449/0.190/0.124 [M] — the Moderate
value (0.449) is again in length-baseline territory.

Circular KDE comparator (`results/clustering/audit2026_controls_g100.csv`,
2000 permutations) [V]: ward ARI 0.0035/0.3185/0.0771/0.0981 at bandwidth ×1,
but 0.1265 at ×0.5 on Easy — strongly bandwidth-unstable on Easy [I]. The KDE
uses train+validation residues while the model saw only the training split;
the comparison is transductive and must be disclosed as such [V].

**[I]** The rewrite's claim that "the limiting factor is the information
content of the unordered per-protein angle densities themselves rather than the
smoothing induced by the flow" is not supported: on Moderate the KDE/histogram
attains 0.32 while the model attains 0.059, so the flow discards
label-relevant structure there. The defensible claim is only that the
limitations are *not specific to the flow* (both representations cluster weakly
at k=4 relative to the length baseline).

## 6. Label hierarchy — VERIFIED codes; proposed mechanism NOT supported

Verified category codes and sizes [M] (from `SCOP/*/data.csv`, loader rule):
Easy = classes a/b/c/d (396/398/399/397 of 1590); Moderate = folds
a.1/b.47/c.1/d.2 (400/399/396/399 of 1594); Hard = superfamilies
c.1.2/c.1.8/c.1.10/c.1.15 (96/98/100/77 of 371); Challenging = families
c.1.8.1/.3/.4/.5 (21/98/53/22 of 194).

**[I]** The rewrite's mechanistic story ("broad classes are distinguished by a
sequential architectural property that an unordered density cannot encode,
whereas families differ in local conformational statistics") is not supported:
broad SCOP classes differ strongly in helix/sheet content, which *is* visible
in (θ,τ) densities (model-free 1-NN on Easy reaches 0.71 vs 0.25 chance
[audit2026_controls]); and the tier pattern tracks the length confound
(η² = 0.125/0.674/0.175/0.830) at least as well as it tracks label granularity.
**[R]** Drop the mechanism; state the composition/length confound instead.

## 7. Unconditional baseline — VERIFIED mathematically and computationally

`circularspline_protein.py` trains the unconditional NCSF with `cond_dim = 1`
and a constant zero condition (`c = torch.zeros_like(x[:, :1])`, lines 100, 123,
138) [V]. Loading `runs/scop_{easy,challenging}/ep20-bs512/uncond/best_flow.pt`
and evaluating with the constant condition yields bit-identical densities;
Hellinger distance between any two per-protein densities is exactly 0 [M].
**[I]** Density-space clustering of the unconditional model is undefined by
construction; the rewrite's statement is correct.

## 8. Statistical-significance hygiene

The primary PC-NCSF ARI on Easy (0.0030) is "significant" under a global
label-permutation test (p≈0.0025 in `audit2026_controls_g100.csv`) while its
null 95% quantile is 0.0012 [V] — a statistically detectable but practically
null effect. **[R]** Never present permutation p-values without the effect size
and the length baseline. All "p = 0.002"-style claims in the rewrite are
replaced by the 1000-shuffle measurements above.

## 9. Summary of incorrect or unsupported claims in the proposed rewrite

1. "67 model-based variants per tier" → actual: 71 model-based (84 total) [V].
2. k-NN nulls "0.249/0.249/0.256/0.441, 500 shuffles, p=0.002" → not supported
   by any committed artifact; the committed control used 3 shuffles [V].
3. "The densities are not uninformative... the limiting factor is the
   information content of the angular densities, not the flow" → contradicted
   by the Moderate KDE/histogram (0.32 vs model 0.059) [M].
4. The class-vs-family mechanism for the ARI gradient → unsupported; the
   gradient is equally consistent with tier-specific length confounding [M].
5. "5-NN accuracies far above chance" framing → true vs chance, but length
   alone is higher on every tier; without the length baseline this framing is
   misleading [M].
6. Challenging NMI 0.143 (rewrite) vs 0.137 (primary pipeline) — the rewrite
   mixed cache-based and primary-pipeline numbers [M].

## 10. Remaining uncertainties / limitations

- The KDE comparator's very high 1-NN accuracies (0.71–0.96) were not
  re-tested against length-stratified nulls (would require recomputing KDE
  grids); retrieval claims in the revision therefore rest on the PC-NCSF
  space and the length baseline, with the KDE reported only for Ward ARI [L].
- The density caches use cell-edge grid phase; the primary pipeline uses cell
  centres. ARI agrees to ≤0.001; NMI on Challenging to 0.009 [M][L].
- Absence of chain IDs / insertion codes in the SCOP CSVs remains (documented
  in `reports/scop_clustering_audit_2026-10-10.md` §11) [L].
- Length here = number of valid consecutive (θ,τ) pairs; residue count gives
  the same picture (Spearman ≈ 1 with pair count on these tiers) [I].
