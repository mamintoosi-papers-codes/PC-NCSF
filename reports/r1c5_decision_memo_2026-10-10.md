# R1-C5 Decision Memo — 2026-10-10

Question: what is the strongest scientifically defensible way to answer
Reviewer 1, Comment 5 (real-data clustering validation on SCOP)?

## Recommendation

Adopt the **"evaluated, controlled, honestly negative"** strategy:

1. Report the real-data clustering experiment as requested, with the
   alignment-verified numbers (ARI 0.003 / 0.059 / 0.076 / 0.103).
2. Show it is robust (84 variants per tier; best model-based ARI ≤ 0.112).
3. Add two controls the current draft lacks:
   - a **model-free comparator** (per-protein histograms / KDE) showing the
     limitation is not specific to the flow, and
   - a **protein-length baseline** showing that `log1p(valid_pair_count)`
     alone clusters the labels far better (ARI up to 0.56) than any angular
     representation — i.e., these benchmark tiers are length-confounded.
4. Conclude: the density representation carries a small, statistically
   detectable amount of SCOP-label information beyond length, but not enough
   to support any downstream structural-clustering utility claim. Section 3.5
   becomes a carefully controlled negative/diagnostic result; the synthetic
   clustering is scoped as a controlled demonstration only.

## Why this beats the existing draft (`R1-C5_rewrite_for_Nooshin.md`)

- The draft's k-NN "local signal" argument is true only against chance; the
  length baseline beats the model's retrieval on every tier (5-NN: 0.45/0.90/
  0.71/0.95 vs 0.36/0.56/0.57/0.57). Without the length control the argument is
  misleading, and it is trivially discoverable by any reviewer who clusters a
  single length feature.
- The draft's mechanism story (class = sequential architecture invisible to
  unordered densities) is unsupported; broad classes do differ in helix/sheet
  content, which angular densities capture.
- The draft's "the bottleneck is the representation, not the flow" is
  contradicted on Moderate (KDE/histogram 0.32 vs model 0.059).
- The draft cites k-NN nulls from an uncommitted run (500 shuffles) that no
  artifact supports.

## Trade-off considered for Section 3.5

Keeping 3.5 as a full subsection (recommended) vs demoting it to a paragraph:
the journal explicitly requested the experiment, so removing it would look
evasive; keeping it with rigorous controls converts a weak result into
evidence of careful science. The section's *role* changes from "validation of
downstream utility" to "controlled negative result plus a dataset-composition
caveat". The Abstract and Conclusion are adjusted so no sentence implies that
likelihood gains translate into classification ability.

## What we explicitly do NOT claim

- No claim that weak ARI reflects an inherent limit of angular densities
  (the length baseline and the Moderate KDE show otherwise).
- No mechanistic explanation of the Easy→Challenging ARI gradient beyond the
  measured composition confound.
- No "significance" claims without effect sizes and baselines.
