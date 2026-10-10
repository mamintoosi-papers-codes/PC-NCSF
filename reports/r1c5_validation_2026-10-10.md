# R1-C5 Validation Report — 2026-10-10

## 1. Experiments executed (this audit)

All run from the repository root, environment
`C:\programs\anaconda3\envs\pth\python.exe` (torch 2.3.1, zuko 1.5.0,
scikit-learn 1.5.1, scipy 1.15.3, numpy 2.0.0), CPU.

| Command | Config | Exit | Outputs |
|---|---|---|---|
| `python scripts/r1c5_length_controls.py --grid 100 --n-shuffle 1000 --n-mantel 200` | seed 20261010; ambiguous-excluded subsets (1586/1594/371/193); 3 representations; ward k=#categories; 1000 global + 1000 length-stratified label shuffles; 200 Mantel permutations | 0 | `results/clustering/r1c5_length_controls_g100.csv`, `results/clustering/r1c5_mantel_g100.csv` |
| `python scripts/r1c5_class_length_summary.py` | loader dropna rule; majority labels; eta^2 of log1p(valid pairs) on category | 0 | `results/clustering/r1c5_class_length_summary.csv` |
| inline Python: unconditional checkpoint check | `runs/scop_{easy,challenging}/ep20-bs512/uncond/best_flow.pt`, constant-zero condition, 50x50 grid | 0 | identical evaluations, Hellinger distance exactly 0 (recorded in §3) |
| inline Python: variants counts + best-ARI extraction | `results/clustering/variants_*.csv` | 0 | 84 rows/tier = 71 model-based + 13 empirical; best model ARIs 0.0116/0.1117/0.0795/0.1001 |
| inline Python: numeric cross-check of tex vs CSVs | regex-parses Tables 8/9 and all text numbers from the revised manuscript; compares to CSV artifacts; whitespace-normalized literal checks of the response letter | 0 | 62/62 numeric checks pass; 13/13 normalized letter/manuscript text checks pass (an initial strict-literal run flagged 2 false alarms caused by line wrapping, resolved by normalized re-matching) |

No committed artifact was modified; all new outputs use new file names.

## 2. Verified numbers (cross-checks, all passing)

- Primary pipeline (committed `audit2026_pcncsf_g100.csv`): ward ARI
  0.0030/0.0593/0.0763/0.1032, NMI 0.0089/0.1244/0.0965/0.1374,
  m = 1586/1594/371/193. Independent recomputation from the committed density
  caches reproduces ARI to ≤0.001 (0.0031/0.0590/0.0761/0.1040).
- Variants (committed): best model-based ARI 0.012/0.112/0.080/0.100 ✓;
  empirical histogram committed 0.116/0.321/0.074/0.094, ambiguous-excluded
  recompute 0.117/0.321/0.074/0.098 ✓.
- Length DIAGNOSTIC rows of the committed hybrid-corrected CSV
  (0.0466/0.4480/0.2198/0.5465, all-protein) reproduced on kept subsets as
  0.0467/0.4480/0.2198/0.5615 ✓ (Challenging differs only via the excluded
  ambiguous id).
- Model 1NN/5NN from the committed `knn_retrieval.csv` (all-protein
  0.3365/0.3591, 0.5276/0.5634, 0.5067/0.5741, 0.5412/0.5670) reproduced on
  kept subsets (0.335/0.359, 0.528/0.563, 0.507/0.574, 0.539/0.565) ✓.
- KDE rows quoted in the response come from the committed
  `audit2026_controls_g100.csv` (ward ARI 0.0035–0.3185 across bandwidths) ✓.
- SCOP category codes/counts per tier verified from `SCOP/*/data.csv` ✓.

## 3. Computational claims verified

- Unconditional baseline: `circularspline_protein.py` trains with
  `cond_dim=1` and `c = torch.zeros_like(x[:, :1])` (lines 100/123/138);
  loading the easy and challenging uncond checkpoints and evaluating with the
  constant condition gives bit-identical densities; Hellinger distance between
  any two per-protein densities is exactly 0. The "undefined by construction"
  statement is therefore verified both mathematically and computationally.
- Ambiguous labels: Easy `1jj2`, `1lt4`, `1nmu`, `1qbk`; Challenging `1ta3`;
  none in Moderate/Hard (matches `audit2026_pcncsf_g100.csv` counts 4/0/0/1).

## 4. LaTeX compilation

- Manuscript `paper/final_after_revision_references.tex` (TeXLive 2026,
  pdflatex): after two passes, **0 undefined references** (new Table 8 =
  `tab:scop_clustering` on p.24, Table 9 = `tab:scop_controls` on p.25,
  both resolve), **0 non-figure errors**, 0 overfull boxes, 26 pages.
- Response letter `paper/Response_Letter_Final_Source.tex`: compiles with
  exit 0, **0 errors**, 14 pages.
- Pre-existing, environment-caused issues (unchanged by this work):
  (a) figure image files are not in the repository, so the manuscript build
  here runs in draft mode for figures; the committed figure-ful PDFs were
  restored with `git checkout` and the author should rebuild locally with the
  figure assets before submission; (b) `references_final.bib` and the
  `elsarticle_harv_doi` style file are absent from the repository, so bibtex
  cannot resolve the 36 citations locally. The revised text adds no new
  `\cite` commands.

## 5. Files changed / added

Modified (uncommitted, per policy):
- `paper/Response_Letter_Final_Source.tex` — R1-C5 response replaced
  (only that response; all other comments untouched).
- `paper/final_after_revision_references.tex` — Abstract sentence, Section 3.3
  label-hierarchy sentence, full Section 3.5 rewrite with Tables 8 and 9,
  Conclusion clustering passage and Limitations addition (all in red, per the
  journal's colored-revision requirement).

New:
- `scripts/r1c5_length_controls.py`, `scripts/r1c5_class_length_summary.py`
- `results/clustering/r1c5_length_controls_g100.csv`,
  `results/clustering/r1c5_mantel_g100.csv`,
  `results/clustering/r1c5_class_length_summary.csv`
- `reports/r1c5_evidence_audit_2026-10-10.md`,
  `reports/r1c5_decision_memo_2026-10-10.md`,
  `reports/r1c5_validation_2026-10-10.md`
- `reports/r1c5_backup_2026-10-10/` — pre-edit copies of the two tex files and
  the superseded `R1-C5_rewrite_for_Nooshin.md`.

## 6. Unresolved issues and limitations

1. The KDE comparator was used only for Ward ARI (from the committed audit
   run); its very high 1-NN accuracies were not re-tested against
   length-stratified nulls, so retrieval claims in the revision rest on the
   PC-NCSF space and the length baseline.
2. Density caches (cell-edge grid phase) vs primary pipeline (cell centres):
   ARI agrees to ≤0.001; NMI on Challenging to 0.009. Both are documented;
   Table 8 uses the primary pipeline, Table 9 the cache-based run.
3. Absence of chain IDs / insertion codes in the SCOP CSVs remains (documented
   in `reports/scop_clustering_audit_2026-10-10.md`).
4. Nothing was committed or pushed; all changes are left in the working tree
   for review. The superseded proposed rewrite is preserved unchanged in the
   backup folder and in git history.
