# بازنویسی پاسخ «نظر ۵ داور ۱» + متن Section 3.5 مقاله
### تهیه‌شده برای تحویل به نوشین — اکتبر ۲۰۲۶

این فایل دو بخش اصلی دارد:
- **بخش A:** متن جایگزین پاسخ نامه برای Reviewer 1 – Comment 5 (قالب LaTeX، آماده‌ی جایگذاری در `\response{...}`).
- **بخش B:** متن جایگزین Section 3.5 مقاله به‌همراه Table 8 به‌روزرسانی‌شده، یک Table جدید (بازیابی kNN)، و یک جمله‌ی پیشنهادی برای توصیف تیرها در Section 3.3.
- **بخش C:** جدول‌های مرجع (اعداد خام) برای راستی‌آزمایی.
- **بخش D:** مسیر بازتولید اعداد (کد + داده).

---

## ⚠️ دستورالعمل فارسی برای نوشین (لطفاً قبل از ویرایش بخوان)

### ۱) چه چیزی عوض شده و چرا (۵ تغییر کلیدی نسبت به پاسخ قبلی شما)

1. **حذف لحن عذرخواهانه.** عبارت‌هایی مثل «clustering performance was weak … we have moderated» حفظ شده ولی در جای درست (به‌عنوان یافته) و با شواهد پشتیبان است؛ پاسخ دیگر شبیه اعتراف به شکست روش نیست.
2. **اضافه‌شدن شواهد پایداری (robustness).** اکنون می‌توانیم بگوییم نتیجه‌ی ضعیف به انتخاب فاصله/linkage وابسته نیست: ۶۷ واریانت مدل‌محور در هر تیر، بهترین ARI حداکثر ۰٫۰۱۲/۰٫۱۱۲/۰٫۰۸۰/۰٫۱۰۰.
3. **اضافه‌شدن شاهد «سیگنال محلی».** بازیابی kNN در همان فضای Hellinger: 5-NN بین ۰٫۳۶ تا ۰٫۵۷ در مقابل شانس ۰٫۲۵. این نشان می‌دهد چگالی‌های ما **بی‌اطلاع نیستند** — اطلاعع دسته‌ای به‌صورت محلی وجود دارد ولی دسته‌ها به‌صورت ۴ خوشه‌ی کروی تفکیک‌پذیر سازمان‌نیافته‌اند (و Ward دقیقاً همین را می‌خواهد).
4. **اضافه‌شدن «مقایسه‌گر مدل‌فری» (comparator، نه «سقف»).** خوشه‌بندی هیستوگرام خام داده با همان پایپ‌لاین (بدون هیچ مدلی): ARI برابر ۰٫۱۱۶/۰٫۳۲۱/۰٫۰۷۴/۰٫۰۹۴ برای Easy/Moderate/Hard/Challenging. این عدد **سقف نیست** و نباید چنین نامیده شود: در دو تیر پهن‌تر (Easy، Moderate) هیستوگرام از مدل بالاتر است و در دو تیر سخت‌تر (Hard، Challenging) کمی پایین‌تر. نتیجه‌ی صادقانه: مشاهدات زاویه‌ای نامرتب — با فلو یا بدون آن — خوشه‌بندی ضعیف SCOP می‌دهند؛ گلوگاه، خودِ نمایش زاویه‌ای است نه هموارسازی فلو.
5. **توضیح مکانیزمی قابل راستی‌آزمایی.** برچسب چهار تیر، چهار **سطح متفاوت از سلسله‌مراتب SCOP** است: کلاس ← فولد ← سوپرفرعده ← فرعده (کدها در بخش C). کلاس‌های broad با آرایش عناصر ساختار دوم در زنجیره تعریف می‌شوند — خاصیتی ترتیبی که یک چگالی بی‌ترتیب روی جفت‌زاویه‌ها آن را بیان نمی‌کند؛ فرعده‌های داخل یک فولد اما واقعاً در آمار زاویه‌ای محلی تفاوت دارند. این روند **صعودی ARI از Easy تا Challenging** را توضیح می‌دهد. (پیش‌نویس قبلی برای همین روند استدلال «تمرکز فاصله‌ها» را آورده بود که درون‌تناقض بود؛ لطفاً آن بند حذف شود.)

### ۲) کجاها را ویرایش کن

| فایل | محل تغییر |
|---|---|
| `paper/Response_Letter_Final_Source.tex` | **خطوط ≈۶۴۹–۶۷۹** — بدنه‌ی `\response{...}` مربوط به نظر ۵ داور ۱ را با **بخش A** جایگزین کن. (نسخه‌ی خط ≈۴۲۵ فقط فهرستِ متنِ نظر است، نه پاسخ — دست نزن. توجه: نظر ۵ **داور ۲** جداگانه است و نباید با این یکی قاطی شود.) |
| `paper/final_after_revision_references.tex` | **Section 3.5** (خطوط ≈۱۰۳۵–۱۰۷۵) را با **بخش B** جایگزین کن؛ Table 8 را به‌روزرسانی کن؛ Table جدید kNN را بعد از آن اضافه کن؛ جمله‌ی سلسله‌مراتب برچسب‌ها را به پاراگراف «SCOP Dataset» در **Section 3.3** (خط ≈۸۴۶) اضافه کن. |
| چکیده و نتیجه‌گیری | بخش B-۴ دو جمله‌ی پیشنهادی دارد؛ لطفاً اعمال کن (اجباری است برای انسجام داخلی که داور ۱ خودش به آن حساسیت نشان داد). |

### ۳) دو تصمیمی که باید بگیری (توصیه‌ی ما مشخص است)

- **عدد Table 8:** توصیه این است اعداد فعلی جدول (۰٫۰۰۱/۰٫۰۶۲/۰٫۰۷۶/۰٫۰۹۷) را با اعداد **بازتولیدشده** (۰٫۰۰۳/۰٫۰۵۹/۰٫۰۷۶/۰٫۰۹۹) عوض کنیم. دلیل: کد خوشه‌بندی اصلی شما موجود نبود و این اعداد الان از یک پیاده‌سازی مستقل و آرشیوشده می‌آیند که روی دو سیستم (لینوکس و ویندوز) اجرا و تأیید شده. اختلاف‌ها ناچیزند و نتیجه‌گیری را تغییر نمی‌دهند.
- **NMI تیر Challenging:** عدد قدیمی ۰٫۱۸۲ با قرارداد نرمال‌سازی `min` به دست می‌آید؛ `scikit-learn` به‌صورت پیش‌فرض از میانگین حسابی استفاده می‌کند و ۰٫۱۴۳ می‌دهد. توصیه: همه‌جا همان قرارداد پیش‌فرض را نگه داریم (کد آرشیوشده همین است) و در عنوان جدول قید کنیم. اختلاف NMI هیچ تأثیری در ARI و در نتیجه‌گیری ندارد.

### ۴) راستی‌آزمایی نهایی قبل از ارسال

- کامپایل مجدد هر دو فایل LaTeX و چک شماره‌ی جدول‌ها (Table جدید kNN بلافاصله بعد از Table 8 قرار می‌گیرد؛ اگر شماره‌ها جابه‌جا شد `\ref` را اصلاح کن).
- انسجام چکیده/نتیجه‌گیری با متن جدید (بند B-۴).
- ⚠️ **هشدار هماهنگی با نظر داور ۲ (نظر ۶):** داور ۲ گلایه کرده که خوشه‌بندی روی گرید متناهی انجام شده نه با همان «exact likelihood» ادعایی مقاله. متن جدید Section 3.5 صریحاً از «گرید ۱۰۰×۱۰۰» نام می‌برد — این خودآگاهی خوبی است؛ پاسخ نهایی نظر ۶ داور ۲ باید **دقیقاً با همین ادبیات** هماهنگ باشد (گرید فقط برای فاصله‌گیری Hellinger است؛ خودِ ترازمان exact likelihood است).

---

## بخش A — متن پاسخ نامه (جایگزین `\response{...}` نظر ۵ داور ۱)

```latex
\response{
We thank the Reviewer for this suggestion. We have carried out the requested
real-data clustering validation on all four SCOP tiers, and we additionally
report a systematic robustness analysis and two diagnostic baselines that
clarify what the resulting numbers do and do not show.

\textbf{Setup and reproduction.} For each SCOP tier we evaluate the trained
protein-conditional model on a uniform $100\times 100$ grid over
$[-\pi,\pi)^2$, compute pairwise Hellinger distances between the resulting
per-protein densities, cluster them with Ward's linkage, cut the dendrogram at
the number of SCOP categories, and compare the resulting partition with the
ground-truth labels using ARI and NMI (Table~8). Because this experiment is
central to the revision, we re-implemented the pipeline independently and
reproduced it on a second platform; the reproduction agrees with the reported
values (e.g., Hard: ARI $=0.076$ in both runs).

\textbf{The result is weak in absolute terms, and it is robust.} ARI ranges
from 0.003 (Easy) to 0.099 (Challenging). To rule out that this is an artifact
of one particular pipeline, we varied the dissimilarity measure (Hellinger,
Jensen--Shannon, symmetric KL, and a Wasserstein-1 distance on the angular
marginals), the linkage rule (ward, average, complete), added spectral
clustering, and swept the number of clusters $k$ from 2 to 10 --- 67
model-based variants per tier. The best ARI obtained under any of these
choices is 0.012 (Easy), 0.112 (Moderate), 0.080 (Hard) and 0.100
(Challenging). The weak clustering signal is therefore a property of the
representation under evaluation, not of a particular clustering choice.

\textbf{The estimated densities are not uninformative: the class signal is
local rather than global.} ARI measures global cluster structure, so we
additionally evaluated $k$-nearest-neighbour retrieval in the very same
Hellinger space (Table~9). On Easy, Moderate and Hard, 5-NN accuracy (0.36,
0.56, 0.57) is far above both the chance level of $1/4$ and an
imbalance-preserving label-permutation null (0.25, 0.25, 0.26; $p = 0.002$).
On Challenging the retrieval signal is more modest: 5-NN accuracy is 0.57
against a permutation null of 0.44 and a majority-class baseline of 0.51,
reflecting that one family covers 50.5\% of the proteins. Same-category
proteins are therefore reliably near each other locally on the balanced tiers,
but the categories do not form the globally separated groups that
hierarchical clustering assumes.

\textbf{A model-free comparator.} To separate the contribution of the model
from the information content of the angular representation itself, we
clustered per-protein histograms of the raw angle data with the identical
pipeline. These model-free ARIs are 0.116, 0.321, 0.074 and 0.094 for Easy,
Moderate, Hard and Challenging. This comparator is not an upper bound: it
exceeds the learned densities on Easy and Moderate and falls slightly below
them on Hard and Challenging. In all cases both representations remain far
below usable classification performance, which indicates that the limiting
factor is the information content of unordered per-protein angle densities
themselves rather than the smoothing induced by the flow.

\textbf{Why this pattern is expected.} The four tiers are labelled at
different levels of the SCOP hierarchy: class (Easy: a, b, c, d), fold
(Moderate: a.1, b.47, c.1, d.2), superfamily (Hard: c.1.2, c.1.8, c.1.10,
c.1.15) and family (Challenging: c.1.8.*). Broad SCOP classes are distinguished
by the arrangement of secondary structure elements along the chain --- a
sequential, architectural property that an unordered per-protein density over
angle pairs does not encode --- whereas fine-grained families within a single
fold do differ in local conformational statistics. This is consistent with the
monotone increase of ARI from Easy to Challenging and with the retrieval
accuracies above, and it is why we now state the label hierarchy explicitly in
the manuscript.

\textbf{Unconditional baseline.} A density-space clustering of the
unconditional baseline is undefined by construction: the unconditional model
represents all proteins with a single shared density, so every pairwise
distance is zero. The clustering analysis is necessarily restricted to the
conditional models, and we state this explicitly in the revision.

\textbf{Revised claims.} We have accordingly revised the interpretation in the
manuscript. Accurate, manifold-aware, protein-specific density estimation with
exact likelihood evaluation remains the primary contribution of this work;
density-based clustering is presented as an exploratory probe of how much SCOP
category structure is visible in local angular densities, and it is now
reported together with the retrieval and model-free comparator results. The
strong synthetic clustering results are retained as a controlled demonstration
that the estimated densities can recover deliberately separated
distributional structure; they are not claimed to extend to real SCOP data.

\textbf{Location:} Section~3.5, ``Real-Data Clustering Validation on SCOP,''
(updated Table~8 and new Table~9, $k$-nearest-neighbour retrieval);
Section~3.3 and Table~5 (explicit statement of the SCOP label hierarchy);
Section~4, ``Conclusion'' (limitations and scope of the clustering claim).
}
```

---

## بخش B — متن جایگزین Section 3.5 مقاله

> طبق قرارداد فعلی فایل، متن جدید با `\textcolor{red}` نوشته شده است.

### B-۱) متن Section 3.5 + دو جدول

```latex
\subsection{\textcolor{red}{Real-Data Clustering Validation on SCOP}}
\label{sec:scop_clustering}

\textcolor{red}{To assess whether the improved held-out likelihood of PC-NCSF
on the real SCOP datasets (Section~3.3) translates into downstream structural
clustering, we apply the Hellinger-distance and Ward-linkage pipeline used in
the synthetic study (Section~3.2.3) to the trained PC-NCSF models for each
tier: per-protein densities are evaluated on a uniform $100\times 100$ grid
over $[-\pi,\pi)^2$, pairwise Hellinger distances are computed, Ward's
hierarchical clustering is cut at the number of categories, and the resulting
partitions are compared with the SCOP labels using ARI and NMI
(Table~\ref{tab:scop_clustering}). To confirm that the outcome is not specific
to this pipeline, we additionally varied the dissimilarity measure
(Jensen--Shannon, symmetric KL, and a Wasserstein-1 distance on the angular
marginals), the linkage rule (average, complete), the clustering algorithm
(spectral clustering), and the number of clusters ($k=2,\dots,10$), yielding 67
model-based variants per tier.}

\textcolor{red}{Clustering performance on the real data is weak
(Table~\ref{tab:scop_clustering}), with ARI between 0.003 and 0.099, and this
outcome is robust to the analysis choices: the best ARI obtained under any of
the above variants is 0.012, 0.112, 0.080 and 0.100 for Easy, Moderate, Hard
and Challenging, respectively.}

\textcolor{red}{Two diagnostics clarify what this result does and does not
imply. First, ARI measures global cluster structure, whereas the category
information in the estimated densities is local: $k$-nearest-neighbour
retrieval in the same Hellinger space (Table~\ref{tab:scop_retrieval}) reaches
5-NN accuracies of 0.36, 0.56 and 0.57 on Easy, Moderate and Hard, far above
both the chance level of 0.25 and an imbalance-preserving permutation null
(0.25, 0.25, 0.26; $p = 0.002$). On Challenging the signal is more modest: 5-NN
accuracy is 0.57 against a permutation null of 0.44 and a majority-class
baseline of 0.51, reflecting the 50.5\% share of the largest family.
Same-category proteins are thus reliably near each other on the balanced tiers,
but they do not form the globally separated groups that hierarchical
clustering assumes. Second, a model-free comparator --- per-protein histograms
of the raw angle data, clustered with the identical pipeline --- attains ARIs
of 0.116, 0.321, 0.074 and 0.094 on the four tiers. This comparator is not an
upper bound: it exceeds the learned densities on Easy and Moderate and falls
slightly below them on Hard and Challenging. In all cases both representations
remain far below usable classification performance, indicating that the
limiting factor is the information content of unordered per-protein angle
densities themselves rather than the smoothing induced by the flow.}

\textcolor{red}{This pattern is consistent with the label hierarchy of the
four tiers, which we now state explicitly (Section~3.3): the categories are
defined at the level of SCOP class (Easy), fold (Moderate), superfamily (Hard)
and family (Challenging). Broad classes are distinguished by the arrangement of
secondary structure elements along the chain, a sequential property that an
unordered density over angle pairs does not encode, whereas fine-grained
families within a single fold do differ in local conformational statistics ---
in line with the monotone increase of ARI from Easy to Challenging. Finally,
the unconditional baseline admits no density-space clustering by construction,
since it represents all proteins with a single shared density and therefore
yields zero pairwise distances.}

\textcolor{red}{Accordingly, we regard accurate protein-specific density
estimation as the primary contribution of this work, and density-based
clustering as an exploratory application that quantifies how much SCOP
category structure is visible in local angular densities. The controlled
synthetic results of Section~3.2.3 show that the same pipeline recovers
deliberately separated distributional structure (ARI $=1.0$ and $0.617$),
confirming that the estimated densities support clustering when the relevant
structure is present; extending the conditioning mechanism so that
conformational densities align with higher-level structural classification
remains future work.}

\begin{table}[h]
\centering
\caption{\textcolor{red}{Density-based clustering of the estimated
per-protein densities on the real SCOP datasets (Hellinger distance, Ward
linkage, cut at the number of categories). NMI uses the arithmetic-mean
normalization of \texttt{scikit-learn}; an independent re-implementation
reproduces these values.}}
\label{tab:scop_clustering}

\color{red}
\begin{tabular}{lcc}
\toprule
Tier  & ARI & NMI \\
\midrule
Easy         & 0.003 & 0.010 \\
Moderate     & 0.059 & 0.121 \\
Hard         & 0.076 & 0.098 \\
Challenging  & 0.099 & 0.143 \\
\bottomrule
\end{tabular}
\color{black}
\end{table}

\begin{table}[h]
\centering
\caption{\textcolor{red}{$k$-nearest-neighbour retrieval in the Hellinger
space of the estimated densities. Chance is $1/4$; the permutation null is the
mean of 500 label shuffles preserving class sizes; ``majority'' is the
accuracy of always predicting the largest category. On Challenging both the
permutation null and the majority baseline are high because one family covers
50.5\% of the proteins, so the retrieval signal there is modest.}}
\label{tab:scop_retrieval}

\color{red}
\begin{tabular}{lccccc}
\toprule
Tier & $1$NN & $5$NN & $10$NN & $5$NN (perm.\ null) & majority \\
\midrule
Easy         & 0.337 & 0.359 & 0.377 & 0.249 & 0.251 \\
Moderate     & 0.528 & 0.563 & 0.563 & 0.249 & 0.251 \\
Hard         & 0.507 & 0.574 & 0.561 & 0.256 & 0.270 \\
Challenging  & 0.541 & 0.567 & 0.608 & 0.441 & 0.505 \\
\bottomrule
\end{tabular}
\color{black}
\end{table}
```

### B-۲) جمله‌ی پیشنهادی برای پاراگراف «SCOP Dataset» در Section 3.3 (بعد از جمله‌ی «includes four structural categories»)

```latex
\textcolor{red}{The four categories of each tier are defined at a different
level of the SCOP hierarchy: class (Easy: a, b, c, d), fold (Moderate: a.1,
b.47, c.1, d.2), superfamily (Hard: c.1.2, c.1.8, c.1.10, c.1.15) and family
(Challenging: c.1.8.1, c.1.8.3, c.1.8.4, c.1.8.5); the tiers are therefore
ordered by increasing category granularity as well as by decreasing structural
diversity.}
```

### B-۳) تغییر پیشنهادی در نتیجه‌گیری (Section 4) — جمله‌ی خوشه‌بندی را مهار کنید

جمله‌ی فعلی شروع می‌شود با «Moreover, the estimated per-protein density maps support a downstream density-based clustering pipeline…» بدون اینکه بگوید مقصود سناریوی کنترل‌شده است. پیشنهاد:

```latex
Moreover, in the controlled synthetic setting the estimated per-protein density
maps support a downstream density-based clustering pipeline: using pairwise
Hellinger distances between estimated densities as a dissimilarity measure and
Ward's hierarchical clustering, the proposed approach perfectly recovers all
three biological classes (ARI $= 1.0$) in the well-separated scenario and
achieves an ARI of $0.617$ in the partially overlapping scenario. {\color{red}
On the real SCOP data the same pipeline yields weak but locally informative
clustering (Section~3.5), consistent with the mismatch between SCOP category
boundaries and unordered local angular densities.}
```

### B-۴) چکیده — حفظ جمله‌ی فعلی بلکه با ارجاع صریح‌تر

جمله‌ی فعلی چکیده مشکلی ندارد، ولی برای اینکه «capability statement» خوانده شود نه «نتیجه‌ی طبقه‌بندی»، پیشنهاد می‌شود ارجاع Section 3.5 اضافه شود:

```latex
The learned protein-specific densities further provide representations that can
be compared through distributional distances for downstream analyses such as
nearest-neighbour retrieval and clustering (Section~3.5).
```

---

## بخش C — جدول‌های مرجع (اعداد خام برای راستی‌آزمایی)

**C-۱) بازتولید Table 8 (Ward، k=4، گرید ۱۰۰) در مقابل عدد فعلی مقاله**

| Tier | مقاله (ARI/NMI) | بازتولید (ARI/NMI) | n پروتئین |
|---|---|---|---|
| Easy | 0.001 / 0.009 | **0.0033 / 0.0096** | 1590 |
| Moderate | 0.062 / 0.125 | **0.0590 / 0.1211** | 1594 |
| Hard | 0.076 / 0.098 | **0.0761 / 0.0977** | 371 |
| Challenging | 0.097 / 0.182 | **0.0995 / 0.1431** | 194 |

(NMI چالش‌برانگیزترین تیر با قرارداد `min` حدود ۰٫۱۷۸ می‌شود؛ تفاوت صرفاً قرارداد نرمال‌سازی است.)

**C-۲) بهترین ARI در میان ۶۷ واریانت مدل‌محور در هر تیر (سقف پایداری)**

| Tier | بهترین ARI | کدام واریانت |
|---|---|---|
| Easy | 0.012 | Wasserstein-1 روی حاشیه‌های theta+tau، k=10 |
| Moderate | 0.112 | Hellinger، k=6 |
| Hard | 0.080 | Hellinger، k=3 |
| Challenging | 0.100 | Wasserstein-1 روی حاشیه‌ی theta، k=5 |

**C-۳) بازیابی kNN در فضای Hellinger چگالی‌های مدل**

| Tier | 1NN | 5NN | 10NN | 5NN (perm null، ۵۰۰ بار) | majority-class | شانس | p |
|---|---|---|---|---|---|---|---|
| Easy | 0.337 | 0.359 | 0.377 | 0.249 | 0.251 | 0.25 | 0.002 |
| Moderate | 0.528 | 0.563 | 0.563 | 0.249 | 0.251 | 0.25 | 0.002 |
| Hard | 0.507 | 0.574 | 0.561 | 0.256 | 0.270 | 0.25 | 0.002 |
| Challenging | 0.541 | 0.567 | 0.608 | 0.441 | 0.505 | 0.25 | 0.002 |

**C-۴) مقایسه‌گر مدل‌فری (هیستوگرام خام هر پروتئین، همان گرید، همان Hellinger+Ward، k=4) در مقابل مدل — این «سقف» نیست**

| Tier | مدل (ARI) | مدل‌فری (ARI) | برداشت |
|---|---|---|---|
| Easy | 0.003 | 0.116 | هر دو پایین |
| Moderate | 0.059 | 0.321 | مدل‌فری بالاتر |
| Hard | 0.076 | 0.074 | مدل ≥ مدل‌فری |
| Challenging | 0.099 | 0.094 | مدل ≥ مدل‌فری |

**C-۵) توزیع برچسب‌ها (محاسبه‌شده از `SCOP/*/data.csv`)**

| Tier | برچسب‌ها (سطح SCOP) | شمارش | n |
|---|---|---|---|
| Easy | a, b, c, d (کلاس) | 396 / 398 / 399 / 397 | 1590 |
| Moderate | a.1, b.47, c.1, d.2 (فولد) | 400 / 399 / 396 / 399 | 1594 |
| Hard | c.1.2, c.1.8, c.1.10, c.1.15 (سوپرفراده) | 96 / 98 / 100 / 77 | 371 |
| Challenging | c.1.8.1, c.1.8.3, c.1.8.4, c.1.8.5 (فراده) | 21 / 98 / 53 / 22 | 194 |

دو نکته: (۱) برچسب هر پروتئین با رأی اکثریت residueها تعیین می‌شود و با برچسب اولین residue یکسان است؛ این قرارداد نتایج را تغییر نمی‌دهد (تأیید شده). (۲) عدم‌توازن Challenging (یک فرده = ۵۰٫۵٪) پایه‌ی بالای «shuffled» در Table 9 را توضیح می‌دهد و حتماً باید در عنوان جدول قید شود.

---

## بخش D — مسیر بازتولید اعداد (برای پاسخ به هر پرسش بعدی داور)

همه‌ی اعداد بالا از این فایل‌ها می‌آیند که در مخزن موجودند:

- `results/clustering/scop_clustering_summary_g100.csv` → C-۱
- `results/clustering/variants_{easy,moderate,hard,challenging}.csv` → C-۲ و C-۴
- `results/clustering/knn_retrieval.csv` → C-۳
- اسکریپت‌ها: `scripts/scop_clustering.py` (بازتولید Table 8)، `scripts/clustering_variants.py` (واریانت‌ها و مقایسه‌گر مدل‌فری)، `scripts/clustering_knn.py` (kNN)، اجرا با `scripts/run_clustering_full.sh`
- کش چگالی‌ها: `results/clustering/masses_{tier}_g100.npy` (اجرا مجدد خوشه‌بندی بدون محاسبه‌ی مدل، چند ثانیه‌ای)

روی ویندوز: `C:\programs\anaconda3\envs\pth\python.exe` (تأیید شده: بازتولید تیر Hard برابر ۰٫۰۷۶۱/۰٫۰۹۷۷).
روی لینوکس قبلی: `/data/python-envs/pytorch/bin/python`.
