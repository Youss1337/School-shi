# Week 3 — Support Vector Machines & Unsupervised Learning

Margins and kernels, then clustering without labels. Everything this week measures
**distance**, which is why every method here needs scaled features.

| File | What it is |
|---|---|
| `homework_week3.ipynb` | **Start here.** The scaffold — plumbing done, each step a `TODO`, answers to fill in. |
| `homework_week3_solution.ipynb` | The worked version. All six parts, outputs and figures executed. |
| `homework_week3.py` / `homework_week3_solution.py` | The same two as `# %%` cell scripts. |
| `lecture_week3.py` | The lecture notebook as a cell script. Reference for every pattern. |
| `P1_Clustering_problem_Week_3_2026.ipynb` | The original course file. The brief is its last markdown cell. |

## Running it

No new packages — the Week 1/2 environment covers this. Open
`homework_week3.ipynb`, pick the `base` kernel, and work down it. The dataset is
pulled from GitHub, so you need a connection the first time.

## The headline finding, and it contradicts the brief

The homework asks how many times larger the income spread is than the spending
spread, and the lecture's section 4.1 sets up the expectation that it is large —
"skip the scaler and you will hand in income brackets".

**On this dataset it is 1.02x.** Income is recorded in *thousands* of dollars
(15–137); the spending score runs 1–99. Two unrelated quantities happen to occupy
almost the same numeric range.

So scaled and unscaled K-Means produce the **identical partition — ARI 1.000, zero
of 200 customers assigned differently**. The scaling lesson is correct in general
and simply does not bite here. Had income been in raw dollars the ratio would be
about 1,000x and everything the lecture warns about would happen.

The solution reports this rather than pretending the expected result appeared.

## What the rest turns on

1. **K = 5**, agreed independently by the elbow (each cluster to the fifth buys
   31–42% of remaining WCSS; the sixth buys 16%) and the silhouette (peak 0.5547).
2. **Five segments**, and income does **not** predict spending — all four high/low
   combinations exist. The two off-diagonal groups (low income/high spend, high
   income/low spend) are where the actionable decisions are.
3. **DBSCAN merges three of the five** into one 115-customer region. Those segments
   are not separated by real gaps; they are convenient cuts through a continuum.
   Worth knowing before anyone calls them natural customer types.
4. **DBSCAN's 15 "noise" customers are mostly the highest earners.** Statistically
   isolated, commercially the most valuable people on the list. Low density does not
   mean low value — which is why the same property makes DBSCAN right for fraud and
   wrong here.

## A caveat stated in the solution

`eps = 0.40` is chosen from the k-distance knee and a sweep, not from the best
silhouette. `eps = 0.20` scores higher (0.586) but labels 77 of 200 customers as
noise — the score flatters it because it is computed only on the points that
survived.
