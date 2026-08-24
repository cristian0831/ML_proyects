# Client Segmentation Analysis

## 1) Problem Statement

Businesses running broad, one-size-fits-all marketing typically waste spend on customers who were never going to convert, while under-serving the customers most likely to respond. Without a systematic way to group customers by behavior, it's hard to answer basic questions like *who should we target first?* or *who is spending beyond what their income would suggest, and worth prioritizing?*

This project addresses that by using unsupervised machine learning to discover natural customer segments from transactional/demographic data — with no pre-labeled "good customer" ground truth to train against — so that marketing and retention efforts can be targeted at the segments most worth pursuing.

## 2) Dataset Description

Source: `data/Mall_Customers.csv` — 200 customer records, 5 columns:

| Column | Type | Description |
|---|---|---|
| `CustomerID` | int | Unique customer identifier |
| `Gender` | categorical | Male / Female |
| `Age` | int | Customer age in years |
| `Annual Income (k$)` | int | Annual income, in thousands of dollars |
| `Spending Score (1-100)` | int | Score assigned based on customer spending behavior/purchasing patterns |

The dataset is small and clean (no missing values), which makes it well suited for demonstrating a full segmentation workflow end to end.

## 3) Goal of the Project

Segment customers into distinct, interpretable groups using unsupervised clustering (KMeans), in order to identify which customers are **easiest to convert** — i.e., which segments show spending behavior disproportionate to their income, or otherwise represent the clearest opportunity for targeted marketing. The output is a set of reusable model artifacts and a scored export, surfaced through an interactive dashboard for business exploration.

## 4) Project Structure

```
Client Segmentation/
├── client_segmentation_analysis.ipynb   # End-to-end analysis: EDA → feature engineering →
│                                         # preprocessing → model training → validation → deployment
├── data/
│   └── Mall_Customers.csv               # Source dataset
├── models/                              # Persisted model artifacts (joblib), produced by the notebook
│   ├── scaler.joblib                    # Fitted StandardScaler
│   ├── kmeans.joblib                    # Fitted KMeans (k=6)
│   └── cluster_features.joblib          # Ordered list of features the model expects
├── exports/                             # Scored data, produced by the notebook
│   ├── customer_segments.csv            # All 200 customers with assigned Cluster + Segment_Name
│   └── segment_summary.csv              # Per-segment aggregate statistics
├── app.py                               # Streamlit dashboard (interactive exploration + live scorer)
├── requirements.txt                     # Pinned dependencies for the dashboard app
└── .venv/                               # Isolated virtual environment for the dashboard (not committed)
```

## 5) Features — Model Training Methodology

**Algorithm**: KMeans clustering, chosen because segments are expected to be roughly convex/globular in the feature space and the goal is a small number of clearly interpretable groups rather than arbitrarily-shaped clusters.

**Clustering features**: `Age`, `Annual Income (k$)`, `Spending Score (1-100)` — standardized via `StandardScaler` (zero mean, unit variance) before fitting, since KMeans relies on Euclidean distance and would otherwise let the larger-magnitude income values dominate the distance calculation over age and spending score.

**Engineered features (used for profiling/interpretation, not for clustering)**:
- `Spending_Income_Ratio` — spending score relative to income, a proxy for "spends beyond means"
- `Age_Group` — binned age (18-25, 26-35, ..., 66+)
- `Income_Tier` — binned income (Low, Mid, High, Very High)

These were deliberately excluded from the clustering inputs because they're derived directly from features already in the model — including them would double-count the same signal and distort the distance metric. They're used afterward to interpret and name the resulting clusters.

`Gender` was label-encoded but also excluded from the clustering distance calculation, for the same reason — a binary dummy variable would distort Euclidean distance disproportionately to its actual signal value; it's retained for post-hoc profiling (e.g., gender mix per segment).

**Choosing k**: swept `k = 2..10`, evaluating both the elbow method (inertia) and silhouette score at each k. Final choice: **k = 6**, selected by the highest silhouette score (~0.43).

**Validation**:
- **Cluster stability** — refit the model across 10 different random seeds and computed the Adjusted Rand Index (ARI) between each run and the reference clustering. Mean ARI ≈ 0.99 (min 0.988), indicating the 6-cluster structure is not an artifact of a lucky initialization.
- **Per-cluster fit quality** — per-point silhouette scores (`silhouette_samples`) were positive on average for every cluster (0.29–0.51), with only ~1.5% of points showing a negative silhouette (borderline/misassigned).
- **Scaling sanity check** — confirmed no single feature dominates the standardized centroid distances, verifying `StandardScaler` did its job before fitting.

## 6) How to Run

### Notebook (full analysis pipeline)
Open `client_segmentation_analysis.ipynb` in Jupyter and run all cells top to bottom. It reads `data/Mall_Customers.csv` and regenerates everything in `models/` and `exports/`.

### Dashboard (interactive exploration)
```bash
cd "Client Segmentation"
source .venv/bin/activate      # create it first if missing: python3 -m venv .venv && pip install -r requirements.txt
streamlit run app.py
```
Then open `http://localhost:8501`. The dashboard loads the artifacts in `models/` and the export in `exports/customer_segments.csv` directly — no need to re-run the notebook unless the underlying data or model changes.

## 7) Main Conclusions

Six segments emerged, each with a distinct income/spending/age profile:

| Segment | Age | Income (k$) | Spending Score | Spend/Income Ratio | Size |
|---|---|---|---|---|---|
| Steady Regulars | 56.3 | 54.3 | 49.1 | 0.9 | 45 |
| Steady Starters | 26.8 | 57.1 | 48.1 | 0.9 | 39 |
| Untapped Opportunity | 41.9 | 88.9 | 17.0 | 0.2 | 33 |
| Premium Customers | 32.7 | 86.5 | 82.1 | 1.0 | 39 |
| Aspirational Spenders | 25.0 | 25.3 | 77.6 | **3.3** | 23 |
| Budget Conscious | 45.5 | 26.3 | 19.4 | 0.8 | 21 |

Key takeaways:

- **Aspirational Spenders are the clearest conversion target.** Despite the lowest income of any segment, they have a spend-to-income ratio more than 3x any other group — strong evidence of high purchase intent and price insensitivity relative to means, making them highly responsive to marketing regardless of budget-tier messaging.
- **Premium Customers are the highest combined-value segment** — high income *and* high spending — and warrant retention-focused rather than acquisition-focused effort, since they're already converting well.
- **Untapped Opportunity is the biggest missed revenue signal** — high income but the lowest spending score of any segment. This group has the financial capacity to spend but isn't currently doing so, making it the prime target for engagement campaigns aimed at unlocking latent spend rather than pure discounting.
- **Steady Regulars and Steady Starters** behave similarly (moderate, stable spending near a 0.9 ratio) despite an ~30-year age gap, suggesting income and spending habits — not age — are the dominant drivers of this behavioral pattern.
- **Budget Conscious customers** show low income and low spending with no ratio outlier — the lowest-priority segment for active marketing spend.
- Cluster assignments were validated as stable (ARI ≈ 0.99 across reseeded refits) and well-separated (positive average silhouette per cluster), so these segments are a reliable basis for targeting decisions rather than an artifact of a single model fit.
