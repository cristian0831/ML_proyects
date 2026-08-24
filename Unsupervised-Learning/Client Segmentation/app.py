from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

# Business-friendly names for the 6 KMeans clusters. Not persisted anywhere in the
# exported CSVs (those only carry the mode-based "Low Income / 18-25" style label) -
# derived here from exports/segment_summary.csv stats: Cluster 4 has the highest
# Spending_Income_Ratio (3.3, low income but spends beyond its means) which anchors
# "Aspirational Spenders"; the rest follow the same income/spending/ratio pattern.
FRIENDLY_SEGMENT_NAMES = {
    0: "Steady Regulars",       # mid income, older, moderate steady spend, largest group
    1: "Steady Starters",       # mid income, younger, moderate steady spend
    2: "Untapped Opportunity",  # high income, lowest spending score, ratio 0.2
    3: "Premium Customers",     # high income AND high spending, ratio 1.0
    4: "Aspirational Spenders", # low income, highest ratio 3.3
    5: "Budget Conscious",      # low income, low spending, ratio 0.8
}

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")


@st.cache_data
def load_customer_segments() -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / "exports" / "customer_segments.csv")
    df["Friendly_Segment"] = df["Cluster"].map(FRIENDLY_SEGMENT_NAMES)
    return df


@st.cache_resource
def load_model_artifacts():
    scaler = joblib.load(BASE_DIR / "models" / "scaler.joblib")
    kmeans = joblib.load(BASE_DIR / "models" / "kmeans.joblib")
    features = joblib.load(BASE_DIR / "models" / "cluster_features.joblib")
    return scaler, kmeans, features


df = load_customer_segments()
scaler, kmeans, cluster_features = load_model_artifacts()

st.title("Customer Segmentation Dashboard")
st.caption("KMeans segmentation of mall customers — Age, Annual Income, and Spending Score.")

# ---------------------------------------------------------------------------
# Filter bar
# ---------------------------------------------------------------------------
with st.container():
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1.2, 1.2, 0.8])
    with c1:
        selected_segments = st.multiselect(
            "Segments",
            options=sorted(df["Friendly_Segment"].unique()),
            default=sorted(df["Friendly_Segment"].unique()),
            key="segment_filter",
        )
    with c2:
        selected_gender = st.selectbox(
            "Gender",
            options=["All"] + sorted(df["Gender"].unique()),
            key="gender_filter",
        )
    with c3:
        age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
        age_range = st.slider("Age", age_min, age_max, (age_min, age_max), key="age_filter")
    with c4:
        inc_min, inc_max = int(df["Annual Income (k$)"].min()), int(df["Annual Income (k$)"].max())
        income_range = st.slider(
            "Annual Income (k$)", inc_min, inc_max, (inc_min, inc_max), key="income_filter"
        )
    with c5:
        st.write("")
        st.write("")
        if st.button("Reset filters"):
            for key in ["segment_filter", "gender_filter", "age_filter", "income_filter"]:
                st.session_state.pop(key, None)
            st.rerun()

filtered_df = df[df["Friendly_Segment"].isin(selected_segments)]
if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]
filtered_df = filtered_df[
    filtered_df["Age"].between(*age_range)
    & filtered_df["Annual Income (k$)"].between(*income_range)
]

if filtered_df.empty:
    st.info("No customers match the current filters.")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers", len(filtered_df))
k2.metric("Segments Represented", filtered_df["Cluster"].nunique())
k3.metric("Avg Spending Score", f"{filtered_df['Spending Score (1-100)'].mean():.1f}")
k4.metric("Avg Income", f"${filtered_df['Annual Income (k$)'].mean():.1f}k")

st.divider()

# ---------------------------------------------------------------------------
# Segment overview + comparison table
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1.4])

with col_left:
    st.subheader("Segment Overview")
    counts = (
        filtered_df.groupby("Friendly_Segment")
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=True)
    )
    fig_bar = px.bar(counts, x="Count", y="Friendly_Segment", orientation="h", text="Count")
    fig_bar.update_layout(yaxis_title=None, xaxis_title="Customers", margin=dict(l=0, r=10, t=10, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("Segment Comparison")
    comparison = (
        filtered_df.groupby(["Cluster", "Friendly_Segment"])
        .agg(
            Age=("Age", "mean"),
            Income=("Annual Income (k$)", "mean"),
            Spending=("Spending Score (1-100)", "mean"),
            Ratio=("Spending_Income_Ratio", "mean"),
            Count=("CustomerID", "count"),
        )
        .reset_index()
        .sort_values("Cluster")
        .drop(columns="Cluster")
        .rename(columns={"Friendly_Segment": "Segment"})
    )

    def highlight_max_ratio(s):
        return ["background-color: #ffedd5; font-weight: 600" if v == s.max() else "" for v in s]

    styled = comparison.style.format(
        {"Age": "{:.1f}", "Income": "{:.1f}", "Spending": "{:.1f}", "Ratio": "{:.2f}"}
    ).apply(highlight_max_ratio, subset=["Ratio"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("Highlighted: highest Spending/Income ratio — strongest conversion propensity signal.")

st.divider()

# ---------------------------------------------------------------------------
# Income vs Spending scatter with centroids
# ---------------------------------------------------------------------------
st.subheader("Income vs. Spending Score")
fig_scatter = px.scatter(
    filtered_df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Friendly_Segment",
    hover_data=["CustomerID", "Age", "Gender"],
)

# kmeans.cluster_centers_ lives in *scaled* space - must inverse-transform before
# plotting against raw axes, otherwise centroids land near (0, 0).
centers_original = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_), columns=cluster_features
)
fig_scatter.add_trace(
    go.Scatter(
        x=centers_original["Annual Income (k$)"],
        y=centers_original["Spending Score (1-100)"],
        mode="markers",
        marker=dict(symbol="x", size=14, color="black", line=dict(width=2)),
        name="Centroids",
    )
)
fig_scatter.update_layout(margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Live segment scorer
# ---------------------------------------------------------------------------
st.subheader("Live Segment Scorer")
st.caption("Enter a hypothetical customer's details to predict which segment they'd fall into.")
with st.form("scorer_form"):
    sc1, sc2, sc3, sc4 = st.columns([1, 1, 1, 0.6])
    in_age = sc1.number_input("Age", min_value=18, max_value=100, value=30)
    in_income = sc2.number_input("Annual Income (k$)", min_value=0, max_value=200, value=60)
    in_spending = sc3.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
    with sc4:
        st.write("")
        st.write("")
        submitted = st.form_submit_button("Predict Segment")

if submitted:
    new_row = pd.DataFrame(
        [
            {
                "Age": in_age,
                "Annual Income (k$)": in_income,
                "Spending Score (1-100)": in_spending,
            }
        ]
    )
    X_new_scaled = pd.DataFrame(
        scaler.transform(new_row[cluster_features]),
        columns=cluster_features,
        index=new_row.index,
    )
    predicted_cluster = int(kmeans.predict(X_new_scaled)[0])
    predicted_name = FRIENDLY_SEGMENT_NAMES[predicted_cluster]
    st.success(f"Predicted segment: **{predicted_name}** (Cluster {predicted_cluster})")

st.divider()

# ---------------------------------------------------------------------------
# Drill-down customer table
# ---------------------------------------------------------------------------
st.subheader("Customer Detail")
st.dataframe(
    filtered_df[
        [
            "CustomerID",
            "Gender",
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)",
            "Spending_Income_Ratio",
            "Friendly_Segment",
        ]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "CustomerID": st.column_config.NumberColumn("Customer ID"),
        "Annual Income (k$)": st.column_config.NumberColumn("Income (k$)", format="$%.0f k"),
        "Spending Score (1-100)": st.column_config.NumberColumn("Spending Score"),
        "Spending_Income_Ratio": st.column_config.NumberColumn("Spend/Income Ratio", format="%.2f"),
        "Friendly_Segment": st.column_config.TextColumn("Segment"),
    },
)
