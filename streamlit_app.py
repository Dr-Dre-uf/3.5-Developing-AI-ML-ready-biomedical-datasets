# --------------------------------------------------------------
# app.py – Interactive genomic‑preprocessing explorer
# --------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Genomic Pre‑processing Explorer",
    layout="wide",
)

# ------------------------------------------------------------------
# 1️⃣  Simulated genomic dataset (identical to the notebook)
# ------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    np.random.seed(42)
    data = {
        "Gene_ID": np.arange(1, 101),
        "Expression_Level": np.append(
            np.random.normal(10, 2, 95), [50, 52, 55, 60, 65]
        ),  # Outliers
        "Mutation_Frequency": np.append(
            np.random.normal(0.05, 0.01, 95), [0.2, 0.22, 0.25, 0.3, 0.35]
        ),  # Outliers
        "Pathway_Score": np.append(
            np.random.normal(5, 1, 95), [15, 16, 18, 20, 22]
        ),  # Outliers
        "Missing_Feature": [
            np.nan if i % 10 == 0 else np.random.normal(5, 1) for i in range(100)
        ],
    }
    return pd.DataFrame(data)


df_raw = load_data()

# ------------------------------------------------------------------
# 2️⃣  Sidebar – user‑controlled preprocessing knobs
# ------------------------------------------------------------------
st.sidebar.header("Pre‑processing Settings")

# Winsorization (capping) percentiles
lower_pct = st.sidebar.slider(
    "Lower Winsorization Percentile", min_value=0, max_value=20, value=5, step=1
)
upper_pct = st.sidebar.slider(
    "Upper Winsorization Percentile", min_value=80, max_value=100, value=95, step=1
)

# Missing‑value imputation strategy
impute_strategy = st.sidebar.selectbox(
    "Missing‑Value Imputation", ["mean", "median", "most_frequent"]
)

# Scaling option for the numeric features
scale_option = st.sidebar.radio(
    "Feature Scaling",
    ("Standard (Z‑score)", "Min‑Max", "None"),
)

# ------------------------------------------------------------------
# 3️⃣  Helper functions
# ------------------------------------------------------------------
def cap_outliers(series: pd.Series, lower: int, upper: int) -> pd.Series:
    """Winsorize a series using the given percentile bounds."""
    lo, hi = np.percentile(series, [lower, upper])
    return np.clip(series, lo, hi)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply outlier capping, imputation, and scaling."""
    df = df.copy()

    # ---- 1️⃣  Outlier handling ------------------------------------
    for col in ["Expression_Level", "Mutation_Frequency", "Pathway_Score"]:
        df[col] = cap_outliers(df[col], lower_pct, upper_pct)

    # ---- 2️⃣  Missing‑value imputation -----------------------------
    imp = SimpleImputer(strategy=impute_strategy)
    df["Missing_Feature"] = imp.fit_transform(df[["Missing_Feature"]])

    # ---- 3️⃣  Scaling ------------------------------------------------
    numeric_cols = ["Expression_Level", "Mutation_Frequency", "Pathway_Score"]
    if scale_option == "Standard (Z‑score)":
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif scale_option == "Min‑Max":
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    # else: leave as‑is (None)

    # Normalise the imputed feature (mirrors the original notebook)
    df[["Missing_Feature"]] = MinMaxScaler().fit_transform(
        df[["Missing_Feature"]]
    )

    return df


df_processed = preprocess(df_raw)

# ------------------------------------------------------------------
# 4️⃣  Main layout – show data, stats, and visualisations
# ------------------------------------------------------------------
st.title("Genomic Data Pre‑processing Explorer")

# ---- Side‑by‑side data snapshots ------------------------------------
col_raw, col_proc = st.columns(2)

with col_raw:
    st.subheader("Raw Data (first 5 rows)")
    st.dataframe(df_raw.head())

with col_proc:
    st.subheader("Processed Data (first 5 rows)")
    st.dataframe(df_processed.head())

st.markdown("---")

# ---- Outlier detection on the raw data -------------------------------
st.subheader("Outlier Detection (Z‑score) – Raw Data")
z_scores_raw = df_raw[
    ["Expression_Level", "Mutation_Frequency", "Pathway_Score"]
].apply(zscore)

outlier_counts = ((z_scores_raw > 3) | (z_scores_raw < -3)).sum()
st.write(outlier_counts)

# ---- Box‑plot comparison ---------------------------------------------
st.subheader("Box‑Plot: Raw vs. Processed Features")
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

sns.boxplot(
    data=df_raw[["Expression_Level", "Mutation_Frequency", "Pathway_Score"]],
    ax=axes[0],
    palette="pastel",
)
axes[0].set_title("Raw")

sns.boxplot(
    data=df_processed[
        ["Expression_Level", "Mutation_Frequency", "Pathway_Score"]
    ],
    ax=axes[1],
    palette="muted",
)
axes[1].set_title("Processed")

st.pyplot(fig)

# ------------------------------------------------------------------
# 5️⃣  Footer
# ------------------------------------------------------------------
st.caption(
    "Adjust the sliders, selectbox, and radio buttons in the sidebar to see "
    "how each preprocessing choice reshapes the dataset in real‑time."
)