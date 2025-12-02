# --------------------------------------------------------------
# app.py – Interactive genomic‑preprocessing explorer (no icons)
# --------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --------------------------------------------------------------
# Page configuration (icon removed)
# --------------------------------------------------------------
st.set_page_config(
    page_title="Genomic Pre‑processing Explorer",
    layout="wide",
)

# --------------------------------------------------------------
# 1️⃣  Simulated genomic dataset (identical to the notebook)
# --------------------------------------------------------------
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
            np.nan if i % 10 == 0 else np.random.normal(5, 1)
            for i in range(100)
        ],
    }
    return pd.DataFrame(data)

df_raw = load_data()

# --------------------------------------------------------------
# 2️⃣  Sidebar – user‑controlled preprocessing knobs
# --------------------------------------------------------------
st.sidebar.header("Pre‑processing Settings")

# Winsorization percentiles
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

# Feature scaling choice
scale_option = st.sidebar.radio(
    "Feature Scaling", ("Standard (Z‑score)", "Min‑Max", "None")
)

# --------------------------------------------------------------
# 3️⃣  Helper functions
# --------------------------------------------------------------
def cap_outliers(series: pd.Series, lower: int, upper: int) -> pd.Series:
    lo, hi = np.percentile(series, [lower, upper])
    return np.clip(series, lo, hi)

def zscore_manual(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Z‑scores with NumPy/Pandas (no scipy)."""
    return (df - df.mean()) / df.std(ddof=0)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- Outlier handling (Winsorization) -----------------------
    for col in ["Expression_Level", "Mutation_Frequency", "Pathway_Score"]:
        df[col] = cap_outliers(df[col], lower_pct, upper_pct)

    # ---- Missing‑value imputation --------------------------------
    imp = SimpleImputer(strategy=impute_strategy)
    df["Missing_Feature"] = imp.fit_transform(df[["Missing_Feature"]])

    # ---- Scaling -------------------------------------------------
    numeric_cols = ["Expression_Level", "Mutation_Frequency", "Pathway_Score"]
    if scale_option == "Standard (Z‑score)":
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif scale_option == "Min‑Max":
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    # else: keep original values

    # Normalise the imputed feature (as in the original notebook)
    df[["Missing_Feature"]] = MinMaxScaler().fit_transform(
        df[["Missing_Feature"]]
    )

    return df

df_processed = preprocess(df_raw)

# --------------------------------------------------------------
# 4️⃣  Layout – data tables & visualisations
# --------------------------------------------------------------
st.title("Genomic Data Pre‑processing Explorer")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Raw data (first 5 rows)")
    st.dataframe(df_raw.head())
with c2:
    st.subheader("Processed data (first 5 rows)")
    st.dataframe(df_processed.head())

st.markdown("---")

# ---- Outlier detection on raw data (manual Z‑score) ----------
st.subheader("Outlier detection (Z‑score) on raw data")
z_raw = zscore_manual(
    df_raw[["Expression_Level", "Mutation_Frequency", "Pathway_Score"]]
)
out_counts = ((z_raw > 3) | (z_raw < -3)).sum()
st.write(out_counts)

# ---- Box‑plot comparison ----------------------------------------
st.subheader("Box‑plot: Raw vs. Processed")
fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

sns.boxplot(
    data=df_raw[["Expression_Level", "Mutation_Frequency", "Pathway_Score"]],
    ax=axs[0],
    palette="pastel",
)
axs[0].set_title("Raw")

sns.boxplot(
    data=df_processed[
        ["Expression_Level", "Mutation_Frequency", "Pathway_Score"]
    ],
    ax=axs[1],
    palette="muted",
)
axs[1].set_title("Processed")

st.pyplot(fig)

st.caption(
    "Use the sidebar to adjust Winsorization percentiles, imputation method, and scaling. "
    "Tables and plots update instantly."
)