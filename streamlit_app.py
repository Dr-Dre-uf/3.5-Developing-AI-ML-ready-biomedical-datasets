# --------------------------------------------------------------
# streamlit_app.py – Genomic preprocessing explorer (no sklearn)
# --------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------
st.set_page_config(page_title="Genomic Pre‑processing Explorer", layout="wide")

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

lower_pct = st.sidebar.slider(
    "Lower Winsorization Percentile", min_value=0, max_value=20, value=5, step=1
)
upper_pct = st.sidebar.slider(
    "Upper Winsorization Percentile", min_value=80, max_value=100, value=95, step=1
)

impute_strategy = st.sidebar.selectbox(
    "Missing‑Value Imputation", ["mean", "median", "most_frequent"]
)

scale_option = st.sidebar.radio(
    "Feature Scaling", ("Standard (Z‑score)", "Min‑Max", "None")
)

# --------------------------------------------------------------
# 3️⃣  Helper functions (pure pandas / NumPy)
# --------------------------------------------------------------
def cap_outliers(series: pd.Series, lower: int, upper: int) -> pd.Series:
    lo, hi = np.percentile(series, [lower, upper])
    return np.clip(series, lo, hi)


def manual_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z‑score computed with pandas (no scipy)."""
    return (df - df.mean()) / df.std(ddof=0)


def impute(series: pd.Series, strategy: str) -> pd.Series:
    if strategy == "mean":
        fill = series.mean()
    elif strategy == "median":
        fill = series.median()
    else:  # most_frequent
        fill = series.mode().iloc[0]
    return series.fillna(fill)


def scale(series: pd.Series, method: str) -> pd.Series:
    if method == "Standard (Z‑score)":
        return (series - series.mean()) / series.std(ddof=0)
    elif method == "Min‑Max":
        return (series - series.min()) / (series.max() - series.min())
    else:  # None
        return series


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- 1️⃣  Winsorize outliers ---------------------------------
    for col in ["Expression_Level", "Mutation_Frequency", "Pathway_Score"]:
        df[col] = cap_outliers(df[col], lower_pct, upper_pct)

    # ---- 2️⃣  Impute missing values -------------------------------
    df["Missing_Feature"] = impute(df["Missing_Feature"], impute_strategy)

    # ---- 3️⃣  Scale numeric features -------------------------------
    for col in ["Expression_Level", "Mutation_Frequency", "Pathway_Score"]:
        df[col] = scale(df[col], scale_option)

    # ---- 4️⃣  Normalise the imputed feature (as in original notebook) --
    df["Missing_Feature"] = scale(df["Missing_Feature"], "Min‑Max")

    return df


df_processed = preprocess(df_raw)

# --------------------------------------------------------------
# 4️⃣  Layout – tables & visualisations
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
z_raw = manual_zscore(
    df_raw[["Expression_Level", "Mutation_Frequency", "Pathway_Score"]]
)
out_counts = ((z_raw > 3) | (z_raw < -3)).sum()
st.write(out_counts)

# ---- Box‑plot comparison ---------------------------------------
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
    "Adjust the sidebar controls to see how Winsorization, imputation, and scaling reshape the dataset."
)