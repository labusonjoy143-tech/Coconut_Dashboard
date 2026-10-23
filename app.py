# app.py
# Coconut Production Analytics Dashboard
# Run locally: streamlit run app.py
# Streamlit Community Cloud: keep app.py and requirements.txt in the repository root.
# Users upload their own CSV; no dataset needs to be stored in GitHub.

import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyRegressor

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

st.set_page_config(
    page_title="Coconut Production Analytics",
    page_icon="🥥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.7rem; padding-bottom: 2rem;}
.dashboard-title {font-size: 2.15rem; font-weight: 750; letter-spacing: -0.04em;}
.muted {color: #737783;}
div[data-testid="stMetric"] {
    background: #f7f8fb; border: 1px solid #e7e9ef;
    padding: 13px 15px; border-radius: 12px;
}
section[data-testid="stSidebar"] {border-right: 1px solid #e7e9ef;}
</style>
""", unsafe_allow_html=True)

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
MONTH_MAP = {month.lower(): i for i, month in enumerate(MONTHS, start=1)}

NUMERIC_HINTS = [
    "year", "harvest", "production", "yield", "area", "price",
    "rainfall", "temperature", "inflation", "supply", "volume",
    "quantity", "amount", "ton", "kg", "hectare", "humidity"
]


def load_csv(uploaded_file):
    """Read CSV files with common encodings, including Windows-encoded files."""
    encodings = ["utf-8-sig", "cp1252", "latin1"]
    last_error = None
    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Unable to decode this CSV file. Details: {last_error}")


def clean_dataset(raw, trim_text=True, remove_duplicates=True):
    data = raw.copy()
    data.columns = [str(c).strip() for c in data.columns]

    # Ensure column names are unique.
    seen, new_columns = {}, []
    for col in data.columns:
        seen[col] = seen.get(col, 0) + 1
        new_columns.append(col if seen[col] == 1 else f"{col}_{seen[col]}")
    data.columns = new_columns

    data = data.dropna(how="all").dropna(axis=1, how="all")

    if trim_text:
        for col in data.select_dtypes(include=["object", "string"]).columns:
            data[col] = data[col].astype("string").str.strip()
            data[col] = data[col].replace({
                "": pd.NA, "NA": pd.NA, "N/A": pd.NA,
                "null": pd.NA, "None": pd.NA, "nan": pd.NA
            })

    for col in data.columns:
        name = str(col).lower()
        if any(hint in name for hint in NUMERIC_HINTS):
            original_nonmissing = int(data[col].notna().sum())
            converted = pd.to_numeric(
                data[col].astype("string")
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace("₱", "", regex=False)
                .str.strip(),
                errors="coerce"
            )
            if original_nonmissing > 0 and converted.notna().sum() >= max(
                1, int(original_nonmissing * 0.5)
            ):
                data[col] = converted

    duplicate_count = int(data.duplicated().sum())
    if remove_duplicates:
        data = data.drop_duplicates().copy()

    # Treat negative values in nonnegative measures as invalid/missing.
    nonnegative_tokens = [
        "harvest", "production", "yield", "area", "rainfall",
        "price", "supply", "volume", "quantity"
    ]
    for col in data.columns:
        name = str(col).lower()
        if (pd.api.types.is_numeric_dtype(data[col])
                and any(token in name for token in nonnegative_tokens)):
            data.loc[data[col] < 0, col] = np.nan

    return data, duplicate_count


def get_numeric_columns(data):
    return data.select_dtypes(include=np.number).columns.tolist()


def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    transformers = []

    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]),
            numeric
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical
        ))
    if not transformers:
        raise ValueError("No usable predictor columns were found.")
    return ColumnTransformer(transformers, remainder="drop")


def plot_histogram(series, title, xlabel):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        st.info("There are no valid numeric values to plot.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=25, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Records")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_correlation(data, columns):
    corr = data[columns].corr(numeric_only=True)
    if corr.empty:
        st.info("Not enough numeric data to calculate correlations.")
        return
    fig, ax = plt.subplots(figsize=(max(7, len(columns) * 0.8), 5.5))
    image = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=65, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.set_title("Correlation between numeric variables")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def quality_report(data):
    return pd.DataFrame({
        "Column": data.columns,
        "Data type": [str(data[c].dtype) for c in data.columns],
        "Missing values": [int(data[c].isna().sum()) for c in data.columns],
        "Missing (%)": [(data[c].isna().mean() * 100).round(2) for c in data.columns],
        "Unique values": [int(data[c].nunique(dropna=True)) for c in data.columns],
    })


# ---------- Sidebar ----------
st.sidebar.markdown("## 🥥 Coconut Analytics")
st.sidebar.caption("Production • Statistics • Prediction")
uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"],
    help="Upload a CSV or Excel file from your device. The dataset does not need to be in GitHub."
)

with st.sidebar.expander("Cleaning settings", expanded=False):
    trim_text = st.checkbox("Trim text and standardize blanks", value=True)
    remove_duplicates = st.checkbox("Remove exact duplicate rows", value=True)
    preview_rows = st.selectbox("Preview rows", [5, 10, 20, 50], index=1)

page = st.sidebar.radio(
    "Dashboard navigation",
    [
        "Overview",
        "Data Cleaning",
        "Statistics & Exploration",
        "Classification & Prediction",
        "Harvest Forecasting",
        "Downloads",
        "Coconut Supply & Inflation",
    ],
)

st.markdown(
    '<div class="dashboard-title">Coconut Production Analytics</div>',
    unsafe_allow_html=True
)
st.caption(
    "Analyze coconut production records, explore trends, evaluate prediction models, "
    "and forecast future harvest."
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel dataset using the left sidebar to begin. No dataset is loaded from GitHub.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. Upload")
        st.write("Choose a dataset from your computer.")
    with col2:
        st.markdown("#### 2. Explore")
        st.write("Review missing values, statistics, and production trends.")
    with col3:
        st.markdown("#### 3. Analyze")
        st.write("Run prediction models, forecasting, and export results.")
    st.markdown(
        "**Recommended fields (your column names may differ):** Year, Month, "
        "Barangay, Harvest (kg), Harvested Area (ha), Rainfall, Temperature, and prices."
    )
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        raw_data = load_csv(uploaded_file)
    else:
        raw_data = pd.read_excel(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded dataset: {exc}")
    st.stop()

if raw_data.empty:
    st.error("The uploaded file contains column headers but no data rows.")
    st.stop()

df, duplicates_found = clean_dataset(raw_data, trim_text, remove_duplicates)
if df.empty:
    st.error("No records remain after cleaning. Review the uploaded file and cleaning settings.")
    st.stop()

for key in [
    "forecast_download", "prediction_download", "classification_prediction_download"
]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------- Overview ----------
if page == "Overview":
    st.markdown("### Dataset overview")
    a, b, c, d = st.columns(4)
    a.metric("Records after cleaning", f"{len(df):,}")
    b.metric("Columns", f"{df.shape[1]:,}")
    c.metric("Duplicates found", f"{duplicates_found:,}")
    d.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    st.markdown("### Uploaded data preview")
    st.dataframe(df.head(preview_rows), width="stretch")

    st.markdown("### Data quality report")
    st.dataframe(quality_report(df), width="stretch", hide_index=True)
    st.caption(
        "Exact duplicates are removed only if enabled. Negative values in production-like "
        "measures are treated as missing, not as valid observations."
    )

# ---------- Data Cleaning ----------
elif page == "Data Cleaning":
    st.markdown("### Data cleaning")
    st.write("Review missing values, inspect duplicates, choose an imputation method, and download a cleaned copy.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Exact duplicates found", f"{duplicates_found:,}")
    c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    st.markdown("#### Missing-value report")
    st.dataframe(
        quality_report(df).sort_values("Missing (%)", ascending=False),
        width="stretch", hide_index=True
    )
    fill_method = st.selectbox(
        "How should remaining missing values be handled?",
        [
            "Leave missing values as-is",
            "Numeric: median; text: most frequent",
            "Numeric: mean; text: most frequent",
            "Drop rows with any missing values"
        ],
        key="clean_fill_method"
    )
    cleaned_output = df.copy()
    if fill_method in (
        "Numeric: median; text: most frequent",
        "Numeric: mean; text: most frequent"
    ):
        for col in cleaned_output.columns:
            if "inflation" in col.lower():
                continue
            if pd.api.types.is_numeric_dtype(cleaned_output[col]):
                if cleaned_output[col].notna().any():
                    value = (
                        cleaned_output[col].median()
                        if fill_method.startswith("Numeric: median")
                        else cleaned_output[col].mean()
                    )
                    cleaned_output[col] = cleaned_output[col].fillna(value)
            elif cleaned_output[col].notna().any():
                mode = cleaned_output[col].mode(dropna=True)
                if not mode.empty:
                    cleaned_output[col] = cleaned_output[col].fillna(mode.iloc[0])
    elif fill_method == "Drop rows with any missing values":
        cleaned_output = cleaned_output.dropna().copy()

    st.markdown("#### Cleaned data preview")
    st.write(f"Rows in downloadable cleaned output: **{len(cleaned_output):,}**")
    st.dataframe(cleaned_output.head(preview_rows), width="stretch")
    st.download_button(
        "Download cleaned dataset (CSV)",
        data=cleaned_output.to_csv(index=False).encode("utf-8-sig"),
        file_name="cleaned_coconut_data.csv",
        mime="text/csv",
        width="stretch"
    )

# ---------- Statistics / EDA ----------
elif page == "Statistics & Exploration":
    numeric_cols = get_numeric_columns(df)
    if numeric_cols:
        st.markdown("### Descriptive statistics")
        st.dataframe(df[numeric_cols].describe().T.round(3), width="stretch")

        st.markdown("### Correlation analysis")
        if len(numeric_cols) >= 2:
            plot_correlation(df, numeric_cols)
        else:
            st.info("At least two numeric columns are needed for correlation analysis.")

        selected_num = st.selectbox("Select a numeric field to explore", numeric_cols)
        plot_histogram(df[selected_num], f"Distribution of {selected_num}", selected_num)
    else:
        st.warning("No numeric columns were detected. Check whether numeric fields contain text or symbols.")

    possible_categories = [
        col for col in df.columns
        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_categorical_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
        )
        and 1 < df[col].nunique(dropna=True) <= 100
    ]
    if possible_categories:
        st.markdown("### Category breakdown")
        category = st.selectbox("Choose a category field", possible_categories)
        counts = df[category].fillna("Missing").astype(str).value_counts().head(20)
        st.dataframe(counts.rename("Records").to_frame(), width="stretch")
        fig, ax = plt.subplots(figsize=(9, 4))
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Records by {category}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    barangay_col = next(
        (c for c in df.columns if c.lower().strip() in ("barangay", "village", "location")),
        None
    )
    harvest_col = next(
        (c for c in df.columns if "harvest" in c.lower() and pd.api.types.is_numeric_dtype(df[c])),
        None
    )
    if barangay_col and harvest_col:
        st.markdown("### Harvest by location")
        grouped = df.groupby(barangay_col, dropna=False)[harvest_col].sum(min_count=1).sort_values(ascending=False)
        st.dataframe(grouped.rename(f"Total {harvest_col}").to_frame(), width="stretch")
        fig, ax = plt.subplots(figsize=(10, 4))
        grouped.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel(harvest_col)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ---------- Classification / Prediction ----------
elif page == "Coconut Supply & Inflation":
    st.markdown("### Audited annual inflation forecasting")
    st.info("Upload rosario_annual_combined.csv on the sidebar for validated annual modeling. Same-year production and weather are not used to forecast that year's inflation.")
    required = ["Year", "Inflation Rate (%)", "Annual Harvest (kg)", "Harvested Area (ha)", "Annual Rainfall (mm)"]
    if not all(c in df.columns for c in required):
        st.warning("This page requires the audited annual combined CSV; monthly records must not be randomly split for annual inflation prediction.")
    else:
        from sklearn.base import clone
        from statsmodels.tsa.arima.model import ARIMA
        annual = df.sort_values("Year").drop_duplicates("Year").copy()
        features = [c for c in annual.columns if c.startswith("Mean ") or c in required[2:]]
        modeling = annual[["Year", "Inflation Rate (%)"]].copy()
        for c in features:
            modeling["Previous year " + c] = pd.to_numeric(annual[c], errors="coerce").shift(1)
        modeling["Previous year inflation"] = pd.to_numeric(annual["Inflation Rate (%)"], errors="coerce").shift(1)
        modeling = modeling.dropna(subset=["Inflation Rate (%)", "Previous year inflation"]).reset_index(drop=True)
        st.caption("One-step-ahead expanding-window predictions; predictors lagged one year; missing predictors imputed inside each training fold. Conflicting inflation targets are excluded.")
        if len(modeling) < 13:
            st.warning("Not enough consecutive, resolved annual inflation observations for evaluation.")
        else:
            X = modeling.drop(columns=["Year", "Inflation Rate (%)"])
            y = modeling["Inflation Rate (%)"].to_numpy()
            years = modeling["Year"].to_numpy()
            test_indices = [i for i, yr in enumerate(years) if yr >= 2020 and i >= 10]
            models = {
                "Multiple Linear Regression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LinearRegression())]),
                "SVR": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", SVR(C=1.0, epsilon=0.1))]),
                "ANN": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", MLPRegressor(hidden_layer_sizes=(8,), max_iter=2000, random_state=42))]),
            }
            scores, preds = [], []
            for name in ["Training mean baseline", "Last observation baseline", *models, "ARIMA(1,0,0)"]:
                actual, estimated = [], []
                for i in test_indices:
                    yt = y[:i]
                    try:
                        if name == "Training mean baseline": value = float(yt.mean())
                        elif name == "Last observation baseline": value = float(yt[-1])
                        elif name.startswith("ARIMA"): value = float(ARIMA(yt, order=(1,0,0)).fit().forecast(1)[0])
                        else:
                            m = clone(models[name]); m.fit(X.iloc[:i], yt)
                            value = float(m.predict(X.iloc[[i]])[0])
                        actual.append(y[i]); estimated.append(value)
                        preds.append({"Model": name, "Year": int(years[i]), "Actual": float(y[i]), "Predicted": value})
                    except Exception as exc:
                        st.warning(f"{name}, {years[i]}: {exc}")
                if len(actual) == len(test_indices) and len(actual) >= 2:
                    mse = mean_squared_error(actual, estimated)
                    scores.append({"Model": name, "N": len(actual), "MSE": mse, "RMSE": mse ** 0.5, "MAE": mean_absolute_error(actual, estimated), "R²": r2_score(actual, estimated)})
            if scores:
                st.dataframe(pd.DataFrame(scores).round(4), hide_index=True, width="stretch")
                st.dataframe(pd.DataFrame(preds), hide_index=True, width="stretch")
                st.download_button("Download model scores", pd.DataFrame(scores).to_csv(index=False), "annual_model_scores.csv")
                st.download_button("Download predictions", pd.DataFrame(preds).to_csv(index=False), "annual_model_predictions.csv")
                st.caption("These are expanding-window test scores, not in-sample fit. R² can be negative. 2025 is not scored while its inflation target remains unresolved.")

elif page == "Classification & Prediction":
    numeric_cols = get_numeric_columns(df)
    suggested_targets = [
        c for c in numeric_cols
        if any(term in c.lower() for term in ("harvest", "production", "yield", "supply", "quantity", "volume"))
    ]
    if not suggested_targets:
        st.warning("No likely numeric production target was detected. Ensure your CSV has a numeric harvest, production, yield, supply, or quantity column.")
    else:
        target = st.selectbox("Choose the value to predict", suggested_targets)
        st.caption(
            "Models use the uploaded dataset only. Results are estimates and do not prove "
            "that a variable causes a change in production. Avoid predictors that directly "
            "contain or were calculated from the target."
        )
        model_data = df[df[target].notna()].copy()
        feature_cols = []
        for col in model_data.columns:
            if col == target:
                continue
            if model_data[col].isna().all() or model_data[col].nunique(dropna=True) <= 1:
                continue
            if (
                (col.lower().strip() == "id" or col.lower().strip().endswith("_id"))
                and model_data[col].nunique() > 0.8 * len(model_data)
            ):
                continue
            feature_cols.append(col)

        if len(model_data) < 10:
            st.warning("At least 10 valid records are recommended for a basic train/test evaluation.")
        elif not feature_cols:
            st.warning("No usable predictor columns were found apart from the selected target.")
        else:
            X = model_data[feature_cols].copy()
            y = pd.to_numeric(model_data[target], errors="coerce")
            valid = y.notna()
            X, y = X.loc[valid], y.loc[valid]
            if len(X) < 10:
                st.warning("There are fewer than 10 valid target records after conversion.")
            else:
                st.markdown("### Regression: predict the selected quantity")
                test_fraction = st.slider("Test dataset proportion", 0.15, 0.40, 0.20, 0.05)
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_fraction, random_state=42
                    )
                    reg = Pipeline([
                        ("preprocess", make_preprocessor(X_train)),
                        ("model", MLPRegressor(
                            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                            alpha=0.001, learning_rate_init=0.001, max_iter=1000,
                            early_stopping=False, random_state=42
                        )),
                    ])
                    reg.fit(X_train, y_train)
                    prediction = reg.predict(X_test)
                    mae = mean_absolute_error(y_test, prediction)
                    rmse = mean_squared_error(y_test, prediction) ** 0.5
                    r2 = r2_score(y_test, prediction) if len(y_test) > 1 else np.nan
                    m1, m2, m3 = st.columns(3)
                    m1.metric("MAE", f"{mae:,.3f}")
                    m2.metric("RMSE", f"{rmse:,.3f}")
                    m3.metric("R²", f"{r2:.3f}" if np.isfinite(r2) else "N/A")

                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(y_test, prediction, alpha=0.7)
                    low = min(float(y_test.min()), float(np.min(prediction)))
                    high = max(float(y_test.max()), float(np.max(prediction)))
                    ax.plot([low, high], [low, high], linestyle="--")
                    ax.set_xlabel(f"Actual {target}")
                    ax.set_ylabel(f"Predicted {target}")
                    ax.set_title("Actual vs predicted")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    baseline = DummyRegressor(strategy="mean")
                    baseline.fit(X_train, y_train)
                    baseline_prediction = baseline.predict(X_test)
                    st.markdown("### Model comparison")
                    comparison = pd.DataFrame([
                        {"Model": "Deep Learning (ANN)", "MAE": mae, "RMSE": rmse, "R²": r2},
                        {
                            "Model": "Mean baseline",
                            "MAE": mean_absolute_error(y_test, baseline_prediction),
                            "RMSE": mean_squared_error(y_test, baseline_prediction) ** 0.5,
                            "R²": r2_score(y_test, baseline_prediction) if len(y_test) > 1 else np.nan,
                        }
                    ])
                    st.dataframe(comparison.round(4), width="stretch", hide_index=True)

                    st.markdown("### Predictor importance")
                    with st.spinner("Calculating permutation importance..."):
                        importance_result = permutation_importance(
                            reg, X_test, y_test, n_repeats=2, random_state=42,
                            scoring="neg_mean_absolute_error", n_jobs=1
                        )
                    importance = pd.DataFrame({
                        "Predictor": X_test.columns,
                        "Importance": importance_result.importances_mean
                    }).sort_values("Importance", ascending=False)
                    st.dataframe(importance, width="stretch", hide_index=True)
                    fig, ax = plt.subplots(figsize=(9, 4))
                    importance.sort_values("Importance").plot(
                        kind="barh", x="Predictor", y="Importance", legend=False, ax=ax
                    )
                    ax.set_title("Permutation importance (higher means more useful to this model)")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.session_state["prediction_download"] = pd.DataFrame({
                        "Actual": y_test.to_numpy(),
                        "Predicted": prediction
                    })
                except Exception as exc:
                    st.error(f"Regression could not run with the selected columns: {exc}")

                st.markdown("### Classification and combination prediction")
                st.write(
                    "Choose an outcome variable and input variables. The model classifies "
                    "the outcome and can estimate a class for a new combination."
                )
                categorical_targets = [
                    c for c in model_data.columns
                    if c != target
                    and 2 <= model_data[c].nunique(dropna=True) <= 30
                    and (
                        pd.api.types.is_object_dtype(model_data[c])
                        or pd.api.types.is_string_dtype(model_data[c])
                        or pd.api.types.is_categorical_dtype(model_data[c])
                        or pd.api.types.is_bool_dtype(model_data[c])
                    )
                ]
                numeric_targets = [
                    c for c in get_numeric_columns(model_data)
                    if c != target and model_data[c].nunique(dropna=True) >= 3
                ]
                target_options = categorical_targets + [
                    c for c in numeric_targets if c not in categorical_targets
                ]
                if not target_options:
                    st.info("No suitable classification outcome was found. Choose a categorical field or a numeric field that can be grouped into Low / Medium / High.")
                else:
                    class_target = st.selectbox(
                        "Classification outcome", target_options, key="classification_target"
                    )
                    raw_y = model_data[class_target]
                    if pd.api.types.is_numeric_dtype(raw_y):
                        numeric_mode = st.radio(
                            "Numeric outcome handling",
                            ["Group into Low / Medium / High", "Use numeric values as classes"],
                            horizontal=True, key="numeric_class_mode"
                        )
                        if numeric_mode == "Group into Low / Medium / High":
                            numeric_y = pd.to_numeric(raw_y, errors="coerce")
                            try:
                                y_class = pd.qcut(
                                    numeric_y, q=3,
                                    labels=["Low", "Medium", "High"],
                                    duplicates="drop"
                                ).astype("string")
                            except ValueError:
                                y_class = pd.Series(pd.NA, index=raw_y.index, dtype="string")
                        else:
                            y_class = raw_y.astype("string")
                    else:
                        y_class = raw_y.astype("string")

                    # Prevent target leakage: neither the selected classification outcome
                    # nor the primary production target can be used as a predictor.
                    feature_options = [
                        c for c in model_data.columns
                        if c not in (class_target, target)
                        and model_data[c].nunique(dropna=True) > 1
                    ]
                    feature_options = [
                        c for c in feature_options
                        if not (
                            (c.lower().strip() == "id" or c.lower().strip().endswith("_id"))
                            and model_data[c].nunique(dropna=True) > 0.8 * len(model_data)
                        )
                    ]
                    selected_features = st.multiselect(
                        "Input variables (choose a combination)",
                        feature_options,
                        default=feature_options[:min(5, len(feature_options))],
                        help="Choose variables available before the outcome occurs. Avoid fields derived from the outcome.",
                        key="classification_features"
                    )
                    if not selected_features:
                        st.warning("Select at least one input variable.")
                    else:
                        X_class = model_data[selected_features].copy()
                        y_class = pd.Series(y_class, index=model_data.index)
                        valid = y_class.notna()
                        X_class, y_class = X_class.loc[valid], y_class.loc[valid].astype(str)
                        counts = y_class.value_counts()
                        keep = counts[counts >= 2].index
                        mask = y_class.isin(keep)
                        X_class, y_class = X_class.loc[mask], y_class.loc[mask]

                        if len(y_class) < 6 or y_class.nunique() < 2:
                            st.warning("There are not enough examples in multiple classes. Choose another outcome or upload more records.")
                        else:
                            test_size_cls = st.slider(
                                "Classification test proportion", .15, .40, .20, .05,
                                key="classification_test_fraction"
                            )
                            stratify = y_class if y_class.value_counts().min() >= 2 else None
                            try:
                                Xtr, Xte, ytr, yte = train_test_split(
                                    X_class, y_class, test_size=test_size_cls,
                                    random_state=42, stratify=stratify
                                )
                                clf = Pipeline([
                                    ("preprocess", make_preprocessor(Xtr)),
                                    ("model", RandomForestClassifier(
                                        n_estimators=250, random_state=42,
                                        class_weight="balanced", min_samples_leaf=1, n_jobs=-1
                                    ))
                                ])
                                clf.fit(Xtr, ytr)
                                yhat = clf.predict(Xte)
                                score1, score2 = st.columns(2)
                                score1.metric("Accuracy", f"{accuracy_score(yte, yhat):.1%}")
                                score2.metric("Balanced accuracy", f"{balanced_accuracy_score(yte, yhat):.1%}")
                                report = pd.DataFrame(
                                    classification_report(
                                        yte, yhat, output_dict=True, zero_division=0
                                    )
                                ).T
                                report.index.name = "Class"
                                st.markdown("#### Classification report")
                                st.dataframe(report.round(4), width="stretch")
                                with st.expander("What do the report columns mean?"):
                                    st.write("**Precision:** how often predictions for a class are correct. **Recall:** how many actual examples of a class were found. **F1-score:** balance between precision and recall. **Support:** number of true test examples in that class.")
                                    st.write("**Macro avg** treats classes equally; **weighted avg** accounts for each class's number of records. Rare classes may score poorly if there are too few examples.")
                                labels = list(clf.named_steps["model"].classes_)
                                cm = confusion_matrix(yte, yhat, labels=labels)
                                fig, ax = plt.subplots(figsize=(max(5, len(labels) * .9), 4.5))
                                im = ax.imshow(cm, aspect="auto")
                                ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
                                ax.set_yticks(range(len(labels)), labels)
                                ax.set_xlabel("Predicted class")
                                ax.set_ylabel("Actual class")
                                ax.set_title("Confusion matrix")
                                for i in range(cm.shape[0]):
                                    for j in range(cm.shape[1]):
                                        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
                                fig.colorbar(im, ax=ax, label="Records")
                                fig.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)

                                try:
                                    perm = permutation_importance(
                                        clf, Xte, yte, n_repeats=5,
                                        random_state=42, scoring="f1_weighted", n_jobs=-1
                                    )
                                    importance = pd.DataFrame({
                                        "Variable": selected_features,
                                        "Importance": perm.importances_mean,
                                        "Description": [
                                            f"Contribution of {f} to weighted F1 when shuffled; this is model usefulness, not causation."
                                            for f in selected_features
                                        ]
                                    }).sort_values("Importance", ascending=False)
                                    st.markdown("#### Variable importance")
                                    try:
                                        import plotly.express as px
                                        fig = px.bar(
                                            importance, x="Importance", y="Variable",
                                            orientation="h", hover_data={"Description": True, "Importance": ":.4f"},
                                            title="Predictor importance — hover over a bar for its explanation"
                                        )
                                        st.plotly_chart(fig, width="stretch")
                                    except ImportError:
                                        st.bar_chart(
                                            importance.set_index("Variable")["Importance"],
                                            horizontal=True
                                        )
                                        st.caption("Importance measures model usefulness, not causation.")
                                    with st.expander("How is variable importance calculated?"):
                                        st.write("Permutation importance shuffles one variable at a time and measures how much the model score drops. Higher positive values suggest the model relies more on that variable; negative values can indicate noise.")
                                except Exception as imp_exc:
                                    st.info(f"Variable importance could not be calculated: {imp_exc}")

                                st.markdown("#### Predict using a combination of values")
                                st.caption("Enter one value for every selected input variable, then click the button to estimate the outcome.")
                                user_values = {}
                                for feature in selected_features:
                                    series = X_class[feature]
                                    if pd.api.types.is_numeric_dtype(series):
                                        vals = pd.to_numeric(series, errors="coerce").dropna()
                                        default = float(vals.median()) if not vals.empty else 0.0
                                        low = float(vals.min()) if not vals.empty else 0.0
                                        high = float(vals.max()) if not vals.empty else 1.0
                                        if low == high:
                                            high = low + 1.0
                                        user_values[feature] = st.number_input(
                                            feature, min_value=low, max_value=high,
                                            value=min(max(default, low), high),
                                            key=f"combo_{feature}"
                                        )
                                    else:
                                        opts = series.dropna().astype(str).value_counts().index.tolist() or ["Unknown"]
                                        user_values[feature] = st.selectbox(
                                            feature, opts, key=f"combo_cat_{feature}"
                                        )

                                if st.button("Predict outcome from combination", type="primary"):
                                    one_row = pd.DataFrame([user_values], columns=selected_features)
                                    outcome = clf.predict(one_row)[0]
                                    st.success(f"Predicted {class_target}: {outcome}")
                                    if hasattr(clf.named_steps["model"], "predict_proba"):
                                        probs = clf.predict_proba(one_row)[0]
                                        prob_df = pd.DataFrame({
                                            "Class": clf.named_steps["model"].classes_,
                                            "Probability": probs,
                                            "Description": [
                                                f"Estimated model probability for class {x}"
                                                for x in clf.named_steps["model"].classes_
                                            ]
                                        }).sort_values("Probability", ascending=False)
                                        try:
                                            import plotly.express as px
                                            fig = px.bar(
                                                prob_df, x="Probability", y="Class",
                                                orientation="h",
                                                hover_data={"Description": True, "Probability": ":.1%"},
                                                title="Estimated class probabilities — hover for details"
                                            )
                                            st.plotly_chart(fig, width="stretch")
                                        except ImportError:
                                            st.bar_chart(
                                                prob_df.set_index("Class")["Probability"],
                                                horizontal=True
                                            )
                                        st.dataframe(
                                            prob_df[["Class", "Probability"]].style.format(
                                                {"Probability": "{:.1%}"}
                                            ),
                                            width="stretch", hide_index=True
                                        )
                                    st.session_state["classification_prediction_download"] = pd.DataFrame(
                                        [{**user_values, "Predicted outcome": outcome}]
                                    )
                            except Exception as exc:
                                st.error(f"Classification could not run with these settings: {exc}")

# ---------- Forecasting ----------
elif page == "Harvest Forecasting":
    st.markdown("### Monthly harvest forecast")
    year_col = next(
        (c for c in df.columns if c.lower().strip() in ("year", "harvest year", "production year")),
        None
    )
    month_col = next(
        (c for c in df.columns if c.lower().strip() in ("month", "harvest month", "production month")),
        None
    )
    target_options = [
        c for c in get_numeric_columns(df)
        if any(term in c.lower() for term in ("harvest", "production", "yield", "supply", "quantity", "volume"))
    ]

    if not year_col or not month_col or not target_options:
        st.warning("Forecasting needs Year, Month, and a numeric harvest/production column. Check your CSV headers.")
    else:
        target = st.selectbox("Production field to forecast", target_options, key="forecast_target")
        forecast_data = df[[year_col, month_col, target]].copy()
        forecast_data[year_col] = pd.to_numeric(forecast_data[year_col], errors="coerce")
        forecast_data[target] = pd.to_numeric(forecast_data[target], errors="coerce")
        month_values = forecast_data[month_col].astype("string").str.strip().str.lower()
        forecast_data["__month_num"] = month_values.map(MONTH_MAP)
        numeric_months = pd.to_numeric(month_values, errors="coerce")
        forecast_data["__month_num"] = forecast_data["__month_num"].fillna(numeric_months)
        forecast_data = forecast_data.dropna(subset=[year_col, "__month_num", target])
        forecast_data = forecast_data[
            forecast_data["__month_num"].between(1, 12)
            & forecast_data[year_col].between(1900, 2200)
        ].copy()

        if forecast_data.empty:
            st.warning("No valid year/month/production records were found. Month can be a full month name or a number from 1 to 12.")
        else:
            forecast_data["Date"] = pd.to_datetime({
                "year": forecast_data[year_col].astype(int),
                "month": forecast_data["__month_num"].astype(int),
                "day": 1,
            })
            monthly = forecast_data.groupby("Date")[target].sum().sort_index().asfreq("MS")
            missing_months = int(monthly.isna().sum())
            # Do not fabricate historical harvest values using future observations.
            # Missing periods remain visible; forecasting is disabled until resolved.
            has_missing_months = missing_months > 0

            st.metric("Monthly observations", f"{len(monthly):,}")
            st.caption(f"Missing calendar months before interpolation: {missing_months}. Interpolation is used for plotting/model continuity and may affect results.")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(monthly.index, monthly.values)
            ax.set_title("Historical monthly total")
            ax.set_ylabel(target)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            if has_missing_months:
                st.warning("Monthly gaps detected: forecasting is disabled rather than interpolating missing historical harvest.")
            elif len(monthly) < 6:
                st.warning("At least 6 monthly observations are recommended for a basic forecast; more history is better.")
            else:
                horizon = st.slider("Months to forecast", 3, 24, 12)
                future_index = pd.date_range(
                    monthly.index[-1] + pd.offsets.MonthBegin(1),
                    periods=horizon, freq="MS"
                )
                forecast_values = None
                method = "Seasonal-naive fallback"
                if HAS_STATSMODELS and len(monthly) >= 24:
                    try:
                        fitted = ExponentialSmoothing(
                            monthly, trend="add", seasonal="add",
                            seasonal_periods=12, initialization_method="estimated"
                        ).fit(optimized=True)
                        forecast_values = np.asarray(fitted.forecast(horizon))
                        method = "Holt-Winters Exponential Smoothing"
                    except Exception:
                        forecast_values = None
                if forecast_values is None:
                    vals = []
                    for date in future_index:
                        prior_year_date = date - pd.DateOffset(months=12)
                        if prior_year_date in monthly.index and pd.notna(monthly.loc[prior_year_date]):
                            vals.append(float(monthly.loc[prior_year_date]))
                        else:
                            vals.append(float(monthly.tail(min(12, len(monthly))).median()))
                    forecast_values = np.asarray(vals)

                forecast_values = np.maximum(forecast_values, 0)
                future = pd.DataFrame({
                    "Date": future_index,
                    f"Forecast {target}": forecast_values,
                    "Method": method,
                })
                st.write(f"Forecast method: **{method}**")
                st.dataframe(future.round(2), width="stretch", hide_index=True)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(monthly.index, monthly.values, label="Historical")
                ax.plot(future_index, forecast_values, marker="o", label="Forecast")
                ax.set_title(f"Forecast for the next {horizon} months")
                ax.set_ylabel(target)
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.session_state["forecast_download"] = future

# ---------- Downloads ----------
elif page == "Downloads":
    st.markdown("### Download your analysis outputs")
    st.write("These files are generated from the dataset uploaded in your current session.")
    st.download_button(
        "Download cleaned dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="cleaned_coconut_data.csv",
        mime="text/csv",
        width="stretch",
    )

    st.download_button(
        "Download data quality report (CSV)",
        data=quality_report(df).to_csv(index=False).encode("utf-8-sig"),
        file_name="data_quality_report.csv",
        mime="text/csv",
        width="stretch",
    )

    forecast = st.session_state.get("forecast_download")
    if forecast is not None:
        st.download_button(
            "Download generated forecast (CSV)",
            data=forecast.to_csv(index=False).encode("utf-8-sig"),
            file_name="coconut_production_forecast.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.info("Open Harvest Forecasting and generate a forecast first to enable the forecast download.")

    prediction = st.session_state.get("prediction_download")
    if prediction is not None:
        st.download_button(
            "Download latest test predictions (CSV)",
            data=prediction.to_csv(index=False).encode("utf-8-sig"),
            file_name="model_test_predictions.csv",
            mime="text/csv",
            width="stretch",
        )

    class_prediction_download = st.session_state.get("classification_prediction_download")
    if class_prediction_download is not None:
        st.download_button(
            "Download latest combination prediction (CSV)",
            data=class_prediction_download.to_csv(index=False).encode("utf-8-sig"),
            file_name="combination_prediction.csv",
            mime="text/csv",
            width="stretch",
        )

st.markdown("---")
st.caption(
    "Coconut Production Analytics • Uploads are processed by this app for the current session. "
    "Model results depend on the quality, quantity, and coverage of the uploaded records."
)
