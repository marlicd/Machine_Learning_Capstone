"""
Stroke Prediction Using Machine Learning
==========================================
A Streamlit decision-support dashboard for exploring stroke risk factors,
comparing machine learning models, and estimating stroke probability.

Author: Marlic Demetrius
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Stroke Prediction Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# DESIGN TOKENS
# ----------------------------------------------------------------------------
# A clinical-chart palette rather than a generic SaaS gradient: dark ink
# background, a single desaturated teal used sparingly, and semantic color
# reserved only for the risk output (green/amber/red). Numbers are set in a
# monospace face, the way a lab report or patient monitor sets a reading.
# ============================================================================
INK = "#0a0f1a"
PANEL = "#111a2c"
LINE = "#26324a"
TEXT = "#e7ecf4"
MUTED = "#8b97ac"
ACCENT = "#4fd8c4"
LOW = "#34d399"
MODERATE = "#f2b544"
HIGH = "#ef5350"

PLOTLY_TEMPLATE = "plotly_dark"
CHART_SEQUENCE = [ACCENT, "#7c93c9", MODERATE, HIGH, "#9a86d1"]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
        color: {TEXT};
    }}
    h1, h2, h3 {{
        font-family: 'Source Serif 4', serif;
        font-weight: 700;
        letter-spacing: -0.01em;
    }}
    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: {ACCENT};
        margin-bottom: 0.4rem;
    }}
    .hero-title {{
        font-family: 'Source Serif 4', serif;
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.15;
        color: {TEXT};
        margin: 0 0 0.6rem 0;
    }}
    .hero-sub {{
        font-size: 1.02rem;
        color: {MUTED};
        max-width: 620px;
        line-height: 1.55;
    }}
    .pulse {{
        width: 100%;
        height: 28px;
        margin: 1.4rem 0 1.6rem 0;
        opacity: 0.9;
    }}
    .section-title {{
        font-family: 'Source Serif 4', serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: {TEXT};
        margin: 2rem 0 0.9rem 0;
    }}
    .section-rule {{
        border: none;
        border-top: 1px solid {LINE};
        margin: -0.6rem 0 1.1rem 0;
    }}
    .reading-strip {{
        display: flex;
        flex-wrap: wrap;
        border-top: 1px solid {LINE};
        border-bottom: 1px solid {LINE};
        margin: 0.5rem 0 1.5rem 0;
    }}
    .reading {{
        flex: 1;
        min-width: 140px;
        padding: 0.9rem 1.2rem;
        border-right: 1px solid {LINE};
    }}
    .reading:last-child {{ border-right: none; }}
    .reading .val {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        color: {ACCENT};
        line-height: 1.2;
    }}
    .reading .lab {{
        font-size: 0.74rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {MUTED};
        margin-top: 0.25rem;
    }}
    .quiet-list {{
        border-left: 2px solid {LINE};
        padding-left: 1rem;
        margin-bottom: 0.5rem;
    }}
    .quiet-list p {{
        margin: 0.35rem 0;
        color: {TEXT};
    }}
    .tech-line {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: {MUTED};
        letter-spacing: 0.02em;
    }}
    .callout {{
        border-left: 2px solid {ACCENT};
        background: {PANEL};
        padding: 0.9rem 1.1rem;
        border-radius: 2px;
        margin: 0.6rem 0 1.2rem 0;
        color: {TEXT};
        font-size: 0.94rem;
    }}
    .note {{
        border-left: 2px solid {MODERATE};
        padding: 0.85rem 1.1rem;
        margin-top: 1.2rem;
        color: {MUTED};
        font-size: 0.88rem;
        line-height: 1.5;
    }}
    .risk-band {{
        padding: 1.4rem 1.6rem;
        border-radius: 4px;
        border: 1px solid {LINE};
    }}
    .risk-band .tag {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        opacity: 0.85;
    }}
    .risk-band .verdict {{
        font-family: 'Source Serif 4', serif;
        font-size: 1.9rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }}
    .selected-model {{
        border: 1px solid {LINE};
        border-left: 3px solid {ACCENT};
        padding: 1.1rem 1.4rem;
        border-radius: 2px;
    }}
    .footer {{
        color: {MUTED};
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid {LINE};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

PULSE_SVG = f"""
<svg class="pulse" viewBox="0 0 600 28" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
  <polyline points="0,14 210,14 230,4 245,24 262,14 600,14"
    fill="none" stroke="{ACCENT}" stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round" />
</svg>
"""

# ============================================================================
# CONSTANTS
# ============================================================================
CSV_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "healthcare-dataset-stroke-data.csv"),
    "healthcare-dataset-stroke-data.csv",
]
CATEGORICAL_COLUMNS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
FEATURE_DICTIONARY = [
    ("gender", "Patient's gender (Male / Female / Other)"),
    ("age", "Patient's age in years"),
    ("hypertension", "Whether the patient has hypertension (0 = No, 1 = Yes)"),
    ("heart_disease", "Whether the patient has a history of heart disease (0 = No, 1 = Yes)"),
    ("ever_married", "Whether the patient has ever been married"),
    ("work_type", "Type of employment"),
    ("Residence_type", "Urban or rural residence"),
    ("avg_glucose_level", "Average blood glucose level (mg/dL)"),
    ("bmi", "Body Mass Index"),
    ("smoking_status", "Smoking history of the patient"),
    ("stroke", "Target variable — whether the patient experienced a stroke (0 = No, 1 = Yes)"),
]

PAGES = [
    "Home",
    "Dataset Overview",
    "Exploratory Data Analysis",
    "Feature Relationships",
    "Machine Learning Models",
    "Model Comparison",
    "Stroke Risk Prediction",
    "Conclusion",
]

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================
@st.cache_data(show_spinner="Loading dataset...")
def load_raw_data():
    for path in CSV_CANDIDATES:
        if os.path.exists(path):
            return pd.read_csv(path)
    st.error(
        "Could not find `healthcare-dataset-stroke-data.csv`. "
        "Place it in the same folder as app.py."
    )
    st.stop()


@st.cache_data(show_spinner="Preprocessing data...")
def preprocess_data(df_raw: pd.DataFrame):
    """Mirrors the notebook's cleaning steps: drop id, fill missing BMI with
    the median, then label-encode the categorical columns."""
    df = df_raw.drop(columns=["id"]).copy()
    bmi_missing_before = int(df["bmi"].isnull().sum())
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    df_readable = df.copy()  # human-readable categories, used for EDA

    df_encoded = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le

    return df_readable, df_encoded, encoders, bmi_missing_before


# ============================================================================
# MODEL TRAINING (cached — runs once per session)
# ============================================================================
def _metrics_bundle(model, y_test, y_pred, best_params):
    return {
        "model": model,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=["No Stroke", "Stroke"], zero_division=0),
        "best_params": best_params,
    }


@st.cache_resource(show_spinner="Training Logistic Regression, Decision Tree, Random Forest and XGBoost (first run only)...")
def train_models(df_encoded: pd.DataFrame):
    X = df_encoded.drop(columns=["stroke"])
    y = df_encoded["stroke"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # ---- Logistic Regression (uses scaled features) ----
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        {"C": [0.01, 0.1, 1, 10, 100], "solver": ["liblinear", "lbfgs"], "class_weight": [None, "balanced"]},
        scoring="recall", cv=5, n_jobs=-1,
    )
    lr_grid.fit(X_train_scaled, y_train)
    best_lr = lr_grid.best_estimator_
    results["Logistic Regression"] = _metrics_bundle(
        best_lr, y_test, best_lr.predict(X_test_scaled), lr_grid.best_params_
    )

    # ---- Decision Tree ----
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        {
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 5, 7, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": [None, "balanced"],
        },
        scoring="recall", cv=5, n_jobs=-1,
    )
    dt_grid.fit(X_train, y_train)
    best_dt = dt_grid.best_estimator_
    results["Decision Tree"] = _metrics_bundle(
        best_dt, y_test, best_dt.predict(X_test), dt_grid.best_params_
    )

    # ---- Random Forest ----
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": [None, "balanced"],
        },
        scoring="recall", cv=5, n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    results["Random Forest"] = _metrics_bundle(
        best_rf, y_test, best_rf.predict(X_test), rf_grid.best_params_
    )

    # ---- XGBoost ----
    xgb_grid = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric="logloss"),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
        scoring="recall", cv=5, n_jobs=-1,
    )
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_
    results["XGBoost"] = _metrics_bundle(
        best_xgb, y_test, best_xgb.predict(X_test), xgb_grid.best_params_
    )

    comparison = pd.DataFrame(
        [
            {
                "Model": name,
                "Accuracy": r["accuracy"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1 Score": r["f1"],
            }
            for name, r in results.items()
        ]
    )

    feature_importance = pd.Series(
        best_rf.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    return {
        "results": results,
        "comparison": comparison,
        "scaler": scaler,
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "needs_scaling": {"Logistic Regression": True, "Decision Tree": False, "Random Forest": False, "XGBoost": False},
    }


# ============================================================================
# SMALL UI HELPERS
# ============================================================================
def pulse_divider():
    st.markdown(PULSE_SVG, unsafe_allow_html=True)


def section_title(text, rule=True):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)
    if rule:
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)


def reading_strip(items):
    """items: list of (value, label) tuples rendered as a lab-report style strip."""
    cells = "".join(
        f'<div class="reading"><div class="val">{v}</div><div class="lab">{l}</div></div>'
        for v, l in items
    )
    st.markdown(f'<div class="reading-strip">{cells}</div>', unsafe_allow_html=True)


def quiet_list(lines):
    items = "".join(f"<p>{line}</p>" for line in lines)
    st.markdown(f'<div class="quiet-list">{items}</div>', unsafe_allow_html=True)


def footer():
    st.markdown(
        """
        <div class="footer">
            Marlic Demetrius · Bachelor of Computer Science ·
            Machine Learning Capstone Project · 2026
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig, height=None):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=INK,
        plot_bgcolor=INK,
        font=dict(family="IBM Plex Sans", color=TEXT, size=13),
        title_font=dict(family="Source Serif 4", size=17, color=TEXT),
        margin=dict(t=50, l=10, r=10, b=10),
    )
    if height:
        fig.update_layout(height=height)
    return fig


# ============================================================================
# PAGE 1 — HOME
# ============================================================================
def page_home(df_raw):
    st.markdown('<div class="eyebrow">Clinical Decision Support · Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Stroke Prediction Using Machine Learning</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Predicting stroke likelihood from patient demographic and health '
        'information, built to support clinical judgement rather than replace it.</div>',
        unsafe_allow_html=True,
    )
    pulse_divider()

    section_title("Why this exists")
    st.markdown(
        """
        A stroke happens when blood flow to part of the brain is cut off, and brain
        tissue starts dying within minutes. It's one of the leading causes of death
        and long-term disability worldwide, and the outcome usually comes down to
        how early it's caught. A patient with well-managed blood pressure and
        glucose looks very different, on paper, from one heading toward a stroke —
        the signal is often there before the event is.

        That's the gap this project tries to close. Trained on demographic and
        health records, a model can scan a patient's profile and flag elevated
        risk long before a clinician would otherwise think to look — not as a
        diagnosis, but as a prompt to look closer.
        """
    )

    section_title("What this dashboard covers")
    quiet_list(
        [
            "Explore the dataset the models were trained on",
            "Look at how age, glucose, hypertension and other factors relate to stroke",
            "Train and compare four classification models",
            "Estimate stroke probability for an individual patient",
        ]
    )

    section_title("Dataset at a glance")
    reading_strip(
        [
            (f"{df_raw.shape[0]:,}", "Rows"),
            (f"{df_raw.shape[1]}", "Columns"),
            (f"{df_raw.shape[1] - 1}", "Features"),
            ("stroke", "Target"),
            (f"{int(df_raw.isnull().sum().sum())}", "Missing values"),
        ]
    )

    section_title("Built with")
    st.markdown(
        '<div class="tech-line">Python · Pandas · NumPy · Scikit-Learn · Matplotlib · Seaborn · XGBoost · Streamlit</div>',
        unsafe_allow_html=True,
    )

    with st.expander("How to use this dashboard"):
        st.markdown(
            """
            Use the sidebar to move between sections. **Dataset Overview** and
            **Exploratory Data Analysis** cover the raw data; **Feature
            Relationships** and **Machine Learning Models** cover how the
            models were built; **Model Comparison** shows how they stack up
            against each other; and **Stroke Risk Prediction** is where you
            enter a patient's details to get an estimate.
            """
        )

    footer()


# ============================================================================
# PAGE 2 — DATASET OVERVIEW
# ============================================================================
def page_dataset_overview(df_raw, bmi_missing_before):
    st.markdown('<div class="eyebrow">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:1.9rem;">Before any analysis, know the data</div>', unsafe_allow_html=True)
    pulse_divider()

    section_title("What's in this dataset")
    st.markdown(
        """
        This is the Healthcare Stroke Prediction Dataset, a commonly used public
        dataset for stroke risk research. Each row is a single patient; each
        column is a demographic, medical, or lifestyle attribute recorded for
        them, plus whether they went on to have a stroke.
        """
    )

    section_title("First look")
    st.dataframe(df_raw.head(20), use_container_width=True)

    section_title("Shape and quality")
    reading_strip(
        [
            (f"{df_raw.shape[0]:,}", "Rows"),
            (f"{df_raw.shape[1]}", "Columns"),
            (f"{df_raw.dtypes.nunique()}", "Data types"),
            (f"{int(df_raw.isnull().sum().sum())}", "Missing values"),
            (f"{int(df_raw.duplicated().sum())}", "Duplicate rows"),
        ]
    )

    with st.expander("Column data types"):
        dtypes_df = pd.DataFrame({"Column": df_raw.dtypes.index, "Dtype": df_raw.dtypes.values.astype(str)})
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

    section_title("Feature dictionary")
    fd = pd.DataFrame(FEATURE_DICTIONARY, columns=["Feature", "Description"])
    st.table(fd)

    section_title("Missing values")
    miss = df_raw.isnull().sum()
    miss = miss[miss > 0]
    if len(miss) > 0:
        fig = px.bar(
            x=miss.index, y=miss.values,
            labels={"x": "Column", "y": "Missing count"},
            color_discrete_sequence=[ACCENT],
        )
        st.plotly_chart(style_fig(fig, 350), use_container_width=True)
    st.markdown(
        f"""
        `bmi` was missing for **{bmi_missing_before} patients** — not every
        visit records it. Those gaps were filled with the median BMI rather
        than the mean, since BMI is skewed by a small number of outliers and
        the median holds up better against that.
        """
    )

    footer()


# ============================================================================
# PAGE 3 — EXPLORATORY DATA ANALYSIS
# ============================================================================
def page_eda(df_readable):
    st.markdown('<div class="eyebrow">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:1.9rem;">What the data looks like, by risk factor</div>', unsafe_allow_html=True)
    pulse_divider()

    filter_choice = st.radio(
        "Filter patients", ["All patients", "Stroke only", "No stroke only"],
        horizontal=True, label_visibility="collapsed",
    )
    if filter_choice == "Stroke only":
        df = df_readable[df_readable["stroke"] == 1]
    elif filter_choice == "No stroke only":
        df = df_readable[df_readable["stroke"] == 0]
    else:
        df = df_readable

    section_title("Stroke distribution")
    counts = df_readable["stroke"].value_counts(normalize=True).mul(100).round(1)
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.histogram(
            df_readable, x="stroke", color="stroke",
            color_discrete_map={0: ACCENT, 1: HIGH},
            labels={"stroke": "Stroke (0 = No, 1 = Yes)"},
        )
        fig.update_layout(showlegend=False, bargap=0.3)
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with c2:
        st.markdown(
            f"""
            The dataset is heavily imbalanced — about **{counts.get(0, 0):.1f}%**
            of patients did not have a stroke, versus **{counts.get(1, 0):.1f}%**
            who did. That skew is worth keeping in mind throughout: a model can
            look accurate while still missing most real stroke cases, which is
            why recall gets more weight than accuracy later in this dashboard.
            """
        )

    section_title("Age")
    fig = px.histogram(
        df, x="age", nbins=30, color="stroke" if filter_choice == "All patients" else None,
        color_discrete_map={0: ACCENT, 1: HIGH}, opacity=0.85,
    )
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    st.caption("Stroke cases skew older — risk climbs steadily rather than appearing at any single age.")

    section_title("BMI")
    fig = px.histogram(df, x="bmi", nbins=30, color_discrete_sequence=[ACCENT])
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    st.caption("BMI is fairly spread out across the population, without a dominant cluster.")

    section_title("Average glucose level")
    fig = px.histogram(df, x="avg_glucose_level", nbins=30, color_discrete_sequence=[MODERATE])
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    st.caption("Glucose readings above roughly 150 mg/dL show up more often among stroke patients.")

    section_title("Age, BMI and glucose against stroke status")
    b1, b2, b3 = st.columns(3)
    with b1:
        fig = px.box(df_readable, x="stroke", y="age", color="stroke", color_discrete_map={0: ACCENT, 1: HIGH})
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    with b2:
        fig = px.box(df_readable, x="stroke", y="bmi", color="stroke", color_discrete_map={0: ACCENT, 1: HIGH})
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    with b3:
        fig = px.box(df_readable, x="stroke", y="avg_glucose_level", color="stroke", color_discrete_map={0: ACCENT, 1: HIGH})
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    section_title("What stands out")
    quiet_list(
        [
            "Stroke risk rises with age, consistently rather than at a threshold.",
            "Hypertension and heart disease both correlate with higher stroke rates.",
            "Elevated glucose is more common in the stroke group than the non-stroke group.",
        ]
    )

    footer()


# ============================================================================
# PAGE 4 — FEATURE RELATIONSHIPS
# ============================================================================
def page_feature_relationships(df_encoded, bundle):
    st.markdown('<div class="eyebrow">Feature Relationships</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:1.9rem;">How the features relate to stroke — and each other</div>', unsafe_allow_html=True)
    pulse_divider()

    section_title("Correlation heatmap")
    corr = df_encoded.corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto",
        zmin=-1, zmax=1,
    )
    st.plotly_chart(style_fig(fig, 560), use_container_width=True)
    st.markdown(
        """
        A positive value means two features tend to move together; a negative
        value means one rises as the other falls. Values near zero are weak
        relationships, values closer to ±1 are strong ones.
        """
    )

    stroke_corr = corr["stroke"].drop("stroke").sort_values(ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Correlation with `stroke`**")
        st.dataframe(stroke_corr.round(3).rename("Correlation"), use_container_width=True)
    with c2:
        st.markdown(
            f"""
            Age ({stroke_corr.get('age', 0):.2f}) is the strongest linear signal here.
            Hypertension ({stroke_corr.get('hypertension', 0):.2f}) and heart disease
            ({stroke_corr.get('heart_disease', 0):.2f}) both trend positive, as expected
            clinically. Average glucose ({stroke_corr.get('avg_glucose_level', 0):.2f})
            is a milder positive signal. BMI ({stroke_corr.get('bmi', 0):.2f}), gender
            ({stroke_corr.get('gender', 0):.2f}) and smoking status
            ({stroke_corr.get('smoking_status', 0):.2f}) are weaker in a purely linear
            sense — they may still matter through interactions the correlation
            table alone can't show.
            """
        )

    section_title("Feature importance — Random Forest")
    fi = bundle["feature_importance"].sort_values(ascending=True)
    fig = px.bar(
        x=fi.values, y=fi.index, orientation="h",
        color_discrete_sequence=[ACCENT],
        labels={"x": "Importance", "y": ""},
    )
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)
    st.markdown(
        """
        Age dominates — stroke risk in this dataset is strongly age-driven.
        Glucose level ranks next, consistent with its role in cardiovascular
        risk more broadly, followed by BMI.
        """
    )

    footer()


# ============================================================================
# PAGE 5 — MACHINE LEARNING MODELS
# ============================================================================
def _model_metrics(r):
    reading_strip(
        [
            (f"{r['accuracy']*100:.1f}%", "Accuracy"),
            (f"{r['precision']*100:.1f}%", "Precision"),
            (f"{r['recall']*100:.1f}%", "Recall"),
            (f"{r['f1']*100:.1f}%", "F1 score"),
        ]
    )


def _confusion_matrix_plot(cm):
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Teal",
        x=["No Stroke", "Stroke"], y=["No Stroke", "Stroke"],
        labels=dict(x="Predicted", y="Actual"),
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 360), use_container_width=True)


def page_ml_models(bundle):
    st.markdown('<div class="eyebrow">Machine Learning Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:1.9rem;">Four models, tuned for recall</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Each model was tuned with 5-fold GridSearchCV, scored on recall — '
        'catching true stroke cases matters more here than avoiding false alarms.</div>',
        unsafe_allow_html=True,
    )
    pulse_divider()

    results = bundle["results"]

    section_title("Logistic Regression")
    with st.expander("How it works"):
        st.markdown(
            """
            Logistic Regression estimates stroke probability as a weighted
            combination of the input features, passed through a sigmoid so the
            output lands between 0 and 1. It's a linear model — fast to train,
            straightforward to interpret, and a strong baseline for this kind
            of problem. Its main limitation is that it can't capture
            interactions between features unless they're built in explicitly,
            so it may miss more complex, non-linear patterns.
            """
        )
    r = results["Logistic Regression"]
    _model_metrics(r)
    c1, c2 = st.columns(2)
    with c1:
        _confusion_matrix_plot(r["confusion_matrix"])
    with c2:
        st.code(r["report"], language="text")

    section_title("Decision Tree")
    with st.expander("How it works"):
        st.markdown(
            """
            A Decision Tree repeatedly splits the data on whichever feature
            and threshold best separates stroke from non-stroke patients,
            building a chain of yes/no decisions. It's easy to trace a single
            prediction back through the tree, and it needs no feature scaling.
            Left unchecked it will overfit — memorizing the training set
            instead of generalizing — so `max_depth` and `min_samples_leaf`
            were tuned via grid search to keep it in check.
            """
        )
    r = results["Decision Tree"]
    _model_metrics(r)
    c1, c2 = st.columns(2)
    with c1:
        _confusion_matrix_plot(r["confusion_matrix"])
    with c2:
        st.code(r["report"], language="text")

    section_title("Random Forest")
    with st.expander("How it works"):
        st.markdown(
            """
            Random Forest trains many Decision Trees, each on a random subset
            of the data and features, then lets them vote on the final
            prediction. Averaging over many trees smooths out the overfitting
            any single tree is prone to, while still keeping some
            interpretability through feature importance.
            """
        )
    r = results["Random Forest"]
    _model_metrics(r)
    c1, c2 = st.columns(2)
    with c1:
        _confusion_matrix_plot(r["confusion_matrix"])
    with c2:
        st.code(r["report"], language="text")

    fi = bundle["feature_importance"].sort_values(ascending=True)
    fig = px.bar(x=fi.values, y=fi.index, orientation="h", color_discrete_sequence=[ACCENT], labels={"x": "Importance", "y": ""})
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    section_title("XGBoost")
    with st.expander("How it works"):
        st.markdown(
            """
            XGBoost builds trees sequentially, with each new tree trained to
            correct the errors of the ones before it — gradient boosting
            rather than the parallel voting of a Random Forest. On this
            dataset it reaches the highest accuracy of the four models, but
            that's partly an artifact of the class imbalance: predicting "no
            stroke" for almost everyone already gets you to roughly 95%
            accuracy. Without explicit class weighting, it leans into that
            majority-class behavior, which is why its recall is the lowest
            here — it misses most of the true stroke cases despite the
            headline accuracy.
            """
        )
    r = results["XGBoost"]
    _model_metrics(r)
    c1, c2 = st.columns(2)
    with c1:
        _confusion_matrix_plot(r["confusion_matrix"])
    with c2:
        st.code(r["report"], language="text")

    footer()


# ============================================================================
# PAGE 6 — MODEL COMPARISON
# ============================================================================
def page_model_comparison(bundle):
    st.markdown('<div class="eyebrow">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:1.9rem;">Putting all four side by side</div>', unsafe_allow_html=True)
    pulse_divider()

    comparison = bundle["comparison"].copy()

    section_title("Comparison table")
    display_df = comparison.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_bytes = comparison.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download comparison table (CSV)",
        data=csv_bytes,
        file_name="model_comparison.csv",
        mime="text/csv",
    )

    section_title("Accuracy, precision, recall, F1")
    melted = comparison.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(
        melted, x="Model", y="Score", color="Metric", barmode="group",
        color_discrete_sequence=CHART_SEQUENCE,
    )
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)


    section_title("Reading the metrics")
    st.markdown(
        """
        Accuracy is the share of predictions that were correct overall — on a
        dataset this imbalanced, it can look strong even when a model is
        barely detecting the minority class. Precision asks: of the patients
        flagged as high-risk, how many actually were? Recall asks the reverse:
        of the patients who actually had a stroke, how many did the model
        catch? F1 balances the two.

        Recall carries the most weight in this project. A missed stroke risk
        (false negative) costs far more than a false alarm — a patient
        flagged unnecessarily just gets an extra check, while a missed case
        gets no intervention at all.
        """
    )

    section_title("Selected model")
    best_row = comparison.loc[comparison["Model"] == "Logistic Regression"].iloc[0]
    st.markdown(
        f"""
        <div class="selected-model">
            <div class="eyebrow" style="margin-bottom:0.2rem;">Selected model</div>
            <div style="font-family:'Source Serif 4',serif;font-size:1.3rem;font-weight:700;">Logistic Regression</div>
            <div class="tech-line" style="margin-top:0.3rem;">
                Recall {best_row['Recall']*100:.1f}% · F1 {best_row['F1 Score']*100:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Chosen for the best balance between recall and F1 score of the four models tested.")

    footer()


# ============================================================================
# PAGE 7 — STROKE RISK PREDICTION
# ============================================================================
def page_prediction(bundle, encoders, feature_names):
    st.markdown('<div class="eyebrow">Stroke Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:1.9rem;">Estimate risk for a patient</div>', unsafe_allow_html=True)
    pulse_divider()

    section_title("Patient information")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", list(encoders["gender"].classes_))
        age = st.slider("Age", 0, 100, 45)
        hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
    with c2:
        heart_disease = st.radio("Heart disease", ["No", "Yes"], horizontal=True)
        ever_married = st.selectbox("Ever married", list(encoders["ever_married"].classes_))
        work_type = st.selectbox("Work type", list(encoders["work_type"].classes_))
    with c3:
        residence_type = st.selectbox("Residence type", list(encoders["Residence_type"].classes_))
        avg_glucose = st.number_input("Average glucose level (mg/dL)", 40.0, 300.0, 100.0, step=0.1)
        bmi = st.number_input("BMI", 10.0, 80.0, 25.0, step=0.1)
    smoking_status = st.selectbox("Smoking status", list(encoders["smoking_status"].classes_))

    with st.expander("Advanced: choose a different model"):
        model_name = st.selectbox(
            "Model used for prediction",
            list(bundle["results"].keys()),
            index=list(bundle["results"].keys()).index("Logistic Regression"),
        )

    predict_clicked = st.button("Predict stroke risk", type="primary", use_container_width=True)

    if predict_clicked:
        raw_input = {
            "gender": gender,
            "age": age,
            "hypertension": 1 if hypertension == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence_type,
            "avg_glucose_level": avg_glucose,
            "bmi": bmi,
            "smoking_status": smoking_status,
        }
        encoded_input = {}
        for col in feature_names:
            val = raw_input[col]
            if col in encoders:
                val = encoders[col].transform([val])[0]
            encoded_input[col] = val
        input_df = pd.DataFrame([encoded_input])[feature_names]

        model = bundle["results"][model_name]["model"]
        if bundle["needs_scaling"][model_name]:
            model_input = bundle["scaler"].transform(input_df)
        else:
            model_input = input_df.values

        prediction = model.predict(model_input)[0]
        proba = model.predict_proba(model_input)[0]
        stroke_probability = proba[1]

        section_title("Result")
        r1, r2 = st.columns(2)
        with r1:
            band_color = HIGH if prediction == 1 else LOW
            verdict = "HIGH" if prediction == 1 else "LOW"
            st.markdown(
                f"""
                <div class="risk-band" style="border-color:{band_color};">
                    <div class="tag" style="color:{band_color};">Stroke risk</div>
                    <div class="verdict" style="color:{band_color};">{verdict}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(f"**Estimated probability: {stroke_probability*100:.1f}%**")
            st.progress(min(max(stroke_probability, 0.0), 1.0))

            distance_from_mid = abs(stroke_probability - 0.5)
            if distance_from_mid >= 0.25:
                confidence = "High confidence"
            elif distance_from_mid >= 0.10:
                confidence = "Medium confidence"
            else:
                confidence = "Low confidence"
            st.markdown(f"**{confidence}**")

        section_title("Interpretation")
        if stroke_probability < 0.30:
            tier_color, tier_label, risk_tier = LOW, "Low risk — probability below 30%.", "low"
        elif stroke_probability < 0.60:
            tier_color, tier_label, risk_tier = MODERATE, "Moderate risk — probability between 30% and 60%.", "moderate"
        else:
            tier_color, tier_label, risk_tier = HIGH, "High risk — probability of 60% or above.", "high"
        st.markdown(
            f'<p style="color:{tier_color};font-weight:600;">{tier_label}</p>',
            unsafe_allow_html=True,
        )

        section_title("Recommended next steps")
        if risk_tier in ("high", "moderate"):
            quiet_list(
                [
                    "Recommend a prompt clinical assessment.",
                    "Review blood glucose levels.",
                    "Assess blood pressure.",
                    "Evaluate cardiovascular history.",
                ]
            )
        else:
            quiet_list(
                [
                    "Continue routine monitoring.",
                    "Maintain current lifestyle and health habits.",
                    "Follow standard physician guidance.",
                ]
            )

    st.markdown(
        """
        <div class="note">
            This prediction is intended to support healthcare professionals.
            It should not replace clinical diagnosis or professional medical
            judgement.
        </div>
        """,
        unsafe_allow_html=True,
    )
    footer()


# ============================================================================
# PAGE 8 — CONCLUSION
# ============================================================================
def page_conclusion(bundle):
    st.markdown('<div class="eyebrow">Conclusion</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:1.9rem;">Summary, limitations and what\'s next</div>', unsafe_allow_html=True)
    pulse_divider()

    section_title("Project summary")
    st.markdown(
        """
        This project set out to predict stroke risk from patient demographic,
        medical, and lifestyle information. The path there: explore the data,
        clean it (drop the identifier column, fill missing BMI with the
        median, label-encode the categorical fields), train four
        classification models, tune each with 5-fold GridSearchCV scored on
        recall, and compare the results. The model with the best balance of
        recall and F1 was carried into this dashboard for deployment.
        """
    )

    section_title("Final result")
    best_row = bundle["comparison"].loc[bundle["comparison"]["Model"] == "Logistic Regression"].iloc[0]
    st.markdown(
        f"""
        <div class="selected-model">
            <div class="eyebrow" style="margin-bottom:0.2rem;">Best model</div>
            <div style="font-family:'Source Serif 4',serif;font-size:1.3rem;font-weight:700;">Logistic Regression</div>
            <div class="tech-line" style="margin-top:0.3rem;">Recall {best_row['Recall']*100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Selected for the best balance between recall and F1 score among the models tested.")

    section_title("Limitations")
    quiet_list(
        [
            "The dataset is small, particularly for the minority stroke class.",
            "Stroke cases make up only about 5% of the data.",
            "The feature set is limited — no cholesterol, blood pressure readings, or family history.",
            "The model has not been clinically validated and should not be used as a sole basis for diagnosis.",
        ]
    )

    section_title("What could improve this")
    quiet_list(
        [
            "A larger, more diverse dataset.",
            "Additional medical features.",
            "Threshold tuning on the deployed model.",
            "External validation on an independent patient cohort.",
            "A deep learning approach for richer feature interactions.",
        ]
    )

    footer()


# ============================================================================
# MAIN
# ============================================================================
def main():
    df_raw = load_raw_data()
    df_readable, df_encoded, encoders, bmi_missing_before = preprocess_data(df_raw)
    feature_names = df_encoded.drop(columns=["stroke"]).columns.tolist()

    st.sidebar.markdown('<div class="eyebrow" style="margin-bottom:0.1rem;">Stroke Prediction</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div style="font-family:\'Source Serif 4\',serif;font-size:1.25rem;font-weight:700;margin-bottom:1rem;">Dashboard</div>', unsafe_allow_html=True)
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
    st.sidebar.markdown(f'<hr style="border-color:{LINE};margin:1.2rem 0;">', unsafe_allow_html=True)
    st.sidebar.caption("Decision-support tool. Not a substitute for clinical judgement.")

    needs_models = page in [
        "Feature Relationships", "Machine Learning Models",
        "Model Comparison", "Stroke Risk Prediction", "Conclusion",
    ]
    bundle = train_models(df_encoded) if needs_models else None

    if page == "Home":
        page_home(df_raw)
    elif page == "Dataset Overview":
        page_dataset_overview(df_raw, bmi_missing_before)
    elif page == "Exploratory Data Analysis":
        page_eda(df_readable)
    elif page == "Feature Relationships":
        page_feature_relationships(df_encoded, bundle)
    elif page == "Machine Learning Models":
        page_ml_models(bundle)
    elif page == "Model Comparison":
        page_model_comparison(bundle)
    elif page == "Stroke Risk Prediction":
        page_prediction(bundle, encoders, feature_names)
    elif page == "Conclusion":
        page_conclusion(bundle)


if __name__ == "__main__":
    main()