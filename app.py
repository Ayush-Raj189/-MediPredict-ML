"""
Multiple Disease Prediction System
Streamlit Web Application
Diseases: Diabetes · Heart Disease · Parkinson's
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediPredict — Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-main:    #070d1a;
    --bg-card:    #0f1929;
    --bg-card2:   #162035;
    --border:     #1e3050;
    --border-glow:#1a4080;
    --text-main:  #e2eaf6;
    --text-dim:   #7a93b8;
    --text-muted: #3d5475;
    --cyan:       #00c8f8;
    --cyan-dim:   #0a8ab8;
    --amber:      #f59e0b;
    --red:        #f43f5e;
    --purple:     #a78bfa;
    --green:      #10b981;
    --yellow:     #facc15;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text-main) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] {
    background: transparent !important;
    border: 0 !important;
}
.block-container { padding: 1.5rem 2.5rem 3rem 2.5rem !important; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #080f1e !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    min-width: 18rem !important;
    max-width: 18rem !important;
}

/* Responsive sizing for different devices */
@media (max-width: 1200px) {
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        min-width: 16rem !important;
        max-width: 16rem !important;
    }
    .block-container {
        padding: 1.2rem 1.4rem 2.4rem 1.4rem !important;
    }
}

@media (max-width: 900px) {
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        min-width: 15rem !important;
        max-width: 15rem !important;
    }
    .block-container {
        padding: 1rem 1.1rem 2rem 1.1rem !important;
    }
}

@media (max-width: 640px) {
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        min-width: min(88vw, 20rem) !important;
        max-width: min(88vw, 20rem) !important;
    }
    .block-container {
        padding: 0.8rem 0.8rem 1.8rem 0.8rem !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        font-size: 0.82rem !important;
    }
}

/* Keep toggle controls clearly visible and easy to tap */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    opacity: 1 !important;
    z-index: 1001 !important;
}
[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button {
    background: #0f1929 !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35) !important;
}
[data-testid="collapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover {
    border-color: var(--cyan) !important;
    box-shadow: 0 8px 22px rgba(0, 200, 248, 0.2) !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span { color: var(--text-main) !important; }

/* ── Radio buttons (nav) ── */
[data-testid="stSidebar"] .stRadio > div { gap: 0.4rem; }
[data-testid="stSidebar"] .stRadio label {
    background: #0c1525 !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    cursor: pointer;
    transition: all .2s;
    width: 100%;
}
[data-testid="stSidebar"] .stRadio label:hover {
    border-color: var(--cyan) !important;
    background: #101e35 !important;
}

/* ── Sliders & number inputs ── */
.stSlider > div > div > div { background: var(--border) !important; }
.stSlider > div > div > div > div { background: var(--cyan) !important; }
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
.stSelectbox select {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-main) !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stNumberInput"] input:focus { border-color: var(--cyan) !important; box-shadow: 0 0 0 2px rgba(0,200,248,.15) !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] div[data-baseweb="select"] div {
    background: var(--bg-card2) !important;
    border-color: var(--border) !important;
    color: var(--text-main) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0060b8, #004a90) !important;
    border: 1px solid var(--cyan-dim) !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2.5rem !important;
    letter-spacing: 0.04em !important;
    transition: all .25s !important;
    box-shadow: 0 4px 20px rgba(0,96,184,.3) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0070d8, #005ab0) !important;
    box-shadow: 0 6px 28px rgba(0,180,255,.35) !important;
    transform: translateY(-2px) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-dim) !important;
    border-radius: 9px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.5rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card2) !important;
    color: var(--cyan) !important;
    border: 1px solid var(--border) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-main) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Labels ── */
label, .stMarkdown p { color: var(--text-main) !important; }
small, .stMarkdown small { color: var(--text-dim) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "models"

@st.cache_resource
def load_models():
    """Load all pkl artifacts once and cache them."""
    base = MODEL_DIR
    d = {
        "diabetes_model"   : joblib.load(os.path.join(base, "diabetes_model.pkl")),
        "diabetes_scaler"  : joblib.load(os.path.join(base, "diabetes_scaler.pkl")),
        "diabetes_imputer" : joblib.load(os.path.join(base, "diabetes_imputer.pkl")),
        "heart_model"      : joblib.load(os.path.join(base, "heart_model.pkl")),
        "heart_scaler"     : joblib.load(os.path.join(base, "heart_scaler.pkl")),
        "park_model"       : joblib.load(os.path.join(base, "parkinsons_model.pkl")),
        "park_scaler"      : joblib.load(os.path.join(base, "parkinsons_scaler.pkl")),
    }
    return d

# ─── Feature engineering helpers (must mirror notebooks exactly) ─────────────
ZERO_INVALID = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def add_features_diabetes(df):
    df = df.copy()
    df['Glucose_BMI']     = df['Glucose'] * df['BMI']
    df['Age_BMI']         = df['Age'] * df['BMI']
    df['Insulin_Glucose'] = df['Insulin'] / (df['Glucose'] + 1e-6)
    df['Preg_Age']        = df['Pregnancies'] / (df['Age'] + 1e-6)
    df['Glucose2']        = df['Glucose'] ** 2
    df['BMI2']            = df['BMI'] ** 2
    df['Age_Glucose']     = df['Age'] * df['Glucose']
    df['DPF_Glucose']     = df['DiabetesPedigreeFunction'] * df['Glucose']
    return df

def cap_outliers(df, factor=1.5):
    df2 = df.copy()
    for col in df2.columns:
        Q1, Q3 = df2[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df2[col] = df2[col].clip(Q1 - factor * IQR, Q3 + factor * IQR)
    return df2

def add_features_parkinsons(df):
    df2 = df.copy()
    df2['Jitter_RAP_x_PPQ']    = df2['MDVP:RAP']      * df2['MDVP:PPQ']
    df2['Shimmer_APQ3_x_APQ5'] = df2['Shimmer:APQ3']  * df2['Shimmer:APQ5']
    df2['HNR_NHR_ratio']       = df2['HNR']            / (df2['NHR'] + 1e-8)
    df2['RPDE_x_DFA']          = df2['RPDE']           * df2['DFA']
    df2['spread1_x_spread2']   = df2['spread1']        * df2['spread2']
    df2['PPE_squared']         = df2['PPE'] ** 2
    df2['D2_squared']          = df2['D2'] ** 2
    df2['Fo_range']            = df2['MDVP:Fhi(Hz)']   - df2['MDVP:Flo(Hz)']
    return df2


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def predict_diabetes(models, patient: dict):
    BASE = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age']
    df_p = pd.DataFrame([patient], columns=BASE)
    for col in ZERO_INVALID:
        df_p[col] = df_p[col].replace(0, np.nan)
    df_imp = pd.DataFrame(
        models["diabetes_imputer"].transform(df_p), columns=BASE)
    df_fe  = add_features_diabetes(df_imp)
    df_sc  = models["diabetes_scaler"].transform(df_fe)
    pred   = models["diabetes_model"].predict(df_sc)[0]
    prob   = models["diabetes_model"].predict_proba(df_sc)[0][1]
    return int(pred), float(prob)

def predict_heart(models, patient: dict):
    COLS = ['age','sex','cp','trestbps','chol','fbs',
            'restecg','thalach','exang','oldpeak','slope','ca','thal']
    df_p = pd.DataFrame([patient], columns=COLS)
    df_sc = models["heart_scaler"].transform(df_p)
    pred  = models["heart_model"].predict(df_sc)[0]
    prob  = models["heart_model"].predict_proba(df_sc)[0][1]
    return int(pred), float(prob)

def predict_parkinsons(models, patient: dict):
    COLS = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)',
            'MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ',
            'Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)',
            'Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA',
            'NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE']
    df_p = pd.DataFrame([patient], columns=COLS)
    df_c = cap_outliers(df_p)
    df_e = add_features_parkinsons(df_c)
    df_s = models["park_scaler"].transform(df_e)
    pred = models["park_model"].predict(df_s)[0]
    prob = models["park_model"].predict_proba(df_s)[0][1]
    return int(pred), float(prob)


# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def render_result_card(pred: int, prob: float, disease: str,
                       pos_label: str, neg_label: str, accent: str):
    """Renders a beautiful animated result card with risk gauge."""
    pct        = prob * 100
    risk_label = "HIGH RISK"   if pct >= 60 else ("MODERATE RISK" if pct >= 35 else "LOW RISK")
    risk_color = "#f43f5e"     if pct >= 60 else ("#f59e0b"        if pct >= 35 else "#10b981")
    result_txt = pos_label if pred == 1 else neg_label
    result_col = "#f43f5e" if pred == 1 else "#10b981"
    result_ico = "⚠️" if pred == 1 else "✅"

    # SVG arc gauge  ────────────────────────────────────────────────
    r, cx, cy = 70, 100, 100
    dash_total = 3.14159 * r         # half circle circumference
    dash_fill  = dash_total * min(pct / 100, 1.0)

    gauge_svg = f"""
    <svg viewBox="0 0 200 120" xmlns="http://www.w3.org/2000/svg" style="width:200px;height:120px">
      <defs>
        <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stop-color="#10b981"/>
          <stop offset="50%"  stop-color="#f59e0b"/>
          <stop offset="100%" stop-color="#f43f5e"/>
        </linearGradient>
      </defs>
      <!-- Track -->
      <path d="M 30 100 A {r} {r} 0 0 1 170 100"
            fill="none" stroke="#1e3050" stroke-width="14"
            stroke-linecap="round"/>
      <!-- Fill -->
      <path d="M 30 100 A {r} {r} 0 0 1 170 100"
            fill="none" stroke="url(#arcGrad)" stroke-width="14"
            stroke-linecap="round"
            stroke-dasharray="{dash_fill:.1f} {dash_total:.1f}"
            style="transition: stroke-dasharray 1s ease"/>
      <!-- Centre text -->
      <text x="100" y="90" text-anchor="middle"
            font-family="DM Mono, monospace"
            font-size="26" font-weight="600" fill="{risk_color}">{pct:.1f}%</text>
      <text x="100" y="108" text-anchor="middle"
            font-family="Space Grotesk, sans-serif"
            font-size="9" fill="#7a93b8">PROBABILITY</text>
      <!-- Min / Max labels -->
      <text x="28" y="115" text-anchor="middle" font-size="8" fill="#3d5475">0%</text>
      <text x="172" y="115" text-anchor="middle" font-size="8" fill="#3d5475">100%</text>
    </svg>"""

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #0f1929 0%, #11203a 100%);
        border: 1px solid {accent}55;
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-top: 1rem;
        box-shadow: 0 8px 40px {accent}22, inset 0 1px 0 {accent}20;
        animation: fadeUp .5s ease both;
    ">
      <!-- Header -->
      <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1.5rem;">
        <span style="font-size:2.2rem">{result_ico}</span>
        <div>
          <div style="font-size:1.5rem; font-weight:700; color:{result_col}; line-height:1.2">
            {result_txt}
          </div>
          <div style="font-size:0.8rem; color:#7a93b8; letter-spacing:.1em; text-transform:uppercase">
            {disease} Prediction Result
          </div>
        </div>
        <div style="margin-left:auto;">
          <span style="
            background:{risk_color}22; color:{risk_color};
            border:1px solid {risk_color}55;
            border-radius:50px; padding:.3rem 1rem;
            font-size:.75rem; font-weight:700; letter-spacing:.1em;
          ">{risk_label}</span>
        </div>
      </div>

      <hr style="border-color:#1e3050; margin:0 0 1.5rem"/>

      <!-- Gauge + Details -->
      <div style="display:flex; gap:2.5rem; align-items:center; flex-wrap:wrap;">
        <div style="text-align:center; flex-shrink:0;">
          {gauge_svg}
        </div>
        <div style="flex:1; min-width:200px;">
          <p style="color:#7a93b8; font-size:.82rem; line-height:1.7; margin:0">
            The model analysed the provided biomarkers and computed a
            <strong style="color:{risk_color}">{pct:.1f}%</strong> probability
            of {disease.lower()}.
            {"This result <strong>warrants immediate medical consultation</strong>. Do not rely solely on this tool." if pred==1
             else "The biomarker profile appears within a healthy range, though regular screening is always recommended."}
          </p>
          <div style="margin-top:1.2rem; display:flex; gap:.8rem; flex-wrap:wrap;">
            <div style="background:#0c1525; border:1px solid #1e3050; border-radius:10px; padding:.6rem 1rem; text-align:center; min-width:90px;">
              <div style="font-family:'DM Mono',monospace; font-size:1.1rem; font-weight:600; color:{accent}">{pct:.1f}%</div>
              <div style="font-size:.7rem; color:#7a93b8">Probability</div>
            </div>
            <div style="background:#0c1525; border:1px solid #1e3050; border-radius:10px; padding:.6rem 1rem; text-align:center; min-width:90px;">
              <div style="font-family:'DM Mono',monospace; font-size:1.1rem; font-weight:600; color:{result_col}">{result_txt.split()[0]}</div>
              <div style="font-size:.7rem; color:#7a93b8">Prediction</div>
            </div>
            <div style="background:#0c1525; border:1px solid #1e3050; border-radius:10px; padding:.6rem 1rem; text-align:center; min-width:90px;">
              <div style="font-family:'DM Mono',monospace; font-size:1.1rem; font-weight:600; color:{risk_color}">{risk_label.split()[0]}</div>
              <div style="font-size:.7rem; color:#7a93b8">Risk Level</div>
            </div>
          </div>
        </div>
      </div>

      <div style="margin-top:1.5rem; background:#0c152033; border-left:3px solid #f59e0b;
                  border-radius:0 8px 8px 0; padding:.7rem 1rem;">
        <span style="color:#f59e0b; font-size:.75rem; font-weight:600">⚕ DISCLAIMER</span>
        <span style="color:#7a93b8; font-size:.75rem; margin-left:.5rem">
          This tool is for educational purposes only. Always consult a qualified medical professional.
        </span>
      </div>
    </div>

    <style>
    @keyframes fadeUp {{
      from {{ opacity:0; transform:translateY(18px); }}
      to   {{ opacity:1; transform:translateY(0); }}
    }}
    </style>
    """, unsafe_allow_html=True)


def section_header(icon, title, subtitle, accent):
    st.markdown(f"""
    <div style="
        border-left: 4px solid {accent};
        padding: .5rem 0 .5rem 1.2rem;
        margin-bottom: 1.5rem;
    ">
      <div style="display:flex; align-items:center; gap:.7rem;">
        <span style="font-size:1.8rem">{icon}</span>
        <div>
          <h2 style="margin:0; font-size:1.55rem; font-weight:700; color:#e2eaf6">{title}</h2>
          <p  style="margin:0; font-size:.85rem; color:#7a93b8">{subtitle}</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def field_card(title, content_fn, accent="#1e3050"):
    st.markdown(f"""
    <div style="
        background:#0f1929; border:1px solid #1e3050;
        border-top:2px solid {accent};
        border-radius:14px; padding:1.2rem 1.4rem 0.4rem;
        margin-bottom:.6rem;
    ">
      <p style="font-size:.72rem; font-weight:600; letter-spacing:.12em;
                color:#3d5475; text-transform:uppercase; margin:0 0 .8rem">{title}</p>
    """, unsafe_allow_html=True)
    content_fn()
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.5rem 0 1rem;">
      <div style="font-size:2.5rem">🩺</div>
      <h1 style="font-size:1.3rem; font-weight:700; margin:.4rem 0 .1rem; color:#e2eaf6">
        MediPredict
      </h1>
      <p style="font-size:.75rem; color:#3d5475; letter-spacing:.12em; text-transform:uppercase">
        Multiple Disease AI
      </p>
    </div>
    <hr style="border-color:#1e3050; margin:.5rem 0 1.5rem"/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home", "🩸  Diabetes", "❤️  Heart Disease", "🧠  Parkinson's"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style="border-color:#1e3050; margin:1.5rem 0 1rem"/>
    <p style="font-size:.72rem; color:#3d5475; text-transform:uppercase; letter-spacing:.1em">Models</p>
    """, unsafe_allow_html=True)

    for lbl, col in [("Diabetes",     "#f59e0b"),
                     ("Heart Disease","#f43f5e"),
                     ("Parkinson's",  "#a78bfa")]:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:.6rem; margin-bottom:.4rem;">
          <div style="width:8px; height:8px; border-radius:50%; background:{col};
                      box-shadow:0 0 6px {col}88;"></div>
          <span style="font-size:.8rem; color:#7a93b8">{lbl}</span>
          <span style="margin-left:auto; font-size:.7rem; color:#3d5475">VotingEnsemble</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <hr style="border-color:#1e3050; margin:1rem 0"/>
    <p style="font-size:.72rem; color:#3d5475; text-align:center; margin-top:1rem">
      Built with Streamlit · scikit-learn<br>For educational purposes only
    </p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
try:
    models = load_models()
    models_ok = True
except Exception as e:
    models_ok = False
    load_error = str(e)


# ═════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    # Hero
    st.markdown("""
    <div style="
        text-align:center;
        padding: 3.5rem 1rem 2.5rem;
        background: radial-gradient(ellipse at 50% -10%, #0a3060 0%, transparent 70%);
        border-bottom: 1px solid #1e3050;
        margin-bottom: 3rem;
    ">
      <div style="font-size:4rem; margin-bottom:.5rem">🩺</div>
      <h1 style="font-size:3rem; font-weight:700; margin:0;
                 background:linear-gradient(135deg,#00c8f8,#a78bfa);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent">
        MediPredict
      </h1>
      <p style="font-size:1.15rem; color:#7a93b8; margin:.8rem auto 0; max-width:520px; line-height:1.7">
        AI-powered prediction system for three major diseases —
        trained on clinical datasets with ensemble ML models.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Disease cards
    cards = [
        ("🩸", "Diabetes",      "#f59e0b", "🩸  Diabetes",
         "PIMA Indians dataset · 8 biomarkers",
         "Analyses glucose, BMI, insulin and family history to assess diabetes risk."),
        ("❤️", "Heart Disease",  "#f43f5e", "❤️  Heart Disease",
         "Cleveland dataset · 13 biomarkers",
         "Evaluates chest pain, ECG results, cholesterol and more for cardiac risk."),
        ("🧠", "Parkinson's",   "#a78bfa", "🧠  Parkinson's",
         "Oxford dataset · 22 voice features",
         "Detects Parkinson's through vocal biomarker analysis (jitter, shimmer, HNR…)."),
    ]

    cols = st.columns(3, gap="large")
    for col, (ico, name, accent, nav, tag, desc) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div style="
                background: linear-gradient(160deg, #0f1929 60%, #141f35);
                border: 1px solid {accent}44;
                border-top: 3px solid {accent};
                border-radius: 18px;
                padding: 1.8rem 1.5rem;
                height: 100%;
                transition: transform .2s, box-shadow .2s;
            " onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 12px 40px {accent}22'"
              onmouseout ="this.style.transform='';this.style.boxShadow=''">
              <div style="font-size:2.4rem; margin-bottom:.8rem">{ico}</div>
              <h3 style="color:#e2eaf6; font-size:1.25rem; margin:0 0 .3rem">{name}</h3>
              <p style="font-size:.72rem; color:{accent}; text-transform:uppercase;
                        letter-spacing:.1em; margin:0 0 .9rem">{tag}</p>
              <p style="color:#7a93b8; font-size:.85rem; line-height:1.6; margin:0">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats row
    st.markdown("""
    <div style="
        display:flex; gap:1rem; flex-wrap:wrap; justify-content:center;
        padding:2rem; background:#0f1929;
        border:1px solid #1e3050; border-radius:16px;
        margin-bottom:2rem;
    ">
    """, unsafe_allow_html=True)

    stats = [("3",    "Diseases",           "#00c8f8"),
             ("10+",  "ML Algorithms",      "#f59e0b"),
             ("95%+", "Ensemble Accuracy",  "#10b981"),
             ("~900", "Training Samples",   "#a78bfa")]
    cols2 = st.columns(4)
    for col, (val, lbl, accent) in zip(cols2, stats):
        with col:
            st.markdown(f"""
            <div style="text-align:center; padding:.5rem;">
              <div style="font-size:2rem; font-weight:700;
                          font-family:'DM Mono',monospace; color:{accent}">{val}</div>
              <div style="font-size:.78rem; color:#7a93b8; text-transform:uppercase;
                          letter-spacing:.1em">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <h3 style="color:#e2eaf6; font-size:1.15rem; font-weight:600; margin:2rem 0 1.2rem">
      ⚙️ How It Works
    </h3>
    """, unsafe_allow_html=True)
    steps = [("1","Select a disease","Use the sidebar to choose Diabetes, Heart Disease or Parkinson's.","#00c8f8"),
             ("2","Enter biomarkers","Fill in the clinical values using the sliders and input fields.","#f59e0b"),
             ("3","Run prediction","Click Predict to run the ML ensemble and compute risk probability.","#a78bfa"),
             ("4","Review results","A detailed result card shows prediction, probability gauge and risk level.","#10b981")]
    hcols = st.columns(4, gap="medium")
    for col, (n, ttl, desc, accent) in zip(hcols, steps):
        with col:
            st.markdown(f"""
            <div style="background:#0f1929; border:1px solid #1e3050; border-radius:14px; padding:1.2rem; text-align:center;">
              <div style="width:36px; height:36px; border-radius:50%;
                          background:{accent}22; border:2px solid {accent};
                          display:flex; align-items:center; justify-content:center;
                          margin:0 auto .8rem;
                          font-weight:700; color:{accent}; font-size:1rem">{n}</div>
              <h4 style="color:#e2eaf6; margin:0 0 .4rem; font-size:.9rem">{ttl}</h4>
              <p style="color:#7a93b8; font-size:.78rem; margin:0; line-height:1.5">{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# DIABETES PAGE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🩸  Diabetes":
    ACCENT = "#f59e0b"
    section_header("🩸", "Diabetes Prediction",
                   "PIMA Indians Diabetes Dataset · 8 clinical biomarkers · Voting Ensemble", ACCENT)

    if not models_ok:
        st.error(f"❌ Could not load models: `{load_error}`\n\nEnsure `models/` folder is in the same directory as `app.py`.")
        st.stop()

    with st.expander("ℹ️ About this model", expanded=False):
        st.markdown("""
        **Features used:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age

        **Pipeline:** Zero→NaN replacement · KNN Imputation · IQR Outlier Capping · Feature Engineering (8 interaction terms) · RobustScaler → Voting Ensemble

        **Target:** `1` = Diabetic · `0` = Non-Diabetic
        """)

    col_form, col_result = st.columns([1.1, 1], gap="large")

    with col_form:
        st.markdown(f'<p style="font-size:.75rem; color:#3d5475; text-transform:uppercase; letter-spacing:.12em; margin-bottom:1rem">Patient Biomarkers</p>', unsafe_allow_html=True)

        with st.container():
            st.markdown(f'<div style="background:#0f1929; border:1px solid #1e3050; border-top:2px solid {ACCENT}; border-radius:14px; padding:1.2rem 1.4rem;">', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                pregnancies = st.number_input("Pregnancies", 0, 20, 1, help="Number of times pregnant")
                blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 130, 70)
                insulin = st.number_input("Insulin (μU/mL)", 0, 900, 80)
                dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.47, step=0.01, format="%.3f")
            with c2:
                glucose = st.number_input("Glucose (mg/dL)", 0, 250, 110)
                skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
                bmi = st.number_input("BMI (kg/m²)", 0.0, 70.0, 28.0, step=0.1, format="%.1f")
                age = st.number_input("Age (years)", 1, 100, 33)

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍  Run Diabetes Prediction", use_container_width=True)

    with col_result:
        st.markdown(f'<p style="font-size:.75rem; color:#3d5475; text-transform:uppercase; letter-spacing:.12em; margin-bottom:1rem">Reference Ranges</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#0f1929; border:1px solid #1e3050; border-radius:14px; padding:1.2rem 1.4rem;">
        {"".join(f'''
        <div style="display:flex; justify-content:space-between; padding:.4rem 0;
                    border-bottom:1px solid #1e3050;">
          <span style="color:#7a93b8; font-size:.82rem">{f}</span>
          <span style="color:#e2eaf6; font-size:.82rem; font-family:'DM Mono',monospace">{v}</span>
        </div>''' for f, v in [
            ("Glucose (normal)", "70–99 mg/dL"),
            ("Glucose (diabetic)", "≥126 mg/dL"),
            ("Blood Pressure", "70–90 mm Hg"),
            ("BMI (normal)", "18.5–24.9"),
            ("Insulin (fasting)", "16–166 μU/mL"),
            ("Skin Thickness", "10–30 mm"),
        ])}
        </div>
        """, unsafe_allow_html=True)

    if predict_btn:
        patient = dict(
            Pregnancies=pregnancies, Glucose=glucose,
            BloodPressure=blood_pressure, SkinThickness=skin_thickness,
            Insulin=insulin, BMI=bmi, DiabetesPedigreeFunction=dpf, Age=age
        )
        with st.spinner("Analysing biomarkers…"):
            pred, prob = predict_diabetes(models, patient)
        render_result_card(pred, prob, "Diabetes", "Diabetic", "Non-Diabetic", ACCENT)


# ═════════════════════════════════════════════════════════════════════════════
# HEART DISEASE PAGE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "❤️  Heart Disease":
    ACCENT = "#f43f5e"
    section_header("❤️", "Heart Disease Prediction",
                   "Cleveland Heart Disease Dataset · 13 clinical biomarkers · Voting Ensemble", ACCENT)

    if not models_ok:
        st.error(f"❌ Could not load models: `{load_error}`")
        st.stop()

    with st.expander("ℹ️ About this model", expanded=False):
        st.markdown("""
        **Features used:** Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Angina, ST Depression, ST Slope, Major Vessels, Thal

        **Pipeline:** IQR Outlier Capping · RobustScaler → Soft Voting Ensemble

        **Target:** `1` = Heart Disease · `0` = No Heart Disease
        """)

    col_form, col_result = st.columns([1.1, 1], gap="large")

    with col_form:
        st.markdown(f'<p style="font-size:.75rem; color:#3d5475; text-transform:uppercase; letter-spacing:.12em; margin-bottom:1rem">Patient Biomarkers</p>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:#0f1929; border:1px solid #1e3050; border-top:2px solid {ACCENT}; border-radius:14px; padding:1.2rem 1.4rem;">', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age_h      = st.number_input("Age (years)", 20, 100, 54, key="h_age")
            sex        = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
            cp         = st.selectbox("Chest Pain Type",
                                      [(0,"Typical Angina"),(1,"Atypical Angina"),
                                       (2,"Non-anginal"),(3,"Asymptomatic")],
                                      format_func=lambda x: f"{x[0]} – {x[1]}")
            trestbps   = st.number_input("Resting BP (mm Hg)", 80, 200, 130)
            chol       = st.number_input("Cholesterol (mg/dL)", 100, 600, 246)
            fbs        = st.selectbox("Fasting Blood Sugar > 120?",
                                      [(0,"No"),(1,"Yes")], format_func=lambda x: x[1])
            restecg    = st.selectbox("Resting ECG",
                                      [(0,"Normal"),(1,"ST-T Abnormality"),(2,"LV Hypertrophy")],
                                      format_func=lambda x: f"{x[0]} – {x[1]}")
        with c2:
            thalach    = st.number_input("Max Heart Rate", 60, 220, 149)
            exang      = st.selectbox("Exercise Angina",
                                      [(0,"No"),(1,"Yes")], format_func=lambda x: x[1], key="h_exang")
            oldpeak    = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1, format="%.1f")
            slope      = st.selectbox("ST Slope",
                                      [(0,"Upsloping"),(1,"Flat"),(2,"Downsloping")],
                                      format_func=lambda x: f"{x[0]} – {x[1]}")
            ca         = st.selectbox("Major Vessels (0–4)", [0,1,2,3,4])
            thal       = st.selectbox("Thal",
                                      [(0,"Normal"),(1,"Fixed Defect"),(2,"Reversible Defect")],
                                      format_func=lambda x: f"{x[0]} – {x[1]}")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn_h = st.button("🔍  Run Heart Disease Prediction", use_container_width=True)

    with col_result:
        st.markdown(f'<p style="font-size:.75rem; color:#3d5475; text-transform:uppercase; letter-spacing:.12em; margin-bottom:1rem">Reference Ranges</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#0f1929; border:1px solid #1e3050; border-radius:14px; padding:1.2rem 1.4rem;">
        {"".join(f'''
        <div style="display:flex; justify-content:space-between; padding:.4rem 0; border-bottom:1px solid #1e3050;">
          <span style="color:#7a93b8; font-size:.82rem">{f}</span>
          <span style="color:#e2eaf6; font-size:.82rem; font-family:'DM Mono',monospace">{v}</span>
        </div>''' for f, v in [
            ("Resting BP (normal)", "90–120 mm Hg"),
            ("Cholesterol (normal)", "< 200 mg/dL"),
            ("Cholesterol (high)", "≥ 240 mg/dL"),
            ("Max Heart Rate (est.)", "220 − age"),
            ("ST Depression", "< 1 mm (normal)"),
            ("Major Vessels", "0 = healthy"),
        ])}
        </div>
        """, unsafe_allow_html=True)

    if predict_btn_h:
        patient = dict(
            age=age_h, sex=sex[1], cp=cp[0], trestbps=trestbps,
            chol=chol, fbs=fbs[0], restecg=restecg[0],
            thalach=thalach, exang=exang[0], oldpeak=oldpeak,
            slope=slope[0], ca=ca, thal=thal[0]
        )
        with st.spinner("Analysing cardiac biomarkers…"):
            pred, prob = predict_heart(models, patient)
        render_result_card(pred, prob, "Heart Disease", "Heart Disease", "No Heart Disease", ACCENT)


# ═════════════════════════════════════════════════════════════════════════════
# PARKINSON'S PAGE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧠  Parkinson's":
    ACCENT = "#a78bfa"
    section_header("🧠", "Parkinson's Disease Prediction",
                   "Oxford Parkinson's Dataset · 22 voice features · Voting Ensemble", ACCENT)

    if not models_ok:
        st.error(f"❌ Could not load models: `{load_error}`")
        st.stop()

    with st.expander("ℹ️ About the voice biomarkers", expanded=False):
        st.markdown("""
        Voice recordings are processed to extract biomarkers across four categories:

        | Category | Features | Meaning |
        |----------|----------|---------|
        | **Fundamental Frequency** | Fo, Fhi, Flo | Average/max/min vocal frequency (Hz) |
        | **Jitter** | Jitter(%), Jitter(Abs), RAP, PPQ, DDP | Frequency variation cycle-to-cycle |
        | **Shimmer** | Shimmer, Shimmer(dB), APQ3, APQ5, APQ, DDA | Amplitude variation cycle-to-cycle |
        | **Nonlinear** | NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE | Noise/complexity measures |

        **Pipeline:** IQR Outlier Capping · Feature Engineering (8 interaction terms) · RobustScaler → Voting Ensemble
        """)

    # Tabs for feature groups
    tab1, tab2, tab3, tab4 = st.tabs(["🎵 Frequency", "📈 Jitter", "📊 Shimmer", "🔬 Nonlinear"])

    with tab1:
        st.markdown(f'<p style="color:#7a93b8; font-size:.82rem; margin:.5rem 0 1rem">Average, maximum and minimum vocal fundamental frequency (Hz)</p>', unsafe_allow_html=True)
        tc1, tc2, tc3 = st.columns(3)
        with tc1: fo  = st.number_input("MDVP:Fo(Hz)",  80.0, 300.0, 154.0, step=0.1, format="%.3f")
        with tc2: fhi = st.number_input("MDVP:Fhi(Hz)", 80.0, 600.0, 197.0, step=0.1, format="%.3f")
        with tc3: flo = st.number_input("MDVP:Flo(Hz)", 60.0, 240.0, 116.0, step=0.1, format="%.3f")

    with tab2:
        st.markdown(f'<p style="color:#7a93b8; font-size:.82rem; margin:.5rem 0 1rem">Measures of variation in fundamental frequency</p>', unsafe_allow_html=True)
        tj1, tj2 = st.columns(2)
        with tj1:
            jitter_pct = st.number_input("MDVP:Jitter(%)",   0.0, 0.1,  0.006,  step=0.0001, format="%.5f")
            jitter_abs = st.number_input("MDVP:Jitter(Abs)", 0.0, 0.001,0.00004,step=0.000001,format="%.6f")
            rap        = st.number_input("MDVP:RAP",          0.0, 0.05, 0.003, step=0.0001, format="%.5f")
        with tj2:
            ppq        = st.number_input("MDVP:PPQ",          0.0, 0.05, 0.003, step=0.0001, format="%.5f")
            ddp        = st.number_input("Jitter:DDP",         0.0, 0.15, 0.010, step=0.0001, format="%.5f")

    with tab3:
        st.markdown(f'<p style="color:#7a93b8; font-size:.82rem; margin:.5rem 0 1rem">Measures of variation in amplitude</p>', unsafe_allow_html=True)
        ts1, ts2 = st.columns(2)
        with ts1:
            shimmer     = st.number_input("MDVP:Shimmer",     0.0, 0.2,  0.030, step=0.0001, format="%.5f")
            shimmer_db  = st.number_input("MDVP:Shimmer(dB)", 0.0, 2.0,  0.282, step=0.001,  format="%.3f")
            apq3        = st.number_input("Shimmer:APQ3",      0.0, 0.1,  0.016, step=0.0001, format="%.5f")
        with ts2:
            apq5        = st.number_input("Shimmer:APQ5",      0.0, 0.15, 0.020, step=0.0001, format="%.5f")
            apq         = st.number_input("MDVP:APQ",          0.0, 0.2,  0.024, step=0.0001, format="%.5f")
            dda         = st.number_input("Shimmer:DDA",       0.0, 0.3,  0.047, step=0.0001, format="%.5f")

    with tab4:
        st.markdown(f'<p style="color:#7a93b8; font-size:.82rem; margin:.5rem 0 1rem">Nonlinear dynamical complexity measures</p>', unsafe_allow_html=True)
        tn1, tn2 = st.columns(2)
        with tn1:
            nhr     = st.number_input("NHR",     0.0, 0.5,  0.025,  step=0.0001, format="%.5f")
            hnr     = st.number_input("HNR",     0.0, 40.0, 21.9,   step=0.01,   format="%.3f")
            rpde    = st.number_input("RPDE",    0.0, 1.0,  0.499,  step=0.001,  format="%.4f")
            dfa     = st.number_input("DFA",     0.0, 1.0,  0.718,  step=0.001,  format="%.4f")
        with tn2:
            spread1 = st.number_input("spread1",-10.0,0.0, -5.684, step=0.001,  format="%.4f")
            spread2 = st.number_input("spread2",  0.0, 0.6, 0.227,  step=0.001,  format="%.4f")
            d2      = st.number_input("D2",       0.0, 5.0,  2.382, step=0.001,  format="%.4f")
            ppe     = st.number_input("PPE",      0.0, 0.6,  0.206, step=0.001,  format="%.4f")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn_p = st.button("🔍  Run Parkinson's Prediction", use_container_width=True)

    if predict_btn_p:
        patient = {
            'MDVP:Fo(Hz)': fo,   'MDVP:Fhi(Hz)': fhi, 'MDVP:Flo(Hz)': flo,
            'MDVP:Jitter(%)': jitter_pct, 'MDVP:Jitter(Abs)': jitter_abs,
            'MDVP:RAP': rap, 'MDVP:PPQ': ppq, 'Jitter:DDP': ddp,
            'MDVP:Shimmer': shimmer, 'MDVP:Shimmer(dB)': shimmer_db,
            'Shimmer:APQ3': apq3, 'Shimmer:APQ5': apq5,
            'MDVP:APQ': apq, 'Shimmer:DDA': dda,
            'NHR': nhr, 'HNR': hnr, 'RPDE': rpde, 'DFA': dfa,
            'spread1': spread1, 'spread2': spread2, 'D2': d2, 'PPE': ppe,
        }
        with st.spinner("Analysing voice biomarkers…"):
            pred, prob = predict_parkinsons(models, patient)
        render_result_card(pred, prob, "Parkinson's", "Parkinson's Positive", "Healthy", ACCENT)
