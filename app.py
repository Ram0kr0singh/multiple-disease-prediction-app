import os
import pickle
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# ── Theme synchronization component ───────────────────────────────────────────
_theme_sync_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theme_sync")
_theme_sync = components.declare_component("theme_sync", path=_theme_sync_dir)

# Run component to reactively receive theme from browser
_detected_theme = _theme_sync(key="_theme_sync_watcher", default=None)
if _detected_theme in ("light", "dark"):
    st.session_state["active_theme"] = _detected_theme

# ══════════════════════════════════════════════════════════════════════════════
# THEME HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_theme() -> str:
    """Return 'light' or 'dark' based on the active Streamlit theme."""
    t = st.session_state.get("active_theme")
    if t in ("light", "dark"):
        return t
    try:
        ctx_t = st.context.theme.type          # "light" | "dark" | None
        if ctx_t in ("light", "dark"):
            return ctx_t
    except Exception:
        pass
    return "dark"


# Per-mode palette — brand colours stay identical in both modes
PALETTE = {
    # ── DARK MODE ─────────────────────────────────────────────────────────────
    "dark": {
        # Core teal + caramel brand
        "primary":        "#4FB3BF",   # teal highlight (UI elements)
        "hero_end":       "#258090",   # darker teal for hero gradient end (WCAG)
        "primary_dark":   "#2C4F5E",   # deep teal (gradients start)
        "accent":         "#E09F55",   # caramel
        # Semantic
        "negative":       "#4CC38F",   # muted green  (healthy)
        "positive":       "#EF5B66",   # warm red  (at-risk)
        "gauge_amber":    "#E0A03A",
        "pos_text":       "#ffffff",   # text on positive banner (3.30:1 with EF5B66)
        "neg_text":       "#0A2A1A",  # text on negative banner (7.00:1 with 4CC38F)
        # Backgrounds
        "bg":             "#0F1A1F",
        "bg2":            "#17262D",
        "card_bg":        "#1B2B33",
        "card_border":    "rgba(79,179,191,.20)",
        "card_shadow":    "rgba(0,0,0,.40)",
        "card_h3":        "#E6EEF0",
        # Text
        "text":           "#E6EEF0",
        "step_bg":        "#1B2B33",
        "step_text":      "#E6EEF0",
        # Badges
        "badge_bg":       "rgba(79,179,191,.14)",
        "badge_border":   "rgba(79,179,191,.45)",
        "badge_text":     "#90DADE",
        # Divider
        "divider":        "rgba(79,179,191,.18)",
        # Plot
        "plot_font":      "#E6EEF0",
        "plot_grid":      "rgba(255,255,255,.10)",
        "plot_template":  "plotly_dark",
        "gauge_number":   "#E6EEF0",
        "gauge_title":    "#E6EEF0",
        "gauge_tick":     "#7DA8B0",
        "vline_zero":     "rgba(255,255,255,.25)",
        # Sidebar  (slightly deeper than bg2)
        "sidebar_bg":          "#0D1D23",
        "sidebar_card_bg":     "rgba(27, 43, 51, 0.72)",
        "sidebar_card_border": "rgba(79, 179, 191, 0.22)",
        "sidebar_text":        "#E6EEF0",
        "sidebar_muted":       "#7DA8B0",
        "sidebar_nav_text":    "#C8DEE2",
        "sidebar_nav_hover":   "rgba(79, 179, 191, 0.16)",
        "sidebar_chip_bg":     "rgba(255, 255, 255, 0.05)",
    },
    # ── LIGHT MODE ────────────────────────────────────────────────────────────
    "light": {
        # Core teal + caramel brand
        "primary":        "#1E6F7C",
        "hero_end":       "#1E6F7C",   # same as primary in light (already passes)
        "primary_dark":   "#164E58",
        "accent":         "#B9772E",
        # Semantic
        "negative":       "#3FA37A",
        "positive":       "#C13040",
        "gauge_amber":    "#E0A03A",
        "pos_text":       "#ffffff",   # white on #C13040 = 5.58:1 ✓
        "neg_text":       "#0F1A1F",  # dark on #3FA37A = 5.67:1 ✓
        # Backgrounds
        "bg":             "#FBF7F0",
        "bg2":            "#F1EADF",
        "card_bg":        "#FFFFFF",
        "card_border":    "rgba(30,111,124,.22)",
        "card_shadow":    "rgba(30,111,124,.10)",
        "card_h3":        "#1F2A30",
        # Text
        "text":           "#1F2A30",
        "step_bg":        "#F1EADF",
        "step_text":      "#1F2A30",
        # Badges
        "badge_bg":       "rgba(30,111,124,.10)",
        "badge_border":   "rgba(30,111,124,.35)",
        "badge_text":     "#1E6F7C",
        # Divider
        "divider":        "rgba(30,111,124,.18)",
        # Plot
        "plot_font":      "#1F2A30",
        "plot_grid":      "rgba(0,0,0,.08)",
        "plot_template":  "plotly_white",
        "gauge_number":   "#1F2A30",
        "gauge_title":    "#1F2A30",
        "gauge_tick":     "#4E7E88",
        "vline_zero":     "rgba(0,0,0,.30)",
        # Sidebar  (slightly deeper than bg2)
        "sidebar_bg":          "#EAE2D5",
        "sidebar_card_bg":     "rgba(255, 255, 255, 0.82)",
        "sidebar_card_border": "rgba(30, 111, 124, 0.20)",
        "sidebar_text":        "#1F2A30",
        "sidebar_muted":       "#4E7E88",
        "sidebar_nav_text":    "#1F2A30",
        "sidebar_nav_hover":   "rgba(30, 111, 124, 0.10)",
        "sidebar_chip_bg":     "rgba(30, 111, 124, 0.07)",
    },
}

# Brand colours — read from active PALETTE so they switch with theme
def _c(key):
    return PALETTE[get_theme()][key]

def C_YOU():   return _c("accent")      # caramel — "You" marker
def C_NEG():   return _c("negative")    # healthy green
def C_POS():   return _c("positive")    # at-risk red
def C_AMBER(): return _c("gauge_amber") # amber zone
PLY_PAPER = "rgba(0,0,0,0)"
PLY_PLOT  = "rgba(0,0,0,0)"


def style_fig(fig):
    """Apply the current theme to any Plotly figure in-place and return it."""
    p = PALETTE[get_theme()]
    fig.update_layout(
        template=p["plot_template"],
        paper_bgcolor=PLY_PAPER,
        plot_bgcolor=PLY_PLOT,
        font=dict(family="Inter, sans-serif", color=p["plot_font"]),
    )
    fig.update_xaxes(gridcolor=p["plot_grid"], zerolinecolor=p["plot_grid"])
    fig.update_yaxes(gridcolor=p["plot_grid"], zerolinecolor=p["plot_grid"])
    if fig.layout.legend:
        fig.update_layout(legend=dict(font=dict(color=p["plot_font"])))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CSS INJECTION
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    p = PALETTE[get_theme()]
    st.markdown(f"""
    <style>
    /* ── fonts & base ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

    /* ── gradient hero banner (same in both modes — white text on gradient always passes WCAG AA) ── */
    .hero-banner {{
        background: linear-gradient(135deg, {p["primary_dark"]} 0%, {p["hero_end"]} 100%);
        border-radius: 18px;
        padding: 2.4rem 2.8rem 2rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 32px rgba(30,111,124,0.28);
    }}
    .hero-banner h1 {{ color: #fff !important; font-size: 2.4rem; font-weight: 700; margin: 0 0 .4rem; }}
    .hero-banner p  {{ color: rgba(255,255,255,.92); font-size: 1.05rem; margin: 0; }}

    /* ── page section title ── */
    .section-title {{
        background: linear-gradient(90deg, {p["primary_dark"]}, {p["primary"]});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.55rem;
        font-weight: 700;
        margin: .4rem 0 .2rem;
    }}

    /* ── pill badges (outside hero banner) ── */
    .badge {{
        display: inline-block;
        background: {p['badge_bg']};
        border: 1px solid {p['badge_border']};
        color: {p['badge_text']};
        border-radius: 999px;
        padding: .25rem .85rem;
        font-size: .82rem;
        font-weight: 600;
        margin: .35rem .3rem .35rem 0;
    }}

    /* ── badges INSIDE the hero banner always white — hero is always dark ── */
    .hero-banner .badge {{
        background: rgba(255,255,255,0.20) !important;
        border: 1px solid rgba(255,255,255,0.60) !important;
        color: #ffffff !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.28);
    }}

    /* ── cards ── */
    .card {{
        background: {p['card_bg']};
        border: 1px solid {p['card_border']};
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 4px 20px {p['card_shadow']};
        height: 100%;
    }}
    .card h3 {{ font-size: 1.05rem; font-weight: 600; color: {p['card_h3']}; margin: 0 0 .5rem; }}

    /* ── result banners (white text on saturated gradient — always WCAG AA) ── */
    .result-positive {{
        background: {p["positive"]};
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        color: {p["pos_text"]};
        font-size: 1.15rem;
        font-weight: 600;
        box-shadow: 0 6px 24px rgba(193,48,64,.30);
        margin-bottom: 1rem;
    }}
    .result-negative {{
        background: {p["negative"]};
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        color: {p["neg_text"]};
        font-size: 1.15rem;
        font-weight: 600;
        box-shadow: 0 6px 24px rgba(63,163,122,.28);
        margin-bottom: 1rem;
    }}
    .result-positive small, .result-negative small {{
        display: block;
        font-size: .82rem;
        font-weight: 400;
        opacity: .88;
        margin-top: .4rem;
        color: inherit;
    }}

    /* ── steps ── */
    .step-num {{
        width: 2.2rem; height: 2.2rem;
        background: linear-gradient(135deg,{p["primary_dark"]},{p["primary"]});
        color: #fff;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 1rem;
        margin: 0 auto .5rem;
    }}
    .step-box {{
        background: {p['step_bg']};
        border: 1px solid {p['card_border']};
        border-radius: 12px;
        padding: .9rem 1rem;
        text-align: center;
    }}
    .step-box p {{ margin: 0; font-size: .9rem; color: {p['step_text']}; }}

    /* ── buttons ── */
    .stButton > button {{
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: transform .12s, box-shadow .12s !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(30,111,124,.40) !important;
    }}

    /* ── metric delta ── */
    [data-testid="stMetricDelta"] {{ font-size: .78rem !important; }}

    /* ── divider ── */
    hr {{ border-color: {p['divider']} !important; }}

    /* ── modern sidebar styling ── */
    [data-testid="stSidebar"] {{
        background-color: {p['sidebar_bg']} !important;
        border-right: 1px solid {p['card_border']} !important;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.2rem !important;
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
    }}
    [data-testid="stSidebar"] ::-webkit-scrollbar {{
        width: 5px;
    }}
    [data-testid="stSidebar"] ::-webkit-scrollbar-track {{
        background: transparent;
    }}
    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb {{
        background: {p['card_border']};
        border-radius: 10px;
    }}
    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {{
        background: {p["primary"]};
    }}

    /* ── sidebar brand card ── */
    .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 0.85rem 0.95rem;
        background: linear-gradient(135deg, rgba(30,111,124,0.18) 0%, rgba(79,179,191,0.12) 100%);
        border: 1px solid {p['sidebar_card_border']};
        border-radius: 14px;
        box-shadow: 0 4px 18px {p['card_shadow']};
        backdrop-filter: blur(10px);
        margin-bottom: 0.75rem;
    }}
    .sidebar-brand-icon {{
        width: 42px;
        height: 42px;
        border-radius: 11px;
        background: linear-gradient(135deg, {p["primary_dark"]} 0%, {p["hero_end"]} 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
        box-shadow: 0 4px 12px rgba(30,111,124,0.40);
        flex-shrink: 0;
    }}
    .sidebar-brand-text {{
        display: flex;
        flex-direction: column;
    }}
    .sidebar-brand-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {p['card_h3']};
        letter-spacing: -0.01em;
        line-height: 1.2;
    }}
    .sidebar-brand-sub {{
        font-size: 0.74rem;
        color: {p['sidebar_muted']};
        font-weight: 500;
        margin-top: 2px;
    }}

    /* ── live status indicator ── */
    .sidebar-status {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: {p['sidebar_card_bg']};
        border: 1px solid {p['sidebar_card_border']};
        border-radius: 10px;
        padding: 0.45rem 0.75rem;
        margin-bottom: 1.1rem;
        font-size: 0.76rem;
        font-weight: 600;
        color: {p['sidebar_muted']};
    }}
    .status-pulse-group {{
        display: flex;
        align-items: center;
        gap: 7px;
    }}
    .status-dot-pulse {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: {p["negative"]};
        position: relative;
    }}
    .status-dot-pulse::after {{
        content: '';
        position: absolute;
        top: -2px; left: -2px; right: -2px; bottom: -2px;
        border-radius: 50%;
        border: 2px solid {p["negative"]};
        animation: pulse-ring 1.8s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
    }}
    @keyframes pulse-ring {{
        0%   {{ transform: scale(0.6); opacity: 1; }}
        100% {{ transform: scale(2.0); opacity: 0; }}
    }}
    .status-latency {{
        background: {p['sidebar_chip_bg']};
        border: 1px solid {p['sidebar_card_border']};
        border-radius: 6px;
        padding: 1px 6px;
        font-size: 0.68rem;
        color: {p["negative"]};
        font-weight: 700;
    }}

    /* ── section header in sidebar ── */
    .sidebar-sec-label {{
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {p['sidebar_muted']};
        margin: 0.9rem 0 0.4rem 0.25rem;
    }}

    /* ── model cards in sidebar ── */
    .sidebar-model-item {{
        background: {p['sidebar_card_bg']};
        border: 1px solid {p['sidebar_card_border']};
        border-radius: 10px;
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.4rem;
        transition: transform 0.14s ease, border-color 0.14s ease;
    }}
    .sidebar-model-item:hover {{
        transform: translateX(3px);
        border-color: {p["primary"]};
    }}
    .model-item-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.82rem;
        font-weight: 600;
        color: {p['card_h3']};
    }}
    .model-item-pill {{
        background: rgba(63, 163, 122, 0.16);
        color: {p["negative"]};
        border-radius: 6px;
        padding: 1px 6px;
        font-size: 0.7rem;
        font-weight: 700;
    }}
    .model-item-sub {{
        font-size: 0.69rem;
        color: {p['sidebar_muted']};
        margin-top: 2px;
    }}

    /* ── sidebar footer info ── */
    .sidebar-info-box {{
        background: {p['sidebar_card_bg']};
        border: 1px solid {p['sidebar_card_border']};
        border-radius: 10px;
        padding: 0.65rem 0.8rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.72rem;
        color: {p['sidebar_muted']};
    }}
    .sidebar-info-header {{
        display: flex;
        justify-content: space-between;
        font-weight: 600;
        color: {p['card_h3']};
        margin-bottom: 3px;
    }}

    /* ── hide theme sync component frame ── */
    [data-testid="stCustomComponentV1"]:has(iframe[src*="theme_sync"]),
    iframe[src*="theme_sync"] {{
        position: fixed !important;
        top: -9999px !important;
        left: -9999px !important;
        width: 1px !important;
        height: 1px !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
# Brand colours and PLY_PAPER/PLY_PLOT are defined in the THEME HELPERS block above.
# C_YOU, C_NEG, C_POS, C_AMBER, PLY_PAPER, PLY_PLOT are available globally.

DISCLAIMER = ("⚠️ **Educational demo built for a hackathon. "
              "Not a medical diagnosis. Consult a doctor.**")

# ══════════════════════════════════════════════════════════════════════════════
# HELP DICTIONARY
# ══════════════════════════════════════════════════════════════════════════════
HELP = {
    "diab_pregnancies": (
        "**What it is:** Number of times the patient has been pregnant.\n\n"
        "**Usual range:** 0–10.\n\n"
        "**In the dataset:** 0–17 (median 3)."
    ),
    "diab_glucose": (
        "**What it is:** Plasma glucose concentration 2 hours after an oral "
        "glucose tolerance test (mg/dL).\n\n"
        "**Usual range:** typically under 140 mg/dL; 140–199 is the prediabetes range.\n\n"
        "**In the dataset:** 0–199 (median 117). Note: 0 = not measured."
    ),
    "diab_bp": (
        "**What it is:** Diastolic blood pressure (mm Hg).\n\n"
        "**Usual range:** typically 60–80 mm Hg; 90+ is usually high.\n\n"
        "**In the dataset:** 0–122 mm Hg (median 72). Note: 0 = not measured."
    ),
    "diab_skin": (
        "**What it is:** Triceps skin-fold thickness (mm).\n\n"
        "**Usual range:** typically 10–40 mm.\n\n"
        "**In the dataset:** 0–99 mm (median 23). Note: 0 = not measured."
    ),
    "diab_insulin": (
        "**What it is:** 2-hour serum insulin (µU/mL).\n\n"
        "**Usual range:** typically 16–166 µU/mL.\n\n"
        "**In the dataset:** 0–846 µU/mL (median 30.5). Note: 0 = not measured."
    ),
    "diab_bmi": (
        "**What it is:** Body Mass Index (weight kg / height m²).\n\n"
        "**Usual range:** 18.5–24.9 normal; 25–29.9 overweight; 30+ obese.\n\n"
        "**In the dataset:** 0–67.1 (median 32). Note: 0 = not measured."
    ),
    "diab_dpf": (
        "**What it is:** Diabetes Pedigree Function — family history score.\n\n"
        "**Usual range:** typically 0.1–1.0.\n\n"
        "**In the dataset:** 0.078–2.42 (median 0.37)."
    ),
    "diab_age": (
        "**What it is:** Age in years.\n\n"
        "**In the dataset:** 21–81 (median 29)."
    ),
    "h_age": (
        "**What it is:** Age in years.\n\n"
        "**In the dataset:** 29–77 (median 55)."
    ),
    "h_sex": "**Options:** 0 = Female, 1 = Male.",
    "h_cp": (
        "**What it is:** Type of chest pain.\n\n"
        "**Options:**\n- 0 = Typical angina\n- 1 = Atypical angina\n"
        "- 2 = Non-anginal pain\n- 3 = Asymptomatic"
    ),
    "h_trestbps": (
        "**What it is:** Resting systolic blood pressure (mm Hg).\n\n"
        "**Usual range:** under 120 mm Hg normal; 130+ high.\n\n"
        "**In the dataset:** 94–200 mm Hg (median 130)."
    ),
    "h_chol": (
        "**What it is:** Serum cholesterol (mg/dL).\n\n"
        "**Usual range:** under 200 desirable; 240+ high.\n\n"
        "**In the dataset:** 126–564 mg/dL (median 240)."
    ),
    "h_fbs": "**Options:** 0 = No (≤ 120 mg/dL), 1 = Yes (> 120 mg/dL).",
    "h_restecg": (
        "**Options:**\n- 0 = Normal\n- 1 = ST-T wave abnormality\n"
        "- 2 = Probable LV hypertrophy"
    ),
    "h_thalach": (
        "**What it is:** Max heart rate reached during the exercise test (bpm).\n\n"
        "**Usual range:** roughly 220 minus age.\n\n"
        "**In the dataset:** 71–202 bpm (median 153)."
    ),
    "h_exang": "**Options:** 0 = No, 1 = Yes (angina triggered by exercise).",
    "h_oldpeak": (
        "**What it is:** ST-segment depression during exercise vs rest.\n\n"
        "**Usual range:** 0 is normal; higher is more notable.\n\n"
        "**In the dataset:** 0–6.2 (median 0.8)."
    ),
    "h_slope": "**Options:**\n- 0 = Upsloping\n- 1 = Flat\n- 2 = Downsloping",
    "h_ca": (
        "**What it is:** Major vessels coloured by fluoroscopy (0–4).\n\n"
        "**In the dataset:** 0–4 (median 0)."
    ),
    "h_thal": (
        "**Options:**\n- 0 = Normal\n- 1 = Fixed defect\n"
        "- 2 = Reversible defect\n- 3 = Other\n\n"
        "**In the dataset:** 0–3 (median 2)."
    ),
    "p_fo":          "**What it is:** Voice — average vocal pitch (Hz). Dataset 88–260 (median 149).",
    "p_fhi":         "**What it is:** Voice — highest vocal pitch (Hz). Dataset 102–592 (median 176).",
    "p_flo":         "**What it is:** Voice — lowest vocal pitch (Hz). Dataset 65–239 (median 104).",
    "p_jitter_pct":  "**What it is:** Cycle-to-cycle pitch variation (%). Dataset 0.0017–0.033 (median 0.0049).",
    "p_jitter_abs":  "**What it is:** Same variation in seconds. Dataset 0.00001–0.00026 (median 0.00003).",
    "p_rap":         "**What it is:** Short-term pitch variation, 3-cycle avg. Dataset 0.0007–0.021 (median 0.0025).",
    "p_ppq":         "**What it is:** Pitch variation over 5 cycles. Dataset 0.0009–0.020 (median 0.0027).",
    "p_ddp":         "**What it is:** Avg diff of pitch differences. Dataset 0.002–0.064 (median 0.0075).",
    "p_shimmer":     "**What it is:** Cycle-to-cycle loudness variation. Dataset 0.0095–0.119 (median 0.023).",
    "p_shimmer_db":  "**What it is:** Loudness variation (dB). Dataset 0.085–1.30 (median 0.22).",
    "p_apq3":        "**What it is:** Loudness variation, 3-cycle avg. Dataset 0.0046–0.056 (median 0.013).",
    "p_apq5":        "**What it is:** Loudness variation, 5-cycle avg. Dataset 0.0057–0.079 (median 0.013).",
    "p_apq":         "**What it is:** Loudness variation, 11-cycle avg. Dataset 0.0072–0.138 (median 0.018).",
    "p_dda":         "**What it is:** Avg diff of loudness differences. Dataset 0.014–0.169 (median 0.038).",
    "p_nhr":         "**What it is:** Noise-to-harmonics ratio; higher = noisier. Dataset 0.0007–0.315 (median 0.012).",
    "p_hnr":         "**What it is:** Harmonics-to-noise ratio; higher = clearer. Dataset 8.4–33.0 (median 22.1).",
    "p_rpde":        "**What it is:** Recurrence period density entropy (irregularity). Dataset 0.26–0.69 (median 0.50).",
    "p_dfa":         "**What it is:** Signal fractal scaling exponent (complexity). Dataset 0.57–0.83 (median 0.72).",
    "p_spread1":     "**What it is:** Nonlinear pitch variation (negative). Dataset −7.96 to −2.43 (median −5.72).",
    "p_spread2":     "**What it is:** Nonlinear pitch variation 2. Dataset 0.006–0.45 (median 0.22).",
    "p_d2":          "**What it is:** Correlation dimension (signal complexity). Dataset 1.42–3.67 (median 2.36).",
    "p_ppe":         "**What it is:** Pitch period entropy (irregularity). Dataset 0.045–0.53 (median 0.19).",
}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL & DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
_base = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model(name):
    with open(os.path.join(_base, 'saved_models', name), 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_datasets():
    d = pd.read_csv(os.path.join(_base, 'dataset', 'diabetes.csv'))
    h = pd.read_csv(os.path.join(_base, 'dataset', 'heart.csv'), encoding='utf-8-sig')
    p = pd.read_csv(os.path.join(_base, 'dataset', 'parkinsons.csv'))
    if 'name' in p.columns:
        p = p.drop(columns=['name'])
    return d, h, p

diabetes_model      = load_model('diabetes_model.sav')
heart_disease_model = load_model('heart_disease_model.sav')
parkinsons_model    = load_model('parkinsons_model.sav')

df_diab, df_heart, df_park = load_datasets()

# Feature / target column names
DIAB_FEAT  = [c for c in df_diab.columns  if c != 'Outcome']
HEART_FEAT = [c for c in df_heart.columns if c != 'target']
PARK_FEAT  = [c for c in df_park.columns  if c != 'status']
# Parkinsons target column is 'status'; PPE is the 22nd feature passed to the model

# Pre-compute dataset means per class (used for contribution chart)
diab_means  = df_diab[DIAB_FEAT].mean()
heart_means = df_heart[HEART_FEAT].mean()
park_means  = df_park[PARK_FEAT].mean()

# ── Helpers ───────────────────────────────────────────────────────────────────
def to_floats(values):
    try:
        return [float(str(v).strip()) for v in values]
    except ValueError:
        st.error("Please fill in every field with a valid number.")
        st.stop()

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def model_score(model, X_arr, use_proba=False):
    """Return 0-100 confidence/probability score."""
    if use_proba:
        return float(model.predict_proba([X_arr])[0][1]) * 100
    else:
        df_val = float(model.decision_function([X_arr])[0])
        return sigmoid(df_val) * 100

# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS  (all theme-aware via style_fig())
# ══════════════════════════════════════════════════════════════════════════════

def class_donut_fig(neg_count, pos_count, title=""):
    p = PALETTE[get_theme()]
    fig = go.Figure(go.Pie(
        labels=["Negative", "Positive"],
        values=[neg_count, pos_count],
        hole=0.62,
        marker_colors=[C_NEG(), C_POS()],
        textinfo="percent",
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=p["plot_font"]), x=0.5),
        showlegend=True,
        legend=dict(font=dict(size=11, color=p["plot_font"])),
        margin=dict(l=10, r=10, t=40, b=10),
        height=230,
    )
    return style_fig(fig)


def accuracy_bar_fig():
    p = PALETTE[get_theme()]
    models = ["Diabetes\n(SVM)", "Heart\n(LogReg)", "Parkinson's\n(SVM)"]
    accs   = [77, 80, 87]
    colors = [C_YOU(), C_POS(), C_NEG()]
    fig = go.Figure(go.Bar(
        x=models, y=accs,
        marker_color=colors,
        text=[f"{a}%" for a in accs],
        textposition="outside",
        textfont=dict(color=p["plot_font"]),
        hovertemplate="%{x}: %{y}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Test Accuracy Comparison", font=dict(color=p["plot_font"])),
        yaxis=dict(range=[0, 100], title="Accuracy (%)"),
        xaxis=dict(tickfont=dict(size=11)),
        bargap=0.4,
        margin=dict(l=20, r=20, t=50, b=20),
        height=300,
    )
    return style_fig(fig)


def hist_fig(df, feature, target_col, user_val, neg_label="Negative", pos_label="Positive"):
    """Overlapping histogram; vertical dashed line = user's current value."""
    p = PALETTE[get_theme()]
    neg = df[df[target_col] == 0][feature]
    pos = df[df[target_col] == 1][feature]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=neg, name=neg_label, opacity=0.58,
        marker_color=C_NEG(), nbinsx=30,
        hovertemplate=f"{neg_label}: %{{x}}<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=pos, name=pos_label, opacity=0.58,
        marker_color=C_POS(), nbinsx=30,
        hovertemplate=f"{pos_label}: %{{x}}<extra></extra>",
    ))
    fig.add_vline(x=user_val, line_dash="dash", line_color=C_YOU(), line_width=2,
                  annotation_text="You", annotation_position="top right",
                  annotation_font_color=C_YOU())
    fig.update_layout(
        barmode="overlay",
        title=dict(text=f"Distribution of {feature}", font=dict(size=14, color=p["plot_font"])),
        xaxis_title=feature,
        yaxis_title="Count",
        legend=dict(orientation="h", y=1.12, font=dict(color=p["plot_font"])),
        margin=dict(l=20, r=20, t=60, b=30),
        height=310,
    )
    return style_fig(fig)


def gauge_fig(score, label):
    p = PALETTE[get_theme()]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 1),
        number=dict(suffix=" %", font=dict(size=36, color=p["gauge_number"])),
        title=dict(text=label, font=dict(size=14, color=p["gauge_title"])),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1,
                      tickcolor=p["gauge_tick"],
                      tickfont=dict(color=p["gauge_tick"])),
            bar=dict(color=C_YOU(), thickness=0.28),
            bgcolor="rgba(0,0,0,0)",
            steps=[
                dict(range=[0,  40], color="rgba(76,195,143,.22)"),   # negative / green
                dict(range=[40, 60], color="rgba(224,160,58,.22)"),   # amber
                dict(range=[60,100], color="rgba(239,91,102,.22)"),   # positive / red
            ],
            threshold=dict(line=dict(color=C_YOU(), width=3),
                           thickness=0.78, value=score),
        ),
    ))
    fig.update_layout(margin=dict(l=30, r=30, t=60, b=20), height=280)
    return style_fig(fig)


def contrib_fig(coef, x_vals, means, feat_names, n_top=6):
    """Horizontal bar chart: top-N feature contributions."""
    p = PALETTE[get_theme()]
    contribs   = coef * (np.array(x_vals) - np.array(means))
    idx_sorted = np.argsort(np.abs(contribs))[::-1][:n_top]
    vals   = contribs[idx_sorted]
    labels = [feat_names[i] for i in idx_sorted]
    colors = [C_POS() if v > 0 else C_NEG() for v in vals]
    fig = go.Figure(go.Bar(
        x=vals[::-1], y=labels[::-1],
        orientation="h",
        marker_color=colors[::-1],
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=p["vline_zero"], line_width=1.5)
    fig.update_layout(
        title=dict(text="Feature Contributions (vs dataset average)",
                   font=dict(color=p["plot_font"])),
        xaxis_title="Contribution",
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
    )
    return style_fig(fig)


def compare_fig(user_vals, neg_means, pos_means, feat_names, raw_user):
    """Normalised grouped bar: you vs negative avg vs positive avg."""
    p = PALETTE[get_theme()]
    df_all = pd.DataFrame({
        "Feature": feat_names * 3,
        "Value":   list(user_vals) + list(neg_means) + list(pos_means),
        "Group":   (["You"] * len(feat_names) +
                    ["Avg Negative"] * len(feat_names) +
                    ["Avg Positive"] * len(feat_names)),
        "RawUser": raw_user * 3,
    })
    color_map = {"You": C_YOU(), "Avg Negative": C_NEG(), "Avg Positive": C_POS()}
    fig = px.bar(df_all, x="Feature", y="Value", color="Group",
                 barmode="group",
                 color_discrete_map=color_map,
                 template=p["plot_template"])
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Normalised: %{y:.3f}<extra></extra>"
    )
    fig.update_layout(
        title=dict(text="Your values vs dataset averages (normalised 0–1)",
                   font=dict(color=p["plot_font"])),
        yaxis_title="Normalised value (0–1)",
        legend=dict(orientation="h", y=1.14, font=dict(color=p["plot_font"])),
        paper_bgcolor=PLY_PAPER,
        plot_bgcolor=PLY_PLOT,
        margin=dict(l=20, r=20, t=60, b=60),
        height=360,
        xaxis_tickangle=-30,
    )
    return style_fig(fig)


def normalise(user_val, col_min, col_max):
    rng = col_max - col_min
    if rng == 0:
        return 0.0
    return float(np.clip((user_val - col_min) / rng, 0, 1))


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA & CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════
DIABETES_POS = dict(diab_pregnancies=6, diab_glucose=148, diab_bp=72,
                    diab_skin=35, diab_insulin=0, diab_bmi=33.6,
                    diab_dpf=0.627, diab_age=50)
DIABETES_NEG = dict(diab_pregnancies=4, diab_glucose=110, diab_bp=92,
                    diab_skin=0, diab_insulin=0, diab_bmi=37.6,
                    diab_dpf=0.191, diab_age=30)

HEART_POS = dict(h_age=63, h_sex=1, h_cp=3, h_trestbps=145, h_chol=233,
                 h_fbs=1, h_restecg=0, h_thalach=150, h_exang=0,
                 h_oldpeak=2.3, h_slope=0, h_ca=0, h_thal=1)
HEART_NEG = dict(h_age=63, h_sex=1, h_cp=0, h_trestbps=130, h_chol=254,
                 h_fbs=0, h_restecg=0, h_thalach=147, h_exang=0,
                 h_oldpeak=1.4, h_slope=1, h_ca=1, h_thal=3)

PARK_POS = dict(p_fo=119.992, p_fhi=157.302, p_flo=74.997,
                p_jitter_pct=0.00784, p_jitter_abs=0.00007,
                p_rap=0.0037, p_ppq=0.00554, p_ddp=0.01109,
                p_shimmer=0.04374, p_shimmer_db=0.426,
                p_apq3=0.02182, p_apq5=0.0313, p_apq=0.02971,
                p_dda=0.06545, p_nhr=0.02211, p_hnr=21.033,
                p_rpde=0.414783, p_dfa=0.815285,
                p_spread1=-4.813031, p_spread2=0.266482,
                p_d2=2.301442, p_ppe=0.284654)
PARK_NEG = dict(p_fo=202.266, p_fhi=211.604, p_flo=197.079,
                p_jitter_pct=0.0018, p_jitter_abs=0.000009,
                p_rap=0.00093, p_ppq=0.00107, p_ddp=0.00278,
                p_shimmer=0.00954, p_shimmer_db=0.085,
                p_apq3=0.00469, p_apq5=0.00606, p_apq=0.00719,
                p_dda=0.01407, p_nhr=0.00072, p_hnr=32.684,
                p_rpde=0.368535, p_dfa=0.742133,
                p_spread1=-7.695734, p_spread2=0.17854,
                p_d2=1.544609, p_ppe=0.056141)

def _clear(prefix):
    for k in list(st.session_state.keys()):
        if k.startswith(prefix + "_result"):
            del st.session_state[k]

def load_diabetes_pos():
    st.session_state.update(DIABETES_POS)
def load_diabetes_neg():
    st.session_state.update(DIABETES_NEG)
def load_heart_pos():
    st.session_state.update(HEART_POS)
def load_heart_neg():
    st.session_state.update(HEART_NEG)
def load_park_pos():
    st.session_state.update(PARK_POS)
def load_park_neg():
    st.session_state.update(PARK_NEG)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    _curr_theme = get_theme()
    _p = PALETTE[_curr_theme]

    # Brand header card
    st.markdown('''
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🩺</div>
        <div class="sidebar-brand-text">
            <span class="sidebar-brand-title">MedPredict AI</span>
            <span class="sidebar-brand-sub">Health Risk Demo</span>
        </div>
    </div>
    <div class="sidebar-status">
        <div class="status-pulse-group">
            <span class="status-dot-pulse"></span>
            <span>3 ML Models Ready</span>
        </div>
    </div>
    <div class="sidebar-sec-label">NAVIGATION</div>
    ''', unsafe_allow_html=True)

    _nav_styles = {
        "container": {
            "padding": "0 !important",
            "background-color": "transparent",
            "border": "none",
        },
        "icon": {
            "font-size": "1.05rem",
            "margin-right": "0.55rem",
        },
        "nav-link": {
            "font-size": "0.90rem",
            "text-align": "left",
            "margin": "4px 0",
            "padding": "0.62rem 0.9rem",
            "border-radius": "10px",
            "color": _p["sidebar_nav_text"],
            "font-weight": "500",
            "--hover-color": _p["sidebar_nav_hover"],
            "transition": "all 0.15s ease-in-out",
            "border": "1px solid transparent",
        },
        "nav-link-selected": {
            "background": f"linear-gradient(135deg, {_p['primary_dark']} 0%, {_p['primary']} 100%)",
            "color": "#FFFFFF !important",
            "font-weight": "600",
            "box-shadow": f"0 4px 16px rgba(30, 111, 124, 0.42)",
            "border": "1px solid rgba(255, 255, 255, 0.18)",
        },
    }

    selected = option_menu(
        menu_title=None,
        options=['Home', 'Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['house-fill', 'droplet-fill', 'heart-pulse-fill', 'activity'],
        default_index=0,
        styles=_nav_styles,
    )

    st.markdown('''
    <div class="sidebar-sec-label">ACTIVE MODELS</div>
    <div class="sidebar-model-item">
        <div class="model-item-row">
            <span>🩸 Diabetes</span>
            <span class="model-item-pill">77% Acc</span>
        </div>
        <div class="model-item-sub">SVM • Linear Kernel</div>
    </div>
    <div class="sidebar-model-item">
        <div class="model-item-row">
            <span>❤️ Heart Disease</span>
            <span class="model-item-pill">80% Acc</span>
        </div>
        <div class="model-item-sub">Logistic Regression</div>
    </div>
    <div class="sidebar-model-item">
        <div class="model-item-row">
            <span>🧠 Parkinson's</span>
            <span class="model-item-pill">87% Acc</span>
        </div>
        <div class="model-item-sub">SVM • Linear Kernel</div>
    </div>

    <div class="sidebar-info-box">
        <div class="sidebar-info-header">
            <span>🛡️ Privacy-First AI</span>
            <span>⚡ Instant</span>
        </div>
        <div>In-memory inference. Zero telemetry or patient data retention.</div>
    </div>
    ''', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if selected == 'Home':
    # Hero
    st.markdown("""
    <div class="hero-banner">
      <h1>🧑‍⚕️ Multiple Disease Prediction</h1>
      <p>AI-powered screening for Diabetes, Heart Disease &amp; Parkinson's — for educational purposes.</p>
      <div style="margin-top:.8rem">
        <span class="badge">🤖 3 ML models</span>
        <span class="badge">⚡ Instant results</span>
        <span class="badge">🎓 Educational demo</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-title">Models &amp; Datasets</p>', unsafe_allow_html=True)

    d_neg = int((df_diab['Outcome'] == 0).sum())
    d_pos = int((df_diab['Outcome'] == 1).sum())
    h_neg = int((df_heart['target'] == 0).sum())
    h_pos = int((df_heart['target'] == 1).sum())
    p_neg = int((df_park['status'] == 0).sum())
    p_pos = int((df_park['status'] == 1).sum())

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**🩸 Diabetes (SVM)**")
            st.metric("Test accuracy", "77%")
            st.plotly_chart(class_donut_fig(d_neg, d_pos, "Class balance"),
                            width="stretch")
    with c2:
        with st.container(border=True):
            st.markdown("**❤️ Heart Disease (Logistic Regression)**")
            st.metric("Test accuracy", "80%")
            st.plotly_chart(class_donut_fig(h_neg, h_pos, "Class balance"),
                            width="stretch")
    with c3:
        with st.container(border=True):
            st.markdown("**🧠 Parkinson's (SVM)**")
            st.metric("Test accuracy", "87%")
            st.plotly_chart(class_donut_fig(p_neg, p_pos, "Class balance"),
                            width="stretch")

    st.divider()
    st.markdown('<p class="section-title">Accuracy Comparison</p>', unsafe_allow_html=True)
    st.plotly_chart(accuracy_bar_fig(), width="stretch")

    st.divider()
    st.markdown('<p class="section-title">How it works</p>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    for col, num, txt in [(s1, "1", "Enter patient values or click **Load sample** on any page"),
                          (s2, "2", "Click the **Test Result** button"),
                          (s3, "3", "See the result, a confidence gauge, and why the model decided")]:
        with col:
            st.markdown(f"""
            <div class="step-box">
              <div class="step-num">{num}</div>
              <p>{txt}</p>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.caption(DISCLAIMER)

# ══════════════════════════════════════════════════════════════════════════════
# DIABETES PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif selected == 'Diabetes Prediction':
    st.markdown('<p class="section-title">🩸 Diabetes Prediction using ML</p>',
                unsafe_allow_html=True)
    st.caption("Hover over the (?) next to any field to see what it means and its usual range.")
    st.info("ℹ️ In this dataset, **0** for Glucose, Blood Pressure, Skin Thickness, "
            "Insulin or BMI means the value was not measured.")

    # Sample buttons
    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("🔴 Load sample: likely positive", key="diab_load_pos",
              on_click=load_diabetes_pos)
    b2.button("🟢 Load sample: likely negative", key="diab_load_neg",
              on_click=load_diabetes_neg)
    st.divider()

    for key, default in {
        'diab_pregnancies': 0,
        'diab_glucose': 0,
        'diab_bp': 0,
        'diab_skin': 0,
        'diab_insulin': 0,
        'diab_bmi': 0.0,
        'diab_dpf': 0.0,
        'diab_age': 25,
    }.items():
        st.session_state.setdefault(key, default)

    # Inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input(
            'Number of Pregnancies', min_value=0, max_value=20, step=1,
            key='diab_pregnancies', help=HELP["diab_pregnancies"])
    with col2:
        Glucose = st.number_input(
            'Plasma Glucose Concentration (mg/dL)',
            min_value=0, max_value=300, step=1,
            key='diab_glucose', help=HELP["diab_glucose"])
    with col3:
        BloodPressure = st.number_input(
            'Diastolic Blood Pressure (mm Hg)',
            min_value=0, max_value=200, step=1,
            key='diab_bp', help=HELP["diab_bp"])
    with col1:
        SkinThickness = st.number_input(
            'Triceps Skin Fold Thickness (mm)',
            min_value=0, max_value=100, step=1,
            key='diab_skin', help=HELP["diab_skin"])
    with col2:
        Insulin = st.number_input(
            '2-Hour Serum Insulin (µU/mL)',
            min_value=0, max_value=900, step=1,
            key='diab_insulin', help=HELP["diab_insulin"])
    with col3:
        BMI = st.number_input(
            'Body Mass Index (BMI)',
            min_value=0.0, max_value=70.0, step=0.1, format="%.1f",
            key='diab_bmi', help=HELP["diab_bmi"])
    with col1:
        DiabetesPedigreeFunction = st.number_input(
            'Diabetes Pedigree Function',
            min_value=0.0, max_value=3.0, step=0.001, format="%.3f",
            key='diab_dpf', help=HELP["diab_dpf"])
    with col2:
        Age = st.number_input(
            'Age (years)', min_value=1, max_value=120, step=1,
            key='diab_age', help=HELP["diab_age"])

    user_vals_diab = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigreeFunction, Age]

    # Dataset explorer
    with st.expander("📊 Explore the dataset", expanded=True):
        default_feat = "Glucose"
        feat_choice_d = st.selectbox("Feature to explore", DIAB_FEAT,
                                     index=DIAB_FEAT.index(default_feat),
                                     key="diab_hist_feat")
        feat_idx_d = DIAB_FEAT.index(feat_choice_d)
        st.plotly_chart(
            hist_fig(df_diab, feat_choice_d, 'Outcome',
                     user_vals_diab[feat_idx_d],
                     neg_label="No Diabetes", pos_label="Diabetes"),
            width="stretch")

    st.divider()

    if st.button('🔍 Diabetes Test Result'):
        ui = to_floats(user_vals_diab)
        with st.spinner("Running model…"):
            pred = int(diabetes_model.predict([ui])[0])
            score = model_score(diabetes_model, ui, use_proba=False)
        st.session_state['diab_result_pred']  = pred
        st.session_state['diab_result_score'] = score
        st.session_state['diab_result_input'] = ui

    # Show results (persist across reruns)
    if 'diab_result_pred' in st.session_state:
        pred  = st.session_state['diab_result_pred']
        score = st.session_state['diab_result_score']
        ui    = st.session_state['diab_result_input']

        if pred == 1:
            st.markdown("""<div class="result-positive">
              🔴 The model predicts a higher likelihood of <strong>Diabetes</strong>.
              <small>Not a medical diagnosis — please consult a doctor.</small>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="result-negative">
              🟢 The model predicts a <strong>lower likelihood</strong> of Diabetes.
              <small>Not a medical diagnosis — please consult a doctor.</small>
            </div>""", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📈 Result", "🔍 Why this result", "📊 You vs dataset"])

        with tab1:
            g1, g2 = st.columns([1.6, 1])
            with g1:
                st.plotly_chart(gauge_fig(score, "Model confidence score"),
                                width="stretch")
                st.caption("Indicative score derived from the model's decision function, not a medical probability.")
            with g2:
                st.metric("Prediction",  "Positive 🔴" if pred else "Negative 🟢")
                st.metric("Confidence",  f"{score:.1f} %")
                st.metric("Features entered", f"{len([v for v in ui if v != 0])} / {len(ui)}")

        with tab2:
            coef = diabetes_model.coef_[0]
            st.plotly_chart(contrib_fig(coef, ui, diab_means.values, DIAB_FEAT),
                            width="stretch")
            st.caption("Bars show how far each of your values pushes the result compared "
                       "to the average person in the training data. Red = toward positive, green = toward negative.")

        with tab3:
            feats_d = DIAB_FEAT
            neg_avg = df_diab[df_diab['Outcome']==0][feats_d].mean().values
            pos_avg = df_diab[df_diab['Outcome']==1][feats_d].mean().values
            col_min = df_diab[feats_d].min().values
            col_max = df_diab[feats_d].max().values
            u_norm  = [normalise(v, col_min[i], col_max[i]) for i, v in enumerate(ui)]
            n_norm  = [normalise(v, col_min[i], col_max[i]) for i, v in enumerate(neg_avg)]
            p_norm  = [normalise(v, col_min[i], col_max[i]) for i, v in enumerate(pos_avg)]
            st.plotly_chart(compare_fig(u_norm, n_norm, p_norm, feats_d, ui),
                            width="stretch")
            st.caption("Values are scaled 0–1 within the dataset min/max range "
                       "so features with different units can be compared side by side.")

    st.divider()
    st.caption(DISCLAIMER)

# ══════════════════════════════════════════════════════════════════════════════
# HEART DISEASE PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif selected == 'Heart Disease Prediction':
    st.markdown('<p class="section-title">❤️ Heart Disease Prediction using ML</p>',
                unsafe_allow_html=True)
    st.caption("Hover over the (?) next to any field to see what it means and its usual range.")

    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("🔴 Load sample: likely positive", key="heart_load_pos",
              on_click=load_heart_pos)
    b2.button("🟢 Load sample: likely negative", key="heart_load_neg",
              on_click=load_heart_neg)
    st.divider()

    col1, col2, col3 = st.columns(3)

    SEX_OPTS     = [0, 1];      SEX_LABELS   = ["0 – Female", "1 – Male"]
    CP_OPTS      = [0,1,2,3];   CP_LABELS    = ["0 – Typical Angina","1 – Atypical Angina",
                                                 "2 – Non-Anginal Pain","3 – Asymptomatic"]
    FBS_OPTS     = [0,1];       FBS_LABELS   = ["0 – No (≤ 120 mg/dL)","1 – Yes (> 120 mg/dL)"]
    ECG_OPTS     = [0,1,2];     ECG_LABELS   = ["0 – Normal","1 – ST-T Abnormality","2 – LV Hypertrophy"]
    EXANG_OPTS   = [0,1];       EXANG_LABELS = ["0 – No","1 – Yes"]
    SLOPE_OPTS   = [0,1,2];     SLOPE_LABELS = ["0 – Upsloping","1 – Flat","2 – Downsloping"]
    CA_OPTS      = [0,1,2,3,4]; CA_LABELS    = ["0","1","2","3","4"]
    THAL_OPTS    = [0,1,2,3];   THAL_LABELS  = ["0 – Normal","1 – Fixed Defect",
                                                 "2 – Reversable Defect","3 – Other"]

    for key, default in {
        'h_age': 40,
        'h_sex': 0,
        'h_cp': 0,
        'h_trestbps': 120,
        'h_chol': 200,
        'h_fbs': 0,
        'h_restecg': 0,
        'h_thalach': 150,
        'h_exang': 0,
        'h_oldpeak': 0.0,
        'h_slope': 0,
        'h_ca': 0,
        'h_thal': 0,
    }.items():
        st.session_state.setdefault(key, default)

    with col1:
        age = st.number_input('Age (years)', min_value=1, max_value=120, step=1,
            key='h_age', help=HELP["h_age"])
    with col2:
        sex = st.selectbox('Sex', SEX_OPTS,
            format_func=lambda value: SEX_LABELS[value],
            key='h_sex', help=HELP["h_sex"])
    with col3:
        cp = st.selectbox('Chest Pain Type', CP_OPTS,
            format_func=lambda value: CP_LABELS[value],
            key='h_cp', help=HELP["h_cp"])
    with col1:
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)',
            min_value=50, max_value=250, step=1,
            key='h_trestbps', help=HELP["h_trestbps"])
    with col2:
        chol = st.number_input('Serum Cholesterol (mg/dL)',
            min_value=50, max_value=700, step=1,
            key='h_chol', help=HELP["h_chol"])
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', FBS_OPTS,
            format_func=lambda value: FBS_LABELS[value],
            key='h_fbs', help=HELP["h_fbs"])
    with col1:
        restecg = st.selectbox('Resting ECG Results', ECG_OPTS,
            format_func=lambda value: ECG_LABELS[value],
            key='h_restecg', help=HELP["h_restecg"])
    with col2:
        thalach = st.number_input('Max Heart Rate Achieved (bpm)',
            min_value=50, max_value=250, step=1,
            key='h_thalach', help=HELP["h_thalach"])
    with col3:
        exang = st.selectbox('Exercise Induced Angina', EXANG_OPTS,
            format_func=lambda value: EXANG_LABELS[value],
            key='h_exang', help=HELP["h_exang"])
    with col1:
        oldpeak = st.number_input('ST Depression (Oldpeak)',
            min_value=0.0, max_value=10.0, step=0.1, format="%.1f",
            key='h_oldpeak', help=HELP["h_oldpeak"])
    with col2:
        slope = st.selectbox('Slope of Peak Exercise ST Segment', SLOPE_OPTS,
            format_func=lambda value: SLOPE_LABELS[value],
            key='h_slope', help=HELP["h_slope"])
    with col3:
        ca = st.selectbox('Major Vessels Colored by Fluoroscopy', CA_OPTS,
            format_func=lambda value: CA_LABELS[value],
            key='h_ca', help=HELP["h_ca"])
    with col1:
        thal = st.selectbox('Thalassemia', THAL_OPTS,
            format_func=lambda value: THAL_LABELS[value],
            key='h_thal', help=HELP["h_thal"])

    user_vals_heart = [age, sex, cp, trestbps, chol, fbs, restecg,
                       thalach, exang, oldpeak, slope, ca, thal]

    # Dataset explorer
    with st.expander("📊 Explore the dataset", expanded=True):
        feat_choice_h = st.selectbox("Feature to explore", HEART_FEAT,
                                     index=HEART_FEAT.index("thalach"),
                                     key="heart_hist_feat")
        feat_idx_h = HEART_FEAT.index(feat_choice_h)
        st.plotly_chart(
            hist_fig(df_heart, feat_choice_h, 'target',
                     user_vals_heart[feat_idx_h],
                     neg_label="No Heart Disease", pos_label="Heart Disease"),
            width="stretch")

    st.divider()

    if st.button('🔍 Heart Disease Test Result'):
        ui = to_floats(user_vals_heart)
        with st.spinner("Running model…"):
            pred  = int(heart_disease_model.predict([ui])[0])
            score = model_score(heart_disease_model, ui, use_proba=True)
        st.session_state['heart_result_pred']  = pred
        st.session_state['heart_result_score'] = score
        st.session_state['heart_result_input'] = ui

    if 'heart_result_pred' in st.session_state:
        pred  = st.session_state['heart_result_pred']
        score = st.session_state['heart_result_score']
        ui    = st.session_state['heart_result_input']

        if pred == 1:
            st.markdown("""<div class="result-positive">
              🔴 The model predicts a higher likelihood of <strong>Heart Disease</strong>.
              <small>Not a medical diagnosis — please consult a doctor.</small>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="result-negative">
              🟢 The model predicts a <strong>lower likelihood</strong> of Heart Disease.
              <small>Not a medical diagnosis — please consult a doctor.</small>
            </div>""", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📈 Result", "🔍 Why this result", "📊 You vs dataset"])

        with tab1:
            g1, g2 = st.columns([1.6, 1])
            with g1:
                st.plotly_chart(gauge_fig(score, "Model probability"), width="stretch")
            with g2:
                st.metric("Prediction",  "Positive 🔴" if pred else "Negative 🟢")
                st.metric("Probability", f"{score:.1f} %")
                st.metric("Features", str(len(ui)))

        with tab2:
            coef = heart_disease_model.coef_[0]
            st.plotly_chart(contrib_fig(coef, ui, heart_means.values, HEART_FEAT),
                            width="stretch")
            st.caption("Bars show how far each of your values pushes the result compared "
                       "to the average person in the training data.")

        with tab3:
            cmp_feats = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
            cmp_idx   = [HEART_FEAT.index(f) for f in cmp_feats]
            neg_avg = df_heart[df_heart['target']==0][cmp_feats].mean().values
            pos_avg = df_heart[df_heart['target']==1][cmp_feats].mean().values
            col_min = df_heart[cmp_feats].min().values
            col_max = df_heart[cmp_feats].max().values
            ui_cmp  = [ui[i] for i in cmp_idx]
            u_norm  = [normalise(v, col_min[j], col_max[j]) for j, v in enumerate(ui_cmp)]
            n_norm  = [normalise(v, col_min[j], col_max[j]) for j, v in enumerate(neg_avg)]
            p_norm  = [normalise(v, col_min[j], col_max[j]) for j, v in enumerate(pos_avg)]
            st.plotly_chart(compare_fig(u_norm, n_norm, p_norm, cmp_feats, ui_cmp),
                            width="stretch")
            st.caption("Values are scaled 0–1 within the dataset min/max range.")

    st.divider()
    st.caption(DISCLAIMER)

# ══════════════════════════════════════════════════════════════════════════════
# PARKINSON'S PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Parkinsons Prediction":
    st.markdown('<p class="section-title">🧠 Parkinson\'s Disease Prediction using ML</p>',
                unsafe_allow_html=True)
    st.caption("Hover over the (?) next to any field to see what it means and its usual range.")
    st.info("🎤 These values come from voice-analysis software. "
            "For a demo, use the **Load sample** buttons.")

    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("🔴 Load sample: likely positive", key="park_load_pos",
              on_click=load_park_pos)
    b2.button("🟢 Load sample: likely negative", key="park_load_neg",
              on_click=load_park_neg)
    st.divider()

    col1, col2, col3 = st.columns(3)

    def pnum(label, key, default, step=0.000001, fmt="%.6f", mn=-1e9, mx=1e9):
        st.session_state.setdefault(key, default)
        return st.number_input(label, min_value=mn, max_value=mx, step=step,
                               format=fmt, key=key,
                               help=HELP[key])

    with col1:
        fo  = pnum("MDVP: Fo – Average Vocal Fundamental Frequency (Hz)",
                   'p_fo', 0.0, step=0.001, fmt="%.3f", mn=0.0, mx=300.0)
    with col2:
        fhi = pnum("MDVP: Fhi – Maximum Fundamental Frequency (Hz)",
                   'p_fhi', 0.0, step=0.001, fmt="%.3f", mn=0.0, mx=600.0)
    with col3:
        flo = pnum("MDVP: Flo – Minimum Fundamental Frequency (Hz)",
                   'p_flo', 0.0, step=0.001, fmt="%.3f", mn=0.0, mx=300.0)

    with col1:
        Jitter_percent = pnum("MDVP: Jitter (%)", 'p_jitter_pct', 0.0,
                              step=0.00001, fmt="%.5f", mn=0.0, mx=1.0)
    with col2:
        Jitter_Abs = pnum("MDVP: Jitter (Abs)", 'p_jitter_abs', 0.0,
                          step=0.000001, fmt="%.6f", mn=0.0, mx=0.01)
    with col3:
        RAP = pnum("MDVP: RAP – Relative Avg Perturbation", 'p_rap', 0.0,
                   step=0.00001, fmt="%.5f", mn=0.0, mx=1.0)

    with col1:
        PPQ = pnum("MDVP: PPQ – Five-Point Period Perturbation Quotient",
                   'p_ppq', 0.0, step=0.00001, fmt="%.5f", mn=0.0, mx=1.0)
    with col2:
        DDP = pnum("Jitter: DDP", 'p_ddp', 0.0,
                   step=0.00001, fmt="%.5f", mn=0.0, mx=1.0)
    with col3:
        Shimmer = pnum("MDVP: Shimmer – Amplitude Variation", 'p_shimmer', 0.0,
                       step=0.00001, fmt="%.5f", mn=0.0, mx=1.0)

    with col1:
        Shimmer_dB = pnum("MDVP: Shimmer (dB)", 'p_shimmer_db', 0.0,
                          step=0.001, fmt="%.3f", mn=0.0, mx=3.0)
    with col2:
        APQ3 = pnum("Shimmer: APQ3 – 3-Point Amplitude Perturbation Quotient",
                    'p_apq3', 0.0, step=0.00001, fmt="%.5f", mn=0.0, mx=1.0)
    with col3:
        APQ5 = pnum("Shimmer: APQ5 – 5-Point Amplitude Perturbation Quotient",
                    'p_apq5', 0.0, step=0.00001, fmt="%.5f", mn=0.0, mx=1.0)

    with col1:
        APQ = pnum("MDVP: APQ – Amplitude Perturbation Quotient", 'p_apq', 0.0,
                   step=0.00001, fmt="%.5f", mn=0.0, mx=1.0)
    with col2:
        DDA = pnum("Shimmer: DDA", 'p_dda', 0.0,
                   step=0.00001, fmt="%.5f", mn=0.0, mx=1.0)
    with col3:
        NHR = pnum("NHR – Noise-to-Harmonics Ratio", 'p_nhr', 0.0,
                   step=0.00001, fmt="%.5f", mn=0.0, mx=1.0)

    with col1:
        HNR = pnum("HNR – Harmonics-to-Noise Ratio", 'p_hnr', 0.0,
                   step=0.001, fmt="%.3f", mn=0.0, mx=50.0)
    with col2:
        RPDE = pnum("RPDE – Recurrence Period Density Entropy", 'p_rpde', 0.0,
                    step=0.000001, fmt="%.6f", mn=0.0, mx=1.0)
    with col3:
        DFA = pnum("DFA – Detrended Fluctuation Analysis", 'p_dfa', 0.0,
                   step=0.000001, fmt="%.6f", mn=0.0, mx=1.0)

    with col1:
        spread1 = pnum("spread1 – Nonlinear Fundamental Frequency Variation",
                       'p_spread1', 0.0, step=0.000001, fmt="%.6f", mn=-10.0, mx=0.0)
    with col2:
        spread2 = pnum("spread2 – Nonlinear Fundamental Frequency Variation 2",
                       'p_spread2', 0.0, step=0.000001, fmt="%.6f", mn=0.0, mx=1.0)
    with col3:
        D2 = pnum("D2 – Correlation Dimension", 'p_d2', 0.0,
                  step=0.000001, fmt="%.6f", mn=0.0, mx=5.0)

    with col1:
        PPE = pnum("PPE – Pitch Period Entropy", 'p_ppe', 0.0,
                   step=0.000001, fmt="%.6f", mn=0.0, mx=1.0)

    user_vals_park = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                      APQ3, APQ5, APQ, DDA, NHR, HNR,
                      RPDE, DFA, spread1, spread2, D2, PPE]

    # Dataset explorer
    with st.expander("📊 Explore the dataset", expanded=True):
        feat_choice_p = st.selectbox("Feature to explore", PARK_FEAT,
                                     index=PARK_FEAT.index("HNR"),
                                     key="park_hist_feat")
        feat_idx_p = PARK_FEAT.index(feat_choice_p)
        st.plotly_chart(
            hist_fig(df_park, feat_choice_p, 'status',
                     user_vals_park[feat_idx_p],
                     neg_label="No Parkinson's", pos_label="Parkinson's"),
            width="stretch")

    st.divider()

    if st.button("🔍 Parkinson's Test Result"):
        ui = to_floats(user_vals_park)
        with st.spinner("Running model…"):
            pred  = int(parkinsons_model.predict([ui])[0])
            score = model_score(parkinsons_model, ui, use_proba=False)
        st.session_state['park_result_pred']  = pred
        st.session_state['park_result_score'] = score
        st.session_state['park_result_input'] = ui

    if 'park_result_pred' in st.session_state:
        pred  = st.session_state['park_result_pred']
        score = st.session_state['park_result_score']
        ui    = st.session_state['park_result_input']

        if pred == 1:
            st.markdown("""<div class="result-positive">
              🔴 The model predicts a higher likelihood of <strong>Parkinson's Disease</strong>.
              <small>Not a medical diagnosis — please consult a doctor.</small>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="result-negative">
              🟢 The model predicts a <strong>lower likelihood</strong> of Parkinson's Disease.
              <small>Not a medical diagnosis — please consult a doctor.</small>
            </div>""", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📈 Result", "🔍 Why this result", "📊 You vs dataset"])

        with tab1:
            g1, g2 = st.columns([1.6, 1])
            with g1:
                st.plotly_chart(gauge_fig(score, "Model confidence score"),
                                width="stretch")
                st.caption("Indicative score derived from the model's decision function, not a medical probability.")
            with g2:
                st.metric("Prediction", "Positive 🔴" if pred else "Negative 🟢")
                st.metric("Confidence", f"{score:.1f} %")
                st.metric("Features",   str(len(ui)))

        with tab2:
            coef = parkinsons_model.coef_[0]
            st.plotly_chart(contrib_fig(coef, ui, park_means.values, PARK_FEAT),
                            width="stretch")
            st.caption("Bars show how far each of your values pushes the result compared "
                       "to the average person in the training data.")

        with tab3:
            # Pick top-8 features by absolute coefficient magnitude
            coef = parkinsons_model.coef_[0]
            top8_idx = np.argsort(np.abs(coef))[::-1][:8]
            cmp_feats_p = [PARK_FEAT[i] for i in top8_idx]
            neg_avg = df_park[df_park['status']==0][cmp_feats_p].mean().values
            pos_avg = df_park[df_park['status']==1][cmp_feats_p].mean().values
            col_min = df_park[cmp_feats_p].min().values
            col_max = df_park[cmp_feats_p].max().values
            ui_cmp  = [ui[i] for i in top8_idx]
            u_norm  = [normalise(v, col_min[j], col_max[j]) for j, v in enumerate(ui_cmp)]
            n_norm  = [normalise(v, col_min[j], col_max[j]) for j, v in enumerate(neg_avg)]
            p_norm  = [normalise(v, col_min[j], col_max[j]) for j, v in enumerate(pos_avg)]
            st.plotly_chart(compare_fig(u_norm, n_norm, p_norm, cmp_feats_p, ui_cmp),
                            width="stretch")
            st.caption("Values are scaled 0–1 within the dataset min/max range. "
                       "Top 8 features by model coefficient magnitude shown.")

    st.divider()
    st.caption(DISCLAIMER)
