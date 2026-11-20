
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detection_engine import detect_domain

st.set_page_config(
    page_title="DNS GUARD | Threat Intelligence",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# SESSION STATE
# ==================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "domain_input" not in st.session_state:
    st.session_state.domain_input = ""

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

# ==================================================
# PREMIUM DARK THEME
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --background: #0b1118;
    --surface: #111c27;
    --surface-light: #162431;
    --border: #263746;
    --text: #e7eef4;
    --muted: #91a3b2;
    --accent: #43a99a;
    --accent-light: #65c5b5;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--background);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #0e1822;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -0.5px;
}

h1 {
    font-size: 2rem !important;
}

h2 {
    font-size: 1.35rem !important;
}

p, label {
    color: #b0bec9;
}

hr {
    border-color: var(--border);
}

[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
    min-height: 110px;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif;
}

.stTextInput input {
    background: #101b25 !important;
    border: 1px solid #35495a !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    min-height: 46px;
}

.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

.stButton button,
.stDownloadButton button {
    background: #226d65;
    color: #ffffff;
    border: 1px solid #32877d;
    border-radius: 8px;
    min-height: 40px;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background: #2d8177;
    color: white;
    border-color: #55b6a8;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 9px;
}

[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
}

.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 11px;
    padding: 20px;
    margin-bottom: 14px;
}

.panel-label {
    color: var(--muted);
    font-size: 0.73rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

.panel-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text);
    margin-top: 8px;
}

.eyebrow {
    color: var(--accent-light);
    font-size: 0.7rem;
    letter-spacing: 2px;
    font-weight: 700;
}

.muted-text {
    color: var(--muted);
    font-size: 0.87rem;
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #48b69c;
    border-radius: 50%;
    margin-right: 7px;
}

div[data-testid="stAlert"] {
    border-radius: 9px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("""
    <div style="padding: 8px 0 24px 0;">
        <div style="display:flex; align-items:center; gap:11px;">
            <div style="
                background:#173c3b;
                color:#70c6b7;
                width:42px;
                height:42px;
                border-radius:10px;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:21px;
                font-weight:700;">
                D
            </div>
            <div>
                <div style="font-size:22px;font-weight:700;
                color:#edf4f8;letter-spacing:1px;">
                    DNS GUARD
                </div>
                <div style="font-size:11px;color:#8295a5;">
                    THREAT INTELLIGENCE
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<div class="eyebrow">WORKSPACE</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Domain Analysis",
            "Detection History",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        '<div class="eyebrow">SYSTEM INFORMATION</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="padding-top:14px;">
        <div class="muted-text">
            <span class="status-dot"></span>
            Detection engine loaded
        </div>
        <br>
        <div class="panel-label">MODEL</div>
        <b style="color:#e7eef4;">XGBoost V5</b>
        <br><br>
        <div class="panel-label">FEATURES</div>
        <b style="color:#e7eef4;">24 input features</b>
        <br><br>
        <div class="panel-label">CLASSIFICATIONS</div>
        <b style="color:#e7eef4;">3 categories</b>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="font-size:11px;color:#788b9a;line-height:1.8;">
        DNS GUARD<br>
        Research Prototype<br>
        Version 1.0.0
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# SHARED DATA
# ==================================================

history = st.session_state.history

total = len(history)

benign_count = sum(
    item["prediction"] == "Benign" for item in history
)

dga_count = sum(
    item["prediction"] == "DGA" for item in history
)

tunnel_count = sum(
    item["prediction"] == "DNS Tunneling" for item in history
)

# ==================================================
# TOP HEADER
# ==================================================

st.markdown("""
<div style="margin-bottom:28px;">
    <div class="eyebrow">
        SECURITY OPERATIONS / DOMAIN INTELLIGENCE
    </div>
    <div class="muted-text" style="margin-top:7px;">
        DNS threat classification and domain analysis platform
    </div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# OVERVIEW
# ==================================================

if page == "Overview":

    st.title("Security Overview")

    st.markdown(
        '<p class="muted-text">Monitor domain analysis activity and classification results.</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Analyses", total)
    c2.metric("Benign Domains", benign_count)
    c3.metric("DGA Domains", dga_count)
    c4.metric("DNS Tunneling", tunnel_count)

    st.divider()

    left, right = st.columns([1.4, 1])

    with left:

        st.subheader("Detection Distribution")

        if total > 0:

            chart = pd.DataFrame({
                "Category": ["Benign", "DGA", "DNS Tunneling"],
                "Analyses": [
                    benign_count,
                    dga_count,
                    tunnel_count
                ]
            })

            st.bar_chart(
                chart.set_index("Category"),
                color="#43a99a"
            )

        else:
            st.info(
                "No analysis data is available yet. "
                "Visit Domain Analysis to begin."
            )

    with right:

        st.subheader("Model Information")

        st.markdown("""
        <div class="panel">
            <div class="panel-label">CLASSIFICATION ENGINE</div>
            <div class="panel-value">XGBoost</div>
            <br>
            <div class="panel-label">MODEL VERSION</div>
            <b>V5</b>
            <br><br>
            <div class="panel-label">INPUT FEATURES</div>
            <b>24 lexical and structural features</b>
            <br><br>
            <div class="panel-label">OUTPUT CLASSES</div>
            <b>Benign · DGA · DNS Tunneling</b>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Recent Activity")

    if history:

        recent = pd.DataFrame(history).tail(5)

        st.dataframe(
            recent[[
                "timestamp",
                "domain",
                "prediction",
                "tunneling_score"
            ]],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("Recent domain activity will appear here.")

# ==================================================
# DOMAIN ANALYSIS
# ==================================================

elif page == "Domain Analysis":

    st.title("Domain Analysis")

    st.markdown(
        '<p class="muted-text">Submit a domain to examine its lexical and structural characteristics.</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="panel">
        <div class="panel-label">ANALYSIS ENGINE</div>
        <div class="panel-value">Domain Classification</div>
        <div class="muted-text" style="margin-top:7px;">
            Enter a domain name or load one of the example samples.
        </div>
    </div>
    """, unsafe_allow_html=True)

    samples = {
        "Benign example": "google.com",
        "DGA example": "skqgowoggookqigy.org",
        "DNS tunneling example": (
            "3eabapdnggydixon2aaaedrlp55zvgcamdmoehp2zp313hh4j5cibdeylkmqih."
            "yddcaeir2gls.securitytesting.online"
        )
    }

    selected_sample = st.selectbox(
        "Example domains",
        ["Select an example"] + list(samples.keys())
    )

    if st.button("Load Example"):

        if selected_sample != "Select an example":
            st.session_state.domain_input = samples[selected_sample]

    domain = st.text_input(
        "Domain name",
        key="domain_input",
        placeholder="example.com"
    )

    if st.button(
        "Run Analysis",
        type="primary",
        use_container_width=True
    ):

        if not domain.strip():

            st.warning("Please enter a domain name.")

        else:

            try:

                result = detect_domain(domain.strip())

                record = {
                    "timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "domain": result["domain"],
                    "prediction": result["prediction"],
                    "tunneling_score": result["tunneling_score"],
                    "benign_score": result["probabilities"]["Benign"],
                    "dga_score": result["probabilities"]["DGA"],
                    "tunneling_probability": (
                        result["probabilities"]["DNS Tunneling"]
                    )
                }

                st.session_state.history.append(record)
                st.session_state.latest_result = result

            except Exception as error:

                st.error(f"Analysis failed: {error}")

    result = st.session_state.latest_result

    if result:

        st.divider()

        st.subheader("Analysis Result")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("""
            <div class="panel">
                <div class="panel-label">ANALYZED DOMAIN</div>
            </div>
            """, unsafe_allow_html=True)

            st.code(result["domain"])

        with col2:

            st.markdown(f"""
            <div class="panel">
                <div class="panel-label">PREDICTED CATEGORY</div>
                <div class="panel-value">
                    {result["prediction"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Classification Scores")

        for category, score in result["probabilities"].items():

            st.write(
                f"**{category}** · {score * 100:.2f}%"
            )

            st.progress(float(score))

        st.caption(
            "Scores are model outputs, not calibrated probabilities "
            "or proof of malicious activity."
        )

# ==================================================
# DETECTION HISTORY
# ==================================================

elif page == "Detection History":

    st.title("Detection History")

    st.markdown(
        '<p class="muted-text">Review and export domain analysis records from this session.</p>',
        unsafe_allow_html=True
    )

    if history:

        df = pd.DataFrame(history)

        search = st.text_input(
            "Search domains or categories",
            placeholder="Search..."
        )

        if search:

            mask = (
                df["domain"].str.contains(
                    search, case=False, na=False
                )
                | df["prediction"].str.contains(
                    search, case=False, na=False
                )
            )

            df = df[mask]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Export Analysis History",
            data=csv_data,
            file_name="dns_detection_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):

            st.session_state.history = []
            st.session_state.latest_result = None

            st.rerun()

    else:

        st.info("No domain analysis records are available.")

# ==================================================
# ABOUT PROJECT
# ==================================================

elif page == "About Project":

    st.title("About DNS GUARD")

    st.markdown("""
    <p class="muted-text">
        DNS Tunneling & Exfiltration Detector
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-label">PROJECT OVERVIEW</div>
        <div class="panel-value">DNS Threat Classification</div>
        <p class="muted-text">
            DNS GUARD is a machine learning-based cybersecurity
            research prototype designed to classify domain names
            into benign, DGA and DNS tunneling categories.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Technology Stack")

    tech = pd.DataFrame({
        "Component": [
            "Programming Language",
            "Machine Learning",
            "Data Processing",
            "Dashboard",
            "Feature Extraction"
        ],
        "Technology": [
            "Python",
            "XGBoost",
            "Pandas and NumPy",
            "Streamlit",
            "Custom Python Module"
        ]
    })

    st.dataframe(
        tech,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Detection Workflow")

    st.markdown("""
    1. User submits a domain name.
    2. The feature extraction module calculates 24 features.
    3. The trained XGBoost model generates classification scores.
    4. The detection engine applies the tunneling threshold.
    5. The dashboard displays the prediction.
    """)

    st.subheader("Current Limitations")

    st.markdown("""
    - Analyzes domain names only.
    - Does not inspect live DNS traffic or packets.
    - Does not monitor query frequency or network behaviour.
    - Performance is affected by differences between training
      and test datasets.
    - Predictions are experimental and require further validation.
    """)

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:15px 0 20px 0;">
    <div style="font-size:12px;color:#91a3b2;">
        DNS GUARD · THREAT INTELLIGENCE
    </div>
    <div style="font-size:11px;color:#657889;margin-top:6px;">
        XGBoost V5 · 24 Features · Research Prototype
    </div>
</div>
""", unsafe_allow_html=True)
