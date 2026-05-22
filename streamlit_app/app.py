"""
LEAF Credit Agent — Streamlit Explainability Dashboard
Sprint 1: L0 through L3

Run with: streamlit run streamlit_app/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

from models.schemas import LoanApplication, JurisdictionCode
from pipeline import run_sprint1_pipeline
from storage.ledger import retrieve_application_ledger

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LEAF Credit Agent",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .leaf-header { font-size: 13px; font-weight: 600; color: #0F6E56; text-transform: uppercase;
                   letter-spacing: 0.05em; margin-bottom: 4px; }
    .xai-note { background: #F0FAF5; border-left: 3px solid #0F6E56; padding: 10px 14px;
                border-radius: 4px; font-size: 13px; color: #444; margin: 8px 0; }
    .layer-badge { display: inline-block; background: #E1F5EE; color: #0F6E56;
                   padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
    .metric-positive { color: #0F6E56; font-weight: 600; }
    .metric-negative { color: #E24B4A; font-weight: 600; }
    .metric-neutral { color: #854F0B; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar — Application input form
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 LEAF Credit Agent")
    st.markdown("*Layered Explainability for AI Finance*")
    st.divider()

    st.markdown("### Loan Application")

    applicant_name = st.text_input("Applicant Name", value="Suresh Kumar")
    amount = st.number_input("Loan Amount (₹)", min_value=50000, max_value=5000000,
                              value=450000, step=10000)
    purpose = st.selectbox("Purpose", ["Home renovation", "Education",
                                        "Business expansion", "Medical emergency",
                                        "Vehicle purchase", "Personal"])
    tenure = st.slider("Tenure (months)", 12, 84, 48, step=12)
    income = st.number_input("Monthly Income (₹)", min_value=15000, max_value=500000,
                              value=72400, step=1000)
    employment = st.selectbox("Employment Type", ["salaried", "self_employed", "business"])
    existing_loans = st.slider("Existing Active Loans", 0, 5, 2)
    age = st.slider("Applicant Age", 21, 65, 34)

    st.divider()
    st.markdown("### LEAF Settings")
    jurisdiction = st.selectbox("Jurisdiction", ["India", "USA", "EU"])
    jurisdiction_map = {"India": JurisdictionCode.INDIA,
                        "USA": JurisdictionCode.USA,
                        "EU": JurisdictionCode.EU}

    run_button = st.button("▶ Run LEAF Pipeline", type="primary", use_container_width=True)


# ─────────────────────────────────────────────
# Main — run pipeline on button click
# ─────────────────────────────────────────────
if run_button:
    application = LoanApplication(
        applicant_id=f"CUST-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        amount_requested=float(amount),
        purpose=purpose,
        tenure_months=tenure,
        applicant_name=applicant_name,
        applicant_age=age,
        employment_type=employment,
        monthly_income_declared=float(income),
        existing_loans=existing_loans,
    )

    with st.spinner("Running LEAF pipeline..."):
        results = run_sprint1_pipeline(
            application,
            jurisdiction=jurisdiction_map[jurisdiction],
            verbose=False
        )
    st.session_state["results"] = results
    st.session_state["application"] = application


# ─────────────────────────────────────────────
# Display results
# ─────────────────────────────────────────────
if "results" not in st.session_state:
    st.markdown("## Welcome to the LEAF Credit Agent")
    st.markdown("""
    This application demonstrates **Layered Explainability in AI Finance (LEAF)** —
    a framework that makes every step of a credit decision visible, traceable, and auditable.

    **Sprint 1 covers Layers L0 through L3:**
    - **L0** — Request & Context: locks the regulatory envelope
    - **L1** — Data Provenance: tracks every data source with integrity hashes
    - **L2** — Grounding Check: scores how trustworthy each source is
    - **L3** — Signal Extraction: computes model features with full source traceability

    👈 Fill in the loan application on the left and click **Run LEAF Pipeline** to see
    explainability at every layer.
    """)
    st.stop()


results = st.session_state["results"]
application = st.session_state["application"]
l0 = results["L0"]
l1 = results["L1"]
l2 = results["L2"]
l3 = results["L3"]

st.markdown(f"## 🌿 LEAF Explainability Dashboard")
st.markdown(f"Application `{l0.application_id}` · {l0.timestamp.strftime('%d %b %Y, %H:%M')}")
st.divider()

# ─── Layer tabs ──────────────────────────────
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "L0 — Context", "L1 — Provenance", "L2 — Grounding",
    "L3 — Signals", "📋 Evidence Ledger"
])


# ══════════════════════════════════════════════
# L0 Tab
# ══════════════════════════════════════════════
with tab0:
    st.markdown('<div class="layer-badge">L0 — Request & Context</div>', unsafe_allow_html=True)
    st.markdown("### What was requested, by whom, under which rules?")

    col1, col2, col3 = st.columns(3)
    col1.metric("Application ID", l0.application_id[-8:])
    col2.metric("Jurisdiction", l0.jurisdiction.value)
    col3.metric("Adverse Notice Required", "Yes ✓" if l0.adverse_action_notice_required else "No")

    st.markdown("#### Regulatory Frameworks Applied")
    for fw in l0.regulatory_frameworks:
        st.success(f"✓ {fw.value}")

    st.markdown("#### Application Details")
    df = pd.DataFrame([
        ("Applicant ID", application.applicant_id),
        ("Loan Amount", f"₹{application.amount_requested:,.0f}"),
        ("Purpose", application.purpose),
        ("Tenure", f"{application.tenure_months} months"),
        ("Employment", application.employment_type.title()),
        ("Declared Income", f"₹{application.monthly_income_declared:,.0f}/month"),
        ("Existing Loans", str(application.existing_loans)),
    ], columns=["Field", "Value"])
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown(
        f'<div class="xai-note">🔍 <b>Explainability note:</b> {l0.xai_note}</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════
# L1 Tab
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="layer-badge">L1 — Data Provenance</div>', unsafe_allow_html=True)
    st.markdown("### Every data source — logged, hashed, timestamped")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sources", l1.total_sources)
    col2.metric("Verified", l1.verified_sources)
    col3.metric("With Warnings", l1.sources_with_warnings,
                delta="⚠ Review needed" if l1.sources_with_warnings > 0 else None,
                delta_color="inverse")

    st.markdown("#### Provenance Certificate")
    for src in l1.sources:
        with st.expander(
            f"{'⚠ ' if src.freshness_warning else '✓ '}{src.source_name}  —  hash: `{src.integrity_hash}`",
            expanded=not src.freshness_warning
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Source Type", src.source_type.value.replace("_", " ").title())
            c2.metric("Age", f"{src.age_hours:.0f} hours")
            c3.metric("Records", src.record_count)
            c4.metric("Verified", "✓ Yes" if src.is_verified else "✗ No")
            if src.freshness_warning:
                st.warning(f"⚠ {src.freshness_warning}")

    st.markdown(
        f'<div class="xai-note">🔍 <b>Explainability note:</b> {l1.xai_note}</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════
# L2 Tab
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="layer-badge">L2 — Grounding Check</div>', unsafe_allow_html=True)
    st.markdown("### How trustworthy is the data this decision is based on?")

    score = l2.composite_grounding_score
    col1, col2, col3 = st.columns(3)
    col1.metric("Composite Score", f"{score:.2f}",
                delta="High confidence" if score >= 0.85 else "Moderate confidence")
    col2.metric("Proceed to Model", "✓ Yes" if l2.proceed_to_model else "✗ Hold for review")
    col3.metric("Weakest Source", l2.weakest_source.split("(")[0].strip())

    st.markdown("#### Grounding Scores — per source")
    fig = go.Figure()
    source_names = [s.source_name.split("(")[0].strip() for s in l2.source_scores]
    composites = [s.composite_score for s in l2.source_scores]
    freshness = [s.freshness_score for s in l2.source_scores]
    completeness = [s.completeness_score for s in l2.source_scores]
    consistency = [s.consistency_score for s in l2.source_scores]

    fig.add_trace(go.Bar(name="Freshness (40%)", x=source_names,
                          y=[f * 0.40 for f in freshness], marker_color="#0F6E56"))
    fig.add_trace(go.Bar(name="Completeness (35%)", x=source_names,
                          y=[c * 0.35 for c in completeness], marker_color="#5DCAA5"))
    fig.add_trace(go.Bar(name="Consistency (25%)", x=source_names,
                          y=[c * 0.25 for c in consistency], marker_color="#9FE1CB"))
    fig.update_layout(
        barmode="stack", height=320,
        yaxis_title="Weighted Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_hline(y=l2.proceed_threshold, line_dash="dash",
                   line_color="#E24B4A", annotation_text="Proceed threshold")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="xai-note">🔍 <b>Explainability note:</b> {l2.xai_note}</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════
# L3 Tab
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="layer-badge">L3 — Signal Extraction</div>', unsafe_allow_html=True)
    st.markdown("### Every model feature — visible, traceable, interpretable")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Signals", l3.total_signals)
    col2.metric("Positive ▲", l3.positive_signals)
    col3.metric("Negative ▼", l3.negative_signals)

    st.markdown("#### Signal Log — with source traceability")
    for sig in l3.signals:
        icon = "▲" if sig.risk_direction == "positive" else ("▼" if sig.risk_direction == "negative" else "─")
        color = "normal" if sig.risk_direction == "positive" else ("off" if sig.risk_direction == "negative" else "normal")
        with st.expander(f"{icon} {sig.signal_name}  —  {sig.display_value}"):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("**Sources used**")
                for src_id in sig.source_ids:
                    src_match = next((s for s in l1.sources if s.source_id == src_id), None)
                    if src_match:
                        st.caption(f"• {src_match.source_name}")
                st.markdown("**Risk direction**")
                if sig.risk_direction == "positive":
                    st.success("▲ Positive")
                elif sig.risk_direction == "negative":
                    st.error("▼ Negative")
                else:
                    st.warning("─ Neutral")
            with c2:
                st.markdown("**How it was computed**")
                st.code(sig.computation_formula, language=None)
                st.markdown("**What it means**")
                st.info(sig.interpretation)

    st.markdown("#### Model-Ready Feature Vector")
    feat_df = pd.DataFrame([
        {"Feature": k, "Value": round(v, 4)}
        for k, v in l3.model_ready_features.items()
    ])
    st.dataframe(feat_df, hide_index=True, use_container_width=True)

    st.markdown(
        f'<div class="xai-note">🔍 <b>Explainability note:</b> {l3.xai_note}</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════
# Evidence Ledger Tab
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### 📋 Evidence Ledger — sealed artifacts")
    st.markdown("Every layer output is hashed and stored. Nothing can be modified after sealing.")

    entries = retrieve_application_ledger(l0.application_id)
    for entry in entries:
        st.markdown(f"**{entry.layer}** — sealed `{entry.artifact_hash}` at {entry.sealed_at.strftime('%H:%M:%S')}")

    st.divider()
    st.markdown(f"**Total sealed artifacts:** {len(entries)}")
    st.markdown(f"**Application ID:** `{l0.application_id}`")
    st.caption("Artifacts are stored in SQLite and retrievable for audit at any time.")
