"""
LEAF Credit Agent — Streamlit Explainability Dashboard
Sprint 2: L0 through L5 with LLM Agent Orchestration

Run with: streamlit run streamlit_app/app.py
Set OPENAI_API_KEY environment variable before running.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

from models.schemas import LoanApplication, JurisdictionCode
from models.llm_provider import LLMProvider
from agent import LEAFCreditAgent
from storage.ledger import retrieve_application_ledger

st.set_page_config(page_title="LEAF Credit Agent", page_icon="🌿",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .leaf-badge { display:inline-block; background:#E1F5EE; color:#0F6E56;
                padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
  .xai-note { background:#F0FAF5; border-left:3px solid #0F6E56; padding:10px 14px;
               border-radius:4px; font-size:13px; color:#444; margin:8px 0; }
  .agent-thought { background:#F5F0FF; border-left:3px solid #534AB7; padding:8px 12px;
                   border-radius:4px; font-size:12px; color:#26215C; margin:4px 0; }
  .decision-approved { background:#E1F5EE; border:1px solid #0F6E56; border-radius:8px;
                       padding:16px; text-align:center; }
  .decision-rejected { background:#FCEBEB; border:1px solid #A32D2D; border-radius:8px;
                       padding:16px; text-align:center; }
  .decision-referred { background:#FAEEDA; border:1px solid #854F0B; border-radius:8px;
                       padding:16px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 LEAF Credit Agent")
    st.markdown("*Layered Explainability AI Framework*")
    st.divider()

    st.markdown("### API Configuration")
    api_key = st.text_input("OpenAI API Key", type="password",
                             value=os.getenv("OPENAI_API_KEY", ""),
                             help="Your OpenAI API key")
    st.divider()

    st.markdown("### Loan Application")
    applicant_name = st.text_input("Applicant Name", value="Suresh Kumar")
    amount = st.number_input("Loan Amount (₹)", 50000, 5000000, 450000, 10000)
    purpose = st.selectbox("Purpose", ["Home renovation","Education",
                                        "Business expansion","Medical emergency",
                                        "Vehicle purchase","Personal"])
    tenure = st.slider("Tenure (months)", 12, 84, 48, 12)
    income = st.number_input("Monthly Income (₹)", 15000, 500000, 72400, 1000)
    employment = st.selectbox("Employment Type", ["salaried","self_employed","business"])
    existing_loans = st.slider("Existing Active Loans", 0, 5, 2)
    age = st.slider("Applicant Age", 21, 65, 34)
    st.divider()

    run_button = st.button("▶ Run LEAF Agent", type="primary", use_container_width=True)
    st.caption("This runs all 5 layers including LLM reasoning")

# ── Run agent ────────────────────────────────────────────────────
if run_button:
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    application = LoanApplication(
        applicant_id=f"CUST-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        amount_requested=float(amount), purpose=purpose,
        tenure_months=tenure, applicant_name=applicant_name,
        applicant_age=age, employment_type=employment,
        monthly_income_declared=float(income), existing_loans=existing_loans,
    )

    with st.spinner("🌿 LEAF Agent is reasoning through your application..."):
        try:
            llm = LLMProvider(api_key=api_key, provider="openai")
            agent = LEAFCreditAgent(llm_provider=llm, verbose=False)
            output = agent.run(application)
            st.session_state["output"] = output
            st.session_state["application"] = application
        except Exception as e:
            st.error(f"Agent error: {e}")
            st.stop()

if "output" not in st.session_state:
    st.markdown("## 🌿 LEAF Credit Agent — Sprint 2")
    st.info("Enter loan details in the sidebar and click **Run LEAF Agent** to see all 5 layers with LLM reasoning.")
    st.stop()

output = st.session_state["output"]
application = st.session_state["application"]
results = output["results"]
l0 = results["L0"]; l1 = results["L1"]; l2 = results["L2"]
l3 = results["L3"]; l4 = results["L4"]; l5 = results["L5"]

# ── Header ───────────────────────────────────────────────────────
st.markdown(f"## 🌿 LEAF Agent — {l0.application_id}")

# Decision banner
dec = l4.decision
if "Approved" in dec:
    st.markdown(f'<div class="decision-approved"><h2 style="color:#0F6E56">✓ {dec}</h2>'
                f'<p style="color:#0F6E56">Approval probability: {l4.approval_probability:.1%} · '
                f'Confidence: {l4.decision_confidence}</p></div>', unsafe_allow_html=True)
elif "Rejected" in dec:
    st.markdown(f'<div class="decision-rejected"><h2 style="color:#A32D2D">✗ {dec}</h2>'
                f'<p style="color:#A32D2D">Approval probability: {l4.approval_probability:.1%}</p></div>',
                unsafe_allow_html=True)
else:
    st.markdown(f'<div class="decision-referred"><h2 style="color:#854F0B">⚠ {dec}</h2>'
                f'<p style="color:#854F0B">Approval probability: {l4.approval_probability:.1%}</p></div>',
                unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────
tab0,tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "L0 Context","L1 Provenance","L2 Grounding",
    "L3 Signals","L4 SHAP","L5 Explanation",
    "🤖 Agent Trace","📋 Ledger"
])

# L0
with tab0:
    st.markdown('<div class="leaf-badge">L0 — Request & Context</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Jurisdiction", l0.jurisdiction.value)
    c2.metric("Adverse Notice", "Required ✓" if l0.adverse_action_notice_required else "Not required")
    c3.metric("Frameworks", len(l0.regulatory_frameworks))
    for fw in l0.regulatory_frameworks:
        st.success(f"✓ {fw.value}")
    st.markdown('<div class="xai-note">🔍 L0 locks the regulatory envelope before any data is processed. '
                'All downstream explanations are generated within these constraints.</div>',
                unsafe_allow_html=True)

# L1
with tab1:
    st.markdown('<div class="leaf-badge">L1 — Data Provenance</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Sources", l1.total_sources)
    c2.metric("Verified", l1.verified_sources)
    c3.metric("Warnings", l1.sources_with_warnings)
    for src in l1.sources:
        icon = "⚠" if src.freshness_warning else "✓"
        with st.expander(f"{icon} {src.source_name} — hash: `{src.integrity_hash}`"):
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Type", src.source_type.value.replace("_"," ").title())
            c2.metric("Age", f"{src.age_hours:.0f}h")
            c3.metric("Records", src.record_count)
            c4.metric("Verified", "Yes ✓" if src.is_verified else "No")
            if src.freshness_warning:
                st.warning(src.freshness_warning)

# L2
with tab2:
    st.markdown('<div class="leaf-badge">L2 — Grounding Check</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    score = l2.composite_grounding_score
    c1.metric("Composite Score", f"{score:.2f}")
    c2.metric("Proceed", "✓ Yes" if l2.proceed_to_model else "✗ Hold")
    c3.metric("Weakest Source", l2.weakest_source.split("(")[0].strip()[:20])
    names = [s.source_name.split("(")[0].strip() for s in l2.source_scores]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Freshness (40%)", x=names,
                          y=[s.freshness_score*0.40 for s in l2.source_scores],
                          marker_color="#0F6E56"))
    fig.add_trace(go.Bar(name="Completeness (35%)", x=names,
                          y=[s.completeness_score*0.35 for s in l2.source_scores],
                          marker_color="#5DCAA5"))
    fig.add_trace(go.Bar(name="Consistency (25%)", x=names,
                          y=[s.consistency_score*0.25 for s in l2.source_scores],
                          marker_color="#9FE1CB"))
    fig.update_layout(barmode="stack", height=300, margin=dict(l=0,r=0,t=20,b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.add_hline(y=l2.proceed_threshold, line_dash="dash",
                   line_color="#E24B4A", annotation_text="Threshold")
    st.plotly_chart(fig, use_container_width=True)

# L3
with tab3:
    st.markdown('<div class="leaf-badge">L3 — Signal Extraction</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Signals", l3.total_signals)
    c2.metric("Positive ▲", l3.positive_signals)
    c3.metric("Negative ▼", l3.negative_signals)
    for sig in l3.signals:
        arrow = "▲" if sig.risk_direction=="positive" else ("▼" if sig.risk_direction=="negative" else "─")
        with st.expander(f"{arrow} {sig.signal_name} — {sig.display_value}"):
            st.code(sig.computation_formula)
            st.info(sig.interpretation)
            st.caption(f"Sources: {', '.join(sig.source_ids)}")

# L4
with tab4:
    st.markdown('<div class="leaf-badge">L4 — Model Reasoning (SHAP)</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Approval Probability", f"{l4.approval_probability:.1%}")
    c2.metric("Decision", l4.decision)
    c3.metric("Model Confidence", l4.decision_confidence)

    st.markdown("#### SHAP Waterfall — this applicant")
    contribs = l4.shap_contributions
    labels = [c.plain_label for c in contribs[:8]]
    values = [c.shap_value for c in contribs[:8]]
    colors = ["#0F6E56" if v > 0 else "#E24B4A" for v in values]
    fig2 = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))
    fig2.update_layout(height=350, margin=dict(l=0,r=60,t=20,b=0),
                       xaxis_title="SHAP contribution to approval",
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig2.add_vline(x=0, line_color="gray", line_width=1)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Counterfactual")
    st.info(f"💡 {l4.counterfactual_hint}")
    st.markdown('<div class="xai-note">🔍 Negative SHAP values (red) increase default risk. '
                'Positive values (green) reduce default risk and support approval.</div>',
                unsafe_allow_html=True)

# L5
with tab5:
    st.markdown('<div class="leaf-badge">L5 — Explanation Card (LLM Generated)</div>',
                unsafe_allow_html=True)
    card = l5.explanation_card
    st.markdown(f"### {card.decision_summary}")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Why approved / factors in your favour:**")
        for f in card.top_positive_factors:
            st.success(f"▲ {f}")
    with c2:
        st.markdown("**Risk factors considered:**")
        for f in card.top_negative_factors:
            st.warning(f"▼ {f}")
    st.markdown("#### Plain language rationale")
    st.write(l5.plain_language_rationale)
    st.markdown("#### What would improve your position")
    st.info(f"💡 {card.counterfactual}")
    st.markdown("#### Interest Rate Band")
    st.metric("Rate", l5.interest_rate_band)
    st.caption(card.confidence_note)
    st.caption(f"📋 {card.applicant_rights}")

# Agent Trace
with tab6:
    st.markdown("### 🤖 Agent Reasoning Trace")
    st.markdown("*This is what makes LEAF agentic — the LLM's reasoning at every layer*")
    trace = output.get("reasoning_trace", {})
    observations = trace.get("agent_observations", {})
    decisions = trace.get("layer_decisions", {})
    for layer in ["L0","L1","L2","L3","L4"]:
        obs = observations.get(layer,"")
        dec = decisions.get(layer,"")
        if obs:
            col1,col2 = st.columns([4,1])
            with col1:
                st.markdown(f'<div class="agent-thought">💭 <b>{layer}:</b> {obs}</div>',
                            unsafe_allow_html=True)
            with col2:
                if dec == "PROCEED":
                    st.success(dec)
                elif dec == "ESCALATE":
                    st.warning(dec)
                else:
                    st.error(dec)
    st.markdown("#### L5 — Explanation Card generation trace")
    st.code(l5.agent_reasoning_trace, language=None)

# Ledger
with tab7:
    st.markdown("### 📋 Evidence Ledger — sealed artifacts")
    entries = retrieve_application_ledger(l0.application_id)
    for entry in entries:
        st.markdown(f"**{entry.layer}** — `{entry.artifact_hash}` — "
                    f"sealed at {entry.sealed_at.strftime('%H:%M:%S')}")
    st.metric("Total artifacts sealed", len(entries))
    st.caption(f"Application: `{l0.application_id}` · Immutable · Auditable")
