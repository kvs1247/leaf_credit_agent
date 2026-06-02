"""
LEAF Credit Agent — Explainability Dashboard
Three screens: Home (history), New Application, Viewing (full dashboard)
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
from storage.ledger import (
    init_ledger, get_all_summaries, clear_all_data, retrieve_application_ledger
)
from storage.loader import load_application_results

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="LEAF Credit Agent — Explainability Dashboard",
    page_icon="🌿", layout="wide",
    initial_sidebar_state="expanded"
)

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
  .decision-referred  { background:#FAEEDA; border:1px solid #854F0B; border-radius:8px;
                        padding:16px; text-align:center; }
  .demo-card { background:#F5F0FF; border:1px solid #C4BCEF; border-radius:8px;
               padding:10px 14px; margin-bottom:6px; }
  .home-title { font-size:28px; font-weight:600; color:#0F6E56; margin-bottom:4px; }
  .home-sub   { font-size:14px; color:#666; margin-bottom:20px; }
</style>
""", unsafe_allow_html=True)

# ── Session state initialisation ─────────────────────────────────
def _init_state():
    if "screen" not in st.session_state:
        st.session_state.screen = "home"
    if "viewing_id" not in st.session_state:
        st.session_state.viewing_id = None
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")

_init_state()
init_ledger()

# ── Navigation helpers ────────────────────────────────────────────
def go_home():
    st.session_state.screen = "home"
    st.session_state.viewing_id = None

def go_new():
    st.session_state.screen = "new_application"

def go_view(application_id: str):
    st.session_state.screen = "viewing"
    st.session_state.viewing_id = application_id

# ── Shared sidebar — API key always visible ───────────────────────
with st.sidebar:
    st.markdown("## 🌿 LEAF Credit Agent")
    st.markdown("*Layered Explainability AI Framework*")
    st.divider()
    st.markdown("### API Configuration")
    api_key = st.text_input(
        "OpenAI API Key", type="password",
        value=st.session_state.api_key,
        help="Required for L5 explanation generation"
    )
    if api_key:
        st.session_state.api_key = api_key

    if st.session_state.screen != "home":
        st.divider()
        if st.button("🏠 Back to Home", use_container_width=True):
            go_home()
            st.rerun()

# ══════════════════════════════════════════════════════════════════
# SCREEN 1 — HOME (Application History Dashboard)
# ══════════════════════════════════════════════════════════════════
if st.session_state.screen == "home":

    st.markdown('<div class="home-title">🌿 LEAF Credit Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="home-sub">Explainability Dashboard — Layered Explainability AI Framework for Finance</div>',
                unsafe_allow_html=True)

    # ── Action buttons ────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        if st.button("➕ New Application", type="primary", use_container_width=True):
            go_new()
            st.rerun()

    with col2:
        if st.button("🎭 Load Demo Scenarios", use_container_width=True,
                     help="Load 4 pre-crafted scenarios covering all LEAF capabilities"):
            if not st.session_state.api_key:
                st.error("Please enter your OpenAI API key first.")
            else:
                from demo_scenarios import load_demo_scenarios
                llm = LLMProvider(api_key=st.session_state.api_key, provider="openai")

                progress_bar = st.progress(0)
                status_text = st.empty()

                def on_progress(i, total, label):
                    progress_bar.progress((i) / total)
                    status_text.info(f"Loading scenario {i+1}/{total}: {label}...")

                with st.spinner("Loading demo scenarios..."):
                    try:
                        load_demo_scenarios(llm, progress_callback=on_progress)
                        progress_bar.progress(1.0)
                        status_text.success("✓ 4 demo scenarios loaded successfully")
                    except Exception as e:
                        st.error(f"Error loading scenarios: {e}")
                st.rerun()

    with col3:
        if st.button("🗑 Reset Demo Data", use_container_width=True,
                     help="Clear all applications and start fresh"):
            clear_all_data()
            st.success("All data cleared.")
            st.rerun()

    st.divider()

    # ── Application history ───────────────────────────────────────
    summaries = get_all_summaries()

    if not summaries:
        st.markdown("### No applications yet")
        st.info(
            "This dashboard will populate as applications are run.\n\n"
            "**To get started:**\n"
            "- Click **New Application** to evaluate a loan manually\n"
            "- Click **Load Demo Scenarios** to instantly load 4 pre-crafted examples "
            "covering all LEAF capabilities\n\n"
            "All decisions are stored permanently in the Evidence Ledger "
            "and retrievable at any time."
        )
    else:
        st.markdown(f"### Application History — {len(summaries)} decision(s)")

        for s in summaries:
            with st.container():
                c1, c2, c3, c4, c5, c6 = st.columns([2, 1.5, 1.2, 1.2, 1, 1.2])

                # Applicant + tag
                tag_html = ""
                if s.get("scenario_tag"):
                    tag_map = {
                        "strong_approval": "🟢 Demo",
                        "clear_rejection": "🔴 Demo",
                        "human_review": "🟡 Demo",
                        "fairness_flag": "🟣 Demo",
                    }
                    tag_html = tag_map.get(s["scenario_tag"], "Demo")

                c1.markdown(f"**{s['applicant_name']}** {tag_html}")
                c1.caption(f"`{s['application_id'][-12:]}`")

                c2.markdown(f"₹{s['amount_requested']:,.0f}")
                c2.caption(s['purpose'])

                # Decision badge
                dec = s['decision']
                if "Approved" in dec and "Conditional" not in dec:
                    c3.success(f"✓ {dec}")
                elif "Rejected" in dec:
                    c3.error(f"✗ {dec}")
                else:
                    c3.warning(f"⚠ {dec}")

                c4.metric("Approval", f"{s['approval_probability']:.0%}")
                c5.caption(s['confidence'])

                # Timestamp
                try:
                    ts = datetime.fromisoformat(s['timestamp']).strftime("%d %b %H:%M")
                except:
                    ts = s['timestamp'][:16]
                c6.caption(ts)

                if c6.button("View →", key=f"view_{s['application_id']}"):
                    go_view(s['application_id'])
                    st.rerun()

                st.divider()


# ══════════════════════════════════════════════════════════════════
# SCREEN 2 — NEW APPLICATION
# ══════════════════════════════════════════════════════════════════
elif st.session_state.screen == "new_application":

    st.markdown("## ➕ New Loan Application")
    st.markdown("Fill in the applicant details and run the LEAF Agent.")

    with st.form("application_form"):
        col1, col2 = st.columns(2)
        with col1:
            applicant_name = st.text_input("Applicant Name", value="Suresh Kumar")
            amount = st.number_input("Loan Amount (₹)", 50000, 5000000, 450000, 10000)
            purpose = st.selectbox("Purpose", [
                "Home renovation", "Education", "Business expansion",
                "Medical emergency", "Vehicle purchase", "Personal"
            ])
            tenure = st.slider("Tenure (months)", 12, 84, 48, 12)

        with col2:
            income = st.number_input("Monthly Income (₹)", 15000, 500000, 72400, 1000)
            employment = st.selectbox("Employment Type",
                                       ["salaried", "self_employed", "business"])
            existing_loans = st.slider("Existing Active Loans", 0, 5, 2)
            age = st.slider("Applicant Age", 21, 65, 34)

        submitted = st.form_submit_button("▶ Run LEAF Agent", type="primary",
                                           use_container_width=True)

    if submitted:
        if not st.session_state.api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
            st.stop()

        application = LoanApplication(
            applicant_id=f"CUST-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            amount_requested=float(amount), purpose=purpose,
            tenure_months=tenure, applicant_name=applicant_name,
            applicant_age=age, employment_type=employment,
            monthly_income_declared=float(income),
            existing_loans=existing_loans,
        )

        with st.spinner("🌿 LEAF Agent is reasoning through this application..."):
            try:
                llm = LLMProvider(api_key=st.session_state.api_key, provider="openai")
                agent = LEAFCreditAgent(llm_provider=llm, verbose=False)
                output = agent.run(application)
                go_view(output["application_id"])
                st.rerun()
            except Exception as e:
                st.error(f"Agent error: {e}")


# ══════════════════════════════════════════════════════════════════
# SCREEN 3 — VIEWING (Full 8-tab explainability dashboard)
# ══════════════════════════════════════════════════════════════════
elif st.session_state.screen == "viewing":

    application_id = st.session_state.viewing_id
    results = load_application_results(application_id)

    if not results:
        st.error(f"Could not load application {application_id}")
        if st.button("Back to Home"):
            go_home()
            st.rerun()
        st.stop()

    l0 = results.get("L0")
    l1 = results.get("L1")
    l2 = results.get("L2")
    l3 = results.get("L3")
    l4 = results.get("L4")
    l5 = results.get("L5")
    trace = results.get("AGENT_TRACE", {})

    # ── Decision banner ───────────────────────────────────────────
    st.markdown("## 🌿 LEAF Credit Agent — Explainability Dashboard")
    st.caption(f"Application `{l0.application_id}` · {l0.timestamp}")

    dec = l4.decision if l4 else "Unknown"
    prob = l4.approval_probability if l4 else 0
    conf = l4.decision_confidence if l4 else ""

    if "Approved" in dec and "Conditional" not in dec:
        st.markdown(
            f'<div class="decision-approved">'
            f'<h2 style="color:#0F6E56;margin:0">✓ {dec}</h2>'
            f'<p style="color:#0F6E56;margin:4px 0">Approval: {prob:.1%} · Confidence: {conf}</p>'
            f'</div>', unsafe_allow_html=True)
    elif "Rejected" in dec:
        st.markdown(
            f'<div class="decision-rejected">'
            f'<h2 style="color:#A32D2D;margin:0">✗ {dec}</h2>'
            f'<p style="color:#A32D2D;margin:4px 0">Approval: {prob:.1%}</p>'
            f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="decision-referred">'
            f'<h2 style="color:#854F0B;margin:0">⚠ {dec}</h2>'
            f'<p style="color:#854F0B;margin:4px 0">Approval: {prob:.1%} · {conf}</p>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────
    tab0,tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
        "L0 Context","L1 Provenance","L2 Grounding",
        "L3 Signals","L4 SHAP","L5 Explanation",
        "🤖 Agent Trace","📋 Ledger"
    ])

    # ── L0 ────────────────────────────────────────────────────────
    with tab0:
        st.markdown('<div class="leaf-badge">L0 — Request & Context</div>',
                    unsafe_allow_html=True)
        if l0:
            c1,c2,c3 = st.columns(3)
            c1.metric("Jurisdiction", l0.jurisdiction.value)
            c2.metric("Adverse Notice", "Required ✓" if l0.adverse_action_notice_required else "No")
            c3.metric("Frameworks", len(l0.regulatory_frameworks))
            for fw in l0.regulatory_frameworks:
                st.success(f"✓ {fw.value}")
            st.markdown(
                '<div class="xai-note">🔍 L0 locks the regulatory envelope before any data '
                'is processed. All downstream explanations satisfy these constraints.</div>',
                unsafe_allow_html=True)

    # ── L1 ────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="leaf-badge">L1 — Data Provenance</div>',
                    unsafe_allow_html=True)
        if l1:
            c1,c2,c3 = st.columns(3)
            c1.metric("Sources", l1.total_sources)
            c2.metric("Verified", l1.verified_sources)
            c3.metric("Warnings", l1.sources_with_warnings)
            for src in l1.sources:
                icon = "⚠" if src.freshness_warning else "✓"
                with st.expander(f"{icon} {src.source_name} — `{src.integrity_hash}`"):
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Type", src.source_type.value.replace("_"," ").title())
                    c2.metric("Age", f"{src.age_hours:.0f}h")
                    c3.metric("Records", src.record_count)
                    c4.metric("Verified", "Yes ✓" if src.is_verified else "No")
                    if src.freshness_warning:
                        st.warning(src.freshness_warning)

    # ── L2 ────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="leaf-badge">L2 — Grounding Check</div>',
                    unsafe_allow_html=True)
        if l2:
            c1,c2,c3 = st.columns(3)
            c1.metric("Composite Score", f"{l2.composite_grounding_score:.2f}")
            c2.metric("Proceed", "✓ Yes" if l2.proceed_to_model else "✗ Hold")
            c3.metric("Weakest Source", l2.weakest_source.split("(")[0][:20])
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
            fig.update_layout(barmode="stack", height=300,
                               margin=dict(l=0,r=0,t=20,b=0),
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)")
            fig.add_hline(y=l2.proceed_threshold, line_dash="dash",
                           line_color="#E24B4A", annotation_text="Threshold")
            st.plotly_chart(fig, use_container_width=True)

    # ── L3 ────────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="leaf-badge">L3 — Signal Extraction</div>',
                    unsafe_allow_html=True)
        if l3:
            c1,c2,c3 = st.columns(3)
            c1.metric("Total Signals", l3.total_signals)
            c2.metric("Positive ▲", l3.positive_signals)
            c3.metric("Negative ▼", l3.negative_signals)
            for sig in l3.signals:
                arrow = ("▲" if sig.risk_direction == "positive"
                         else "▼" if sig.risk_direction == "negative" else "─")
                with st.expander(f"{arrow} {sig.signal_name} — {sig.display_value}"):
                    st.code(sig.computation_formula)
                    st.info(sig.interpretation)
                    src_names = []
                    if l1:
                        for sid in sig.source_ids:
                            match = next((s for s in l1.sources if s.source_id == sid), None)
                            if match:
                                src_names.append(match.source_name)
                    st.caption(f"Sources: {', '.join(src_names) if src_names else ', '.join(sig.source_ids)}")

    # ── L4 ────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="leaf-badge">L4 — Model Reasoning (SHAP)</div>',
                    unsafe_allow_html=True)
        if l4:
            c1,c2,c3 = st.columns(3)
            c1.metric("Approval Probability", f"{l4.approval_probability:.1%}")
            c2.metric("Decision", l4.decision)
            c3.metric("Confidence", l4.decision_confidence)

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
            fig2.update_layout(
                height=350, margin=dict(l=0,r=60,t=20,b=0),
                xaxis_title="SHAP contribution to approval probability",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig2.add_vline(x=0, line_color="gray", line_width=1)
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("#### Counterfactual")
            st.info(f"💡 {l4.counterfactual_hint}")
            st.markdown(
                '<div class="xai-note">🔍 Green bars support approval. Red bars increase '
                'default risk. This individual attribution — not global averages — '
                'makes the adverse action notice legally defensible.</div>',
                unsafe_allow_html=True)

    # ── L5 ────────────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="leaf-badge">L5 — Explanation Card (LLM Generated)</div>',
                    unsafe_allow_html=True)
        if l5:
            card = l5.explanation_card
            st.markdown(f"### {card.decision_summary}")
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**Factors in favour:**")
                for f in card.top_positive_factors:
                    st.success(f"▲ {f}")
            with c2:
                st.markdown("**Risk factors:**")
                for f in card.top_negative_factors:
                    st.warning(f"▼ {f}")

            st.markdown("#### Plain language rationale")
            st.write(l5.plain_language_rationale)

            st.markdown("#### What would improve your position")
            st.info(f"💡 {card.counterfactual}")

            c1, c2 = st.columns(2)
            c1.metric("Interest Rate Band", l5.interest_rate_band)
            c2.metric("Tenure", f"{l5.recommended_tenure_months} months")

            st.caption(card.confidence_note)
            st.caption(f"📋 {card.applicant_rights}")

    # ── Agent Trace ───────────────────────────────────────────────
    with tab6:
        st.markdown("### 🤖 Agent Reasoning Trace")
        st.markdown("*The LLM's reasoning at every layer — this is what makes LEAF agentic*")

        if trace:
            observations = trace.get("agent_observations", {})
            decisions_map = trace.get("layer_decisions", {})
            for layer in ["L0","L1","L2","L3","L4"]:
                obs = observations.get(layer,"")
                dec_val = decisions_map.get(layer,"")
                if obs:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(
                            f'<div class="agent-thought">💭 <b>{layer}:</b> {obs}</div>',
                            unsafe_allow_html=True)
                    with col2:
                        if dec_val == "PROCEED":
                            st.success(dec_val)
                        elif dec_val == "ESCALATE":
                            st.warning(dec_val)
                        else:
                            st.error(dec_val)

        if l5 and l5.agent_reasoning_trace:
            st.markdown("#### L5 — Explanation generation reasoning")
            st.code(l5.agent_reasoning_trace, language=None)
        else:
            st.info("No agent trace available for this application.")

    # ── Evidence Ledger ───────────────────────────────────────────
    with tab7:
        st.markdown("### 📋 Evidence Ledger — sealed artifacts")
        st.markdown("Every layer output is hashed and stored immutably.")
        entries = retrieve_application_ledger(l0.application_id)
        for entry in entries:
            st.markdown(
                f"**{entry.layer}** — `{entry.artifact_hash}` — "
                f"sealed {entry.sealed_at.strftime('%H:%M:%S')}"
            )
        st.divider()
        st.metric("Total artifacts sealed", len(entries))
        st.caption(f"Application: `{l0.application_id}` · Stored in SQLite · "
                   f"Retrievable for 7 years (RBI requirement)")
