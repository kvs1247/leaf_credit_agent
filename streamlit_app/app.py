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
                st.error("Please enter your OpenAI API key in the sidebar first.")
            else:
                from demo_scenarios import load_demo_scenarios
                llm = LLMProvider(
                    api_key=st.session_state.api_key, provider="openai"
                )
                progress_bar = st.progress(0)
                status_text = st.empty()

                def on_progress(i, total, label):
                    progress_bar.progress(i / total)
                    status_text.info(
                        f"Loading scenario {i+1}/{total}: {label}..."
                    )

                with st.spinner("Loading demo scenarios — this takes 2-3 minutes..."):
                    try:
                        load_demo_scenarios(
                            llm_provider=llm,
                            progress_callback=on_progress
                        )
                        progress_bar.progress(1.0)
                        status_text.success(
                            "✓ 4 demo scenarios loaded with genuine AI explanations"
                        )
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
    l6 = results.get("L6")
    l7 = results.get("L7")
    l8 = results.get("L8")
    trace = results.get("AGENT_TRACE", {})

    # ── Decision banner ───────────────────────────────────────────
    st.markdown("## 🌿 LEAF Credit Agent — Explainability Dashboard")
    st.caption(f"Application `{l0.application_id}` · {l0.timestamp}")

    dec = l4.decision if l4 else "Unknown"
    prob = l4.approval_probability if l4 else 0
    conf = l4.decision_confidence if l4 else ""

    # ── Governance-aware banner ───────────────────────────────────
    # L7 overrides the model decision presentation when governance
    # blocks or conditionally approves. The banner reflects the
    # FINAL governance-adjusted outcome — not just the model output.
    # Core insight: a model approval ≠ a legitimate decision.

    if l7 and l7.legitimacy_verdict.value == "Blocked":
        st.markdown(
            f'<div class="decision-rejected">'
            f'<h2 style="color:#A32D2D;margin:0">🚫 Blocked by Governance</h2>'
            f'<p style="color:#A32D2D;margin:4px 0">'
            f'Model decision: {dec} ({prob:.1%}) — '
            f'overridden by L7 compliance violation</p>'
            f'<p style="color:#A32D2D;font-size:13px;margin:2px 0">'
            f'{l7.compliance.blocking_violations[0] if l7.compliance.blocking_violations else "Governance rule violated"}'
            f'</p></div>', unsafe_allow_html=True)

    elif l7 and l7.legitimacy_verdict.value == "Conditional":
        st.markdown(
            f'<div class="decision-referred">'
            f'<h2 style="color:#854F0B;margin:0">⚠ Conditional — Human Review Required</h2>'
            f'<p style="color:#854F0B;margin:4px 0">'
            f'Model decision: {dec} ({prob:.1%}) · Confidence: {conf}</p>'
            f'<p style="color:#854F0B;font-size:13px;margin:2px 0">'
            f'{l7.override_reason or "Officer review required before issuing decision"}'
            f'</p></div>', unsafe_allow_html=True)

    elif "Approved" in dec and "Conditional" not in dec:
        st.markdown(
            f'<div class="decision-approved">'
            f'<h2 style="color:#0F6E56;margin:0">✓ {dec}</h2>'
            f'<p style="color:#0F6E56;margin:4px 0">'
            f'Approval: {prob:.1%} · Confidence: {conf}</p>'
            f'<p style="color:#0F6E56;font-size:13px;margin:2px 0">'
            f'Governance: Legitimate ✓</p>'
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
            f'<p style="color:#854F0B;margin:4px 0">'
            f'Approval: {prob:.1%} · {conf}</p>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────
    tab0,tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10 = st.tabs([
        "L0 Context","L1 Provenance","L2 Grounding",
        "L3 Signals","L4 SHAP","L5 Explanation","L6 Confidence",
        "L7 Governance","L8 Fairness",
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
        if l7 and l7.legitimacy_verdict.value == "Blocked":
                st.warning(
                    "⚖️ **Governance Notice:** This decision was blocked by L7. "
                    "The Explanation Card below was generated for full transparency "
                    "and audit purposes — it would NOT be issued to the applicant. "
                    "A governance violation notice is issued instead."
                )
        elif l7 and l7.legitimacy_verdict.value == "Conditional":
                st.info(
                    "⚠️ **Pending Review:** This explanation card is generated "
                    "but held pending human officer review."
                )

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
        else:
            st.info(
                    "L5 explanation not available for this run. "
                    "This application was loaded from a previous session "
                    "before the LLM was connected, or the API key was "
                    "not supplied when this application was processed."
                )

    # ── L6 ────────────────────────────────────────────────────────
    with tab6:
        st.markdown('<div class="leaf-badge">L6 — Confidence Grade</div>',
                    unsafe_allow_html=True)
        st.markdown("### How trustworthy is this recommendation?")

        if l6:
            # Grade banner
            grade_colors = {
                "A": ("#E1F5EE", "#0F6E56"),
                "B": ("#EAF4FF", "#185FA5"),
                "C": ("#FAEEDA", "#854F0B"),
                "D": ("#FCEBEB", "#A32D2D"),
            }
            bg, fg = grade_colors.get(l6.grade, ("#F5F5F5", "#333"))
            st.markdown(
                f'<div style="background:{bg};border:1px solid {fg};'
                f'border-radius:8px;padding:16px;text-align:center;'
                f'margin-bottom:16px;">'
                f'<h2 style="color:{fg};margin:0">Grade {l6.grade}</h2>'
                f'<p style="color:{fg};margin:4px 0">'
                f'{l6.grade_meaning}</p>'
                f'<p style="color:{fg};font-size:13px;margin:4px 0">'
                f'Composite Score: {l6.composite_score:.3f}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

            # HITL / Block status
            if l6.decision_blocked:
                st.error("🚫 Decision BLOCKED — Grade D requires mandatory "
                         "human authorisation before any decision is issued.")
            elif l6.hitl_required:
                st.warning("⚠ Human review REQUIRED — Grade C. "
                           "Loan officer must review before issuing decision.")
            else:
                st.success("✓ Auto-proceed — confidence is sufficient "
                           "for automated decision.")

            st.markdown("#### Five-Component Breakdown")
            st.markdown("*Confidence = 0.30·Grounding + 0.25·Freshness + "
                        "0.20·Consistency + 0.15·Calibration + 0.10·Compliance*")

            # Component scores bar chart
            components = {
                "Grounding Fidelity\n(w=0.30)":
                    l6.grounding_fidelity.composite,
                "Freshness & Coverage\n(w=0.25)":
                    l6.freshness_coverage.composite,
                "Model Consistency\n(w=0.20)":
                    l6.model_consistency.composite,
                "Calibration\n(w=0.15)":
                    l6.calibration.composite,
                "Compliance\n(w=0.10)":
                    l6.compliance_suitability.composite,
            }
            weighted = {
                "Grounding Fidelity\n(w=0.30)":
                    l6.grounding_fidelity.composite * 0.30,
                "Freshness & Coverage\n(w=0.25)":
                    l6.freshness_coverage.composite * 0.25,
                "Model Consistency\n(w=0.20)":
                    l6.model_consistency.composite * 0.20,
                "Calibration\n(w=0.15)":
                    l6.calibration.composite * 0.15,
                "Compliance\n(w=0.10)":
                    l6.compliance_suitability.composite * 0.10,
            }

            fig_l6 = __import__('plotly.graph_objects',
                                 fromlist=['graph_objects']).Figure()
            fig_l6.add_trace(__import__('plotly.graph_objects',
                             fromlist=['graph_objects']).Bar(
                x=list(components.keys()),
                y=list(components.values()),
                name="Raw Score",
                marker_color="#9FE1CB",
            ))
            fig_l6.add_trace(__import__('plotly.graph_objects',
                             fromlist=['graph_objects']).Bar(
                x=list(weighted.keys()),
                y=list(weighted.values()),
                name="Weighted Contribution",
                marker_color="#0F6E56",
            ))
            fig_l6.update_layout(
                barmode="group", height=320,
                yaxis=dict(range=[0, 1], title="Score"),
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            fig_l6.add_hline(y=l6.composite_score, line_dash="dash",
                              line_color="#E24B4A",
                              annotation_text=f"Grade {l6.grade} "
                                              f"({l6.composite_score:.3f})")
            st.plotly_chart(fig_l6, use_container_width=True)

            # Component detail expanders
            with st.expander("Component 1 — Grounding Fidelity "
                             f"({l6.grounding_fidelity.composite:.3f})"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Retrieval Quality",
                           f"{l6.grounding_fidelity.retrieval_quality:.3f}")
                c2.metric("Semantic Similarity",
                           f"{l6.grounding_fidelity.semantic_similarity:.3f}")
                c3.metric("Contradiction Penalty",
                           f"{l6.grounding_fidelity.contradiction_penalty:.3f}")
                if l6.grounding_fidelity.contradictions_detected:
                    st.markdown("**Contradictions detected:**")
                    for c in l6.grounding_fidelity.contradictions_detected:
                        st.warning(f"⚠ {c}")
                else:
                    st.success("✓ No contradictions detected")

            with st.expander("Component 2 — Data Freshness & Coverage "
                             f"({l6.freshness_coverage.composite:.3f})"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Freshness",
                           f"{l6.freshness_coverage.freshness:.3f}")
                c2.metric("Source Diversity",
                           f"{l6.freshness_coverage.source_diversity:.3f}")
                c3.metric("Evidence Completeness",
                           f"{l6.freshness_coverage.evidence_completeness:.3f}")
                st.caption(
                    f"Sources used: "
                    f"{', '.join(l6.freshness_coverage.sources_used)}"
                )

            with st.expander("Component 3 — Model Consistency "
                             f"({l6.model_consistency.composite:.3f})"):
                c1, c2, c3 = st.columns(3)
                c1.metric("XGBoost vs Rules",
                           f"{l6.model_consistency.xgboost_vs_rules_agreement:.3f}")
                c2.metric("SHAP Stability",
                           f"{l6.model_consistency.shap_stability:.3f}")
                c3.metric("Boundary Distance",
                           f"{l6.model_consistency.decision_boundary_distance:.3f}")
                if l6.model_consistency.consistency_flags:
                    for f in l6.model_consistency.consistency_flags:
                        st.warning(f"⚠ {f}")
                else:
                    st.success("✓ Model and rules are consistent")

            with st.expander("Component 4 — Calibration & Backtesting "
                             f"({l6.calibration.composite:.3f})"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Historical Applications",
                           l6.calibration.historical_applications)
                c2.metric("Hit Ratio",
                           f"{l6.calibration.hit_ratio:.3f}")
                c3.metric("Calibration Confidence",
                           f"{l6.calibration.calibration_confidence:.3f}")
                st.info(l6.calibration.baseline_note)

            with st.expander("Component 5 — Compliance & Suitability "
                             f"({l6.compliance_suitability.composite:.3f})"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Fair Lending",
                           f"{l6.compliance_suitability.fair_lending_score:.3f}")
                c2.metric("KYC Completeness",
                           f"{l6.compliance_suitability.kyc_completeness:.3f}")
                c3.metric("Regulatory Alignment",
                           f"{l6.compliance_suitability.regulatory_alignment:.3f}")
                if l6.compliance_suitability.suitability_flags:
                    for f in l6.compliance_suitability.suitability_flags:
                        st.warning(f"⚠ {f}")
                else:
                    st.success("✓ All compliance checks passed")

            # Confidence narrative
            st.markdown("#### Confidence Narrative")
            st.info(l6.confidence_narrative)
            st.markdown(
                '<div class="xai-note">🔍 '
                '<b>Explainability note:</b> '
                'L6 transforms confidence from a vague probability into a '
                'transparent, decomposable, evidence-backed reliability '
                'assessment. This is meta-explainability — explaining the '
                'quality of the explanation itself. No prior XAI framework '
                'provides this.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("L6 not available for this application.")
    # ── L7 ────────────────────────────────────────────────────────
    with tab7:
        st.markdown('<div class="leaf-badge">L7 — Compliance & Suitability</div>',
                    unsafe_allow_html=True)
        st.markdown("### Is this decision legitimate — not just correct?")
        st.markdown(
            '*This layer introduces* ***Governance Explainability*** *— '
            'distinct from all previous layers. L4-L6 ask: is the decision '
            'correct? L7 asks: is the decision legitimate?*'
        )

        if l7:
            # Legitimacy verdict banner
            verdict_colors = {
                "Legitimate":   ("#E1F5EE", "#0F6E56"),
                "Conditional":  ("#FAEEDA", "#854F0B"),
                "Blocked":      ("#FCEBEB", "#A32D2D"),
            }
            bg, fg = verdict_colors.get(
                l7.legitimacy_verdict.value, ("#F5F5F5", "#333")
            )
            st.markdown(
                f'<div style="background:{bg};border:1px solid {fg};'
                f'border-radius:8px;padding:16px;text-align:center;'
                f'margin-bottom:16px;">'
                f'<h2 style="color:{fg};margin:0">'
                f'{l7.legitimacy_verdict.value}</h2>'
                f'<p style="color:{fg};margin:4px 0">'
                f'Decision Legitimate: '
                f'{"Yes ✓" if l7.decision_legitimate else "No ✗"} · '
                f'Override Required: '
                f'{"Yes ⚠" if l7.override_required else "No"}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

            if l7.override_reason:
                st.warning(f"⚠ {l7.override_reason}")

            # Two columns — compliance and suitability
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Component 1 — Compliance")
                st.markdown(
                    f"*Is this recommendation legally and "
                    f"procedurally acceptable?*"
                )
                status_color = (
                    "green" if l7.compliance.overall_status == "Compliant"
                    else "orange" if "Warning" in l7.compliance.overall_status
                    else "red"
                )
                st.markdown(
                    f"**Status:** :{status_color}[{l7.compliance.overall_status}]"
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Pass", l7.compliance.passed)
                c2.metric("Fail", l7.compliance.failed)
                c3.metric("Warnings", l7.compliance.warnings)

                st.markdown("**Checks:**")
                for check in l7.compliance.checks:
                    if check.result.value == "pass":
                        st.success(
                            f"✓ {check.rule_name} — {check.actual_value}"
                        )
                    elif check.result.value == "warning":
                        st.warning(
                            f"⚠ {check.rule_name} — {check.actual_value}"
                        )
                    else:
                        st.error(
                            f"✗ {check.rule_name} — {check.actual_value}"
                        )
                    with st.expander(
                        f"Regulation reference — {check.rule_id}"
                    ):
                        st.caption(
                            f"**Regulation:** {check.regulation_reference}"
                        )
                        st.caption(f"**Threshold:** {check.threshold}")
                        st.write(check.details)

            with col2:
                st.markdown("#### Component 2 — Suitability")
                st.markdown(
                    "*Is this recommendation appropriate for THIS borrower?*"
                )
                suit_color = (
                    "green" if l7.suitability.suitability_label.value
                    == "Suitable"
                    else "orange" if l7.suitability.suitability_label.value
                    == "Partial Match"
                    else "red"
                )
                st.markdown(
                    f"**Label:** :{suit_color}"
                    f"[{l7.suitability.suitability_label.value}] "
                    f"(score: {l7.suitability.overall_score:.3f})"
                )

                st.markdown("**Five dimensions:**")
                for dim in l7.suitability.dimensions:
                    color = (
                        "green" if dim.label == "Suitable"
                        else "orange" if dim.label == "Partial Match"
                        else "red"
                    )
                    with st.expander(
                        f":{color}[{dim.label}] {dim.dimension} "
                        f"— {dim.score:.2f}"
                    ):
                        st.caption(f"**Actual:** {dim.actual}")
                        st.caption(f"**Benchmark:** {dim.benchmark}")
                        st.write(dim.assessment)

                if l7.suitability.primary_concern:
                    st.warning(
                        f"⚠ Primary concern: {l7.suitability.primary_concern}"
                    )

            # Governance narrative
            st.markdown("#### Governance Narrative")
            st.info(l7.governance_narrative)
            st.markdown(
                '<div class="xai-note">🔍 '
                '<b>Governance Explainability:</b> '
                'A hospital analogy — L4 is the diagnosis, L5 is the '
                'treatment recommendation, L6 is confidence in the diagnosis. '
                'L7 asks: can this treatment legitimately be prescribed '
                'to THIS patient? Same logic applies here.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("L7 not available for this application.")

    # ── L8 ────────────────────────────────────────────────────────
    with tab8:
        st.markdown('<div class="leaf-badge">L8 — Bias & Fairness Diagnostics</div>',
                    unsafe_allow_html=True)
        st.markdown("### Is the system behaving fairly over time?")
        st.markdown(
            '*L7 asks: is this decision allowed? '
            'L8 asks: **is the system fair over time?** '
            'These are fundamentally different questions.*'
        )

        if l8:
            # Verdict banner
            verdict_styles = {
                "Fair":    ("#E1F5EE", "#0F6E56", "✓"),
                "Caution": ("#FAEEDA", "#854F0B", "⚠"),
                "Biased":  ("#FCEBEB", "#A32D2D", "🚫"),
            }
            bg, fg, icon = verdict_styles.get(
                l8.verdict, ("#F5F5F5", "#333", "?")
            )
            st.markdown(
                f'<div style="background:{bg};border:1px solid {fg};'
                f'border-radius:8px;padding:16px;text-align:center;'
                f'margin-bottom:16px;">'
                f'<h2 style="color:{fg};margin:0">'
                f'{icon} Fairness Verdict: {l8.verdict}</h2>'
                f'<p style="color:{fg};margin:4px 0">'
                f'Overall Bias Score: {l8.obs:.3f} · '
                f'Flags: {l8.total_flags} · '
                f'Investigation: {"Required" if l8.investigation_required else "Not required"}'
                f'</p></div>',
                unsafe_allow_html=True
            )

            # Key metrics row
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("EII", f"{l8.eii:.3f}",
                      help="Exposure Imbalance Index")
            c2.metric("ARP", f"{l8.arp:.3f}",
                      help="Approval Rate Parity gap")
            c3.metric("SCI", f"{l8.sci:.3f}",
                      help="Source Concentration Index")
            c4.metric("TDS", f"{l8.tds:.3f}",
                      help="Temporal Drift Score")
            c5.metric("ICS", f"{l8.ics:.3f}",
                      help="Individual Consistency Score")

            st.markdown(f"*Analysis based on {l8.total_applications_analysed} "
                        f"historical applications*")

            # Five diagnostic areas
            st.markdown("#### Five Diagnostic Areas")
            areas = [
                ("1. Representation Fairness",
                 l8.representation.score,
                 l8.representation.flags,
                 f"EII Employment: {l8.representation.eii_employment:.3f} · "
                 f"EII Income: {l8.representation.eii_income:.3f}",
                 l8.representation.data_basis),
                ("2. Outcome Fairness",
                 l8.outcome.score,
                 l8.outcome.flags,
                 f"ARP disparity: {l8.outcome.arp_disparity:.3f} · "
                 f"CP disparity: {l8.outcome.cp_disparity:.3f}",
                 l8.outcome.data_basis),
                ("3. Evidence & Source Fairness",
                 l8.evidence_source.score,
                 l8.evidence_source.flags,
                 f"SCI: {l8.evidence_source.sci:.3f} · "
                 f"Bureau dependency: {l8.evidence_source.cibil_dependency:.0%} · "
                 f"Alt data: {l8.evidence_source.alternative_data_utilisation:.0%}",
                 "Individual application analysis"),
                ("4. Temporal Fairness",
                 l8.temporal.score,
                 l8.temporal.flags,
                 f"TDS: {l8.temporal.tds:.3f} · "
                 f"Historical rate: {l8.temporal.historical_approval_rate:.0%} · "
                 f"Recent rate: {l8.temporal.recent_approval_rate:.0%}",
                 l8.temporal.data_basis),
                ("5. Individual/Profile Fairness",
                 l8.individual_profile.score,
                 l8.individual_profile.flags,
                 f"ICS: {l8.individual_profile.ics:.3f} · "
                 f"Similar profiles: {l8.individual_profile.similar_profiles_found}",
                 l8.individual_profile.data_basis),
            ]

            for area_name, score, flags, detail, basis in areas:
                color = ("green" if score >= 0.75
                         else "orange" if score >= 0.55
                         else "red")
                with st.expander(
                    f":{color}[Score: {score:.3f}] {area_name}"
                ):
                    st.caption(detail)
                    st.caption(basis)
                    if flags:
                        for flag in flags:
                            st.warning(f"⚠ {flag}")
                    else:
                        st.success("✓ No issues detected in this area")

            # Proxy bias
            if l8.evidence_source.proxy_bias_detected:
                st.markdown("#### 🚨 Proxy Bias Alert")
                st.error(
                    f"**Demographic proxy detected:** "
                    f"{l8.evidence_source.proxy_details}"
                )
            if l8.individual_profile.proxy_risks:
                st.markdown("#### Proxy Risk Indicators")
                for risk in l8.individual_profile.proxy_risks:
                    st.warning(f"⚠ {risk}")

            # Bias flags with full detail
            if l8.bias_flags:
                st.markdown("#### Detected Bias Flags")
                for flag in l8.bias_flags:
                    sev_color = (
                        "red" if flag.severity in ("high","critical")
                        else "orange"
                    )
                    with st.expander(
                        f":{sev_color}[{flag.severity.upper()}] "
                        f"{flag.bias_type} — {flag.dimension}"
                    ):
                        st.write(flag.description)
                        c1, c2 = st.columns(2)
                        c1.caption(f"**Observed:** {flag.observed_value}")
                        c2.caption(f"**Threshold:** {flag.threshold}")
                        st.markdown("**Root cause:**")
                        st.write(flag.root_cause_indicator)
                        st.markdown("**Recommended action:**")
                        st.info(flag.recommended_action)
                        if flag.regulation_reference:
                            st.caption(
                                f"📋 Regulation: {flag.regulation_reference}"
                            )

            # Remediation
            st.markdown("#### Remediation Actions")
            for action in l8.remediation_actions:
                st.info(f"→ {action}")

            # Fairness summary
            st.markdown("#### Fairness Summary")
            if l8.verdict == "Fair":
                st.success(l8.fairness_summary)
            elif l8.verdict == "Caution":
                st.warning(l8.fairness_summary)
            else:
                st.error(l8.fairness_summary)

            st.markdown(
                '<div class="xai-note">🔍 '
                '<b>Fairness Explainability:</b> '
                'L8 detects unjustified patterns across decisions. '
                'Everything may be compliant and accurate — yet the system '
                'could still drift into systematic unfairness without anyone '
                'noticing. L8 is the continuous fairness assurance loop. '
                'As more applications accumulate, this analysis becomes '
                'richer and more reliable.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("L8 not available for this application.")

    # ── Agent Trace ───────────────────────────────────────────────
    with tab9:
        st.markdown("### 🤖 Agent Reasoning Trace")
        st.markdown("*The LLM's reasoning at every layer — this is what makes LEAF agentic*")

        if trace:
            observations = trace.get("agent_observations", {})
            decisions_map = trace.get("layer_decisions", {})
            for layer in ["L0","L1","L2","L3","L4","L6","L7","L8"]:
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
    with tab10:
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
