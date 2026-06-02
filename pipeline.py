"""
LEAF Credit Agent — Pipeline Runner
Chains L0 → L1 → L2 → L3 → L4 → L5 → L6 and seals every
artifact to the Evidence Ledger.

Use this for testing individual layers without the LLM agent.
L5 requires an LLM provider — pass one in or set OPENAI_API_KEY.
"""

from models.schemas import LoanApplication, JurisdictionCode
from layers.l0_request import L0RequestContext
from layers.l1_provenance import L1DataProvenance
from layers.l2_grounding import L2GroundingCheck
from layers.l3_signals import L3SignalExtraction
from layers.l4_model import L4ModelReasoning_Layer
from layers.l6_confidence import L6ConfidenceGrade_Layer
from layers.l7_compliance import L7Compliance_Layer
from layers.l8_fairness import L8FairnessDiagnostics_Layer
from layers.l9_human_loop import L9HumanLoop_Layer, L9PendingReview
from layers.l10_audit import L10AuditLedger_Layer
from storage.ledger import init_ledger, seal_artifact
from typing import Optional


def run_pipeline(
    application: LoanApplication,
    jurisdiction: JurisdictionCode = JurisdictionCode.INDIA,
    llm_provider=None,
    verbose: bool = True,
) -> dict:
    """
    Run the full LEAF pipeline L0 through L6.
    L5 (LLM recommendation) is included only when llm_provider is supplied.
    L6 (Confidence Grade) always runs — it does not need an LLM.

    Returns a dict with all layer artifacts for inspection.
    """

    init_ledger()
    results = {}

    # ── L0 ───────────────────────────────────────────────────────
    if verbose:
        print("\n[LEAF] ▶ L0 — Request & Context")
    l0 = L0RequestContext(jurisdiction=jurisdiction).process(application)
    seal_artifact(l0.application_id, "L0", l0.model_dump())
    results["L0"] = l0
    if verbose:
        print(f"       Application ID : {l0.application_id}")
        print(f"       Jurisdiction   : {l0.jurisdiction}")
        print(f"       Frameworks     : "
              f"{[f.value for f in l0.regulatory_frameworks]}")
        print(f"       Adverse Notice : {l0.adverse_action_notice_required}")

    # ── L1 ───────────────────────────────────────────────────────
    if verbose:
        print("\n[LEAF] ▶ L1 — Data Provenance")
    l1, raw_data = L1DataProvenance().process(application, l0.application_id)
    seal_artifact(l0.application_id, "L1", l1.model_dump())
    results["L1"] = l1
    if verbose:
        for src in l1.sources:
            warn = f"  ⚠ {src.freshness_warning}" if src.freshness_warning else ""
            print(f"       {src.source_name:<40} "
                  f"hash={src.integrity_hash}  age={src.age_hours:.0f}h{warn}")

    # ── L2 ───────────────────────────────────────────────────────
    if verbose:
        print("\n[LEAF] ▶ L2 — Grounding Check")
    l2 = L2GroundingCheck().process(l1, raw_data, l0.application_id)
    seal_artifact(l0.application_id, "L2", l2.model_dump())
    results["L2"] = l2
    if verbose:
        for s in l2.source_scores:
            print(f"       {s.source_name:<40} "
                  f"score={s.composite_score:.2f}  status={s.status.value}")
        print(f"       Composite score : {l2.composite_grounding_score:.2f}")
        print(f"       Proceed to model: "
              f"{'✓ Yes' if l2.proceed_to_model else '✗ No'}")

    if not l2.proceed_to_model:
        if verbose:
            print("\n[LEAF] ✗ Grounding too low — halting pipeline")
        return {"status": "halted", "reason": "grounding_below_threshold",
                "results": results}

    # ── L3 ───────────────────────────────────────────────────────
    if verbose:
        print("\n[LEAF] ▶ L3 — Signal Extraction")
    l3 = L3SignalExtraction().process(raw_data, application, l0.application_id)
    seal_artifact(l0.application_id, "L3", l3.model_dump())
    results["L3"] = l3
    if verbose:
        for sig in l3.signals:
            arrow = ("▲" if sig.risk_direction == "positive"
                     else "▼" if sig.risk_direction == "negative" else "─")
            print(f"       {arrow} {sig.signal_name:<35} {sig.display_value}")
        print(f"\n       Model-ready features : {len(l3.model_ready_features)}")
        print(f"       Positive signals    : {l3.positive_signals}")
        print(f"       Negative signals    : {l3.negative_signals}")

    # ── L4 ───────────────────────────────────────────────────────
    if verbose:
        print("\n[LEAF] ▶ L4 — Model Reasoning + SHAP")
    l4 = L4ModelReasoning_Layer().process(
        l3.model_ready_features, l0.application_id
    )
    seal_artifact(l0.application_id, "L4", l4.model_dump())
    results["L4"] = l4
    if verbose:
        print(f"       Decision            : {l4.decision}")
        print(f"       Approval probability: {l4.approval_probability:.1%}")
        print(f"       Confidence          : {l4.decision_confidence}")
        print(f"       Top approval factors: {l4.top_approval_factors}")
        print(f"       Top rejection factors:{l4.top_rejection_factors}")

    # ── L6 ───────────────────────────────────────────────────────
    if verbose:
        print("\n[LEAF] ▶ L6 — Confidence Grade")
    l6 = L6ConfidenceGrade_Layer().process(
        l1, l2, l3, l4, l0.application_id
    )
    seal_artifact(l0.application_id, "L6", l6.model_dump())
    results["L6"] = l6
    if verbose:
        print(f"       Grade               : {l6.grade}")
        print(f"       Composite Score     : {l6.composite_score:.3f}")
        print(f"       Meaning             : {l6.grade_meaning}")
        print(f"       HITL Required       : {l6.hitl_required}")
        print(f"       Decision Blocked    : {l6.decision_blocked}")
        print(f"       Grounding Fidelity  : {l6.grounding_fidelity.composite:.3f}")
        print(f"       Freshness/Coverage  : {l6.freshness_coverage.composite:.3f}")
        print(f"       Model Consistency   : {l6.model_consistency.composite:.3f}")
        print(f"       Calibration         : {l6.calibration.composite:.3f}")
        print(f"       Compliance          : {l6.compliance_suitability.composite:.3f}")
        if l6.grounding_fidelity.contradictions_detected:
            for c in l6.grounding_fidelity.contradictions_detected:
                print(f"       ⚠ Contradiction: {c}")
        if l6.model_consistency.consistency_flags:
            for f in l6.model_consistency.consistency_flags:
                print(f"       ⚠ Consistency : {f}")

    if l6.decision_blocked:
        if verbose:
            print("\n[LEAF] ✗ Grade D — decision blocked, mandatory human review")
        return {"status": "blocked", "reason": "grade_D", "results": results}

    # ── L7 ───────────────────────────────────────────────────────
    if verbose:
        print("\n[LEAF] ▶ L7 — Compliance & Suitability")
    l7 = L7Compliance_Layer().process(
        l0, l3, l4, l6, application_id=l0.application_id
    )
    seal_artifact(l0.application_id, "L7", l7.model_dump())
    results["L7"] = l7
    if verbose:
        print(f"       Legitimacy Verdict  : {l7.legitimacy_verdict.value}")
        print(f"       Decision Legitimate : {l7.decision_legitimate}")
        print(f"       Override Required   : {l7.override_required}")
        print(f"       Compliance          : {l7.compliance.overall_status}")
        print(f"       Suitability         : {l7.suitability.suitability_label.value} "
              f"({l7.suitability.overall_score:.3f})")
        if l7.compliance.blocking_violations:
            for v in l7.compliance.blocking_violations:
                print(f"       ✗ Blocking: {v}")
        if l7.override_reason:
            print(f"       ⚠ Override: {l7.override_reason}")

    # ── L8 ───────────────────────────────────────────────────────
    if verbose:
        print("\n[LEAF] ▶ L8 — Bias & Fairness Diagnostics")
    l8 = L8FairnessDiagnostics_Layer().process(
        l0, l3, l4, l6, l7, application_id=l0.application_id
    )
    seal_artifact(l0.application_id, "L8", l8.model_dump())
    results["L8"] = l8
    if verbose:
        print(f"       Verdict            : {l8.verdict}")
        print(f"       Overall Bias Score : {l8.obs:.3f}")
        print(f"       Investigation      : {l8.investigation_required}")
        print(f"       Flags detected     : {l8.total_flags}")
        for flag in l8.bias_flags:
            print(f"       ⚠ [{flag.severity}] {flag.bias_type}")

    # ── L9 ───────────────────────────────────────────────────────
    if verbose:
        print("\n[LEAF] ▶ L9 — Human-in-the-Loop")
    l9_pending = L9HumanLoop_Layer().process(
        l0, l4, l6, l7, l8,
        l5=results.get("L5"),
        application_id=l0.application_id,
    )
    seal_artifact(l0.application_id, "L9_PENDING", l9_pending.model_dump())
    results["L9"] = l9_pending
    if verbose:
        print(f"       HITL Required   : {l9_pending.hitl_required}")
        print(f"       Triggers        : "
              f"{[t.value for t in l9_pending.triggers]}")
        print(f"       Trigger Summary : {l9_pending.trigger_summary}")
        print(f"       Review ID       : {l9_pending.review_id}")
        status = ("⚠ Awaiting human review" if l9_pending.hitl_required
                  else "✓ Auto-logged")
        print(f"       Status          : {status}")

    # ── L5 (optional — needs LLM) ─────────────────────────────────
    if llm_provider:
        if verbose:
            print("\n[LEAF] ▶ L5 — Recommendation & Explanation Card")
        from layers.l5_recommendation import L5Recommendation_Layer
        l5 = L5Recommendation_Layer(llm_provider=llm_provider).process(
            l0, l3, l4, l0.application_id
        )
        seal_artifact(l0.application_id, "L5", l5.model_dump())
        results["L5"] = l5
        if verbose:
            print(f"       Decision   : {l5.decision}")
            print(f"       Rate Band  : {l5.interest_rate_band}")
            print(f"       Rationale  : {l5.plain_language_rationale[:80]}...")
    else:
        if verbose:
            print("\n[LEAF] ℹ L5 skipped — no LLM provider supplied")

    # ── L10 ──────────────────────────────────────────────────────
    if verbose:
        print("\n[LEAF] ▶ L10 — Auditability & Reproducibility Ledger")
    l10 = L10AuditLedger_Layer().process(l0.application_id)
    seal_artifact(l0.application_id, "L10", l10.model_dump())
    results["L10"] = l10
    if verbose:
        cert = l10.audit_certificate
        print(f"       Audit Verdict      : {cert.audit_verdict}")
        print(f"       Completeness       : "
              f"{l10.completeness.completeness_score:.0%} "
              f"({l10.completeness.completeness_label})")
        print(f"       Integrity          : "
              f"{l10.integrity.integrity_status}")
        print(f"       Reproducibility    : "
              f"{l10.reproducibility.reproducibility_score:.0%}")
        print(f"       Binding Hash       : {l10.integrity.binding_hash}")
        print(f"       Certificate ID     : {cert.certificate_id}")

    if verbose:
        layers = "L0, L1, L2, L3, L4, L6, L7, L8, L9, L10" + \
                 (", L5" if llm_provider else "")
        print(f"\n[LEAF] ✓ Pipeline complete — {l0.application_id}")
        print(f"       Layers sealed: {layers}\n")

    return {
        "status": "complete",
        "application_id": l0.application_id,
        "results": results,
    }


if __name__ == "__main__":
    sample = LoanApplication(
        applicant_id="CUST-2024-00471",
        amount_requested=450000,
        purpose="Home renovation",
        tenure_months=48,
        applicant_name="Suresh Kumar",
        applicant_age=34,
        employment_type="salaried",
        monthly_income_declared=72400,
        existing_loans=2,
    )
    run_pipeline(sample, verbose=True)
