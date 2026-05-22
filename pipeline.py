"""
LEAF Credit Agent — Pipeline Runner
Chains L0 → L1 → L2 → L3 and seals every artifact to the Evidence Ledger.
This is the entry point for running a credit application through LEAF.
"""

from models.schemas import LoanApplication, JurisdictionCode
from layers.l0_request import L0RequestContext
from layers.l1_provenance import L1DataProvenance
from layers.l2_grounding import L2GroundingCheck
from layers.l3_signals import L3SignalExtraction
from storage.ledger import init_ledger, seal_artifact


def run_sprint1_pipeline(
    application: LoanApplication,
    jurisdiction: JurisdictionCode = JurisdictionCode.INDIA,
    verbose: bool = True
) -> dict:
    """
    Run the Sprint 1 LEAF pipeline: L0 through L3.
    Each layer output is sealed to the Evidence Ledger.

    Returns a dict with all layer artifacts for inspection.
    """

    init_ledger()

    results = {}

    # ── L0: Request & Context ─────────────────────────────────────
    if verbose:
        print("\n[LEAF] ▶ L0 — Request & Context")
    l0 = L0RequestContext(jurisdiction=jurisdiction)
    intake = l0.process(application)
    seal_artifact(intake.application_id, "L0", intake.model_dump())
    results["L0"] = intake
    if verbose:
        print(f"       Application ID : {intake.application_id}")
        print(f"       Jurisdiction   : {intake.jurisdiction}")
        print(f"       Frameworks     : {[f.value for f in intake.regulatory_frameworks]}")
        print(f"       Adverse Notice : {intake.adverse_action_notice_required}")

    # ── L1: Data Provenance ───────────────────────────────────────
    if verbose:
        print("\n[LEAF] ▶ L1 — Data Provenance")
    l1 = L1DataProvenance()
    provenance, raw_data = l1.process(application, intake.application_id)
    seal_artifact(intake.application_id, "L1", provenance.model_dump())
    results["L1"] = provenance
    if verbose:
        for src in provenance.sources:
            warn = f"  ⚠ {src.freshness_warning}" if src.freshness_warning else ""
            print(f"       {src.source_name:<40} hash={src.integrity_hash}  age={src.age_hours:.0f}h{warn}")

    # ── L2: Grounding Check ───────────────────────────────────────
    if verbose:
        print("\n[LEAF] ▶ L2 — Grounding Check")
    l2 = L2GroundingCheck()
    grounding = l2.process(provenance, raw_data, intake.application_id)
    seal_artifact(intake.application_id, "L2", grounding.model_dump())
    results["L2"] = grounding
    if verbose:
        for s in grounding.source_scores:
            print(f"       {s.source_name:<40} score={s.composite_score:.2f}  status={s.status.value}")
        print(f"       Composite score : {grounding.composite_grounding_score:.2f}")
        print(f"       Proceed to model: {'✓ Yes' if grounding.proceed_to_model else '✗ No — human review required'}")

    # ── L3: Signal Extraction ─────────────────────────────────────
    if verbose:
        print("\n[LEAF] ▶ L3 — Signal Extraction")
    l3 = L3SignalExtraction()
    signals = l3.process(raw_data, application, intake.application_id)
    seal_artifact(intake.application_id, "L3", signals.model_dump())
    results["L3"] = signals
    if verbose:
        for sig in signals.signals:
            arrow = "▲" if sig.risk_direction == "positive" else ("▼" if sig.risk_direction == "negative" else "─")
            print(f"       {arrow} {sig.signal_name:<35} {sig.display_value}")
        print(f"\n       Model-ready features: {len(signals.model_ready_features)}")
        print(f"       Positive signals   : {signals.positive_signals}")
        print(f"       Negative signals   : {signals.negative_signals}")

    if verbose:
        print(f"\n[LEAF] ✓ Sprint 1 complete. Evidence Ledger sealed for {intake.application_id}")
        print(f"       Layers sealed: L0, L1, L2, L3\n")

    return results


# ─────────────────────────────────────────────
# Quick test runner
# ─────────────────────────────────────────────
if __name__ == "__main__":
    sample_application = LoanApplication(
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

    results = run_sprint1_pipeline(sample_application, verbose=True)
