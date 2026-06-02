"""
LEAF Layer 9 — Human-in-the-Loop & Overrides
=====================================================================
Core question: "How can humans supervise, modify, challenge,
approve, or reject AI recommendations while maintaining full
explainability?"

Layer 9 introduces Accountability Explainability — the ninth type
in the LEAF explainability taxonomy:

    L1 — Source Explainability
    L2 — Retrieval Explainability
    L3 — Signal Explainability
    L4 — Reasoning Explainability
    L5 — Decision Explainability
    L6 — Confidence Explainability
    L7 — Governance Explainability
    L8 — Fairness Explainability
    L9 — Accountability Explainability  ← this layer

The fundamental truth this layer recognises:
    Even a perfectly explainable AI should not always have
    the final say. Humans possess information, judgment, context,
    and responsibility that AI does not.

Five core functions:
    1. Human Review    — inspect all layer outputs before deciding
    2. Human Decision  — Approve / Modify / Reject / Escalate
    3. Override Types  — recommendation, confidence, risk, suitability
    4. Reason & Justification — mandatory documented rationale
    5. Override Logging — who, when, what changed, why

Two states:
    Pending   — awaiting human action (created by pipeline)
    Completed — human has acted (created by Streamlit form)

HITL is triggered when:
    L6 grade C or D        → hitl_required = True
    L7 Conditional/Blocked → override_required = True
    L8 Caution/Biased      → investigation_required = True
    Manual                 → loan officer chooses to review

Doctoral contribution:
    L9 operationalises the human oversight requirements of
    EU AI Act Article 14, RBI Model Risk Management, and
    SR 11-7 model governance — making them visible, auditable,
    and demonstrably implemented in the system.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from models.schemas import L0IntakeRecord, L3SignalLog
from layers.l4_model import L4ModelReasoning
from layers.l5_recommendation import L5Recommendation
from layers.l6_confidence import L6ConfidenceGrade
from layers.l7_compliance import L7GovernanceVerdict
from layers.l8_fairness import L8FairnessDiagnostics


# ─────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────

class ReviewDecision(str, Enum):
    APPROVE   = "Approve"
    MODIFY    = "Modify"
    REJECT    = "Reject"
    ESCALATE  = "Escalate"
    PENDING   = "Pending"


class ReviewTrigger(str, Enum):
    L6_GRADE_C    = "L6 Grade C — Moderate confidence"
    L6_GRADE_D    = "L6 Grade D — Low confidence"
    L7_CONDITIONAL = "L7 Conditional — Human review required"
    L7_BLOCKED    = "L7 Blocked — Governance violation"
    L8_CAUTION    = "L8 Caution — Fairness concerns"
    L8_BIASED     = "L8 Biased — Bias detected"
    MANUAL        = "Manual — Loan officer initiated"
    AUTO          = "Auto-logged — No trigger (record only)"


class OverrideType(str, Enum):
    RECOMMENDATION = "Recommendation"
    CONFIDENCE     = "Confidence score"
    RISK_RATING    = "Risk rating"
    SUITABILITY    = "Suitability assessment"
    INTEREST_RATE  = "Interest rate band"
    ASSUMPTION     = "Underlying assumption"
    NONE           = "No override"


# ─────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────

class LayerSummaryForReviewer(BaseModel):
    """
    Compact summary of all previous layers shown to the reviewer.
    The reviewer sees this before making their decision.
    """
    # L4 — What did the model decide?
    model_decision: str
    model_approval_probability: float
    model_confidence_label: str
    top_approval_factors: List[str]
    top_rejection_factors: List[str]
    counterfactual_hint: str

    # L5 — What explanation was generated?
    explanation_summary: str
    interest_rate_band: str

    # L6 — How confident is the system?
    confidence_grade: str
    confidence_score: float
    confidence_meaning: str
    hitl_required_by_l6: bool

    # L7 — Is it compliant and suitable?
    legitimacy_verdict: str
    compliance_status: str
    suitability_label: str
    override_required_by_l7: bool
    blocking_violations: List[str]

    # L8 — Is it fair?
    fairness_verdict: str
    fairness_obs: float
    fairness_flags: int
    investigation_required_by_l8: bool


class L9PendingReview(BaseModel):
    """
    Created by the pipeline when HITL is triggered.
    Awaits human action through the Streamlit interface.
    """
    layer: str = "L9"
    review_id: str
    application_id: str

    # Why was review triggered?
    triggers: List[ReviewTrigger]
    hitl_required: bool
    trigger_summary: str

    # Context for reviewer
    layer_summary: LayerSummaryForReviewer

    # Application context
    applicant_name: str
    loan_amount: float
    loan_purpose: str
    tenure_months: int

    # Status
    status: str = "Pending"
    created_at: datetime

    explainability_type: str = "Accountability Explainability"


class OverrideRecord(BaseModel):
    """Details of what was changed during a Modify action."""
    override_type: OverrideType
    original_value: str
    new_value: str
    reason_for_change: str


class L9HumanReviewRecord(BaseModel):
    """
    Completed after human action.
    Sealed to the Evidence Ledger.
    This is the accountability artifact — who, when, what, why.
    """
    layer: str = "L9"
    review_id: str
    application_id: str

    # Reviewer identity
    reviewer_id: str
    reviewer_name: str
    reviewer_role: str

    # Decision
    decision: ReviewDecision
    decision_rationale: str = Field(...,
        description="Mandatory justification — cannot be empty")

    # Before and after
    original_model_decision: str
    final_decision: str
    original_confidence: float
    final_confidence: float
    confidence_adjusted: bool

    # Override details
    overrides_applied: List[OverrideRecord] = Field(default_factory=list)
    evidence_considered: str

    # Escalation details (if escalated)
    escalation_target: Optional[str] = None
    escalation_reason: Optional[str] = None

    # Timestamps
    review_started_at: datetime
    review_completed_at: datetime
    time_taken_seconds: float

    # Governance
    approval_status: str = Field(...,
        description="Approved / Modified / Rejected / Escalated")
    regulatory_basis: str = Field(
        default="RBI Model Risk Management — Human oversight requirements. "
                "EU AI Act Article 14 — Human oversight measures.",
    )

    # Audit
    audit_trail_complete: bool = True
    explainability_type: str = "Accountability Explainability"

    xai_note: str = Field(
        default="L9 provides Accountability Explainability — "
                "explaining who changed a recommendation, "
                "why they changed it, what changed, and when. "
                "This operationalises EU AI Act Article 14 and "
                "RBI model governance requirements.",
    )


# ─────────────────────────────────────────────────────────────────
# Trigger detection
# ─────────────────────────────────────────────────────────────────

def _detect_triggers(
    l6: L6ConfidenceGrade,
    l7: L7GovernanceVerdict,
    l8: L8FairnessDiagnostics,
) -> tuple:
    """
    Determine which triggers require human review.
    Returns (triggers list, hitl_required bool, summary string)
    """
    triggers = []
    summary_parts = []

    # L6 triggers
    if l6.grade == "D":
        triggers.append(ReviewTrigger.L6_GRADE_D)
        summary_parts.append(f"Grade D confidence ({l6.composite_score:.2f})")
    elif l6.grade == "C":
        triggers.append(ReviewTrigger.L6_GRADE_C)
        summary_parts.append(f"Grade C confidence ({l6.composite_score:.2f})")

    # L7 triggers
    if l7.legitimacy_verdict.value == "Blocked":
        triggers.append(ReviewTrigger.L7_BLOCKED)
        summary_parts.append("Governance blocked")
    elif l7.legitimacy_verdict.value == "Conditional":
        triggers.append(ReviewTrigger.L7_CONDITIONAL)
        summary_parts.append("Conditional legitimacy")

    # L8 triggers
    if l8.verdict == "Biased":
        triggers.append(ReviewTrigger.L8_BIASED)
        summary_parts.append("Bias detected")
    elif l8.verdict == "Caution":
        triggers.append(ReviewTrigger.L8_CAUTION)
        summary_parts.append("Fairness caution")

    hitl_required = len(triggers) > 0

    if not triggers:
        triggers.append(ReviewTrigger.AUTO)
        summary = "No critical triggers. Auto-logged for governance record."
    else:
        summary = f"Review required: {'; '.join(summary_parts)}."

    return triggers, hitl_required, summary


def _build_layer_summary(
    l4: L4ModelReasoning,
    l5: Optional[L5Recommendation],
    l6: L6ConfidenceGrade,
    l7: L7GovernanceVerdict,
    l8: L8FairnessDiagnostics,
) -> LayerSummaryForReviewer:
    """Build the compact reviewer summary from all layer outputs."""
    explanation_summary = ""
    interest_rate_band = "Pending L5"
    if l5:
        explanation_summary = l5.plain_language_rationale[:200]
        interest_rate_band = l5.interest_rate_band

    return LayerSummaryForReviewer(
        model_decision=l4.decision,
        model_approval_probability=l4.approval_probability,
        model_confidence_label=l4.decision_confidence,
        top_approval_factors=l4.top_approval_factors,
        top_rejection_factors=l4.top_rejection_factors,
        counterfactual_hint=l4.counterfactual_hint,
        explanation_summary=explanation_summary,
        interest_rate_band=interest_rate_band,
        confidence_grade=l6.grade,
        confidence_score=l6.composite_score,
        confidence_meaning=l6.grade_meaning,
        hitl_required_by_l6=l6.hitl_required,
        legitimacy_verdict=l7.legitimacy_verdict.value,
        compliance_status=l7.compliance.overall_status,
        suitability_label=l7.suitability.suitability_label.value,
        override_required_by_l7=l7.override_required,
        blocking_violations=l7.compliance.blocking_violations,
        fairness_verdict=l8.verdict,
        fairness_obs=l8.obs,
        fairness_flags=l8.total_flags,
        investigation_required_by_l8=l8.investigation_required,
    )


# ─────────────────────────────────────────────────────────────────
# Layer implementation
# ─────────────────────────────────────────────────────────────────

class L9HumanLoop_Layer:
    """
    Layer 9: Human-in-the-Loop & Overrides

    Input  : L0, L4, L5 (optional), L6, L7, L8
    Output : L9PendingReview (awaits human action via Streamlit)

    The pipeline creates the PendingReview.
    The human completes it through the interactive interface.
    The CompletedReview is sealed to the Evidence Ledger.
    """

    def process(
        self,
        l0: L0IntakeRecord,
        l4: L4ModelReasoning,
        l6: L6ConfidenceGrade,
        l7: L7GovernanceVerdict,
        l8: L8FairnessDiagnostics,
        l5: Optional[L5Recommendation] = None,
        application_id: str = "",
    ) -> L9PendingReview:

        app_id = application_id or l0.application_id
        review_id = f"REV-{datetime.now().strftime('%Y%m%d%H%M%S')}-" \
                    f"{str(uuid.uuid4())[:6].upper()}"

        # Detect triggers
        triggers, hitl_required, trigger_summary = _detect_triggers(
            l6, l7, l8
        )

        # Build reviewer summary
        layer_summary = _build_layer_summary(l4, l5, l6, l7, l8)

        return L9PendingReview(
            review_id=review_id,
            application_id=app_id,
            triggers=triggers,
            hitl_required=hitl_required,
            trigger_summary=trigger_summary,
            layer_summary=layer_summary,
            applicant_name=l0.applicant_id,
            loan_amount=l0.amount_requested,
            loan_purpose=l0.purpose,
            tenure_months=l0.tenure_months,
            status="Pending",
            created_at=datetime.now(),
        )


# ─────────────────────────────────────────────────────────────────
# Review completion helper (called from Streamlit form)
# ─────────────────────────────────────────────────────────────────

def complete_review(
    pending: L9PendingReview,
    reviewer_name: str,
    reviewer_role: str,
    decision: ReviewDecision,
    reason: str,
    evidence_considered: str,
    overrides: Optional[List[OverrideRecord]] = None,
    new_confidence: Optional[float] = None,
    escalation_target: Optional[str] = None,
    review_started_at: Optional[datetime] = None,
) -> L9HumanReviewRecord:
    """
    Called from the Streamlit form when the reviewer submits.
    Creates the completed accountability record.
    """
    now = datetime.now()
    started = review_started_at or now
    time_taken = (now - started).total_seconds()

    original_decision = pending.layer_summary.model_decision
    original_confidence = pending.layer_summary.confidence_score

    # Determine final decision text
    if decision == ReviewDecision.APPROVE:
        final_decision = original_decision
        approval_status = "Approved"
    elif decision == ReviewDecision.MODIFY:
        # Use the override to determine final decision
        rec_override = next(
            (o for o in (overrides or [])
             if o.override_type == OverrideType.RECOMMENDATION),
            None
        )
        final_decision = (
            rec_override.new_value if rec_override
            else original_decision
        )
        approval_status = "Modified"
    elif decision == ReviewDecision.REJECT:
        final_decision = "Rejected by reviewer"
        approval_status = "Rejected"
    else:  # ESCALATE
        final_decision = "Escalated for senior review"
        approval_status = "Escalated"

    final_confidence = new_confidence if new_confidence else original_confidence
    confidence_adjusted = (
        new_confidence is not None
        and abs(new_confidence - original_confidence) > 0.01
    )

    reviewer_id = f"REV-{reviewer_name[:3].upper()}-" \
                  f"{datetime.now().strftime('%H%M')}"

    return L9HumanReviewRecord(
        review_id=pending.review_id,
        application_id=pending.application_id,
        reviewer_id=reviewer_id,
        reviewer_name=reviewer_name,
        reviewer_role=reviewer_role,
        decision=decision,
        decision_rationale=reason,
        original_model_decision=original_decision,
        final_decision=final_decision,
        original_confidence=original_confidence,
        final_confidence=final_confidence,
        confidence_adjusted=confidence_adjusted,
        overrides_applied=overrides or [],
        evidence_considered=evidence_considered,
        escalation_target=escalation_target,
        escalation_reason=reason if decision == ReviewDecision.ESCALATE
        else None,
        review_started_at=started,
        review_completed_at=now,
        time_taken_seconds=round(time_taken, 1),
        approval_status=approval_status,
    )
