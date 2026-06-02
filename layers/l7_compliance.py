"""
LEAF Layer 7 — Compliance & Suitability (Governance Explainability)
=====================================================================
Core question: "Even if the recommendation is correct, should it be allowed?"

This layer introduces a fundamentally new type of explainability —
Governance Explainability — which is distinct from all previous layers:

    L1 — Source Explainability      (what data was used?)
    L2 — Retrieval Explainability   (how trustworthy is the data?)
    L3 — Signal Explainability      (how were features computed?)
    L4 — Reasoning Explainability   (why did the model decide this?)
    L5 — Decision Explainability    (what does the decision mean?)
    L6 — Confidence Explainability  (how trustworthy is the explanation?)
    L7 — Governance Explainability  (is this decision legitimate?)

The distinction that matters:
    L4-L6 answer: Is the decision CORRECT?
    L7  answers:  Is the decision LEGITIMATE?

These are fundamentally different questions.
A decision can be correct (model is right) but illegitimate (unsuitable
for this specific borrower, violates a regulation, or breaches a policy).

Two components:
    Component 1 — Compliance: Is this recommendation legally and
                               procedurally acceptable?
    Component 2 — Suitability: Is this recommendation appropriate
                                for THIS borrower?

Inputs from earlier layers:
    L0 — applicant profile, jurisdiction, regulatory context
    L3 — extracted signals (DTI, income, CIBIL, etc.)
    L5 — recommendation (decision, rate band, tenure)
    L6 — confidence grade

Doctoral contribution:
    L7 directly addresses Research Gap 5 (disconnect between regulatory
    expectations and operational practices) from Chapter 2.
    It operationalises governance as a first-class system component,
    not a post-hoc compliance check.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from models.schemas import L0IntakeRecord, L3SignalLog
from layers.l4_model import L4ModelReasoning
from layers.l5_recommendation import L5Recommendation
from layers.l6_confidence import L6ConfidenceGrade


# ─────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────

class CheckSeverity(str, Enum):
    HARD = "hard"       # Blocks decision entirely
    SOFT = "soft"       # Advisory — documented but proceeds


class CheckResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class SuitabilityLabel(str, Enum):
    SUITABLE = "Suitable"
    PARTIAL = "Partial Match"
    UNSUITABLE = "Unsuitable"


class LegitimacyVerdict(str, Enum):
    LEGITIMATE = "Legitimate"
    CONDITIONAL = "Conditional"
    BLOCKED = "Blocked"


# ─────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────

class ComplianceCheck(BaseModel):
    """A single compliance rule check with full audit trail."""
    rule_id: str
    rule_name: str
    regulation_reference: str = Field(...,
        description="Specific regulation or guideline this rule derives from")
    result: CheckResult
    severity: CheckSeverity
    actual_value: str = Field(...,
        description="The actual value that was checked")
    threshold: str = Field(...,
        description="The threshold or limit that was applied")
    details: str = Field(...,
        description="Plain language explanation of this check")
    blocks_decision: bool = Field(default=False)


class ComplianceResult(BaseModel):
    """
    Component 1 — Compliance
    Answers: Is this recommendation legally and procedurally acceptable?
    """
    checks: List[ComplianceCheck]
    total_checks: int
    passed: int
    failed: int
    warnings: int
    blocking_violations: List[str] = Field(default_factory=list,
        description="Hard failures that block the decision")
    advisory_warnings: List[str] = Field(default_factory=list,
        description="Soft warnings that are documented but do not block")
    overall_status: str = Field(...,
        description="Compliant / Non-Compliant / Compliant with Warnings")
    is_blocking: bool = Field(default=False,
        description="True if any hard failure exists")


class SuitabilityDimension(BaseModel):
    """Assessment of one suitability dimension."""
    dimension: str
    score: float = Field(..., ge=0, le=1)
    label: str = Field(..., description="Suitable / Partial / Unsuitable")
    actual: str = Field(..., description="Applicant's actual value")
    benchmark: str = Field(..., description="What would be ideal for this product")
    assessment: str = Field(..., description="Plain language assessment")


class SuitabilityResult(BaseModel):
    """
    Component 2 — Suitability
    Answers: Is this recommendation appropriate for THIS borrower?
    """
    dimensions: List[SuitabilityDimension]
    overall_score: float = Field(..., ge=0, le=1)
    suitability_label: SuitabilityLabel
    primary_concern: Optional[str] = Field(None,
        description="The most significant suitability concern if any")
    suitability_narrative: str = Field(...,
        description="Plain language suitability assessment")


class L7GovernanceVerdict(BaseModel):
    """
    L7 complete output — the Governance Certificate.

    This is the layer that answers:
    'Is this decision not only correct and confident,
     but also compliant, appropriate, and permissible
     for the specific borrower receiving it?'
    """
    layer: str = "L7"
    application_id: str

    # Two components
    compliance: ComplianceResult
    suitability: SuitabilityResult

    # Combined legitimacy verdict — unique to L7
    legitimacy_verdict: LegitimacyVerdict = Field(...,
        description="Legitimate / Conditional / Blocked")
    decision_legitimate: bool = Field(...,
        description="True only when both compliance and suitability pass")
    override_required: bool = Field(...,
        description="True when human override needed before proceeding")
    override_reason: Optional[str] = Field(None)

    # Governance narrative — separate from model explanation
    governance_narrative: str = Field(...,
        description="Plain language governance assessment. "
                    "Distinct from L5's model explanation — "
                    "this addresses legitimacy not correctness.")

    # Explainability metadata
    explainability_type: str = "Governance Explainability"
    timestamp: datetime

    xai_note: str = Field(
        default="L7 introduces Governance Explainability — distinct from "
                "all previous layers. It answers not 'is this correct?' "
                "but 'is this legitimate?' This directly operationalises "
                "the regulatory expectations described in Chapter 2 "
                "Research Gap 5.",
    )


# ─────────────────────────────────────────────────────────────────
# Compliance checks — credit domain
# ─────────────────────────────────────────────────────────────────

def _run_compliance_checks(
    l0: L0IntakeRecord,
    l3: L3SignalLog,
    l4: L4ModelReasoning,
) -> ComplianceResult:
    """
    Run all compliance checks for credit lending.
    Hard failures block the decision.
    Soft warnings are documented but do not block.
    """
    checks = []
    signals = {s.signal_key: s.raw_value for s in l3.signals}

    cibil = signals.get("cibil_score", 700)
    dti = signals.get("dti_ratio", 0.4)
    income = signals.get("verified_monthly_income", 50000)
    existing_loans = signals.get("existing_loan_burden", 0)
    amount = l0.amount_requested

    # ── Check 1: CIBIL minimum threshold ────────────────────────
    cibil_pass = cibil >= 650
    checks.append(ComplianceCheck(
        rule_id="CC-001",
        rule_name="CIBIL Minimum Score",
        regulation_reference="RBI Fair Lending Guidelines — Credit Assessment Standards",
        result=CheckResult.PASS if cibil_pass else CheckResult.FAIL,
        severity=CheckSeverity.HARD,
        actual_value=f"{cibil:.0f}",
        threshold="≥ 650",
        details=(
            f"Applicant CIBIL score is {cibil:.0f}. "
            + ("Meets minimum threshold of 650." if cibil_pass
               else "Below minimum threshold of 650. Loan cannot proceed without senior approval.")
        ),
        blocks_decision=not cibil_pass,
    ))

    # ── Check 2: DTI ceiling ─────────────────────────────────────
    dti_pass = dti <= 0.55
    dti_warning = dti > 0.45
    checks.append(ComplianceCheck(
        rule_id="CC-002",
        rule_name="Debt-to-Income Ceiling",
        regulation_reference="RBI Model Risk Draft 2024 — Responsible Lending Norms",
        result=(CheckResult.PASS if dti_pass and not dti_warning
                else CheckResult.WARNING if dti_pass
                else CheckResult.FAIL),
        severity=CheckSeverity.HARD if not dti_pass else CheckSeverity.SOFT,
        actual_value=f"{dti:.1%}",
        threshold="≤ 55% (warning above 45%)",
        details=(
            f"Debt-to-income ratio is {dti:.1%}. "
            + ("Well within acceptable limits." if dti <= 0.45
               else "Elevated — above advisory threshold of 45% but within hard limit."
               if dti_pass else
               "Exceeds hard ceiling of 55%. Additional documentation required.")
        ),
        blocks_decision=not dti_pass,
    ))

    # ── Check 3: Loan amount to annual income ratio ──────────────
    annual_income = income * 12
    loan_to_income = amount / annual_income if annual_income > 0 else 99
    lti_pass = loan_to_income <= 10
    lti_warning = loan_to_income > 7
    checks.append(ComplianceCheck(
        rule_id="CC-003",
        rule_name="Loan-to-Income Ratio",
        regulation_reference="RBI Circular on Household Debt — Responsible Lending",
        result=(CheckResult.PASS if lti_pass and not lti_warning
                else CheckResult.WARNING if lti_pass
                else CheckResult.FAIL),
        severity=CheckSeverity.HARD if not lti_pass else CheckSeverity.SOFT,
        actual_value=f"{loan_to_income:.1f}x annual income",
        threshold="≤ 10x (warning above 7x)",
        details=(
            f"Loan amount ₹{amount:,.0f} is {loan_to_income:.1f}x "
            f"annual income of ₹{annual_income:,.0f}. "
            + ("Within acceptable range." if not lti_warning
               else "Elevated but within hard limit — document income verification."
               if lti_pass else "Exceeds maximum loan-to-income ratio of 10x.")
        ),
        blocks_decision=not lti_pass,
    ))

    # ── Check 4: Maximum existing loans ─────────────────────────
    max_loans_pass = existing_loans <= 4
    max_loans_warning = existing_loans >= 3
    checks.append(ComplianceCheck(
        rule_id="CC-004",
        rule_name="Maximum Active Loans",
        regulation_reference="RBI Guidelines on Multiple Lending — Credit Concentration",
        result=(CheckResult.PASS if max_loans_pass and not max_loans_warning
                else CheckResult.WARNING if max_loans_pass
                else CheckResult.FAIL),
        severity=CheckSeverity.SOFT,
        actual_value=f"{existing_loans:.0f} active loans",
        threshold="≤ 4 (warning at 3+)",
        details=(
            f"Applicant has {existing_loans:.0f} existing active loan(s). "
            + ("Within acceptable range." if not max_loans_warning
               else "High number of active loans — review total obligation carefully."
               if max_loans_pass else "Exceeds recommended maximum of 4 concurrent loans.")
        ),
        blocks_decision=False,
    ))

    # ── Check 5: Adverse action notice readiness ─────────────────
    adverse_ready = l0.adverse_action_notice_required
    checks.append(ComplianceCheck(
        rule_id="CC-005",
        rule_name="Adverse Action Notice Readiness",
        regulation_reference="RBI Fair Lending Guidelines — Consumer Protection",
        result=CheckResult.PASS if adverse_ready else CheckResult.WARNING,
        severity=CheckSeverity.SOFT,
        actual_value="Required" if adverse_ready else "Not flagged",
        threshold="Must be generated for all rejections",
        details=(
            "Adverse action notice requirement is correctly identified and "
            "will be generated if the application is declined. "
            if adverse_ready else
            "Adverse action notice requirement not flagged — review jurisdiction settings."
        ),
        blocks_decision=False,
    ))

    # ── Check 6: Minimum income for loan product ─────────────────
    min_income_required = max(15000, amount / 300)
    income_eligible = income >= min_income_required
    checks.append(ComplianceCheck(
        rule_id="CC-006",
        rule_name="Minimum Income Eligibility",
        regulation_reference="Product Policy — Retail Lending Eligibility Criteria",
        result=CheckResult.PASS if income_eligible else CheckResult.FAIL,
        severity=CheckSeverity.HARD,
        actual_value=f"₹{income:,.0f}/month",
        threshold=f"≥ ₹{min_income_required:,.0f}/month for this loan amount",
        details=(
            f"Verified monthly income ₹{income:,.0f} "
            + ("meets minimum eligibility requirement." if income_eligible
               else f"is below minimum ₹{min_income_required:,.0f} required for "
                    f"a ₹{amount:,.0f} loan.")
        ),
        blocks_decision=not income_eligible,
    ))

    # ── Aggregate results ────────────────────────────────────────
    passed = sum(1 for c in checks if c.result == CheckResult.PASS)
    failed = sum(1 for c in checks if c.result == CheckResult.FAIL)
    warnings = sum(1 for c in checks if c.result == CheckResult.WARNING)
    blocking = [c.rule_name for c in checks if c.blocks_decision]
    advisory = [c.details for c in checks if c.result == CheckResult.WARNING]

    if blocking:
        overall_status = "Non-Compliant"
        is_blocking = True
    elif warnings:
        overall_status = "Compliant with Warnings"
        is_blocking = False
    else:
        overall_status = "Compliant"
        is_blocking = False

    return ComplianceResult(
        checks=checks,
        total_checks=len(checks),
        passed=passed,
        failed=failed,
        warnings=warnings,
        blocking_violations=blocking,
        advisory_warnings=advisory,
        overall_status=overall_status,
        is_blocking=is_blocking,
    )


# ─────────────────────────────────────────────────────────────────
# Suitability assessment — credit domain
# ─────────────────────────────────────────────────────────────────

def _assess_suitability(
    l0: L0IntakeRecord,
    l3: L3SignalLog,
    l4: L4ModelReasoning,
    l5: Optional[L5Recommendation],
) -> SuitabilityResult:
    """
    Assess suitability across five dimensions.
    Same loan, same evidence, different suitability outcome
    depending on the specific borrower profile.
    This is the 'User A vs User B' insight from the LEAF document.
    """
    signals = {s.signal_key: s.raw_value for s in l3.signals}

    income = signals.get("verified_monthly_income", 50000)
    dti = signals.get("dti_ratio", 0.4)
    stability = signals.get("income_stability", 0.8)
    cibil = signals.get("cibil_score", 700)
    existing_loans = signals.get("existing_loan_burden", 0)
    amount = l0.amount_requested
    tenure = l0.tenure_months

    # Approximate EMI for this loan at 11.5%
    monthly_rate = 11.5 / 1200
    emi = (amount * monthly_rate) / (1 - (1 + monthly_rate) ** (-tenure))

    dimensions = []

    # ── Dimension 1: Affordability ────────────────────────────────
    emi_to_income = emi / income if income > 0 else 1.0
    if emi_to_income <= 0.20:
        aff_score, aff_label = 0.95, "Suitable"
        aff_assessment = (
            f"New EMI of ₹{emi:,.0f} is only {emi_to_income:.0%} of income — "
            f"very affordable."
        )
    elif emi_to_income <= 0.30:
        aff_score, aff_label = 0.78, "Suitable"
        aff_assessment = (
            f"New EMI of ₹{emi:,.0f} is {emi_to_income:.0%} of income — "
            f"within comfortable range."
        )
    elif emi_to_income <= 0.40:
        aff_score, aff_label = 0.58, "Partial Match"
        aff_assessment = (
            f"New EMI of ₹{emi:,.0f} is {emi_to_income:.0%} of income — "
            f"manageable but leaves limited financial buffer."
        )
    else:
        aff_score, aff_label = 0.30, "Unsuitable"
        aff_assessment = (
            f"New EMI of ₹{emi:,.0f} is {emi_to_income:.0%} of income — "
            f"leaves insufficient buffer for living expenses and emergencies."
        )

    dimensions.append(SuitabilityDimension(
        dimension="Affordability",
        score=round(aff_score, 3),
        label=aff_label,
        actual=f"EMI = ₹{emi:,.0f} ({emi_to_income:.0%} of income)",
        benchmark="EMI ≤ 30% of monthly income",
        assessment=aff_assessment,
    ))

    # ── Dimension 2: Tenure Alignment (life stage) ───────────────
    # Estimate age from application context — use a default of 35
    # In production this comes from L0 applicant profile
    # We proxy using income and loan purpose patterns
    loan_end_approx_age = 35 + (tenure / 12)  # proxy

    if tenure <= 36:
        ten_score, ten_label = 0.92, "Suitable"
        ten_assessment = (
            f"Short tenure of {tenure} months aligns well with "
            f"standard retail lending profiles."
        )
    elif tenure <= 60:
        ten_score, ten_label = 0.80, "Suitable"
        ten_assessment = (
            f"Medium tenure of {tenure} months is standard and appropriate "
            f"for this loan amount."
        )
    elif tenure <= 84:
        ten_score, ten_label = 0.65, "Partial Match"
        ten_assessment = (
            f"Extended tenure of {tenure} months increases total interest "
            f"burden — borrower should be advised of total cost."
        )
    else:
        ten_score, ten_label = 0.40, "Unsuitable"
        ten_assessment = (
            f"Very long tenure of {tenure} months significantly increases "
            f"total interest burden and extends financial commitment."
        )

    dimensions.append(SuitabilityDimension(
        dimension="Tenure Alignment",
        score=round(ten_score, 3),
        label=ten_label,
        actual=f"{tenure} months",
        benchmark="≤ 60 months for standard retail loans",
        assessment=ten_assessment,
    ))

    # ── Dimension 3: Debt Capacity ────────────────────────────────
    if dti <= 0.30:
        debt_score, debt_label = 0.95, "Suitable"
        debt_assessment = (
            f"Total DTI of {dti:.0%} is low — borrower has substantial "
            f"remaining debt capacity."
        )
    elif dti <= 0.45:
        debt_score, debt_label = 0.75, "Suitable"
        debt_assessment = (
            f"Total DTI of {dti:.0%} is moderate — borrower can manage "
            f"this obligation with disciplined budgeting."
        )
    elif dti <= 0.55:
        debt_score, debt_label = 0.50, "Partial Match"
        debt_assessment = (
            f"Total DTI of {dti:.0%} is elevated — borrower is near their "
            f"debt capacity ceiling. Limited financial headroom."
        )
    else:
        debt_score, debt_label = 0.20, "Unsuitable"
        debt_assessment = (
            f"Total DTI of {dti:.0%} exceeds recommended ceiling — "
            f"borrower does not have sufficient debt capacity for "
            f"this additional obligation."
        )

    dimensions.append(SuitabilityDimension(
        dimension="Debt Capacity",
        score=round(debt_score, 3),
        label=debt_label,
        actual=f"DTI = {dti:.0%} (including new loan)",
        benchmark="DTI ≤ 45% recommended",
        assessment=debt_assessment,
    ))

    # ── Dimension 4: Income Stability Suitability ─────────────────
    if stability >= 0.85:
        stab_score, stab_label = 0.95, "Suitable"
        stab_assessment = (
            f"Income stability of {stability:.2f} is strong — "
            f"consistent income pattern supports long-term repayment."
        )
    elif stability >= 0.70:
        stab_score, stab_label = 0.75, "Suitable"
        stab_assessment = (
            f"Income stability of {stability:.2f} is adequate — "
            f"minor income variability noted but within acceptable range."
        )
    elif stability >= 0.55:
        stab_score, stab_label = 0.50, "Partial Match"
        stab_assessment = (
            f"Income stability of {stability:.2f} is moderate — "
            f"income variability may affect repayment consistency."
        )
    else:
        stab_score, stab_label = 0.25, "Unsuitable"
        stab_assessment = (
            f"Income stability of {stability:.2f} is low — "
            f"irregular income pattern creates significant repayment risk."
        )

    dimensions.append(SuitabilityDimension(
        dimension="Income Stability",
        score=round(stab_score, 3),
        label=stab_label,
        actual=f"Stability score = {stability:.2f}",
        benchmark="≥ 0.70 recommended for loan commitments",
        assessment=stab_assessment,
    ))

    # ── Dimension 5: Purpose-Product Alignment ────────────────────
    productive_purposes = [
        "home renovation", "education", "business expansion",
        "vehicle purchase"
    ]
    consumption_purposes = ["personal", "medical emergency"]
    purpose_lower = l0.purpose.lower()

    if any(p in purpose_lower for p in productive_purposes):
        purp_score, purp_label = 0.90, "Suitable"
        purp_assessment = (
            f"Loan purpose '{l0.purpose}' is productive — "
            f"directly aligned with standard retail loan products. "
            f"Asset creation or income-enhancing use."
        )
    elif any(p in purpose_lower for p in consumption_purposes):
        purp_score, purp_label = 0.70, "Partial Match"
        purp_assessment = (
            f"Loan purpose '{l0.purpose}' is consumption-based — "
            f"acceptable but does not create productive assets. "
            f"Suitable if borrower has strong repayment capacity."
        )
    else:
        purp_score, purp_label = 0.60, "Partial Match"
        purp_assessment = (
            f"Loan purpose '{l0.purpose}' requires standard "
            f"documentation and review."
        )

    dimensions.append(SuitabilityDimension(
        dimension="Purpose-Product Alignment",
        score=round(purp_score, 3),
        label=purp_label,
        actual=f"Purpose: {l0.purpose}",
        benchmark="Productive use preferred",
        assessment=purp_assessment,
    ))

    # ── Aggregate suitability ────────────────────────────────────
    weights = [0.30, 0.15, 0.25, 0.20, 0.10]
    overall = round(
        sum(d.score * w for d, w in zip(dimensions, weights)), 3
    )

    if overall >= 0.75:
        suit_label = SuitabilityLabel.SUITABLE
    elif overall >= 0.55:
        suit_label = SuitabilityLabel.PARTIAL
    else:
        suit_label = SuitabilityLabel.UNSUITABLE

    # Primary concern — worst-scoring dimension
    worst = min(dimensions, key=lambda d: d.score)
    primary_concern = (
        f"{worst.dimension}: {worst.assessment}"
        if worst.score < 0.65 else None
    )

    # Suitability narrative
    narrative_parts = [
        f"Suitability assessment: {suit_label.value} "
        f"(score: {overall:.2f})."
    ]
    unsuitable_dims = [d for d in dimensions if d.label == "Unsuitable"]
    partial_dims = [d for d in dimensions if d.label == "Partial Match"]

    if unsuitable_dims:
        narrative_parts.append(
            f"Critical concerns: "
            f"{', '.join(d.dimension for d in unsuitable_dims)}."
        )
    elif partial_dims:
        narrative_parts.append(
            f"Areas requiring attention: "
            f"{', '.join(d.dimension for d in partial_dims)}."
        )
    else:
        narrative_parts.append(
            "All suitability dimensions are within acceptable range."
        )

    return SuitabilityResult(
        dimensions=dimensions,
        overall_score=overall,
        suitability_label=suit_label,
        primary_concern=primary_concern,
        suitability_narrative=" ".join(narrative_parts),
    )


# ─────────────────────────────────────────────────────────────────
# Governance verdict
# ─────────────────────────────────────────────────────────────────

def _build_governance_verdict(
    compliance: ComplianceResult,
    suitability: SuitabilityResult,
    l4: L4ModelReasoning,
    l6: L6ConfidenceGrade,
) -> tuple:
    """
    Determine legitimacy verdict from compliance and suitability.
    Returns (verdict, decision_legitimate, override_required,
             override_reason, governance_narrative)
    """
    # Hard compliance failure → blocked
    if compliance.is_blocking:
        verdict = LegitimacyVerdict.BLOCKED
        legitimate = False
        override_required = True
        override_reason = (
            f"Hard compliance violation(s): "
            f"{', '.join(compliance.blocking_violations)}. "
            f"Senior officer authorisation required."
        )
    # Unsuitable → conditional
    elif suitability.suitability_label == SuitabilityLabel.UNSUITABLE:
        verdict = LegitimacyVerdict.CONDITIONAL
        legitimate = False
        override_required = True
        override_reason = (
            f"Suitability assessment is Unsuitable "
            f"(score: {suitability.overall_score:.2f}). "
            f"Primary concern: {suitability.primary_concern}. "
            f"Loan officer must confirm before proceeding."
        )
    # Partial suitability + warnings → conditional
    elif (suitability.suitability_label == SuitabilityLabel.PARTIAL
          and compliance.warnings > 0):
        verdict = LegitimacyVerdict.CONDITIONAL
        legitimate = True
        override_required = True
        override_reason = (
            f"Partial suitability match with {compliance.warnings} "
            f"compliance warning(s). "
            f"Loan officer review recommended."
        )
    # All good → legitimate
    else:
        verdict = LegitimacyVerdict.LEGITIMATE
        legitimate = True
        override_required = False
        override_reason = None

    # Build governance narrative
    narrative_parts = [
        f"Governance verdict: {verdict.value}.",
        f"Compliance: {compliance.overall_status} "
        f"({compliance.passed}/{compliance.total_checks} checks passed).",
        f"Suitability: {suitability.suitability_label.value} "
        f"(score: {suitability.overall_score:.2f}).",
    ]

    if verdict == LegitimacyVerdict.BLOCKED:
        narrative_parts.append(
            "This decision CANNOT proceed without resolving compliance "
            "violations. The model may be correct — but the decision "
            "is not legitimate under current governance rules."
        )
    elif verdict == LegitimacyVerdict.CONDITIONAL:
        narrative_parts.append(
            "This decision requires human review before proceeding. "
            "The recommendation may be technically sound but requires "
            "explicit officer confirmation of suitability."
        )
    else:
        narrative_parts.append(
            "This decision is both correct and legitimate — "
            "it satisfies all compliance requirements and is "
            "suitable for this specific borrower."
        )

    return (
        verdict, legitimate, override_required,
        override_reason, " ".join(narrative_parts)
    )


# ─────────────────────────────────────────────────────────────────
# Layer implementation
# ─────────────────────────────────────────────────────────────────

class L7Compliance_Layer:
    """
    Layer 7: Compliance & Suitability

    Input  : L0, L3, L4, L5 (optional), L6
    Output : L7GovernanceVerdict

    Key distinction from all previous layers:
        This layer does not ask 'is the decision correct?'
        It asks 'is the decision legitimate?'
    """

    def process(
        self,
        l0: L0IntakeRecord,
        l3: L3SignalLog,
        l4: L4ModelReasoning,
        l6: L6ConfidenceGrade,
        l5: Optional[L5Recommendation] = None,
        application_id: str = "",
    ) -> L7GovernanceVerdict:

        # Component 1 — Compliance
        compliance = _run_compliance_checks(l0, l3, l4)

        # Component 2 — Suitability
        suitability = _assess_suitability(l0, l3, l4, l5)

        # Combined governance verdict
        (verdict, legitimate, override_req,
         override_reason, narrative) = _build_governance_verdict(
            compliance, suitability, l4, l6
        )

        return L7GovernanceVerdict(
            application_id=application_id or l0.application_id,
            compliance=compliance,
            suitability=suitability,
            legitimacy_verdict=verdict,
            decision_legitimate=legitimate,
            override_required=override_req,
            override_reason=override_reason,
            governance_narrative=narrative,
            timestamp=datetime.now(),
        )
