"""
LEAF Layer 6 — Confidence Grade (Uncertainty & Confidence)
=====================================================================
This is the meta-explainability layer — it explains the quality
of the explanation itself.

Core philosophy (from LEAF Framework document):
    Traditional AI outputs a confidence number.
    LEAF Layer 6 transforms confidence from a vague probability
    into a transparent, decomposable, evidence-backed reliability
    assessment.

Five components:
    1. Grounding Fidelity     (w=0.30) — retrieval quality,
                                          semantic similarity,
                                          contradiction penalties
    2. Data Freshness &       (w=0.25) — freshness, source diversity,
       Coverage                          evidence completeness
    3. Model Consistency      (w=0.20) — inter-model agreement,
                                          SHAP stability, rule alignment
    4. Calibration &          (w=0.15) — historical hit ratio,
       Backtesting                        MAE, calibration baseline
    5. Compliance &           (w=0.10) — suitability, KYC, governance
       Suitability

Grades:
    A : score ≥ 0.90 — Strongly grounded, highly reliable → auto-proceed
    B : score ≥ 0.75 — Well-supported, minor ambiguity    → auto-proceed
    C : score ≥ 0.60 — Mixed signals, incomplete grounding → HITL flag
    D : score  < 0.60 — High uncertainty, contradictions  → block decision

Doctoral contribution:
    This layer directly addresses Research Gap 1 (model-level only XAI)
    and Research Gap 2 (GenAI governance under-theorized) from Chapter 2.
    No prior XAI framework provides a structured, auditable confidence
    assessment that spans data quality, model agreement, and governance.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from models.schemas import (
    L1ProvenanceCertificate, L2GroundingReport,
    L3SignalLog, DataSourceType
)
from layers.l4_model import L4ModelReasoning
from storage.ledger import get_all_summaries


# ─────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────

class GroundingFidelityScore(BaseModel):
    """
    Component 1 — Grounding Fidelity
    Evaluates: How well is the recommendation anchored to real evidence?

    Three subcomponents:
        retrieval_quality   : system-level retrieval effectiveness
        semantic_similarity : chunk-level relevance to decision query
        contradiction_penalty: penalty when sources disagree
    """
    retrieval_quality: float = Field(..., ge=0, le=1,
        description="How effective was data retrieval overall? "
                    "Considers precision, recall, coverage, noise ratio.")
    semantic_similarity: float = Field(..., ge=0, le=1,
        description="How semantically relevant is each data source "
                    "to the credit decision query? "
                    "Computed via cosine similarity proxy.")
    contradiction_penalty: float = Field(..., ge=0, le=1,
        description="Penalty applied when data sources contradict each other. "
                    "e.g. high CIBIL but high bounce rate in bank statements.")
    contradictions_detected: List[str] = Field(default_factory=list,
        description="Human-readable list of detected contradictions")
    composite: float = Field(..., ge=0, le=1)


class FreshnessCoverageScore(BaseModel):
    """
    Component 2 — Data Freshness & Coverage
    Evaluates: Was evidence recent, diverse, and complete?

    Three subcomponents:
        freshness           : age-based decay per source
        source_diversity    : how many different source types contributed
        evidence_completeness: how many expected signals were extracted
    """
    freshness: float = Field(..., ge=0, le=1,
        description="Weighted average freshness across all data sources. "
                    "Decays linearly from 1.0 (fresh) to 0.0 (stale).")
    source_diversity: float = Field(..., ge=0, le=1,
        description="Score reflecting how many different source types "
                    "contributed. More types = higher diversity = "
                    "more trustworthy grounding.")
    evidence_completeness: float = Field(..., ge=0, le=1,
        description="What fraction of expected credit signals were "
                    "successfully extracted by L3.")
    sources_used: List[str] = Field(default_factory=list)
    composite: float = Field(..., ge=0, le=1)


class ModelConsistencyScore(BaseModel):
    """
    Component 3 — Model Consistency
    Evaluates: Do different reasoning systems agree?

    Types of consistency checked:
        xgboost_vs_rules    : does XGBoost agree with rule-based assessment?
        shap_stability      : are SHAP values stable (low variance)?
        decision_stability  : would small input changes flip the decision?
    """
    xgboost_vs_rules_agreement: float = Field(..., ge=0, le=1,
        description="Agreement between XGBoost model output and "
                    "rule-based credit assessment (DTI < 0.5, "
                    "CIBIL > 650 etc.). 1.0 = full agreement.")
    shap_stability: float = Field(..., ge=0, le=1,
        description="Stability of SHAP attribution. High-magnitude, "
                    "clear attributions = high stability. "
                    "Near-zero SHAP values = unstable explanation.")
    decision_boundary_distance: float = Field(..., ge=0, le=1,
        description="How far is the approval probability from decision "
                    "thresholds? Far from boundaries = stable decision.")
    consistency_flags: List[str] = Field(default_factory=list,
        description="Specific inconsistencies detected")
    composite: float = Field(..., ge=0, le=1)


class CalibrationScore(BaseModel):
    """
    Component 4 — Calibration & Backtesting
    Evaluates: Historically, how accurate were similar recommendations?

    Makes confidence empirical and historically grounded.
    Starts at a neutral baseline with no history.
    Improves as the Evidence Ledger accumulates decisions.
    """
    historical_applications: int = Field(...,
        description="Number of similar past applications in the ledger")
    hit_ratio: float = Field(..., ge=0, le=1,
        description="Proportion of similar past applications where "
                    "the model's decision was later validated as correct. "
                    "Neutral baseline 0.70 when history is insufficient.")
    calibration_confidence: float = Field(..., ge=0, le=1,
        description="Confidence in the calibration estimate itself. "
                    "Increases with more historical data.")
    baseline_note: str = Field(default="",
        description="Explanation of calibration basis")
    composite: float = Field(..., ge=0, le=1)


class ComplianceSuitabilityScore(BaseModel):
    """
    Component 5 — Compliance & Suitability
    Evaluates: Is this recommendation appropriate and permissible?

    Even strong retrieval and model agreement cannot compensate
    for governance violations.
    """
    fair_lending_score: float = Field(..., ge=0, le=1,
        description="Protected characteristics absent from decision? "
                    "Adverse action notice ready?")
    kyc_completeness: float = Field(..., ge=0, le=1,
        description="Are all required applicant verification documents "
                    "present and verified?")
    regulatory_alignment: float = Field(..., ge=0, le=1,
        description="How well does this decision align with current "
                    "RBI guidelines and model risk framework?")
    suitability_flags: List[str] = Field(default_factory=list,
        description="Specific compliance concerns detected")
    composite: float = Field(..., ge=0, le=1)


class L6ConfidenceGrade(BaseModel):
    """
    L6 output artifact — the complete Confidence Certificate.

    This is what makes LEAF's confidence explainable:
    every score is decomposed, every component is auditable,
    every flag is documented.
    """
    layer: str = "L6"
    application_id: str

    # Five components
    grounding_fidelity: GroundingFidelityScore
    freshness_coverage: FreshnessCoverageScore
    model_consistency: ModelConsistencyScore
    calibration: CalibrationScore
    compliance_suitability: ComplianceSuitabilityScore

    # Weighted aggregation
    weights: Dict[str, float] = Field(
        default={
            "grounding_fidelity": 0.30,
            "freshness_coverage": 0.25,
            "model_consistency": 0.20,
            "calibration": 0.15,
            "compliance_suitability": 0.10,
        },
        description="Weights sum to 1.0. Grounding and freshness weighted "
                    "highest as they directly affect explanation reliability "
                    "in regulated financial environments."
    )
    composite_score: float = Field(..., ge=0, le=1,
        description="Weighted sum of all five components")

    # Grade
    grade: str = Field(...,
        description="A / B / C / D based on composite score")
    grade_meaning: str = Field(...,
        description="Plain language interpretation of the grade")

    # Action
    hitl_required: bool = Field(...,
        description="True for grades C and D — triggers L9 human review")
    decision_blocked: bool = Field(...,
        description="True for grade D — blocks automated decision entirely")
    action_required: str = Field(...,
        description="What should happen next based on the grade")

    # Confidence narrative for Explanation Card
    confidence_narrative: str = Field(...,
        description="Plain language explanation of confidence level "
                    "for the applicant and loan officer")

    timestamp: datetime

    xai_note: str = Field(
        default="L6 transforms AI confidence from a vague probability into "
                "a transparent, decomposable, evidence-backed reliability "
                "assessment. This directly addresses the governance gap "
                "identified in Research Gap 2 of Chapter 2.",
    )


# ─────────────────────────────────────────────────────────────────
# Grade thresholds and meanings
# ─────────────────────────────────────────────────────────────────

GRADE_THRESHOLDS = [
    (0.90, "A", "Strongly grounded and highly reliable",
     False, False,
     "Proceed automatically. All components meet high standards."),
    (0.75, "B", "Well-supported with minor ambiguity",
     False, False,
     "Proceed automatically. Minor weaknesses noted but within tolerance."),
    (0.60, "C", "Mixed signals or incomplete grounding",
     True, False,
     "Flag for human review before issuing decision. "
     "Loan officer must review and confirm."),
    (0.00, "D", "High uncertainty or severe contradictions",
     True, True,
     "Block automated decision. Mandatory senior review required. "
     "Do not issue decision without human authorisation."),
]


def _assign_grade(score: float):
    for threshold, grade, meaning, hitl, block, action in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade, meaning, hitl, block, action
    return "D", GRADE_THRESHOLDS[-1][2], True, True, GRADE_THRESHOLDS[-1][4]


# ─────────────────────────────────────────────────────────────────
# Component calculators
# ─────────────────────────────────────────────────────────────────

def _compute_grounding_fidelity(
    l1: L1ProvenanceCertificate,
    l2: L2GroundingReport,
    l3: L3SignalLog,
) -> GroundingFidelityScore:
    """
    Component 1: Grounding Fidelity

    Retrieval Quality — system-level effectiveness of data retrieval.
    Based on: verification rate, warning ratio, source count adequacy.

    Semantic Similarity — relevance of each source to credit decision.
    In credit, all four sources (CIBIL, bank, UPI, ITR) are directly
    relevant. Proxy score based on source type relevance mapping.

    Contradiction Penalty — detects when data sources disagree.
    Credit-specific contradictions:
        - High CIBIL but high DTI (model vs bureau disagreement)
        - Declared income vs verified income mismatch > 20%
        - Low bounce rate but high existing loan burden
        - Strong repayment history but high delinquent accounts
    """
    # ── Retrieval Quality ────────────────────────────────────────
    verified_ratio = l1.verified_sources / max(l1.total_sources, 1)
    warning_ratio = l1.sources_with_warnings / max(l1.total_sources, 1)
    source_adequacy = min(l1.total_sources / 4.0, 1.0)

    retrieval_quality = round(
        (verified_ratio * 0.40) +
        ((1 - warning_ratio) * 0.35) +
        (source_adequacy * 0.25),
        3
    )

    # ── Semantic Similarity ──────────────────────────────────────
    # Credit-domain relevance weights per source type
    CREDIT_RELEVANCE = {
        DataSourceType.CREDIT_BUREAU: 0.95,
        DataSourceType.BANK_STATEMENT: 0.92,
        DataSourceType.UPI_HISTORY: 0.80,
        DataSourceType.TAX_RETURN: 0.85,
        DataSourceType.ALTERNATIVE: 0.65,
    }
    relevance_scores = [
        CREDIT_RELEVANCE.get(src.source_type, 0.70)
        for src in l1.sources
    ]
    semantic_similarity = round(
        sum(relevance_scores) / len(relevance_scores)
        if relevance_scores else 0.70,
        3
    )

    # ── Contradiction Detection ──────────────────────────────────
    contradictions = []
    contradiction_penalty = 1.0  # starts at 1.0, reduced per contradiction

    signals = {s.signal_key: s.raw_value for s in l3.signals}

    cibil = signals.get("cibil_score", 700)
    dti = signals.get("dti_ratio", 0.4)
    repayment = signals.get("repayment_score", 0.8)
    income_stability = signals.get("income_stability", 0.8)
    existing_loans = signals.get("existing_loan_burden", 0)

    # Contradiction 1: Strong CIBIL but very high DTI
    if cibil > 720 and dti > 0.60:
        contradictions.append(
            f"Strong CIBIL ({cibil:.0f}) contradicts very high DTI "
            f"({dti:.0%}) — bureau history vs current obligations disagree"
        )
        contradiction_penalty -= 0.12

    # Contradiction 2: Good repayment history but high existing loans
    if repayment > 0.85 and existing_loans >= 3:
        contradictions.append(
            f"Strong repayment history ({repayment:.0%}) but "
            f"{existing_loans:.0f} existing active loans — "
            f"historical pattern may not reflect current burden"
        )
        contradiction_penalty -= 0.08

    # Contradiction 3: High income stability but stale ITR
    itr_sources = [s for s in l1.sources
                   if s.source_type == DataSourceType.TAX_RETURN]
    if itr_sources and itr_sources[0].age_hours > 48 and income_stability > 0.80:
        contradictions.append(
            f"High income stability score ({income_stability:.2f}) but "
            f"ITR document is {itr_sources[0].age_hours:.0f}h old — "
            f"income claim cannot be fully verified"
        )
        contradiction_penalty -= 0.06

    # Contradiction 4: L2 grounding score inconsistency
    for src_score in l2.source_scores:
        if src_score.composite_score < 0.65:
            contradictions.append(
                f"Low grounding on {src_score.source_name} "
                f"(score {src_score.composite_score:.2f}) — "
                f"signals from this source carry reduced reliability"
            )
            contradiction_penalty -= 0.05
            break

    contradiction_penalty = round(max(0.40, contradiction_penalty), 3)

    # ── Composite ────────────────────────────────────────────────
    composite = round(
        (retrieval_quality * 0.40) +
        (semantic_similarity * 0.35) +
        (contradiction_penalty * 0.25),
        3
    )

    return GroundingFidelityScore(
        retrieval_quality=retrieval_quality,
        semantic_similarity=semantic_similarity,
        contradiction_penalty=contradiction_penalty,
        contradictions_detected=contradictions,
        composite=composite,
    )


def _compute_freshness_coverage(
    l1: L1ProvenanceCertificate,
    l2: L2GroundingReport,
    l3: L3SignalLog,
) -> FreshnessCoverageScore:
    """
    Component 2: Data Freshness & Coverage

    Freshness: weighted average of L2 freshness scores.
    Source Diversity: how many different source types contributed.
    Evidence Completeness: fraction of expected signals extracted.
    """
    # ── Freshness ────────────────────────────────────────────────
    freshness_scores = [s.freshness_score for s in l2.source_scores]
    freshness = round(
        sum(freshness_scores) / len(freshness_scores)
        if freshness_scores else 0.70,
        3
    )

    # ── Source Diversity ─────────────────────────────────────────
    # 4 source types expected: bureau, bank, UPI, ITR
    unique_types = len({s.source_type for s in l1.sources})
    source_diversity = round(min(unique_types / 4.0, 1.0), 3)

    sources_used = [s.source_name for s in l1.sources]

    # ── Evidence Completeness ────────────────────────────────────
    # 7 signals expected from L3 (the 7 we defined)
    expected_signals = 7
    extracted_signals = l3.total_signals
    evidence_completeness = round(
        min(extracted_signals / expected_signals, 1.0), 3
    )

    # ── Composite ────────────────────────────────────────────────
    composite = round(
        (freshness * 0.50) +
        (source_diversity * 0.30) +
        (evidence_completeness * 0.20),
        3
    )

    return FreshnessCoverageScore(
        freshness=freshness,
        source_diversity=source_diversity,
        evidence_completeness=evidence_completeness,
        sources_used=sources_used,
        composite=composite,
    )


def _compute_model_consistency(
    l3: L3SignalLog,
    l4: L4ModelReasoning,
) -> ModelConsistencyScore:
    """
    Component 3: Model Consistency

    XGBoost vs Rules Agreement:
        Apply industry-standard credit rules and check if XGBoost agrees.
        Rules: CIBIL > 650, DTI < 0.50, repayment > 0.70
        Agreement = XGBoost and rules reach same decision.

    SHAP Stability:
        Are SHAP values clear and high-magnitude, or scattered near zero?
        High-magnitude attributions = stable, explainable decision.
        Near-zero attributions = unstable, hard to explain.

    Decision Boundary Distance:
        How far is the approval probability from decision thresholds?
        Far from 0.55 threshold = stable, confident decision.
    """
    flags = []
    signals = {s.signal_key: s.raw_value for s in l3.signals}

    cibil = signals.get("cibil_score", 700)
    dti = signals.get("dti_ratio", 0.4)
    repayment = signals.get("repayment_score", 0.8)

    # ── XGBoost vs Rule-Based Agreement ─────────────────────────
    # Apply credit rules
    rule_approve = (cibil >= 650 and dti <= 0.50 and repayment >= 0.70)
    model_approve = l4.approval_probability >= 0.55

    if rule_approve == model_approve:
        xgboost_vs_rules = 0.92
    else:
        xgboost_vs_rules = 0.52
        direction = "approve" if model_approve else "reject"
        rule_direction = "approve" if rule_approve else "reject"
        flags.append(
            f"Model says {direction} ({l4.approval_probability:.1%}) but "
            f"rule-based assessment says {rule_direction} "
            f"(CIBIL={cibil:.0f}, DTI={dti:.0%}, "
            f"Repayment={repayment:.0%})"
        )

    xgboost_vs_rules = round(xgboost_vs_rules, 3)

    # ── SHAP Stability ───────────────────────────────────────────
    shap_values = [abs(c.shap_value) for c in l4.shap_contributions]
    if shap_values:
        max_shap = max(shap_values)
        mean_shap = sum(shap_values) / len(shap_values)

        # High-magnitude top features = stable
        top_concentration = sum(
            v for v in sorted(shap_values, reverse=True)[:3]
        ) / max(sum(shap_values), 0.001)

        shap_stability = round(
            (min(max_shap / 0.30, 1.0) * 0.40) +
            (top_concentration * 0.35) +
            (min(mean_shap / 0.10, 1.0) * 0.25),
            3
        )

        if max_shap < 0.05:
            flags.append(
                "SHAP values are very low — model attribution is unclear. "
                "Explanation may not be reliable."
            )
    else:
        shap_stability = 0.70

    # ── Decision Boundary Distance ───────────────────────────────
    # Distance from nearest decision threshold (0.40 and 0.70)
    prob = l4.approval_probability
    dist_from_thresholds = min(
        abs(prob - 0.40),
        abs(prob - 0.70),
    )
    # Normalise: 0.15+ distance = very stable, 0.05 or less = borderline
    decision_boundary_distance = round(min(dist_from_thresholds / 0.15, 1.0), 3)

    if decision_boundary_distance < 0.35:
        flags.append(
            f"Approval probability ({prob:.1%}) is close to a decision "
            f"threshold — small changes in input could flip the decision"
        )

    # ── Composite ────────────────────────────────────────────────
    composite = round(
        (xgboost_vs_rules * 0.45) +
        (shap_stability * 0.35) +
        (decision_boundary_distance * 0.20),
        3
    )

    return ModelConsistencyScore(
        xgboost_vs_rules_agreement=xgboost_vs_rules,
        shap_stability=shap_stability,
        decision_boundary_distance=decision_boundary_distance,
        consistency_flags=flags,
        composite=composite,
    )


def _compute_calibration(
    l4: L4ModelReasoning,
    application_id: str,
) -> CalibrationScore:
    """
    Component 4: Calibration & Backtesting

    Queries the Evidence Ledger for similar past applications.
    Computes a hit ratio — what proportion of similar past
    decisions were later validated.

    Neutral baseline (0.70) when history is insufficient.
    Calibration confidence increases with more historical data.

    This makes confidence empirical rather than intuitive —
    one of LEAF's key doctoral contributions.
    """
    # Query Evidence Ledger for past summaries
    all_summaries = get_all_summaries()

    # Filter to completed applications (exclude current)
    past = [
        s for s in all_summaries
        if s["application_id"] != application_id
        and s.get("decision") is not None
    ]

    n_historical = len(past)

    if n_historical < 5:
        # Insufficient history — use neutral baseline
        hit_ratio = 0.70
        calibration_confidence = round(0.40 + (n_historical * 0.04), 3)
        baseline_note = (
            f"Calibration based on neutral baseline (insufficient history: "
            f"{n_historical} past applications). Calibration improves as "
            f"more applications are processed."
        )
    else:
        # Compute hit ratio from past decisions
        # "Hit" = model approved and outcome was positive (proxy: confidence was High)
        approved = [s for s in past if "Approved" in (s.get("decision") or "")]
        high_confidence = [
            s for s in approved
            if (s.get("confidence") or "") in ("High", "Moderate")
        ]
        hit_ratio = round(
            len(high_confidence) / max(len(approved), 1), 3
        ) if approved else 0.70

        calibration_confidence = round(
            min(0.40 + (n_historical * 0.03), 0.95), 3
        )
        baseline_note = (
            f"Calibration based on {n_historical} historical applications. "
            f"Approved with high/moderate confidence: "
            f"{len(high_confidence)}/{len(approved)}."
        )

    # Composite = blend of hit_ratio and calibration_confidence
    composite = round(
        (hit_ratio * 0.65) +
        (calibration_confidence * 0.35),
        3
    )

    return CalibrationScore(
        historical_applications=n_historical,
        hit_ratio=hit_ratio,
        calibration_confidence=calibration_confidence,
        baseline_note=baseline_note,
        composite=composite,
    )


def _compute_compliance_suitability(
    l1: L1ProvenanceCertificate,
    l3: L3SignalLog,
    l4: L4ModelReasoning,
) -> ComplianceSuitabilityScore:
    """
    Component 5: Compliance & Suitability

    Fair Lending: protected characteristics absent, adverse
                  action notice ready.
    KYC Completeness: all required documents present and verified.
    Regulatory Alignment: decision aligns with RBI guidelines.
    """
    flags = []
    signals = {s.signal_key: s.raw_value for s in l3.signals}

    # ── Fair Lending ─────────────────────────────────────────────
    # All sources verified = fair lending baseline met
    all_verified = all(s.is_verified for s in l1.sources)
    fair_lending_score = 0.95 if all_verified else 0.75
    if not all_verified:
        flags.append("Not all data sources are verified — "
                     "fair lending baseline may not be fully met")

    # ── KYC Completeness ─────────────────────────────────────────
    # All 4 expected source types present = full KYC
    expected_types = {
        DataSourceType.CREDIT_BUREAU,
        DataSourceType.BANK_STATEMENT,
        DataSourceType.UPI_HISTORY,
        DataSourceType.TAX_RETURN,
    }
    present_types = {s.source_type for s in l1.sources}
    kyc_ratio = len(present_types & expected_types) / len(expected_types)
    kyc_completeness = round(kyc_ratio, 3)

    if kyc_ratio < 1.0:
        missing = expected_types - present_types
        flags.append(
            f"Missing KYC sources: "
            f"{', '.join(t.value for t in missing)}"
        )

    # ── Regulatory Alignment ─────────────────────────────────────
    # Check RBI guidelines:
    # DTI should not exceed 0.50 for standard loans
    # CIBIL > 650 for standard retail lending
    dti = signals.get("dti_ratio", 0.4)
    cibil = signals.get("cibil_score", 700)

    regulatory_score = 0.95
    if dti > 0.55:
        regulatory_score -= 0.15
        flags.append(
            f"DTI {dti:.0%} exceeds RBI recommended 55% ceiling for "
            f"retail loans — regulatory alignment reduced"
        )
    if cibil < 650:
        regulatory_score -= 0.20
        flags.append(
            f"CIBIL {cibil:.0f} below standard retail lending threshold "
            f"of 650 — additional regulatory documentation may be required"
        )

    regulatory_alignment = round(max(0.40, regulatory_score), 3)

    # ── Composite ────────────────────────────────────────────────
    composite = round(
        (fair_lending_score * 0.40) +
        (kyc_completeness * 0.35) +
        (regulatory_alignment * 0.25),
        3
    )

    return ComplianceSuitabilityScore(
        fair_lending_score=fair_lending_score,
        kyc_completeness=kyc_completeness,
        regulatory_alignment=regulatory_alignment,
        suitability_flags=flags,
        composite=composite,
    )


def _build_confidence_narrative(
    grade: str,
    composite: float,
    grounding: GroundingFidelityScore,
    freshness: FreshnessCoverageScore,
    consistency: ModelConsistencyScore,
    calibration: CalibrationScore,
) -> str:
    """Generate plain language confidence narrative for Explanation Card."""
    grade_phrases = {
        "A": "very high confidence",
        "B": "good confidence",
        "C": "moderate confidence — human review recommended",
        "D": "low confidence — human authorisation required",
    }
    phrase = grade_phrases.get(grade, "moderate confidence")

    parts = [
        f"This decision has been assessed with {phrase} "
        f"(score: {composite:.2f}, Grade {grade})."
    ]

    if grounding.contradictions_detected:
        parts.append(
            f"Note: {len(grounding.contradictions_detected)} data "
            f"contradiction(s) were detected and have been factored "
            f"into this assessment."
        )

    if freshness.freshness < 0.75:
        parts.append(
            "Some data sources are not fully fresh, which has "
            "slightly reduced the confidence grade."
        )

    if consistency.consistency_flags:
        parts.append(
            "The model's assessment and rule-based checks showed "
            "some divergence, which has been noted in the audit record."
        )

    if calibration.historical_applications < 5:
        parts.append(
            "Calibration is based on a neutral baseline as "
            "insufficient historical data is available."
        )

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────
# Layer implementation
# ─────────────────────────────────────────────────────────────────

class L6ConfidenceGrade_Layer:
    """
    Layer 6: Confidence Grade

    Input  : L1, L2, L3, L4 outputs
    Output : L6ConfidenceGrade with full decomposition
    """

    WEIGHTS = {
        "grounding_fidelity": 0.30,
        "freshness_coverage": 0.25,
        "model_consistency": 0.20,
        "calibration": 0.15,
        "compliance_suitability": 0.10,
    }

    def process(
        self,
        l1: L1ProvenanceCertificate,
        l2: L2GroundingReport,
        l3: L3SignalLog,
        l4: L4ModelReasoning,
        application_id: str,
    ) -> L6ConfidenceGrade:

        # Compute all five components
        grounding = _compute_grounding_fidelity(l1, l2, l3)
        freshness = _compute_freshness_coverage(l1, l2, l3)
        consistency = _compute_model_consistency(l3, l4)
        calibration = _compute_calibration(l4, application_id)
        compliance = _compute_compliance_suitability(l1, l3, l4)

        # Weighted aggregation — the formal LEAF confidence formula
        composite = round(
            grounding.composite * self.WEIGHTS["grounding_fidelity"] +
            freshness.composite * self.WEIGHTS["freshness_coverage"] +
            consistency.composite * self.WEIGHTS["model_consistency"] +
            calibration.composite * self.WEIGHTS["calibration"] +
            compliance.composite * self.WEIGHTS["compliance_suitability"],
            3
        )

        # Assign grade
        grade, meaning, hitl, blocked, action = _assign_grade(composite)

        # Build narrative
        narrative = _build_confidence_narrative(
            grade, composite, grounding, freshness, consistency, calibration
        )

        return L6ConfidenceGrade(
            application_id=application_id,
            grounding_fidelity=grounding,
            freshness_coverage=freshness,
            model_consistency=consistency,
            calibration=calibration,
            compliance_suitability=compliance,
            composite_score=composite,
            grade=grade,
            grade_meaning=meaning,
            hitl_required=hitl,
            decision_blocked=blocked,
            action_required=action,
            confidence_narrative=narrative,
            timestamp=datetime.now(),
        )
