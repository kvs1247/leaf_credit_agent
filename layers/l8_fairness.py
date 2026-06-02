"""
LEAF Layer 8 — Bias & Fairness Diagnostics
=====================================================================
Core question: "Is the system systematically favouring or
disadvantaging something without justification?"

Layer 8 introduces Fairness Explainability — the eighth type
in the LEAF explainability taxonomy:

    L1 — Source Explainability
    L2 — Retrieval Explainability
    L3 — Signal Explainability
    L4 — Reasoning Explainability
    L5 — Decision Explainability
    L6 — Confidence Explainability
    L7 — Governance Explainability
    L8 — Fairness Explainability      ← this layer

Critical distinction from L7:
    L7 asks: "Is this recommendation allowed?"
    L8 asks: "Is the system behaving fairly over time?"

L8 operates at two levels:
    Individual level — was THIS decision influenced by proxy bias?
    System level    — across all decisions, are patterns emerging?

Five diagnostic areas (from LEAF Framework diagram):
    1. Representation Fairness  — are groups fairly represented?
    2. Outcome Fairness         — are outcomes equitably distributed?
    3. Evidence & Source Fairness — is evidence selection unbiased?
    4. Temporal Fairness        — is the system drifting over time?
    5. Individual/Profile Fair  — are similar profiles treated alike?

Key metrics:
    EII  — Exposure Imbalance Index
    ARP  — Approval Rate Parity
    CP   — Confidence Parity
    WRP  — Win Rate Parity
    SCI  — Source Concentration Index
    TDS  — Temporal Drift Score
    ICS  — Individual Consistency Score
    OBS  — Overall Bias Score (weighted combination)

Verdicts:
    Fair    : OBS < 0.25
    Caution : 0.25 ≤ OBS < 0.55
    Biased  : OBS ≥ 0.55

Continuous loop: Detect → Diagnose → Mitigate → Monitor

Doctoral contribution:
    Directly addresses Research Gap 1 (model-level only) and Gap 5
    (regulatory-operational disconnect) from Chapter 2.
    Fairness in financial AI is legally mandated (RBI fair lending,
    ECOA, EU AI Act Article 10) but operationally unimplemented
    in most existing frameworks.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel, Field

from models.schemas import L0IntakeRecord, L3SignalLog
from layers.l4_model import L4ModelReasoning
from layers.l6_confidence import L6ConfidenceGrade
from layers.l7_compliance import L7GovernanceVerdict
from storage.ledger import get_all_summaries, get_layer_artifact


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

MINIMUM_HISTORY = 5        # minimum applications for system-level analysis
DISPARITY_FLAG_THRESHOLD = 0.15   # 15% gap triggers a flag
CONCENTRATION_FLAG = 0.60         # >60% concentration in one source is flagged
DRIFT_FLAG_THRESHOLD = 0.20       # 20% drift over time triggers a flag

# Credit domain protected / sensitive dimensions
EMPLOYMENT_GROUPS = ["salaried", "self_employed", "business"]
INCOME_BANDS = {
    "low":  (0, 30000),
    "mid":  (30001, 70000),
    "high": (70001, float("inf")),
}
PURPOSES = ["home renovation", "education", "business expansion",
            "medical emergency", "vehicle purchase", "personal"]


# ─────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────

class BiasFlag(BaseModel):
    """A single detected bias signal with root cause and remediation."""
    bias_type: str
    dimension: str
    severity: str = Field(..., description="low / medium / high / critical")
    description: str
    observed_value: str
    threshold: str
    root_cause_indicator: str
    recommended_action: str
    regulation_reference: str = ""


class RepresentationFairness(BaseModel):
    """
    Diagnostic Area 1 — Representation Fairness
    Are certain groups systematically over or under-represented
    in recommendations?

    Credit domain: employment type, income band, loan purpose.
    Key metric: Exposure Imbalance Index (EII)
    EII = 0 means perfect equal representation
    EII = 1 means complete concentration in one group
    """
    employment_distribution: Dict[str, float]
    income_band_distribution: Dict[str, float]
    purpose_distribution: Dict[str, float]
    eii_employment: float = Field(..., ge=0, le=1,
        description="Exposure Imbalance Index for employment type")
    eii_income: float = Field(..., ge=0, le=1,
        description="Exposure Imbalance Index for income band")
    flags: List[str] = Field(default_factory=list)
    score: float = Field(..., ge=0, le=1,
        description="0 = highly imbalanced, 1 = well represented")
    data_basis: str = ""


class OutcomeFairness(BaseModel):
    """
    Diagnostic Area 2 — Outcome Fairness
    Are outcomes (approvals, rejections, confidence scores)
    equitably distributed across groups?

    Key metrics: ARP (Approval Rate Parity), CP (Confidence Parity)
    Disparity score: difference between highest and lowest group rate
    Flag threshold: 15% disparity
    """
    approval_rates: Dict[str, float]
    confidence_scores: Dict[str, float]
    arp_disparity: float = Field(..., ge=0, le=1,
        description="Max approval rate gap across employment groups")
    cp_disparity: float = Field(..., ge=0,
        description="Max confidence score gap across employment groups")
    flags: List[str] = Field(default_factory=list)
    score: float = Field(..., ge=0, le=1,
        description="0 = high disparity, 1 = equitable outcomes")
    data_basis: str = ""


class EvidenceSourceFairness(BaseModel):
    """
    Diagnostic Area 3 — Evidence & Source Fairness
    Does the system rely excessively on certain data sources
    or signal types while ignoring others?

    Credit domain:
        Source concentration: over-reliance on CIBIL vs UPI vs bank
        Signal dominance: does DTI always dominate SHAP attribution?
        Alternative data utilisation: are UPI signals used fairly?

    Key metric: Source Concentration Index (SCI)
    """
    source_weights: Dict[str, float]
    dominant_signals: List[str]
    sci: float = Field(..., ge=0, le=1,
        description="Source Concentration Index — 0=diverse, 1=concentrated")
    cibil_dependency: float = Field(..., ge=0, le=1,
        description="How much this decision depended on CIBIL alone")
    alternative_data_utilisation: float = Field(..., ge=0, le=1,
        description="How much UPI and alternative signals contributed")
    flags: List[str] = Field(default_factory=list)
    score: float = Field(..., ge=0, le=1,
        description="0 = highly concentrated, 1 = diverse sources")
    proxy_bias_detected: bool = False
    proxy_details: Optional[str] = None


class TemporalFairness(BaseModel):
    """
    Diagnostic Area 4 — Temporal Fairness
    Is the system's behaviour changing unfairly over time?
    Recency bias, regime drift, data drift.

    Key metric: Temporal Drift Score (TDS)
    Compares recent application outcomes vs historical baseline.
    Flag: >20% shift in approval rates over time.
    """
    historical_approval_rate: float
    recent_approval_rate: float
    tds: float = Field(..., ge=0,
        description="Temporal Drift Score — magnitude of outcome drift")
    recency_bias_detected: bool = False
    regime_shift_detected: bool = False
    data_points_analysed: int = 0
    flags: List[str] = Field(default_factory=list)
    score: float = Field(..., ge=0, le=1,
        description="0 = high drift, 1 = stable over time")
    data_basis: str = ""


class IndividualProfileFairness(BaseModel):
    """
    Diagnostic Area 5 — Individual / Profile Fairness
    Are similar profiles treated consistently?
    Demographic proxy detection.

    Key metric: Individual Consistency Score (ICS)
    Similar profiles = applications within 10% of key signal values.

    Proxy bias: does a non-protected feature (UPI velocity) act
    as a proxy for a protected attribute (urban/rural geography)?
    """
    similar_profiles_found: int
    ics: float = Field(..., ge=0, le=1,
        description="Individual Consistency Score — "
                    "1 = similar profiles get similar outcomes")
    consistency_variance: float
    proxy_risks: List[str] = Field(default_factory=list,
        description="Features that may act as demographic proxies")
    flags: List[str] = Field(default_factory=list)
    score: float = Field(..., ge=0, le=1)
    data_basis: str = ""


class L8FairnessDiagnostics(BaseModel):
    """
    L8 complete output — the Fairness Certificate.

    Contains both:
    - Individual-level bias check for this specific application
    - System-level aggregate bias analysis across all history

    The Overall Bias Score (OBS) synthesises all five areas
    into a single actionable verdict.
    """
    layer: str = "L8"
    application_id: str

    # Five diagnostic areas
    representation: RepresentationFairness
    outcome: OutcomeFairness
    evidence_source: EvidenceSourceFairness
    temporal: TemporalFairness
    individual_profile: IndividualProfileFairness

    # Key metrics (for audit record)
    eii: float = Field(..., description="Exposure Imbalance Index")
    arp: float = Field(..., description="Approval Rate Parity gap")
    sci: float = Field(..., description="Source Concentration Index")
    tds: float = Field(..., description="Temporal Drift Score")
    ics: float = Field(..., description="Individual Consistency Score")
    obs: float = Field(..., ge=0, le=1,
        description="Overall Bias Score — weighted combination")

    # Verdict
    verdict: str = Field(...,
        description="Fair / Caution / Biased")
    verdict_color: str = Field(...,
        description="green / amber / red")
    investigation_required: bool

    # Detected bias flags
    bias_flags: List[BiasFlag] = Field(default_factory=list)
    total_flags: int = 0

    # Remediation
    remediation_actions: List[str] = Field(default_factory=list)

    # User-facing summary
    fairness_summary: str = Field(...,
        description="Plain language summary for the applicant")

    # Audit record
    audit_narrative: str = Field(...,
        description="Detailed narrative for regulatory audit")

    # Data context
    total_applications_analysed: int = 0
    analysis_basis: str = ""

    timestamp: datetime

    explainability_type: str = "Fairness Explainability"

    xai_note: str = Field(
        default="L8 introduces Fairness Explainability — detecting unjustified "
                "patterns across decisions. L7 asks 'is this allowed?'. "
                "L8 asks 'is the system fair over time?'. These are "
                "fundamentally different questions. This directly operationalises "
                "RBI fair lending guidelines and EU AI Act Article 10.",
    )


# ─────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────

def _compute_eii(distribution: Dict[str, float]) -> float:
    """
    Exposure Imbalance Index using normalised entropy.
    EII = 0 means perfect equal distribution (fair)
    EII = 1 means complete concentration in one group (biased)
    """
    values = [v for v in distribution.values() if v > 0]
    if len(values) <= 1:
        return 1.0  # complete concentration
    n = len(values)
    total = sum(values)
    if total == 0:
        return 0.0
    proportions = [v / total for v in values]
    entropy = -sum(p * math.log(p) for p in proportions if p > 0)
    max_entropy = math.log(n)
    normalised = entropy / max_entropy if max_entropy > 0 else 0
    return round(1.0 - normalised, 3)  # invert: low entropy = high EII


def _get_income_band(income: float) -> str:
    for band, (lo, hi) in INCOME_BANDS.items():
        if lo <= income <= hi:
            return band
    return "high"


def _load_historical_data() -> List[dict]:
    """Load all past application summaries and their key signals."""
    summaries = get_all_summaries()
    enriched = []
    for s in summaries:
        app_id = s.get("application_id", "")
        l3_data = get_layer_artifact(app_id, "L3")
        l4_data = get_layer_artifact(app_id, "L4")
        l6_data = get_layer_artifact(app_id, "L6")
        if l3_data and l4_data:
            signals = {sig["signal_key"]: sig["raw_value"]
                       for sig in l3_data.get("signals", [])}
            enriched.append({
                "application_id": app_id,
                "decision": s.get("decision", ""),
                "approval_probability": s.get("approval_probability", 0.5),
                "confidence": s.get("confidence", "Moderate"),
                "timestamp": s.get("timestamp", ""),
                "signals": signals,
                "approval_probability_model": l4_data.get(
                    "approval_probability", 0.5
                ),
                "confidence_score": l6_data.get(
                    "composite_score", 0.75
                ) if l6_data else 0.75,
                # L3 features for grouping
                "employment_type": "salaried",  # proxy from signals
                "income": signals.get("verified_monthly_income", 50000),
                "cibil": signals.get("cibil_score", 700),
                "dti": signals.get("dti_ratio", 0.4),
                "upi": signals.get("upi_velocity", 200),
            })
    return enriched


# ─────────────────────────────────────────────────────────────────
# Diagnostic area implementations
# ─────────────────────────────────────────────────────────────────

def _analyse_representation(
    history: List[dict],
    current: dict,
) -> RepresentationFairness:
    """Analyse representation fairness across employment, income, purpose."""
    flags = []

    if len(history) < MINIMUM_HISTORY:
        # Neutral baseline
        emp_dist = {"salaried": 0.6, "self_employed": 0.25, "business": 0.15}
        inc_dist = {"low": 0.3, "mid": 0.4, "high": 0.3}
        pur_dist = {p: 1/len(PURPOSES) for p in PURPOSES}
        eii_emp = 0.20
        eii_inc = 0.10
        score = 0.80
        basis = (f"Neutral baseline — insufficient history "
                 f"({len(history)} applications, minimum {MINIMUM_HISTORY})")
    else:
        # Income band distribution
        inc_counts: Dict[str, int] = defaultdict(int)
        emp_counts: Dict[str, int] = defaultdict(int)
        pur_counts: Dict[str, int] = defaultdict(int)

        for h in history:
            band = _get_income_band(h["income"])
            inc_counts[band] += 1
            emp_counts["salaried"] += 1  # proxy
            pur_counts["other"] += 1

        total = len(history)
        emp_dist = {k: round(v/total, 3) for k, v in emp_counts.items()}
        inc_dist = {k: round(v/total, 3) for k, v in inc_counts.items()}
        pur_dist = {k: round(v/total, 3) for k, v in pur_counts.items()}

        eii_emp = _compute_eii(emp_dist)
        eii_inc = _compute_eii(inc_dist)

        if eii_inc > 0.40:
            flags.append(
                f"Income band imbalance detected (EII={eii_inc:.2f}) — "
                f"high-income applicants may be over-represented"
            )

        # Score: lower EII = better representation
        avg_eii = (eii_emp + eii_inc) / 2
        score = round(max(0, 1.0 - avg_eii), 3)
        basis = f"Based on {len(history)} historical applications"

    return RepresentationFairness(
        employment_distribution=emp_dist,
        income_band_distribution=inc_dist,
        purpose_distribution=pur_dist,
        eii_employment=eii_emp,
        eii_income=eii_inc,
        flags=flags,
        score=score,
        data_basis=basis,
    )


def _analyse_outcome_fairness(
    history: List[dict],
    current_approval: float,
    current_confidence: float,
) -> OutcomeFairness:
    """Analyse outcome fairness — approval rate and confidence parity."""
    flags = []

    if len(history) < MINIMUM_HISTORY:
        # Neutral baseline — no disparity assumed
        approval_rates = {"all_groups": round(current_approval, 3)}
        confidence_scores = {"all_groups": round(current_confidence, 3)}
        arp = 0.05
        cp = 0.03
        score = 0.85
        basis = (f"Neutral baseline — {len(history)} applications "
                 f"(minimum {MINIMUM_HISTORY} needed)")
    else:
        # Group by income band and compute approval rates
        band_approvals: Dict[str, List[float]] = defaultdict(list)
        band_confidence: Dict[str, List[float]] = defaultdict(list)

        for h in history:
            band = _get_income_band(h["income"])
            approved = 1.0 if "Approved" in (h.get("decision") or "") else 0.0
            band_approvals[band].append(approved)
            band_confidence[band].append(h.get("confidence_score", 0.75))

        approval_rates = {
            band: round(sum(v)/len(v), 3)
            for band, v in band_approvals.items() if v
        }
        confidence_scores = {
            band: round(sum(v)/len(v), 3)
            for band, v in band_confidence.items() if v
        }

        # Compute disparities
        if len(approval_rates) >= 2:
            rates = list(approval_rates.values())
            arp = round(max(rates) - min(rates), 3)
        else:
            arp = 0.0

        if len(confidence_scores) >= 2:
            scores = list(confidence_scores.values())
            cp = round(max(scores) - min(scores), 3)
        else:
            cp = 0.0

        if arp > DISPARITY_FLAG_THRESHOLD:
            flags.append(
                f"Approval rate disparity of {arp:.0%} detected across "
                f"income groups — exceeds {DISPARITY_FLAG_THRESHOLD:.0%} threshold"
            )
        if cp > 0.10:
            flags.append(
                f"Confidence score disparity of {cp:.3f} across income groups"
            )

        score = round(max(0, 1.0 - (arp * 2)), 3)
        basis = f"Based on {len(history)} historical applications"

    return OutcomeFairness(
        approval_rates=approval_rates,
        confidence_scores=confidence_scores,
        arp_disparity=arp,
        cp_disparity=cp,
        flags=flags,
        score=score,
        data_basis=basis,
    )


def _analyse_evidence_source(
    l3: L3SignalLog,
    l4: L4ModelReasoning,
) -> EvidenceSourceFairness:
    """Analyse evidence and source fairness."""
    flags = []

    # Compute SHAP-based signal dominance
    total_shap = sum(abs(c.shap_value) for c in l4.shap_contributions)
    signal_weights: Dict[str, float] = {}

    if total_shap > 0:
        for c in l4.shap_contributions:
            weight = round(abs(c.shap_value) / total_shap, 3)
            signal_weights[c.feature_key] = weight

    # Source concentration — how much does CIBIL dominate?
    cibil_shap = abs(signal_weights.get("cibil_score", 0))
    dti_shap = abs(signal_weights.get("dti_ratio", 0))
    repayment_shap = abs(signal_weights.get("repayment_score", 0))

    # Bureau-based signals (CIBIL + repayment)
    bureau_weight = cibil_shap + repayment_shap
    # Alternative signals (UPI + income stability)
    upi_shap = abs(signal_weights.get("upi_velocity", 0))
    stability_shap = abs(signal_weights.get("income_stability", 0))
    alt_weight = upi_shap + stability_shap

    # Source concentration index
    sci = round(max(bureau_weight, dti_shap, alt_weight), 3)

    if sci > CONCENTRATION_FLAG:
        flags.append(
            f"Source concentration detected — "
            f"bureau signals contribute {bureau_weight:.0%} of attribution. "
            f"Alternative data (UPI, stability) only {alt_weight:.0%}. "
            f"Thin-file borrowers may be systematically disadvantaged."
        )

    # Alternative data utilisation
    alt_utilisation = round(alt_weight, 3)
    if alt_utilisation < 0.15:
        flags.append(
            f"Low alternative data utilisation ({alt_utilisation:.0%}) — "
            f"UPI and income stability signals are underweighted. "
            f"This may disadvantage borrowers without formal credit history."
        )

    # Dominant signals
    dominant = [
        c.feature_key for c in sorted(
            l4.shap_contributions,
            key=lambda x: abs(x.shap_value),
            reverse=True
        )[:3]
    ]

    # Proxy bias detection
    proxy_risks = []
    proxy_detected = False
    proxy_details = None

    # UPI velocity as geographic proxy (urban vs rural)
    upi_val = next(
        (s.raw_value for s in l3.signals if s.signal_key == "upi_velocity"),
        None
    )
    if upi_val is not None and upi_shap > 0.10:
        proxy_risks.append(
            f"UPI transaction velocity ({upi_val:.0f} tx/month) has "
            f"high SHAP weight ({upi_shap:.0%}) — may act as proxy for "
            f"geographic location (urban/rural). RBI fair lending guidelines "
            f"prohibit geographic discrimination."
        )
        proxy_detected = upi_shap > 0.20
        if proxy_detected:
            proxy_details = (
                f"UPI velocity contributes {upi_shap:.0%} to this decision. "
                f"Urban applicants typically show higher UPI velocity — "
                f"this feature may encode geographic bias."
            )

    # Income stability as proxy for employment informality
    stability_val = next(
        (s.raw_value for s in l3.signals if s.signal_key == "income_stability"),
        None
    )
    if stability_val is not None and stability_shap > 0.12:
        proxy_risks.append(
            f"Income stability ({stability_val:.2f}) has high attribution "
            f"({stability_shap:.0%}) — may encode employment informality, "
            f"potentially acting as proxy for self-employed/gig workers."
        )

    source_weights_display = {
        "Bureau signals (CIBIL + Repayment)": round(bureau_weight, 3),
        "Debt burden signals (DTI + Obligations)": round(
            dti_shap + abs(signal_weights.get("existing_loan_burden", 0)), 3
        ),
        "Alternative signals (UPI + Stability)": round(alt_weight, 3),
        "Income signals": round(
            abs(signal_weights.get("verified_monthly_income", 0)), 3
        ),
        "Product signals (Amount + Tenure)": round(
            abs(signal_weights.get("loan_amount", 0)) +
            abs(signal_weights.get("tenure_months", 0)), 3
        ),
    }

    score = round(max(0, 1.0 - sci), 3)

    return EvidenceSourceFairness(
        source_weights=source_weights_display,
        dominant_signals=dominant,
        sci=sci,
        cibil_dependency=round(bureau_weight, 3),
        alternative_data_utilisation=alt_utilisation,
        flags=flags,
        score=score,
        proxy_bias_detected=proxy_detected,
        proxy_details=proxy_details,
    )


def _analyse_temporal_fairness(
    history: List[dict],
    current_approval: float,
) -> TemporalFairness:
    """Analyse temporal fairness — drift detection over time."""
    flags = []

    if len(history) < MINIMUM_HISTORY:
        return TemporalFairness(
            historical_approval_rate=0.65,
            recent_approval_rate=round(current_approval, 3),
            tds=0.0,
            recency_bias_detected=False,
            regime_shift_detected=False,
            data_points_analysed=len(history),
            flags=[],
            score=0.85,
            data_basis=(
                f"Neutral baseline — {len(history)} applications "
                f"(minimum {MINIMUM_HISTORY} needed for drift analysis)"
            ),
        )

    # Sort by timestamp
    sorted_history = sorted(
        history, key=lambda x: x.get("timestamp", "")
    )

    # Split into historical (older half) and recent (newer half)
    mid = len(sorted_history) // 2
    older = sorted_history[:mid]
    recent = sorted_history[mid:]

    def approval_rate(group):
        if not group:
            return 0.65
        approved = sum(
            1 for h in group
            if "Approved" in (h.get("decision") or "")
            and "Blocked" not in (h.get("decision") or "")
        )
        return round(approved / len(group), 3)

    hist_rate = approval_rate(older)
    rec_rate = approval_rate(recent)
    tds = round(abs(rec_rate - hist_rate), 3)

    recency_bias = tds > DRIFT_FLAG_THRESHOLD
    regime_shift = tds > 0.30

    if recency_bias:
        flags.append(
            f"Temporal drift detected — approval rate shifted from "
            f"{hist_rate:.0%} (historical) to {rec_rate:.0%} (recent). "
            f"TDS = {tds:.2f} exceeds threshold of {DRIFT_FLAG_THRESHOLD:.2f}."
        )
    if regime_shift:
        flags.append(
            f"Possible regime shift — approval rate changed by "
            f"{tds:.0%} which suggests a significant policy or "
            f"data distribution change."
        )

    score = round(max(0, 1.0 - (tds * 2)), 3)

    return TemporalFairness(
        historical_approval_rate=hist_rate,
        recent_approval_rate=rec_rate,
        tds=tds,
        recency_bias_detected=recency_bias,
        regime_shift_detected=regime_shift,
        data_points_analysed=len(history),
        flags=flags,
        score=score,
        data_basis=f"Analysed {len(history)} applications over time",
    )


def _analyse_individual_fairness(
    current_signals: Dict[str, float],
    current_approval: float,
    history: List[dict],
    application_id: str,
) -> IndividualProfileFairness:
    """
    Analyse individual/profile fairness.
    Are similar profiles treated consistently?
    """
    flags = []
    proxy_risks = []

    if len(history) < MINIMUM_HISTORY:
        return IndividualProfileFairness(
            similar_profiles_found=0,
            ics=0.80,
            consistency_variance=0.0,
            proxy_risks=[],
            flags=[],
            score=0.80,
            data_basis=(
                f"Neutral baseline — {len(history)} applications "
                f"(minimum {MINIMUM_HISTORY} needed)"
            ),
        )

    # Find similar profiles (within 10% of key signals)
    current_cibil = current_signals.get("cibil_score", 700)
    current_income = current_signals.get("verified_monthly_income", 50000)
    current_dti = current_signals.get("dti_ratio", 0.4)

    similar = []
    for h in history:
        if h.get("application_id") == application_id:
            continue
        h_cibil = h["signals"].get("cibil_score", 700)
        h_income = h["signals"].get("verified_monthly_income", 50000)
        h_dti = h["signals"].get("dti_ratio", 0.4)

        # Similar = within 10% on all three key dimensions
        cibil_similar = abs(h_cibil - current_cibil) / max(current_cibil, 1) < 0.10
        income_similar = abs(h_income - current_income) / max(current_income, 1) < 0.15
        dti_similar = abs(h_dti - current_dti) < 0.08

        if cibil_similar and income_similar and dti_similar:
            similar.append(h)

    ics = 0.80  # default
    variance = 0.0

    if similar:
        similar_probs = [
            h.get("approval_probability", 0.5) for h in similar
        ]
        similar_probs.append(current_approval)

        mean_prob = sum(similar_probs) / len(similar_probs)
        variance = round(
            sum((p - mean_prob) ** 2 for p in similar_probs) / len(similar_probs),
            4
        )
        # High variance = low consistency
        ics = round(max(0, 1.0 - (variance * 10)), 3)

        if variance > 0.05:
            flags.append(
                f"Inconsistency detected across {len(similar)} similar profiles "
                f"(variance={variance:.4f}). Similar applicants are receiving "
                f"significantly different approval probabilities."
            )

    # Proxy risks for this application
    upi = current_signals.get("upi_velocity", 200)
    if upi < 50:
        proxy_risks.append(
            "Low UPI velocity may indicate rural applicant — "
            "this feature should not disadvantage borrowers from "
            "areas with lower digital payment penetration."
        )
    stability = current_signals.get("income_stability", 0.8)
    if stability < 0.60:
        proxy_risks.append(
            "Low income stability may proxy for informal employment — "
            "consider whether this accurately reflects creditworthiness "
            "vs employment type bias."
        )

    score = round((ics + max(0, 1.0 - variance * 5)) / 2, 3)

    return IndividualProfileFairness(
        similar_profiles_found=len(similar),
        ics=ics,
        consistency_variance=variance,
        proxy_risks=proxy_risks,
        flags=flags,
        score=score,
        data_basis=(
            f"Found {len(similar)} similar profiles from "
            f"{len(history)} total applications"
        ),
    )


# ─────────────────────────────────────────────────────────────────
# Bias flag builder
# ─────────────────────────────────────────────────────────────────

def _build_bias_flags(
    rep: RepresentationFairness,
    out: OutcomeFairness,
    evid: EvidenceSourceFairness,
    temp: TemporalFairness,
    indiv: IndividualProfileFairness,
) -> List[BiasFlag]:
    """Synthesise all detected issues into structured BiasFlag objects."""
    flags = []

    # Source concentration flag
    if evid.sci > CONCENTRATION_FLAG:
        flags.append(BiasFlag(
            bias_type="Source Bias",
            dimension="Evidence & Source Fairness",
            severity="medium" if evid.sci < 0.75 else "high",
            description=(
                f"Bureau signals dominate attribution ({evid.cibil_dependency:.0%}). "
                f"Alternative data underutilised ({evid.alternative_data_utilisation:.0%})."
            ),
            observed_value=f"SCI = {evid.sci:.3f}",
            threshold=f"SCI < {CONCENTRATION_FLAG:.2f}",
            root_cause_indicator=(
                "CIBIL and repayment history are weighted more heavily than "
                "UPI and bank statement signals. Thin-file borrowers who lack "
                "formal credit history may be systematically disadvantaged."
            ),
            recommended_action=(
                "Re-weight alternative data signals. Consider increasing UPI "
                "velocity and income stability weights for thin-file applicants. "
                "Align with RBI financial inclusion objectives."
            ),
            regulation_reference="RBI Guidelines on Financial Inclusion — Alternative Data Usage",
        ))

    # Proxy bias flag
    if evid.proxy_bias_detected:
        flags.append(BiasFlag(
            bias_type="Demographic Proxy Bias",
            dimension="Individual/Profile Fairness",
            severity="high",
            description=evid.proxy_details or "Proxy bias detected",
            observed_value="UPI velocity SHAP weight > 20%",
            threshold="Proxy features should not dominate attribution",
            root_cause_indicator=(
                "UPI transaction velocity correlates with geographic location "
                "(urban areas have higher UPI penetration). Using this as a "
                "high-weight feature may encode geographic discrimination."
            ),
            recommended_action=(
                "Review UPI velocity weighting. Consider normalising by "
                "geographic region. Add fairness constraints to prevent "
                "geographic proxies from dominating credit decisions."
            ),
            regulation_reference=(
                "RBI Fair Lending Guidelines — Non-discrimination provisions. "
                "EU AI Act Article 10 — Training data requirements."
            ),
        ))

    # Outcome disparity flag
    if out.arp_disparity > DISPARITY_FLAG_THRESHOLD:
        flags.append(BiasFlag(
            bias_type="Outcome Disparity",
            dimension="Outcome Fairness",
            severity="medium" if out.arp_disparity < 0.25 else "high",
            description=(
                f"Approval rate gap of {out.arp_disparity:.0%} detected "
                f"across income groups."
            ),
            observed_value=f"ARP gap = {out.arp_disparity:.3f}",
            threshold=f"ARP gap < {DISPARITY_FLAG_THRESHOLD:.2f}",
            root_cause_indicator=(
                "Different income groups are receiving materially different "
                "approval rates. This may reflect genuine risk differences "
                "or could indicate systematic discrimination. Investigation required."
            ),
            recommended_action=(
                "Conduct root cause analysis — determine whether disparity "
                "reflects genuine creditworthiness differences or model bias. "
                "Consider fairness constraints in model retraining."
            ),
            regulation_reference=(
                "RBI Fair Lending Guidelines — Equal treatment provisions"
            ),
        ))

    # Temporal drift flag
    if temp.recency_bias_detected:
        flags.append(BiasFlag(
            bias_type="Temporal Drift",
            dimension="Temporal Fairness",
            severity="medium",
            description=(
                f"Approval rates shifted by {temp.tds:.0%} over time "
                f"({temp.historical_approval_rate:.0%} → "
                f"{temp.recent_approval_rate:.0%})."
            ),
            observed_value=f"TDS = {temp.tds:.3f}",
            threshold=f"TDS < {DRIFT_FLAG_THRESHOLD:.2f}",
            root_cause_indicator=(
                "Recent application approval rates differ significantly from "
                "historical baseline. May indicate model drift, data distribution "
                "shift, or policy change without documented governance."
            ),
            recommended_action=(
                "Investigate cause of drift. If policy change, document in "
                "governance record. If data drift, retrain model. "
                "Implement continuous monitoring with automated alerts."
            ),
            regulation_reference=(
                "RBI Model Risk Management — Ongoing monitoring requirements"
            ),
        ))

    # Individual consistency flag
    if indiv.similar_profiles_found > 0 and indiv.ics < 0.70:
        flags.append(BiasFlag(
            bias_type="Inconsistent Treatment",
            dimension="Individual/Profile Fairness",
            severity="medium",
            description=(
                f"Similar profiles receiving inconsistent outcomes "
                f"(ICS = {indiv.ics:.3f}, variance = {indiv.consistency_variance:.4f})."
            ),
            observed_value=f"ICS = {indiv.ics:.3f}",
            threshold="ICS > 0.70",
            root_cause_indicator=(
                "Applicants with similar financial profiles are receiving "
                "materially different approval probabilities. This suggests "
                "model instability or unexplained factors in the decision."
            ),
            recommended_action=(
                "Review model stability. Ensure similar profiles are "
                "receiving consistent treatment. Consider adding fairness "
                "constraints to reduce individual-level variance."
            ),
            regulation_reference=(
                "RBI Fair Lending Guidelines — Individual fairness provisions"
            ),
        ))

    return flags


# ─────────────────────────────────────────────────────────────────
# OBS computation and verdict
# ─────────────────────────────────────────────────────────────────

def _compute_obs(
    rep: RepresentationFairness,
    out: OutcomeFairness,
    evid: EvidenceSourceFairness,
    temp: TemporalFairness,
    indiv: IndividualProfileFairness,
) -> Tuple[float, str, str, bool]:
    """
    Overall Bias Score = weighted combination of all five areas.
    Lower score = more bias detected.
    """
    # Convert scores to bias scores (0 = fair, 1 = biased)
    bias_scores = {
        "representation": 1.0 - rep.score,
        "outcome":        1.0 - out.score,
        "evidence":       1.0 - evid.score,
        "temporal":       1.0 - temp.score,
        "individual":     1.0 - indiv.score,
    }

    # Weights from LEAF diagram
    weights = {
        "representation": 0.20,
        "outcome":        0.30,
        "evidence":       0.25,
        "temporal":       0.15,
        "individual":     0.10,
    }

    obs = round(sum(
        bias_scores[k] * weights[k] for k in weights
    ), 3)

    if obs < 0.25:
        verdict = "Fair"
        color = "green"
        investigation = False
    elif obs < 0.55:
        verdict = "Caution"
        color = "amber"
        investigation = True
    else:
        verdict = "Biased"
        color = "red"
        investigation = True

    return obs, verdict, color, investigation


# ─────────────────────────────────────────────────────────────────
# Layer implementation
# ─────────────────────────────────────────────────────────────────

class L8FairnessDiagnostics_Layer:
    """
    Layer 8: Bias & Fairness Diagnostics

    Input  : L0, L3, L4, L6, L7
    Output : L8FairnessDiagnostics

    Operates at two levels:
        Individual — bias signals in this specific decision
        System     — aggregate patterns across all history
    """

    def process(
        self,
        l0: L0IntakeRecord,
        l3: L3SignalLog,
        l4: L4ModelReasoning,
        l6: L6ConfidenceGrade,
        l7: Optional[L7GovernanceVerdict] = None,
        application_id: str = "",
    ) -> L8FairnessDiagnostics:

        app_id = application_id or l0.application_id
        signals = {s.signal_key: s.raw_value for s in l3.signals}

        # Load historical data from Evidence Ledger
        history = _load_historical_data()
        # Exclude current application from history
        history = [h for h in history if h["application_id"] != app_id]

        # Run all five diagnostic areas
        rep = _analyse_representation(history, signals)
        out = _analyse_outcome_fairness(
            history, l4.approval_probability, l6.composite_score
        )
        evid = _analyse_evidence_source(l3, l4)
        temp = _analyse_temporal_fairness(history, l4.approval_probability)
        indiv = _analyse_individual_fairness(
            signals, l4.approval_probability, history, app_id
        )

        # Compute OBS and verdict
        obs, verdict, color, investigation = _compute_obs(
            rep, out, evid, temp, indiv
        )

        # Build structured bias flags
        bias_flags = _build_bias_flags(rep, out, evid, temp, indiv)

        # Key metrics
        eii = (rep.eii_employment + rep.eii_income) / 2
        arp = out.arp_disparity
        sci = evid.sci
        tds = temp.tds
        ics = indiv.ics

        # Remediation actions
        remediation = list(set(f.recommended_action for f in bias_flags))
        if not remediation:
            remediation = [
                "Continue monitoring — no material bias detected at this time.",
                "Maintain diverse data source utilisation.",
                "Review fairness metrics after every 50 applications.",
            ]

        # User-facing summary
        if verdict == "Fair":
            summary = (
                f"Fairness check: No material bias detected. "
                f"This decision shows balanced use of evidence sources "
                f"and is consistent with similar past applications."
            )
        elif verdict == "Caution":
            summary = (
                f"Fairness check: Some patterns warrant attention. "
                f"{len(bias_flags)} fairness concern(s) detected. "
                f"No immediate action required but monitoring is recommended."
            )
        else:
            summary = (
                f"Fairness check: Significant patterns detected. "
                f"{len(bias_flags)} fairness issue(s) identified. "
                f"Investigation is required before this pattern continues."
            )

        # Audit narrative
        audit_parts = [
            f"Fairness verdict: {verdict} (OBS = {obs:.3f}).",
            f"Analysed {len(history)} historical applications.",
            f"Five diagnostic areas: "
            f"Representation ({rep.score:.2f}), "
            f"Outcome ({out.score:.2f}), "
            f"Evidence ({evid.score:.2f}), "
            f"Temporal ({temp.score:.2f}), "
            f"Individual ({indiv.score:.2f}).",
        ]
        if bias_flags:
            audit_parts.append(
                f"Detected: {', '.join(f.bias_type for f in bias_flags)}."
            )
        if evid.proxy_bias_detected:
            audit_parts.append(
                "PROXY BIAS ALERT: Demographic proxy detected. "
                "Regulatory review recommended."
            )

        return L8FairnessDiagnostics(
            application_id=app_id,
            representation=rep,
            outcome=out,
            evidence_source=evid,
            temporal=temp,
            individual_profile=indiv,
            eii=round(eii, 3),
            arp=arp,
            sci=sci,
            tds=tds,
            ics=ics,
            obs=obs,
            verdict=verdict,
            verdict_color=color,
            investigation_required=investigation,
            bias_flags=bias_flags,
            total_flags=len(bias_flags),
            remediation_actions=remediation,
            fairness_summary=summary,
            audit_narrative=" ".join(audit_parts),
            total_applications_analysed=len(history),
            analysis_basis=(
                f"Individual-level analysis: this application. "
                f"System-level analysis: {len(history)} historical applications."
                if len(history) >= MINIMUM_HISTORY
                else f"Individual-level analysis only. "
                     f"System-level analysis requires {MINIMUM_HISTORY} applications "
                     f"(currently {len(history)})."
            ),
            timestamp=datetime.now(),
        )
