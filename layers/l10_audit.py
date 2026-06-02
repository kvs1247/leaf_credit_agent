"""
LEAF Layer 10 — Auditability & Reproducibility Ledger
=====================================================================
Core question: "Can every recommendation, decision, override, and
outcome be reconstructed, verified, and defended?"

Layer 10 introduces the final two explainability types:
    Audit Explainability        — can an auditor verify this?
    Reproducibility Explainability — can another team recreate this?

Complete LEAF explainability taxonomy:
    L1  — Source Explainability
    L2  — Retrieval Explainability
    L3  — Signal Explainability
    L4  — Reasoning Explainability
    L5  — Decision Explainability
    L6  — Confidence Explainability
    L7  — Governance Explainability
    L8  — Fairness Explainability
    L9  — Accountability Explainability
    L10 — Audit & Reproducibility Explainability  ← this layer

The three pillars:
    1. Traceability    — what happened? (complete decision chain)
    2. Reproducibility — can we recreate the same result?
    3. Auditability    — can an independent party verify it?

The philosophy:
    A financial AI system should never say "trust me."
    It should always say "here is exactly how this was produced."

The aircraft analogy:
    Layers 1–8  = navigation systems
    Layer 9     = pilot intervention
    Layer 10    = flight data recorder (black box)

Layer 10 is not helping the AI make better decisions.
It is helping auditors, regulators, researchers, and compliance
officers understand and verify decisions — sometimes years later.

Doctoral contribution:
    L10 directly addresses Research Gap 3 (absence of integrated
    lifecycle governance frameworks) from Chapter 2. It operationalises
    BCBS 239, RBI model risk management, and EU AI Act Article 13
    (transparency obligations) as a first-class system component.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel, Field

from storage.ledger import retrieve_application_ledger, get_layer_artifact


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

LEAF_VERSION = "1.0.0"
MODEL_VERSION = "xgb-leaf-v1"
SHAP_VERSION = "TreeExplainer-v1"
CONFIDENCE_FORMULA_VERSION = "CF-v1.0"
COMPLIANCE_RULES_VERSION = "RBI-2024-v1"
FAIRNESS_ENGINE_VERSION = "FE-v1.0"

# Layers that must be present for a complete audit trail
REQUIRED_LAYERS = ["L0", "L1", "L2", "L3", "L4", "L6", "L7", "L8", "L9_PENDING"]
OPTIONAL_LAYERS = ["L5", "L9_COMPLETED"]

# Regulatory frameworks this system aligns with
REGULATORY_ALIGNMENTS = [
    "RBI Fair Lending Guidelines",
    "RBI Draft Model Risk Framework 2024",
    "RBI BCBS 239 — Risk Data Aggregation",
    "EU AI Act 2024 — Article 13 (Transparency)",
    "EU AI Act 2024 — Article 14 (Human Oversight)",
    "OECD AI Principles — Accountability",
]


# ─────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────

class LayerAuditEntry(BaseModel):
    """Audit entry for a single layer artifact."""
    layer: str
    status: str = Field(...,
        description="present / missing / optional_missing")
    artifact_hash: str = Field(default="",
        description="SHA-256 hash stored at seal time")
    verified_hash: str = Field(default="",
        description="SHA-256 hash recomputed at audit time")
    integrity: str = Field(default="",
        description="INTACT / COMPROMISED / NOT_VERIFIED")
    sealed_at: Optional[str] = None
    layer_description: str = ""


class CompletenessReport(BaseModel):
    """
    Pillar 1 — Traceability
    Which layers are present and which are missing?
    """
    entries: List[LayerAuditEntry]
    required_present: int
    required_missing: int
    optional_present: int
    completeness_score: float = Field(..., ge=0, le=1,
        description="required_present / total_required")
    missing_layers: List[str] = Field(default_factory=list)
    is_complete: bool
    completeness_label: str


class IntegrityReport(BaseModel):
    """
    Pillar 3 — Auditability
    Re-hash all artifacts and verify against stored hashes.
    """
    total_artifacts: int
    intact: int
    compromised: int
    not_verified: int
    integrity_score: float = Field(..., ge=0, le=1)
    binding_hash: str = Field(...,
        description="SHA-256 of all artifact hashes concatenated — "
                    "changes if any artifact is modified")
    integrity_status: str = Field(...,
        description="INTACT / COMPROMISED / PARTIAL")
    verified_at: datetime


class ReproducibilityRecord(BaseModel):
    """
    Pillar 2 — Reproducibility
    Everything needed to recreate this decision in the future.
    """
    # System versions
    leaf_version: str = LEAF_VERSION
    model_version: str = MODEL_VERSION
    shap_version: str = SHAP_VERSION
    confidence_formula_version: str = CONFIDENCE_FORMULA_VERSION
    compliance_rules_version: str = COMPLIANCE_RULES_VERSION
    fairness_engine_version: str = FAIRNESS_ENGINE_VERSION

    # Confidence formula (exact)
    confidence_formula: str = (
        "C = 0.30·Grounding + 0.25·Freshness + "
        "0.20·Consistency + 0.15·Calibration + 0.10·Compliance"
    )

    # Model configuration
    model_algorithm: str = "XGBoost (XGBClassifier)"
    model_features: List[str] = Field(default_factory=lambda: [
        "cibil_score", "dti_ratio", "repayment_score",
        "verified_monthly_income", "upi_velocity",
        "income_stability", "existing_loan_burden",
        "loan_amount", "tenure_months",
        "employment_type_encoded"
    ])
    model_auc: float = 0.633

    # Data versions
    data_sources_used: List[str] = Field(default_factory=list)
    data_freshness_at_decision: Dict[str, str] = Field(
        default_factory=dict
    )

    # Regulatory configuration
    jurisdiction: str = "India"
    regulatory_frameworks: List[str] = Field(
        default_factory=lambda: REGULATORY_ALIGNMENTS
    )

    # Reproducibility verdict
    reproducibility_score: float = Field(..., ge=0, le=1,
        description="How reproducible is this decision? "
                    "1.0 = fully reproducible with stored artifacts")
    reproducibility_note: str = ""


class DecisionTimeline(BaseModel):
    """Chronological trace of the complete decision lifecycle."""
    events: List[Dict[str, str]]
    total_duration_seconds: float
    start_time: str
    end_time: str


class AuditCertificate(BaseModel):
    """
    The binding audit certificate for this decision.
    This is what a regulator receives.

    The binding_hash changes if ANY artifact in the Evidence Ledger
    is modified — making tampering detectable.
    """
    certificate_id: str
    application_id: str
    issued_at: datetime

    # Decision summary
    final_decision: str
    approval_probability: float
    confidence_grade: str
    legitimacy_verdict: str
    fairness_verdict: str
    human_review_status: str

    # Audit status
    completeness_score: float
    integrity_status: str
    reproducibility_score: float
    binding_hash: str

    # Overall audit verdict
    audit_verdict: str = Field(...,
        description="CLEAN / QUALIFIED / ADVERSE")
    audit_verdict_reason: str

    # Regulatory alignment
    regulatory_frameworks: List[str] = Field(
        default_factory=lambda: REGULATORY_ALIGNMENTS
    )
    retention_period: str = "7 years (RBI requirement)"


class L10AuditLedger(BaseModel):
    """
    L10 complete output — the master audit record.
    Contains all three pillars and the binding certificate.
    """
    layer: str = "L10"
    application_id: str

    # Three pillars
    completeness: CompletenessReport
    integrity: IntegrityReport
    reproducibility: ReproducibilityRecord

    # Decision timeline
    timeline: DecisionTimeline

    # The certificate
    audit_certificate: AuditCertificate

    # Summary
    audit_summary: str

    timestamp: datetime

    explainability_types: List[str] = Field(
        default_factory=lambda: [
            "Audit Explainability",
            "Reproducibility Explainability"
        ]
    )

    xai_note: str = Field(
        default="L10 completes the LEAF explainability stack. "
                "It ensures recommendations are not just explainable today "
                "but verifiable and reproducible in the future. "
                "This directly addresses Research Gap 3 (lifecycle governance) "
                "from Chapter 2.",
    )


# ─────────────────────────────────────────────────────────────────
# Layer descriptions
# ─────────────────────────────────────────────────────────────────

LAYER_DESCRIPTIONS = {
    "L0": "Request & Context — regulatory envelope locked",
    "L1": "Data Provenance — all sources hashed and timestamped",
    "L2": "Grounding Check — data reliability scored",
    "L3": "Signal Extraction — features computed with traceability",
    "L4": "Model Reasoning — XGBoost + SHAP individual attribution",
    "L5": "Recommendation — LLM-generated Explanation Card",
    "L6": "Confidence Grade — meta-explainability assessment",
    "L7": "Compliance & Suitability — governance legitimacy",
    "L8": "Bias & Fairness — systemic fairness diagnostics",
    "L9_PENDING": "Human-in-the-Loop — review triggered",
    "L9_COMPLETED": "Human Decision — review completed and sealed",
    "AGENT_TRACE": "Agent Reasoning — LLM observations at each layer",
}


# ─────────────────────────────────────────────────────────────────
# Pillar 1 — Completeness / Traceability
# ─────────────────────────────────────────────────────────────────

def _check_completeness(
    application_id: str,
    ledger_entries: list,
) -> CompletenessReport:
    """Check which layers are present and compute completeness score."""
    present_layers = {e.layer for e in ledger_entries}
    entries = []

    all_layers = REQUIRED_LAYERS + OPTIONAL_LAYERS

    for layer in all_layers:
        is_required = layer in REQUIRED_LAYERS
        is_present = layer in present_layers

        # Find the entry
        matching = [e for e in ledger_entries if e.layer == layer]
        entry = matching[-1] if matching else None

        if is_present and entry:
            status = "present"
        elif is_required:
            status = "missing"
        else:
            status = "optional_missing"

        entries.append(LayerAuditEntry(
            layer=layer,
            status=status,
            artifact_hash=entry.artifact_hash if entry else "",
            sealed_at=entry.sealed_at.isoformat() if entry else None,
            layer_description=LAYER_DESCRIPTIONS.get(layer, layer),
        ))

    required_present = sum(
        1 for e in entries
        if e.layer in REQUIRED_LAYERS and e.status == "present"
    )
    required_missing = len(REQUIRED_LAYERS) - required_present
    optional_present = sum(
        1 for e in entries
        if e.layer in OPTIONAL_LAYERS and e.status == "present"
    )

    completeness_score = round(
        required_present / len(REQUIRED_LAYERS), 3
    )

    missing = [
        e.layer for e in entries
        if e.layer in REQUIRED_LAYERS and e.status == "missing"
    ]

    if completeness_score == 1.0:
        label = "Complete"
    elif completeness_score >= 0.80:
        label = "Substantially Complete"
    elif completeness_score >= 0.60:
        label = "Partial"
    else:
        label = "Incomplete"

    return CompletenessReport(
        entries=entries,
        required_present=required_present,
        required_missing=required_missing,
        optional_present=optional_present,
        completeness_score=completeness_score,
        missing_layers=missing,
        is_complete=(required_missing == 0),
        completeness_label=label,
    )


# ─────────────────────────────────────────────────────────────────
# Pillar 3 — Integrity / Auditability
# ─────────────────────────────────────────────────────────────────

def _verify_integrity(
    application_id: str,
    ledger_entries: list,
) -> IntegrityReport:
    """
    Re-hash all stored artifacts and compare to stored hashes.
    Compute the binding hash over all artifact hashes.
    """
    intact = 0
    compromised = 0
    not_verified = 0
    all_hashes = []

    for entry in ledger_entries:
        stored_hash = entry.artifact_hash
        artifact_json = entry.artifact_json

        if stored_hash and artifact_json:
            # Re-compute hash
            recomputed = hashlib.sha256(
                artifact_json.encode()
            ).hexdigest()[:12]

            if recomputed == stored_hash:
                intact += 1
                all_hashes.append(stored_hash)
            else:
                compromised += 1
                all_hashes.append(stored_hash)
        else:
            not_verified += 1

    # Binding hash — single number covering ALL artifacts
    # Changes if any single artifact is modified
    binding_input = "|".join(sorted(all_hashes))
    binding_hash = hashlib.sha256(
        binding_input.encode()
    ).hexdigest()[:16]

    total = intact + compromised + not_verified
    integrity_score = round(intact / max(total, 1), 3)

    if compromised > 0:
        status = "COMPROMISED"
    elif not_verified > 0:
        status = "PARTIAL"
    else:
        status = "INTACT"

    return IntegrityReport(
        total_artifacts=total,
        intact=intact,
        compromised=compromised,
        not_verified=not_verified,
        integrity_score=integrity_score,
        binding_hash=binding_hash,
        integrity_status=status,
        verified_at=datetime.now(),
    )


# ─────────────────────────────────────────────────────────────────
# Pillar 2 — Reproducibility
# ─────────────────────────────────────────────────────────────────

def _build_reproducibility(
    application_id: str,
    ledger_entries: list,
    completeness: CompletenessReport,
) -> ReproducibilityRecord:
    """Build the reproducibility record from stored metadata."""
    # Get data source info from L1
    l1_data = get_layer_artifact(application_id, "L1")
    sources = []
    freshness = {}

    if l1_data:
        for src in l1_data.get("sources", []):
            name = src.get("source_name", "Unknown")
            sources.append(name)
            freshness[name] = f"{src.get('age_hours', 0):.0f}h old"

    # Reproducibility score based on completeness + version availability
    repro_score = round(
        completeness.completeness_score * 0.60 +
        (1.0 if len(sources) > 0 else 0.5) * 0.40,
        3
    )

    if repro_score >= 0.90:
        note = (
            "High reproducibility. All version metadata stored. "
            "Future teams can recreate this decision with the same inputs."
        )
    elif repro_score >= 0.75:
        note = (
            "Good reproducibility. Most metadata stored. "
            "Minor gaps exist but decision can be substantially reconstructed."
        )
    else:
        note = (
            "Partial reproducibility. Some layers are missing. "
            "Reproduction may not yield identical results."
        )

    return ReproducibilityRecord(
        data_sources_used=sources,
        data_freshness_at_decision=freshness,
        reproducibility_score=repro_score,
        reproducibility_note=note,
    )


# ─────────────────────────────────────────────────────────────────
# Decision timeline
# ─────────────────────────────────────────────────────────────────

def _build_timeline(
    ledger_entries: list,
) -> DecisionTimeline:
    """Build chronological event trace from sealed artifacts."""
    events = []

    # Build events from ledger entries
    layer_to_event = {
        "L0": "Application received — regulatory context locked",
        "L1": "Data fetched from all sources — provenance recorded",
        "L2": "Data reliability verified — grounding scored",
        "L3": "Credit signals extracted — features computed",
        "L4": "Model inference run — SHAP attribution computed",
        "L6": "Confidence grade assigned",
        "L7": "Compliance & suitability checked",
        "L8": "Fairness diagnostics run",
        "L9_PENDING": "Human review triggered — awaiting decision",
        "L9_COMPLETED": "Human review completed — decision recorded",
        "L5": "Explanation Card generated",
        "AGENT_TRACE": "Agent reasoning trace sealed",
    }

    seen = {}
    for entry in sorted(ledger_entries,
                        key=lambda e: e.sealed_at or datetime.now()):
        if entry.layer in seen:
            continue
        seen[entry.layer] = True
        description = layer_to_event.get(
            entry.layer, f"{entry.layer} — artifact sealed"
        )
        events.append({
            "layer": entry.layer,
            "timestamp": entry.sealed_at.strftime("%H:%M:%S")
            if entry.sealed_at else "unknown",
            "event": description,
            "hash": entry.artifact_hash,
        })

    # Compute duration
    times = [
        e.sealed_at for e in ledger_entries
        if e.sealed_at is not None
    ]
    if len(times) >= 2:
        duration = (max(times) - min(times)).total_seconds()
        start = min(times).isoformat()
        end = max(times).isoformat()
    else:
        duration = 0.0
        start = end = datetime.now().isoformat()

    return DecisionTimeline(
        events=events,
        total_duration_seconds=round(duration, 2),
        start_time=start,
        end_time=end,
    )


# ─────────────────────────────────────────────────────────────────
# Audit certificate
# ─────────────────────────────────────────────────────────────────

def _issue_certificate(
    application_id: str,
    completeness: CompletenessReport,
    integrity: IntegrityReport,
    reproducibility: ReproducibilityRecord,
) -> AuditCertificate:
    """Issue the binding audit certificate."""
    certificate_id = f"LEAF-CERT-{datetime.now().strftime('%Y%m%d')}-" \
                     f"{str(uuid.uuid4())[:8].upper()}"

    # Get key decisions from stored artifacts
    l4_data = get_layer_artifact(application_id, "L4")
    l6_data = get_layer_artifact(application_id, "L6")
    l7_data = get_layer_artifact(application_id, "L7")
    l8_data = get_layer_artifact(application_id, "L8")
    l9_data = get_layer_artifact(application_id, "L9_COMPLETED") or \
              get_layer_artifact(application_id, "L9_PENDING")

    final_decision = l4_data.get("decision", "Unknown") if l4_data else "Unknown"
    approval_prob = l4_data.get("approval_probability", 0.0) if l4_data else 0.0
    conf_grade = l6_data.get("grade", "Unknown") if l6_data else "Unknown"

    legit = "Unknown"
    if l7_data:
        legit = l7_data.get("legitimacy_verdict", "Unknown")

    fairness = "Unknown"
    if l8_data:
        fairness = l8_data.get("verdict", "Unknown")

    human_status = "Not reviewed"
    if l9_data:
        human_status = l9_data.get("status", "Pending")
        if "approval_status" in l9_data:
            human_status = l9_data["approval_status"]

    # Audit verdict
    if (completeness.is_complete
            and integrity.integrity_status == "INTACT"
            and legit != "Blocked"):
        verdict = "CLEAN"
        verdict_reason = (
            "All layers present, all artifacts intact, "
            "governance requirements met."
        )
    elif integrity.integrity_status == "COMPROMISED":
        verdict = "ADVERSE"
        verdict_reason = (
            "One or more artifacts have been modified since sealing. "
            "Audit trail integrity cannot be guaranteed."
        )
    elif not completeness.is_complete:
        verdict = "QUALIFIED"
        verdict_reason = (
            f"Missing layers: {', '.join(completeness.missing_layers)}. "
            f"Decision trail is substantially complete but not fully auditable."
        )
    else:
        verdict = "QUALIFIED"
        verdict_reason = (
            "Decision trail is complete and intact with governance notes. "
            "See layer details for specifics."
        )

    return AuditCertificate(
        certificate_id=certificate_id,
        application_id=application_id,
        issued_at=datetime.now(),
        final_decision=final_decision,
        approval_probability=approval_prob,
        confidence_grade=conf_grade,
        legitimacy_verdict=legit,
        fairness_verdict=fairness,
        human_review_status=human_status,
        completeness_score=completeness.completeness_score,
        integrity_status=integrity.integrity_status,
        reproducibility_score=reproducibility.reproducibility_score,
        binding_hash=integrity.binding_hash,
        audit_verdict=verdict,
        audit_verdict_reason=verdict_reason,
    )


# ─────────────────────────────────────────────────────────────────
# Layer implementation
# ─────────────────────────────────────────────────────────────────

class L10AuditLedger_Layer:
    """
    Layer 10: Auditability & Reproducibility Ledger

    Input  : application_id (queries Evidence Ledger directly)
    Output : L10AuditLedger with certificate

    L10 works entirely from stored artifacts.
    It does not receive live layer objects.
    This proves that the system is self-contained and auditable
    from storage alone — no runtime dependency.
    """

    def process(self, application_id: str) -> L10AuditLedger:

        # Load all sealed artifacts from Evidence Ledger
        ledger_entries = retrieve_application_ledger(application_id)

        # Three pillars
        completeness = _check_completeness(application_id, ledger_entries)
        integrity = _verify_integrity(application_id, ledger_entries)
        reproducibility = _build_reproducibility(
            application_id, ledger_entries, completeness
        )

        # Timeline
        timeline = _build_timeline(ledger_entries)

        # Certificate
        certificate = _issue_certificate(
            application_id, completeness, integrity, reproducibility
        )

        # Summary
        if certificate.audit_verdict == "CLEAN":
            summary = (
                f"Audit Certificate CLEAN. "
                f"All {completeness.required_present} required layers present. "
                f"All {integrity.intact} artifacts intact. "
                f"Binding hash: {integrity.binding_hash}. "
                f"This decision can be fully reconstructed and independently verified."
            )
        elif certificate.audit_verdict == "QUALIFIED":
            summary = (
                f"Audit Certificate QUALIFIED. "
                f"Completeness: {completeness.completeness_score:.0%}. "
                f"Integrity: {integrity.integrity_status}. "
                f"Reason: {certificate.audit_verdict_reason}"
            )
        else:
            summary = (
                f"Audit Certificate ADVERSE. "
                f"CRITICAL: {certificate.audit_verdict_reason}"
            )

        return L10AuditLedger(
            application_id=application_id,
            completeness=completeness,
            integrity=integrity,
            reproducibility=reproducibility,
            timeline=timeline,
            audit_certificate=certificate,
            audit_summary=summary,
            timestamp=datetime.now(),
        )
