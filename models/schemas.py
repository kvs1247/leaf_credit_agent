"""
LEAF Credit Agent — Pydantic Schemas
Every layer produces a typed, serialisable artifact.
These schemas are the contract between layers and the Evidence Ledger.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


# ─────────────────────────────────────────────
# Shared enums
# ─────────────────────────────────────────────

class JurisdictionCode(str, Enum):
    INDIA = "IN"
    USA = "US"
    EU = "EU"


class RegulatoryFramework(str, Enum):
    RBI_FAIR_LENDING = "RBI Fair Lending Guidelines"
    RBI_MODEL_RISK = "RBI Draft Model Risk Framework 2024"
    SR_11_7 = "Federal Reserve SR 11-7"
    EU_AI_ACT = "EU AI Act 2024"
    ECOA = "Equal Credit Opportunity Act"


class DataSourceType(str, Enum):
    CREDIT_BUREAU = "credit_bureau"
    BANK_STATEMENT = "bank_statement"
    UPI_HISTORY = "upi_history"
    TAX_RETURN = "tax_return"
    ALTERNATIVE = "alternative_data"


class GroundingStatus(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    FAILED = "failed"


# ─────────────────────────────────────────────
# L0 — Request & Context
# ─────────────────────────────────────────────

class LoanApplication(BaseModel):
    """Raw input from the applicant."""
    applicant_id: str
    amount_requested: float = Field(..., description="Loan amount in INR")
    purpose: str
    tenure_months: int
    applicant_name: str
    applicant_age: int
    employment_type: str = Field(..., description="salaried / self_employed / business")
    monthly_income_declared: float
    existing_loans: int = Field(default=0, description="Number of active loans")


class L0IntakeRecord(BaseModel):
    """L0 output artifact — locked regulatory context."""
    layer: str = "L0"
    application_id: str
    applicant_id: str
    amount_requested: float
    purpose: str
    tenure_months: int
    jurisdiction: JurisdictionCode
    regulatory_frameworks: List[RegulatoryFramework]
    adverse_action_notice_required: bool
    timestamp: datetime
    leaf_version: str = "1.0.0"

    # Explainability metadata
    xai_note: str = Field(
        default="L0 locks the regulatory context before any data is processed. "
                "All downstream explanations are generated within this regulatory envelope.",
        description="Why this layer matters for explainability"
    )


# ─────────────────────────────────────────────
# L1 — Data Provenance
# ─────────────────────────────────────────────

class DataSourceRecord(BaseModel):
    """A single data source with provenance metadata."""
    source_id: str
    source_type: DataSourceType
    source_name: str
    fetched_at: datetime
    age_hours: float
    integrity_hash: str
    record_count: int
    is_verified: bool
    freshness_warning: Optional[str] = None


class L1ProvenanceCertificate(BaseModel):
    """L1 output artifact — complete data lineage."""
    layer: str = "L1"
    application_id: str
    sources: List[DataSourceRecord]
    total_sources: int
    verified_sources: int
    sources_with_warnings: int
    timestamp: datetime

    xai_note: str = Field(
        default="L1 answers the question: what data was used to make this decision? "
                "Every source is timestamped and hashed — the decision cannot be disputed "
                "on grounds of unknown data inputs.",
        description="Why this layer matters for explainability"
    )


# ─────────────────────────────────────────────
# L2 — Grounding Check
# ─────────────────────────────────────────────

class SourceGroundingScore(BaseModel):
    """Grounding score for a single data source."""
    source_id: str
    source_name: str
    freshness_score: float = Field(..., ge=0, le=1)
    completeness_score: float = Field(..., ge=0, le=1)
    consistency_score: float = Field(..., ge=0, le=1)
    composite_score: float = Field(..., ge=0, le=1)
    status: GroundingStatus
    warning: Optional[str] = None


class L2GroundingReport(BaseModel):
    """L2 output artifact — data reliability assessment."""
    layer: str = "L2"
    application_id: str
    source_scores: List[SourceGroundingScore]
    composite_grounding_score: float = Field(..., ge=0, le=1)
    weakest_source: str
    proceed_to_model: bool
    proceed_threshold: float = 0.75
    timestamp: datetime

    xai_note: str = Field(
        default="L2 answers: how trustworthy is the data this decision is based on? "
                "A low grounding score means the explanation is built on uncertain foundations "
                "and should trigger human review before a decision is issued.",
        description="Why this layer matters for explainability"
    )


# ─────────────────────────────────────────────
# L3 — Signal Extraction
# ─────────────────────────────────────────────

class ExtractedSignal(BaseModel):
    """A single computed feature with full source traceability."""
    signal_name: str
    signal_key: str
    raw_value: float
    display_value: str
    source_ids: List[str] = Field(..., description="Which L1 sources contributed to this signal")
    computation_formula: str = Field(..., description="How this signal was computed")
    interpretation: str = Field(..., description="What this value means for credit assessment")
    risk_direction: str = Field(..., description="positive / negative / neutral")


class L3SignalLog(BaseModel):
    """L3 output artifact — feature engineering with traceability."""
    layer: str = "L3"
    application_id: str
    signals: List[ExtractedSignal]
    total_signals: int
    positive_signals: int
    negative_signals: int
    model_ready_features: Dict[str, float] = Field(
        ..., description="Clean feature dict ready for model input"
    )
    timestamp: datetime

    xai_note: str = Field(
        default="L3 makes every model input visible and traceable. If the model rejects this "
                "application because DTI is 0.41, the applicant can be told exactly how that "
                "number was computed and from which documents.",
        description="Why this layer matters for explainability"
    )


# ─────────────────────────────────────────────
# Evidence Ledger entry (used by L10)
# ─────────────────────────────────────────────

class LedgerEntry(BaseModel):
    """A sealed layer artifact stored in the Evidence Ledger."""
    application_id: str
    layer: str
    artifact_json: str
    artifact_hash: str
    sealed_at: datetime
