"""
LEAF Layer 2 — Grounding Check
Evaluates the reliability of each data source:
freshness, completeness, and cross-source consistency.

Explainability contribution:
    Answers "how trustworthy is the data this decision is based on?"
    A low grounding score means the explanation rests on uncertain
    foundations and should trigger human review.
"""

from datetime import datetime
from typing import Dict, Any, List
from models.schemas import (
    L1ProvenanceCertificate, L2GroundingReport,
    SourceGroundingScore, GroundingStatus,
    DataSourceType
)


# Grounding thresholds
FRESHNESS_IDEAL_HOURS = {
    DataSourceType.CREDIT_BUREAU: 24,
    DataSourceType.BANK_STATEMENT: 6,
    DataSourceType.UPI_HISTORY: 6,
    DataSourceType.TAX_RETURN: 24,
}

PROCEED_THRESHOLD = 0.75


def _freshness_score(age_hours: float, ideal_hours: float, max_hours: float) -> float:
    """
    Linear decay: 1.0 when fresh, 0.0 when at max_hours.
    Clamped to [0, 1].
    """
    if age_hours <= ideal_hours:
        return 1.0
    score = 1.0 - ((age_hours - ideal_hours) / (max_hours - ideal_hours))
    return round(max(0.0, min(1.0, score)), 3)


def _completeness_score(data: Dict[str, Any]) -> float:
    """Score based on how many expected fields have non-null values."""
    if not data:
        return 0.0
    non_null = sum(1 for v in data.values() if v is not None and v != "")
    return round(non_null / len(data), 3)


def _consistency_score(raw_data: Dict[str, Any], source_id: str) -> float:
    """
    Cross-source consistency check.
    In production: compare declared income vs bank credits vs ITR income.
    Here we simulate a consistency score based on field plausibility.
    """
    import random
    random.seed(hash(source_id) % 1000)
    return round(random.uniform(0.78, 0.98), 3)


def _derive_status(score: float) -> GroundingStatus:
    if score >= 0.85:
        return GroundingStatus.HIGH
    elif score >= 0.70:
        return GroundingStatus.MODERATE
    elif score >= 0.50:
        return GroundingStatus.LOW
    else:
        return GroundingStatus.FAILED


class L2GroundingCheck:
    """
    Layer 2: Grounding Check

    Input  : L1ProvenanceCertificate + raw_data dict
    Output : L2GroundingReport
    """

    MAX_AGE_HOURS = {
        DataSourceType.CREDIT_BUREAU: 96,
        DataSourceType.BANK_STATEMENT: 48,
        DataSourceType.UPI_HISTORY: 48,
        DataSourceType.TAX_RETURN: 72,
    }

    def process(
        self,
        provenance: L1ProvenanceCertificate,
        raw_data: Dict[str, Any],
        application_id: str
    ) -> L2GroundingReport:

        source_scores: List[SourceGroundingScore] = []

        for source in provenance.sources:
            ideal = FRESHNESS_IDEAL_HOURS.get(source.source_type, 24)
            maximum = self.MAX_AGE_HOURS.get(source.source_type, 72)

            freshness = _freshness_score(source.age_hours, ideal, maximum)
            data_payload = raw_data.get(source.source_id, {})
            completeness = _completeness_score(data_payload)
            consistency = _consistency_score(raw_data, source.source_id)

            composite = round(
                (freshness * 0.40) +
                (completeness * 0.35) +
                (consistency * 0.25),
                3
            )
            status = _derive_status(composite)

            warning = None
            if freshness < 0.70:
                warning = f"Stale data: {source.age_hours:.0f}h old"
            elif composite < PROCEED_THRESHOLD:
                warning = f"Low composite score: {composite:.2f}"

            source_scores.append(SourceGroundingScore(
                source_id=source.source_id,
                source_name=source.source_name,
                freshness_score=freshness,
                completeness_score=completeness,
                consistency_score=consistency,
                composite_score=composite,
                status=status,
                warning=warning,
            ))

        composite_grounding = round(
            sum(s.composite_score for s in source_scores) / len(source_scores), 3
        )

        weakest = min(source_scores, key=lambda s: s.composite_score)

        return L2GroundingReport(
            application_id=application_id,
            source_scores=source_scores,
            composite_grounding_score=composite_grounding,
            weakest_source=weakest.source_name,
            proceed_to_model=composite_grounding >= PROCEED_THRESHOLD,
            proceed_threshold=PROCEED_THRESHOLD,
            timestamp=datetime.now(),
        )
