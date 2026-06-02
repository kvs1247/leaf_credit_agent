"""
LEAF Credit Agent — Storage Loader
Deserialises layer artifacts from the Evidence Ledger
back into typed Pydantic objects for display.

Used when loading a past application from the home screen.
"""

from typing import Optional
from storage.ledger import get_layer_artifact, get_all_summaries, retrieve_application_ledger
from models.schemas import (
    L0IntakeRecord, L1ProvenanceCertificate,
    L2GroundingReport, L3SignalLog
)
from layers.l4_model import L4ModelReasoning
from layers.l5_recommendation import L5Recommendation
from layers.l6_confidence import L6ConfidenceGrade


def load_application_results(application_id: str) -> Optional[dict]:
    """
    Load all layer artifacts for a past application and
    reconstruct the results dict used by the dashboard.
    Returns None if application not found.
    """
    results = {}

    # L0
    l0_data = get_layer_artifact(application_id, "L0")
    if not l0_data:
        return None
    results["L0"] = L0IntakeRecord(**l0_data)

    # L1
    l1_data = get_layer_artifact(application_id, "L1")
    if l1_data:
        results["L1"] = L1ProvenanceCertificate(**l1_data)

    # L2
    l2_data = get_layer_artifact(application_id, "L2")
    if l2_data:
        results["L2"] = L2GroundingReport(**l2_data)

    # L3
    l3_data = get_layer_artifact(application_id, "L3")
    if l3_data:
        results["L3"] = L3SignalLog(**l3_data)

    # L4
    l4_data = get_layer_artifact(application_id, "L4")
    if l4_data:
        results["L4"] = L4ModelReasoning(**l4_data)

    # L5
    l5_data = get_layer_artifact(application_id, "L5")
    if l5_data:
        results["L5"] = L5Recommendation(**l5_data)

    # L6
    l6_data = get_layer_artifact(application_id, "L6")
    if l6_data:
        results["L6"] = L6ConfidenceGrade(**l6_data)

    # Agent trace
    trace_data = get_layer_artifact(application_id, "AGENT_TRACE")
    if trace_data:
        results["AGENT_TRACE"] = trace_data

    return results
