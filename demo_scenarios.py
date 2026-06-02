"""
LEAF Credit Agent — Demo Scenarios
4 pre-crafted applications covering all LEAF capabilities.
Uses LLM for genuine L5 explanation generation.
L6 confidence grade is always computed (no LLM needed).

Scenario 1 — Strong Approval    : High CIBIL, low DTI
Scenario 2 — Clear Rejection    : Low CIBIL, very high DTI
Scenario 3 — Human Review       : Borderline — Grade C triggers HITL
Scenario 4 — Fairness Flag      : Compliance concerns detected
"""

import random
import numpy as np
from models.schemas import LoanApplication, JurisdictionCode
from models.llm_provider import LLMProvider


DEMO_SCENARIOS = [
    {
        "tag": "strong_approval",
        "seed": 101,
        "label": "Strong Approval",
        "description": "Demonstrates LEAF explaining a clear approval with high confidence",
        "application": {
            "applicant_id": "DEMO-001",
            "amount_requested": 300000,
            "purpose": "Home renovation",
            "tenure_months": 36,
            "applicant_name": "Rajesh Sharma",
            "applicant_age": 38,
            "employment_type": "salaried",
            "monthly_income_declared": 95000,
            "existing_loans": 0,
        },
    },
    {
        "tag": "clear_rejection",
        "seed": 202,
        "label": "Clear Rejection",
        "description": "Demonstrates LEAF generating an adverse action notice with counterfactuals",
        "application": {
            "applicant_id": "DEMO-002",
            "amount_requested": 800000,
            "purpose": "Business expansion",
            "tenure_months": 60,
            "applicant_name": "Priya Patel",
            "applicant_age": 29,
            "employment_type": "self_employed",
            "monthly_income_declared": 28000,
            "existing_loans": 4,
        },
    },
    {
        "tag": "human_review",
        "seed": 303,
        "label": "Human Review Triggered",
        "description": "Demonstrates L6 confidence grade triggering L9 human-in-the-loop",
        "application": {
            "applicant_id": "DEMO-003",
            "amount_requested": 550000,
            "purpose": "Education",
            "tenure_months": 48,
            "applicant_name": "Amit Kumar",
            "applicant_age": 32,
            "employment_type": "salaried",
            "monthly_income_declared": 52000,
            "existing_loans": 2,
        },
    },
    {
        "tag": "fairness_flag",
        "seed": 404,
        "label": "Fairness & Compliance Flag",
        "description": "Demonstrates L8 fairness audit and compliance concerns",
        "application": {
            "applicant_id": "DEMO-004",
            "amount_requested": 250000,
            "purpose": "Medical emergency",
            "tenure_months": 24,
            "applicant_name": "Lakshmi Devi",
            "applicant_age": 44,
            "employment_type": "salaried",
            "monthly_income_declared": 41000,
            "existing_loans": 1,
        },
    },
]


def load_demo_scenarios(llm_provider: LLMProvider, progress_callback=None):
    """
    Run all 4 demo scenarios through the full LEAF Agent pipeline.
    LLM is used for genuine L5 explanation generation.
    L6 is always computed — no LLM needed for confidence grading.
    Each scenario uses a fixed random seed for consistent results.
    Returns list of application IDs created.
    """
    from storage.ledger import clear_all_data
    from agent import LEAFCreditAgent

    # Clear existing data first
    clear_all_data()
    created_ids = []

    for i, scenario in enumerate(DEMO_SCENARIOS):
        if progress_callback:
            progress_callback(i, len(DEMO_SCENARIOS), scenario["label"])

        # Fixed seed for reproducible synthetic data
        random.seed(scenario["seed"])
        np.random.seed(scenario["seed"])

        app = LoanApplication(**scenario["application"])

        agent = LEAFCreditAgent(
            llm_provider=llm_provider,
            jurisdiction=JurisdictionCode.INDIA,
            verbose=False,
            scenario_tag=scenario["tag"],
        )

        output = agent.run(app)
        if output.get("status") in ("complete", "escalated", "blocked"):
            created_ids.append(output.get("application_id"))

        # Reset seed
        random.seed(None)
        np.random.seed(None)

    return created_ids
