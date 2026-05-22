"""
LEAF Layer 1 — Data Provenance
Fetches data from all sources and logs their full provenance:
source identity, timestamp, record count, and integrity hash.

Explainability contribution:
    Every data input used in this decision is visible and verifiable.
    If disputed, the applicant and regulator can see exactly what
    data the system had access to and when it was retrieved.
"""

import hashlib
import random
from datetime import datetime, timedelta
from typing import Dict, Any
from models.schemas import (
    L1ProvenanceCertificate, DataSourceRecord,
    DataSourceType, LoanApplication
)


# ─────────────────────────────────────────────
# Synthetic data simulators
# In production these call real bureau APIs,
# bank statement parsers, and UPI data feeds.
# ─────────────────────────────────────────────

def _simulate_cibil_data(application: LoanApplication) -> Dict[str, Any]:
    """Simulate CIBIL bureau response."""
    return {
        "cibil_score": random.randint(680, 800),
        "credit_history_months": random.randint(24, 120),
        "total_accounts": random.randint(2, 8),
        "delinquent_accounts": random.randint(0, 1),
        "enquiries_6months": random.randint(0, 3),
        "repayment_record_24m": round(random.uniform(0.82, 0.99), 2),
    }


def _simulate_bank_statements(application: LoanApplication) -> Dict[str, Any]:
    """Simulate 6-month bank statement summary."""
    monthly_avg = application.monthly_income_declared * random.uniform(0.88, 1.12)
    return {
        "months_analyzed": 6,
        "avg_monthly_credit": round(monthly_avg, 2),
        "avg_monthly_debit": round(monthly_avg * random.uniform(0.55, 0.78), 2),
        "salary_credits_detected": random.randint(5, 6),
        "emi_debits_detected": application.existing_loans,
        "avg_emi_amount": round(random.uniform(8000, 22000), 2) if application.existing_loans > 0 else 0,
        "min_monthly_balance": round(random.uniform(5000, 25000), 2),
        "bounce_count": random.randint(0, 1),
    }


def _simulate_upi_history(application: LoanApplication) -> Dict[str, Any]:
    """Simulate UPI transaction history from NPCI."""
    return {
        "months_analyzed": 6,
        "avg_monthly_transactions": random.randint(180, 450),
        "avg_monthly_upi_volume": round(random.uniform(15000, 65000), 2),
        "merchant_diversity_score": round(random.uniform(0.6, 0.95), 2),
        "p2p_ratio": round(random.uniform(0.3, 0.6), 2),
        "failed_transaction_rate": round(random.uniform(0.01, 0.05), 2),
    }


def _simulate_itr_data(application: LoanApplication) -> Dict[str, Any]:
    """Simulate ITR (Income Tax Return) data from CBDT."""
    return {
        "assessment_year": "2023-24",
        "gross_total_income": round(application.monthly_income_declared * 12 * random.uniform(0.9, 1.1), 2),
        "tax_paid": round(application.monthly_income_declared * 12 * 0.15 * random.uniform(0.85, 1.1), 2),
        "itr_form": "ITR-1",
        "filing_date": (datetime.now() - timedelta(days=random.randint(30, 200))).strftime("%Y-%m-%d"),
        "verified": True,
    }


def _compute_hash(data: Dict[str, Any]) -> str:
    """Compute a SHA-256 integrity hash for a data payload."""
    payload = str(sorted(data.items())).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


# ─────────────────────────────────────────────
# Layer implementation
# ─────────────────────────────────────────────

class L1DataProvenance:
    """
    Layer 1: Data Provenance

    Input  : LoanApplication + application_id from L0
    Output : L1ProvenanceCertificate + raw data dict for L2/L3
    """

    # Maximum acceptable age in hours per source type
    FRESHNESS_THRESHOLDS = {
        DataSourceType.CREDIT_BUREAU: 72,
        DataSourceType.BANK_STATEMENT: 24,
        DataSourceType.UPI_HISTORY: 24,
        DataSourceType.TAX_RETURN: 48,
    }

    def process(self, application: LoanApplication, application_id: str):
        """
        Fetch all data sources, log provenance, and return the certificate.
        Returns (L1ProvenanceCertificate, raw_data_dict)
        """
        now = datetime.now()
        raw_data = {}
        source_records = []

        # Define sources to fetch
        sources_config = [
            {
                "source_type": DataSourceType.CREDIT_BUREAU,
                "source_name": "CIBIL TransUnion",
                "fetch_delay_hours": random.uniform(0, 2),
                "simulator": _simulate_cibil_data,
            },
            {
                "source_type": DataSourceType.BANK_STATEMENT,
                "source_name": "Account Aggregator (AA Framework)",
                "fetch_delay_hours": random.uniform(0, 0.5),
                "simulator": _simulate_bank_statements,
            },
            {
                "source_type": DataSourceType.UPI_HISTORY,
                "source_name": "NPCI UPI Transaction Feed",
                "fetch_delay_hours": random.uniform(0, 0.5),
                "simulator": _simulate_upi_history,
            },
            {
                "source_type": DataSourceType.TAX_RETURN,
                "source_name": "CBDT ITR Portal",
                "fetch_delay_hours": random.uniform(40, 52),   # intentionally older
                "simulator": _simulate_itr_data,
            },
        ]

        for idx, cfg in enumerate(sources_config):
            data = cfg["simulator"](application)
            age_hours = round(cfg["fetch_delay_hours"], 1)
            fetched_at = now - timedelta(hours=age_hours)
            integrity_hash = _compute_hash(data)
            threshold = self.FRESHNESS_THRESHOLDS[cfg["source_type"]]
            is_stale = age_hours > threshold
            warning = f"Data is {age_hours:.0f}h old (threshold: {threshold}h)" if is_stale else None

            source_id = f"SRC-{idx+1:02d}"
            raw_data[source_id] = data
            raw_data[f"{source_id}_type"] = cfg["source_type"].value

            source_records.append(DataSourceRecord(
                source_id=source_id,
                source_type=cfg["source_type"],
                source_name=cfg["source_name"],
                fetched_at=fetched_at,
                age_hours=age_hours,
                integrity_hash=integrity_hash,
                record_count=len(data),
                is_verified=True,
                freshness_warning=warning,
            ))

        sources_with_warnings = sum(1 for s in source_records if s.freshness_warning)

        certificate = L1ProvenanceCertificate(
            application_id=application_id,
            sources=source_records,
            total_sources=len(source_records),
            verified_sources=sum(1 for s in source_records if s.is_verified),
            sources_with_warnings=sources_with_warnings,
            timestamp=now,
        )

        return certificate, raw_data
