"""
LEAF Layer 0 — Request & Context
Parses the loan application, identifies the regulatory context,
and locks the explainability envelope for all downstream layers.

Explainability contribution:
    Every downstream explanation knows what rules it must satisfy
    before a single byte of applicant data is processed.
"""

import uuid
from datetime import datetime
from models.schemas import (
    LoanApplication, L0IntakeRecord,
    JurisdictionCode, RegulatoryFramework
)


# Jurisdiction → applicable regulatory frameworks
REGULATORY_MAP = {
    JurisdictionCode.INDIA: [
        RegulatoryFramework.RBI_FAIR_LENDING,
        RegulatoryFramework.RBI_MODEL_RISK,
    ],
    JurisdictionCode.USA: [
        RegulatoryFramework.SR_11_7,
        RegulatoryFramework.ECOA,
    ],
    JurisdictionCode.EU: [
        RegulatoryFramework.EU_AI_ACT,
    ],
}

# Jurisdictions that legally require an adverse action notice
ADVERSE_ACTION_JURISDICTIONS = {
    JurisdictionCode.INDIA,
    JurisdictionCode.USA,
    JurisdictionCode.EU,
}


class L0RequestContext:
    """
    Layer 0: Request & Context

    Input  : LoanApplication (raw applicant form)
    Output : L0IntakeRecord  (sealed regulatory context)
    """

    def __init__(self, jurisdiction: JurisdictionCode = JurisdictionCode.INDIA):
        self.jurisdiction = jurisdiction

    def process(self, application: LoanApplication) -> L0IntakeRecord:
        """
        Parse the application and lock the regulatory context.
        This artifact is the first entry in the Evidence Ledger.
        """
        application_id = f"LEAF-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"

        frameworks = REGULATORY_MAP.get(self.jurisdiction, [])
        adverse_action_required = self.jurisdiction in ADVERSE_ACTION_JURISDICTIONS

        record = L0IntakeRecord(
            application_id=application_id,
            applicant_id=application.applicant_id,
            amount_requested=application.amount_requested,
            purpose=application.purpose,
            tenure_months=application.tenure_months,
            jurisdiction=self.jurisdiction,
            regulatory_frameworks=frameworks,
            adverse_action_notice_required=adverse_action_required,
            timestamp=datetime.now(),
        )

        return record
