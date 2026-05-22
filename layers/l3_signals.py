"""
LEAF Layer 3 — Signal Extraction
Computes every model feature from raw data with full source traceability.
Each signal records which data sources contributed, how it was computed,
and what it means for the credit decision.

Explainability contribution:
    Makes every model input visible and traceable.
    If the model rejects because DTI is 0.41, the applicant can be told
    exactly how that number was computed and from which documents.
"""

from datetime import datetime
from typing import Dict, Any, List
from models.schemas import L3SignalLog, ExtractedSignal


class L3SignalExtraction:
    """
    Layer 3: Signal Extraction

    Input  : raw_data dict from L1, application object
    Output : L3SignalLog with full traceability
    """

    def process(
        self,
        raw_data: Dict[str, Any],
        application,
        application_id: str
    ) -> L3SignalLog:

        bureau = raw_data.get("SRC-01", {})
        bank = raw_data.get("SRC-02", {})
        upi = raw_data.get("SRC-03", {})
        itr = raw_data.get("SRC-04", {})

        signals: List[ExtractedSignal] = []

        # ── Signal 1: CIBIL Score Band ──────────────────────────────
        cibil_score = bureau.get("cibil_score", 650)
        if cibil_score >= 750:
            cibil_band, direction = "Excellent", "positive"
        elif cibil_score >= 700:
            cibil_band, direction = "Good", "positive"
        elif cibil_score >= 650:
            cibil_band, direction = "Fair", "neutral"
        else:
            cibil_band, direction = "Poor", "negative"

        signals.append(ExtractedSignal(
            signal_name="CIBIL Score Band",
            signal_key="cibil_score",
            raw_value=float(cibil_score),
            display_value=f"{cibil_score} — {cibil_band}",
            source_ids=["SRC-01"],
            computation_formula="Direct from CIBIL TransUnion bureau report",
            interpretation=f"Credit score of {cibil_score} indicates {cibil_band.lower()} creditworthiness. "
                           f"Scores above 700 are considered good by most lenders.",
            risk_direction=direction,
        ))

        # ── Signal 2: Debt-to-Income Ratio ─────────────────────────
        monthly_income = bank.get("avg_monthly_credit", application.monthly_income_declared)
        emi_amount = bank.get("avg_emi_amount", 0)
        existing_emis = application.existing_loans * emi_amount

        new_emi = (application.amount_requested * (11.5 / 1200)) / (
            1 - (1 + 11.5 / 1200) ** (-application.tenure_months)
        )
        total_obligations = existing_emis + new_emi
        dti = round(total_obligations / monthly_income, 3) if monthly_income > 0 else 1.0

        dti_direction = "positive" if dti < 0.35 else ("negative" if dti > 0.50 else "neutral")
        signals.append(ExtractedSignal(
            signal_name="Debt-to-Income Ratio",
            signal_key="dti_ratio",
            raw_value=dti,
            display_value=f"{dti:.2f} ({dti*100:.0f}%)",
            source_ids=["SRC-02", "SRC-04"],
            computation_formula="(Existing EMIs + New EMI) ÷ Average Monthly Income. "
                                 f"= (₹{existing_emis:,.0f} + ₹{new_emi:,.0f}) ÷ ₹{monthly_income:,.0f}",
            interpretation=f"DTI of {dti:.0%} means {dti:.0%} of monthly income goes to loan repayments. "
                           f"RBI guidelines recommend keeping DTI below 50%.",
            risk_direction=dti_direction,
        ))

        # ── Signal 3: Repayment History Score ─────────────────────
        repayment_score = bureau.get("repayment_record_24m", 0.75)
        rep_direction = "positive" if repayment_score > 0.85 else ("negative" if repayment_score < 0.70 else "neutral")
        signals.append(ExtractedSignal(
            signal_name="Repayment History Score",
            signal_key="repayment_score",
            raw_value=round(repayment_score, 3),
            display_value=f"{repayment_score:.2f} ({'Strong' if repayment_score > 0.85 else 'Moderate'})",
            source_ids=["SRC-01"],
            computation_formula="Proportion of on-time payments over last 24 months from CIBIL report",
            interpretation=f"Applicant made on-time payments {repayment_score*100:.0f}% of the time "
                           f"over the last 24 months. This is a strong predictor of future repayment behavior.",
            risk_direction=rep_direction,
        ))

        # ── Signal 4: Monthly Income (verified) ───────────────────
        avg_income = bank.get("avg_monthly_credit", application.monthly_income_declared)
        declared = application.monthly_income_declared
        income_variance = abs(avg_income - declared) / declared if declared > 0 else 0
        inc_direction = "positive" if income_variance < 0.10 else ("negative" if income_variance > 0.25 else "neutral")
        signals.append(ExtractedSignal(
            signal_name="Verified Monthly Income",
            signal_key="verified_monthly_income",
            raw_value=round(avg_income, 2),
            display_value=f"₹{avg_income:,.0f}/month",
            source_ids=["SRC-02", "SRC-04"],
            computation_formula="Average of monthly credit entries over 6 months from bank statements. "
                                 f"Declared: ₹{declared:,.0f}. Variance: {income_variance:.0%}",
            interpretation=f"Bank statements confirm average monthly income of ₹{avg_income:,.0f}. "
                           f"Variance from declared income is {income_variance:.0%}.",
            risk_direction=inc_direction,
        ))

        # ── Signal 5: UPI Transaction Velocity ────────────────────
        upi_txn = upi.get("avg_monthly_transactions", 0)
        upi_direction = "positive" if upi_txn > 200 else ("negative" if upi_txn < 50 else "neutral")
        signals.append(ExtractedSignal(
            signal_name="UPI Transaction Velocity",
            signal_key="upi_velocity",
            raw_value=float(upi_txn),
            display_value=f"{upi_txn:.0f} transactions/month",
            source_ids=["SRC-03"],
            computation_formula="Average monthly UPI transaction count over last 6 months from NPCI feed",
            interpretation=f"Applicant conducts {upi_txn:.0f} UPI transactions per month, indicating "
                           f"{'high' if upi_txn > 200 else 'moderate'} digital financial activity. "
                           f"Higher velocity suggests financial engagement and stability.",
            risk_direction=upi_direction,
        ))

        # ── Signal 6: Income Stability Score ──────────────────────
        bounce_rate = bank.get("bounce_count", 0) / 6.0
        stability = round(1.0 - (bounce_rate * 0.3) - (income_variance * 0.4), 3)
        stability = max(0.0, min(1.0, stability))
        stab_direction = "positive" if stability > 0.80 else ("negative" if stability < 0.60 else "neutral")
        signals.append(ExtractedSignal(
            signal_name="Income Stability Score",
            signal_key="income_stability",
            raw_value=stability,
            display_value=f"{stability:.2f} ({'Stable' if stability > 0.80 else 'Variable'})",
            source_ids=["SRC-02"],
            computation_formula=f"1 - (bounce_rate × 0.3) - (income_variance × 0.4). "
                                 f"Bounce rate: {bounce_rate:.2f}, Income variance: {income_variance:.2f}",
            interpretation=f"Income stability score of {stability:.2f} reflects consistency of "
                           f"income over 6 months. Values above 0.80 indicate reliable income patterns.",
            risk_direction=stab_direction,
        ))

        # ── Signal 7: Existing Loan Burden ────────────────────────
        burden_direction = "negative" if application.existing_loans >= 3 else ("neutral" if application.existing_loans >= 1 else "positive")
        signals.append(ExtractedSignal(
            signal_name="Existing Loan Obligations",
            signal_key="existing_loan_burden",
            raw_value=float(application.existing_loans),
            display_value=f"{application.existing_loans} active loan(s) — ₹{existing_emis:,.0f}/month",
            source_ids=["SRC-01", "SRC-02"],
            computation_formula="Count of active loans from CIBIL + total EMI debit from bank statements",
            interpretation=f"Applicant has {application.existing_loans} existing loan(s) with total EMI "
                           f"obligations of ₹{existing_emis:,.0f}/month. This directly increases DTI.",
            risk_direction=burden_direction,
        ))

        # ── Build model-ready feature dict ───────────────────────
        model_features = {
            "cibil_score": float(cibil_score),
            "dti_ratio": dti,
            "repayment_score": round(repayment_score, 3),
            "verified_monthly_income": round(avg_income, 2),
            "upi_velocity": float(upi_txn),
            "income_stability": stability,
            "existing_loan_burden": float(application.existing_loans),
            "loan_amount": application.amount_requested,
            "tenure_months": float(application.tenure_months),
            "employment_type_encoded": {
                "salaried": 1.0, "self_employed": 0.7, "business": 0.8
            }.get(application.employment_type, 0.5),
        }

        positive = sum(1 for s in signals if s.risk_direction == "positive")
        negative = sum(1 for s in signals if s.risk_direction == "negative")

        return L3SignalLog(
            application_id=application_id,
            signals=signals,
            total_signals=len(signals),
            positive_signals=positive,
            negative_signals=negative,
            model_ready_features=model_features,
            timestamp=datetime.now(),
        )
