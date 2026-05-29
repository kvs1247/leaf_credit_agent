"""
LEAF Layer 4 — Model Reasoning
Runs the XGBoost credit model and computes individual SHAP attribution.

Important: SHAP values here are computed against DEFAULT probability.
Negative SHAP = reduces default risk = GOOD for applicant (▲)
Positive SHAP  = increases default risk = BAD for applicant (▼)
L4 handles this inversion so all downstream layers see
approval-oriented attribution.

Explainability contribution:
    Shows exactly which features pushed toward approval and which
    toward rejection — for THIS specific applicant, not global averages.
    This is what makes the adverse action notice possible.
"""

from datetime import datetime
from typing import Dict, List
from pydantic import BaseModel, Field
from models.credit_model import predict_with_shap, FEATURE_COLS


# ─────────────────────────────────────────────
# L4 Schema
# ─────────────────────────────────────────────

class SHAPContribution(BaseModel):
    feature_name: str
    feature_key: str
    feature_value: float
    shap_value: float = Field(..., description="Signed contribution to APPROVAL probability")
    direction: str = Field(..., description="positive = helps approval, negative = hurts approval")
    magnitude: str = Field(..., description="high / medium / low")
    plain_label: str = Field(..., description="Human-readable feature name")


class L4ModelReasoning(BaseModel):
    layer: str = "L4"
    application_id: str

    # Model outputs
    default_probability: float
    approval_probability: float
    decision: str
    decision_confidence: str
    base_value: float

    # SHAP attribution
    shap_contributions: List[SHAPContribution]
    top_approval_factors: List[str]
    top_rejection_factors: List[str]

    # Counterfactual hint
    counterfactual_hint: str

    # Model metadata
    model_version: str = "xgb-leaf-v1"
    model_auc: float = 0.633
    timestamp: datetime

    xai_note: str = Field(
        default="L4 shows exactly which features pushed toward approval (▲) and rejection (▼) "
                "for this specific applicant. This individual-level attribution — not global averages — "
                "is what makes the adverse action notice legally defensible.",
    )


# ─────────────────────────────────────────────
# Human-readable labels for features
# ─────────────────────────────────────────────

FEATURE_LABELS = {
    "cibil_score": "CIBIL Credit Score",
    "dti_ratio": "Debt-to-Income Ratio",
    "repayment_score": "Repayment History",
    "verified_monthly_income": "Verified Monthly Income",
    "upi_velocity": "UPI Transaction Activity",
    "income_stability": "Income Stability",
    "existing_loan_burden": "Existing Loan Obligations",
    "loan_amount": "Loan Amount Requested",
    "tenure_months": "Loan Tenure",
    "employment_type_encoded": "Employment Type",
}


def _magnitude(shap_abs: float) -> str:
    if shap_abs >= 0.20:
        return "high"
    elif shap_abs >= 0.08:
        return "medium"
    else:
        return "low"


def _build_counterfactual(contributions: List[SHAPContribution],
                           approval_prob: float) -> str:
    """Generate a simple counterfactual hint based on top negative factors."""
    negatives = [c for c in contributions if c.direction == "negative"
                 and c.magnitude in ("high", "medium")]
    negatives.sort(key=lambda x: x.shap_value)  # most negative first

    if not negatives:
        return "Current profile is strong. No major improvements needed."

    top = negatives[0]
    hints = {
        "dti_ratio": "Reducing total debt obligations would lower your DTI ratio "
                     "and significantly improve approval prospects.",
        "existing_loan_burden": "Closing one or more existing loans before applying "
                                "would reduce your obligation burden.",
        "cibil_score": "Improving your CIBIL score by 30-50 points through timely "
                       "payments over 6 months would strengthen this application.",
        "loan_amount": "Reducing the loan amount requested would lower risk exposure "
                       "and improve approval probability.",
        "repayment_score": "A consistent repayment record over the next 12 months "
                           "would substantially improve this profile.",
    }
    return hints.get(top.feature_key,
                     f"Improving {top.plain_label} would positively impact this decision.")


# ─────────────────────────────────────────────
# Layer implementation
# ─────────────────────────────────────────────

class L4ModelReasoning_Layer:
    """
    Layer 4: Model Reasoning

    Input  : model_ready_features dict from L3
    Output : L4ModelReasoning with SHAP attribution
    """

    def process(
        self,
        model_features: Dict[str, float],
        application_id: str
    ) -> L4ModelReasoning:

        # Run model + SHAP
        result = predict_with_shap(model_features)

        shap_values = result["shap_values"]
        approval_prob = result["approval_probability"]

        # Build SHAP contributions
        # NOTE: SHAP values are w.r.t. default probability
        # We INVERT sign so positive = good for applicant (approval-oriented)
        contributions = []
        for feat_key in FEATURE_COLS:
            raw_shap = shap_values.get(feat_key, 0.0)
            # Invert: negative SHAP on default = positive for approval
            approval_shap = round(-raw_shap, 4)
            direction = "positive" if approval_shap > 0 else "negative"
            feat_val = model_features.get(feat_key, 0.0)

            contributions.append(SHAPContribution(
                feature_name=FEATURE_LABELS.get(feat_key, feat_key),
                feature_key=feat_key,
                feature_value=round(feat_val, 4),
                shap_value=approval_shap,
                direction=direction,
                magnitude=_magnitude(abs(approval_shap)),
                plain_label=FEATURE_LABELS.get(feat_key, feat_key),
            ))

        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)

        top_approval = [
            c.plain_label for c in contributions
            if c.direction == "positive" and c.magnitude in ("high", "medium")
        ][:3]

        top_rejection = [
            c.plain_label for c in contributions
            if c.direction == "negative" and c.magnitude in ("high", "medium")
        ][:3]

        counterfactual = _build_counterfactual(contributions, approval_prob)

        return L4ModelReasoning(
            application_id=application_id,
            default_probability=result["default_probability"],
            approval_probability=approval_prob,
            decision=result["decision"],
            decision_confidence=result["confidence"],
            base_value=result["base_value"],
            shap_contributions=contributions,
            top_approval_factors=top_approval,
            top_rejection_factors=top_rejection,
            counterfactual_hint=counterfactual,
            timestamp=datetime.now(),
        )
