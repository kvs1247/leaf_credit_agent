"""
LEAF Layer 5 — Recommendation & Explanation Card
The LLM takes SHAP values from L4 and generates:
1. Plain language decision rationale
2. Counterfactual explanation
3. Full Explanation Card (applicant-facing)

This is the first layer where genuine AI reasoning appears.
The LLM does not make the credit decision — XGBoost does.
The LLM translates the model's numerical output into
a human-understandable explanation that satisfies
the adverse action notice requirement.

Explainability contribution:
    Bridges the gap between SHAP numbers and human understanding.
    The explanation is grounded in L4's SHAP values — not hallucinated.
"""

import json
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional
from layers.l4_model import L4ModelReasoning
from models.schemas import L0IntakeRecord, L3SignalLog
from models.llm_provider import LLMProvider


# ─────────────────────────────────────────────
# L5 Schema
# ─────────────────────────────────────────────

class ExplanationCard(BaseModel):
    """
    Applicant-facing explanation — plain language, legally defensible.
    Satisfies RBI adverse action notice requirement.
    """
    decision: str
    decision_summary: str
    top_positive_factors: List[str]
    top_negative_factors: List[str]
    counterfactual: str
    confidence_note: str
    applicant_rights: str
    generated_by: str = "LEAF Agent — L5 Recommendation"


class L5Recommendation(BaseModel):
    layer: str = "L5"
    application_id: str

    # Decision
    decision: str
    approval_probability: float
    interest_rate_band: str
    recommended_tenure_months: int

    # LLM-generated content
    plain_language_rationale: str
    counterfactual_explanation: str
    explanation_card: ExplanationCard

    # Agent reasoning trace — logged for Evidence Ledger
    agent_reasoning_trace: str = Field(
        ..., description="The LLM's step-by-step reasoning — the agentic contribution"
    )

    timestamp: datetime

    xai_note: str = Field(
        default="L5 translates SHAP numbers into human language. The LLM is grounded "
                "by L4's attribution values — it cannot fabricate reasons. "
                "The reasoning trace shows exactly how the explanation was constructed.",
    )


# ─────────────────────────────────────────────
# Interest rate bands
# ─────────────────────────────────────────────

def _get_rate_band(approval_prob: float) -> str:
    if approval_prob >= 0.80:
        return "Band A — 9.5% to 10.5% p.a."
    elif approval_prob >= 0.65:
        return "Band B — 10.5% to 12.0% p.a."
    elif approval_prob >= 0.50:
        return "Band C — 12.0% to 14.0% p.a."
    else:
        return "Not applicable — application not approved"


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are LEAF — a Layered Explainability AI Framework for financial decisions.
Your role is to translate a credit model's numerical output into a clear, honest,
and legally defensible explanation for the applicant.

Rules you must follow:
1. Base your explanation ONLY on the SHAP values and signals provided. Do not invent reasons.
2. Use plain language. Avoid jargon. The applicant may not be financially literate.
3. Be honest about both positive and negative factors.
4. The counterfactual must be specific and actionable — not generic advice.
5. Never promise or guarantee a future outcome.
6. Always mention the applicant's right to request a detailed review.

You must respond with a valid JSON object — no markdown, no preamble, just JSON."""


def _build_user_prompt(
    l0: L0IntakeRecord,
    l3: L3SignalLog,
    l4: L4ModelReasoning,
) -> str:
    """Build the structured prompt grounded in layer outputs."""

    shap_summary = []
    for c in l4.shap_contributions[:6]:
        direction = "HELPS approval" if c.direction == "positive" else "HURTS approval"
        shap_summary.append(
            f"  - {c.plain_label}: SHAP={c.shap_value:+.3f} ({direction}, {c.magnitude} impact)"
        )

    signals_summary = []
    for s in l3.signals:
        signals_summary.append(f"  - {s.signal_name}: {s.display_value} ({s.risk_direction})")

    return f"""
Generate a LEAF Explanation Card for this credit decision.

APPLICATION CONTEXT:
- Loan amount: ₹{l0.amount_requested:,.0f}
- Purpose: {l0.purpose}
- Jurisdiction: India (RBI guidelines apply)
- Adverse action notice required: {l0.adverse_action_notice_required}

MODEL DECISION:
- Decision: {l4.decision}
- Approval probability: {l4.approval_probability:.1%}
- Confidence: {l4.decision_confidence}

SHAP ATTRIBUTION (top 6 factors):
{chr(10).join(shap_summary)}

EXTRACTED SIGNALS:
{chr(10).join(signals_summary)}

TOP APPROVAL FACTORS: {l4.top_approval_factors}
TOP REJECTION FACTORS: {l4.top_rejection_factors}

COUNTERFACTUAL HINT FROM MODEL: {l4.counterfactual_hint}

Respond with this exact JSON structure:
{{
  "reasoning_trace": "Your step-by-step reasoning about how you constructed this explanation",
  "plain_language_rationale": "2-3 sentence explanation of the decision in plain language",
  "counterfactual_explanation": "Specific, actionable advice on what would change the outcome",
  "explanation_card": {{
    "decision": "{l4.decision}",
    "decision_summary": "One clear sentence summarising the decision and main reason",
    "top_positive_factors": ["factor 1", "factor 2", "factor 3"],
    "top_negative_factors": ["factor 1", "factor 2"],
    "counterfactual": "Specific thing the applicant can do to improve their position",
    "confidence_note": "One sentence about confidence level in plain terms",
    "applicant_rights": "One sentence about their right to request a review"
  }}
}}
"""


# ─────────────────────────────────────────────
# Layer implementation
# ─────────────────────────────────────────────

class L5Recommendation_Layer:
    """
    Layer 5: Recommendation & Explanation Card

    Input  : L0IntakeRecord, L3SignalLog, L4ModelReasoning, LLMProvider
    Output : L5Recommendation with LLM-generated Explanation Card
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def process(
        self,
        l0: L0IntakeRecord,
        l3: L3SignalLog,
        l4: L4ModelReasoning,
        application_id: str,
    ) -> L5Recommendation:

        # Call LLM — grounded by SHAP values from L4
        user_prompt = _build_user_prompt(l0, l3, l4)
        raw_response = self.llm.complete(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1200,
            temperature=0.2,
        )

        # Parse JSON response
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            # Fallback if LLM wraps in markdown
            import re
            match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                raise ValueError(f"LLM returned invalid JSON: {raw_response[:200]}")

        card_data = parsed.get("explanation_card", {})
        explanation_card = ExplanationCard(
            decision=card_data.get("decision", l4.decision),
            decision_summary=card_data.get("decision_summary", ""),
            top_positive_factors=card_data.get("top_positive_factors", []),
            top_negative_factors=card_data.get("top_negative_factors", []),
            counterfactual=card_data.get("counterfactual", l4.counterfactual_hint),
            confidence_note=card_data.get("confidence_note", ""),
            applicant_rights=card_data.get("applicant_rights", ""),
        )

        return L5Recommendation(
            application_id=application_id,
            decision=l4.decision,
            approval_probability=l4.approval_probability,
            interest_rate_band=_get_rate_band(l4.approval_probability),
            recommended_tenure_months=l0.tenure_months,
            plain_language_rationale=parsed.get("plain_language_rationale", ""),
            counterfactual_explanation=parsed.get("counterfactual_explanation",
                                                   l4.counterfactual_hint),
            explanation_card=explanation_card,
            agent_reasoning_trace=parsed.get("reasoning_trace", ""),
            timestamp=datetime.now(),
        )
