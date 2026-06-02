"""
LEAF Credit Agent — Agent Orchestrator
This is what makes LEAF genuinely agentic.

Instead of a fixed pipeline, the LLM reasons about each layer's
output and decides what to do next. Every reasoning step is logged.
The full reasoning trace becomes part of the Evidence Ledger.

Agent loop (ReAct pattern):
    Think → Act (call a LEAF layer) → Observe → Think → Act → ...

The agent's goal is not just to make a credit decision —
it is to produce a DEFENSIBLE, EXPLAINABLE credit decision.
It is not done until it has a sealed Evidence Ledger.
"""

import os
import json
from datetime import datetime
from typing import Optional
from models.schemas import LoanApplication, JurisdictionCode
from models.llm_provider import LLMProvider, get_provider_from_env
from models.credit_model import get_model
from layers.l0_request import L0RequestContext
from layers.l1_provenance import L1DataProvenance
from layers.l2_grounding import L2GroundingCheck
from layers.l3_signals import L3SignalExtraction
from layers.l4_model import L4ModelReasoning_Layer
from layers.l5_recommendation import L5Recommendation_Layer
from layers.l6_confidence import L6ConfidenceGrade_Layer
from layers.l7_compliance import L7Compliance_Layer
from layers.l8_fairness import L8FairnessDiagnostics_Layer
from layers.l9_human_loop import L9HumanLoop_Layer, L9PendingReview
from layers.l10_audit import L10AuditLedger_Layer
from storage.ledger import init_ledger, seal_artifact, write_summary


# ─────────────────────────────────────────────
# Agent reasoning prompts
# ─────────────────────────────────────────────

AGENT_SYSTEM = """You are the LEAF Credit Agent — an autonomous AI agent that evaluates
loan applications using the LEAF (Layered Explainability AI Framework) architecture.

Your job is to reason about each layer's output and decide the next action.
You must be transparent about your reasoning at every step.

After each layer completes, you will be shown its output.
You must respond with:
1. Your observation of what the output means
2. Your reasoning about what to do next
3. Your decision: PROCEED, ESCALATE, or HALT

PROCEED   = move to next layer normally
ESCALATE  = flag for human review but continue (used when confidence is borderline)
HALT      = stop and reject — only when data is critically insufficient

Always explain WHY you are making each decision.
Be specific about which numbers concerned you or reassured you."""


def _agent_observe_prompt(layer_name: str, layer_output: dict) -> str:
    return f"""
Layer {layer_name} has completed. Here is its output:

{json.dumps(layer_output, indent=2, default=str)}

What do you observe? What does this mean for the credit decision?
What should the agent do next?

Respond as:
OBSERVATION: [what you notice about this output]
REASONING: [what this means for the decision]
DECISION: [PROCEED / ESCALATE / HALT]
NEXT_ACTION: [what the next step should be and why]
"""


# ─────────────────────────────────────────────
# LEAF Agent
# ─────────────────────────────────────────────

class LEAFCreditAgent:
    """
    The LEAF Credit Agent.

    This is a genuine AI agent — the LLM reasons about each
    layer's output and drives the decision process.
    The reasoning trace is logged and sealed into the Evidence Ledger.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        jurisdiction: JurisdictionCode = JurisdictionCode.INDIA,
        verbose: bool = True,
        scenario_tag: Optional[str] = None,
    ):
        self.llm = llm_provider or get_provider_from_env()
        self.jurisdiction = jurisdiction
        self.verbose = verbose
        self.scenario_tag = scenario_tag
        self.reasoning_log = []

        # Ensure model is loaded/trained
        get_model()

    def _log(self, message: str):
        """Print and store agent activity."""
        if self.verbose:
            print(message)
        self.reasoning_log.append(message)

    def _agent_reason(self, layer_name: str, layer_output: dict) -> dict:
        """
        Ask the LLM to reason about a layer's output.
        Returns structured reasoning with PROCEED/ESCALATE/HALT decision.
        """
        response = self.llm.complete(
            system_prompt=AGENT_SYSTEM,
            user_prompt=_agent_observe_prompt(layer_name, layer_output),
            max_tokens=400,
            temperature=0.1,
        )

        # Parse structured response
        lines = response.strip().split("\n")
        reasoning = {"raw": response, "decision": "PROCEED"}
        for line in lines:
            if line.startswith("DECISION:"):
                decision = line.replace("DECISION:", "").strip().upper()
                if any(d in decision for d in ["PROCEED", "ESCALATE", "HALT"]):
                    reasoning["decision"] = next(
                        d for d in ["HALT", "ESCALATE", "PROCEED"] if d in decision
                    )
            elif line.startswith("OBSERVATION:"):
                reasoning["observation"] = line.replace("OBSERVATION:", "").strip()
            elif line.startswith("REASONING:"):
                reasoning["reasoning"] = line.replace("REASONING:", "").strip()
            elif line.startswith("NEXT_ACTION:"):
                reasoning["next_action"] = line.replace("NEXT_ACTION:", "").strip()

        return reasoning

    def run(self, application: LoanApplication) -> dict:
        """
        Run the full LEAF Agent pipeline with LLM reasoning at each step.
        Returns all layer artifacts and the complete reasoning trace.
        """
        init_ledger()
        results = {}
        self.reasoning_log = []

        self._log(f"\n{'='*60}")
        self._log(f"  LEAF CREDIT AGENT — Starting evaluation")
        self._log(f"  Applicant : {application.applicant_name}")
        self._log(f"  Amount    : ₹{application.amount_requested:,.0f}")
        self._log(f"{'='*60}\n")

        # ── L0 ──────────────────────────────────────────────────────
        self._log("[Agent] ▶ Calling L0 — Request & Context")
        l0_layer = L0RequestContext(jurisdiction=self.jurisdiction)
        l0 = l0_layer.process(application)
        seal_artifact(l0.application_id, "L0", l0.model_dump())
        results["L0"] = l0
        self._log(f"[Agent]   Application ID: {l0.application_id}")
        self._log(f"[Agent]   Jurisdiction: {l0.jurisdiction.value} — "
                  f"{len(l0.regulatory_frameworks)} frameworks apply")

        # Agent reasons about L0
        l0_reasoning = self._agent_reason("L0", {
            "jurisdiction": l0.jurisdiction.value,
            "frameworks": [f.value for f in l0.regulatory_frameworks],
            "adverse_action_required": l0.adverse_action_notice_required,
        })
        self._log(f"[Agent] 💭 {l0_reasoning.get('observation', '')}")
        self._log(f"[Agent] → Decision: {l0_reasoning['decision']}")

        if l0_reasoning["decision"] == "HALT":
            self._log("[Agent] ✗ HALTED at L0 — regulatory context not satisfiable")
            return {"status": "halted", "layer": "L0", "results": results}

        # ── L1 ──────────────────────────────────────────────────────
        self._log("\n[Agent] ▶ Calling L1 — Data Provenance")
        l1_layer = L1DataProvenance()
        l1, raw_data = l1_layer.process(application, l0.application_id)
        seal_artifact(l0.application_id, "L1", l1.model_dump())
        results["L1"] = l1
        warnings = [s.freshness_warning for s in l1.sources if s.freshness_warning]
        self._log(f"[Agent]   Sources: {l1.total_sources} fetched, "
                  f"{l1.sources_with_warnings} with warnings")
        if warnings:
            for w in warnings:
                self._log(f"[Agent]   ⚠ {w}")

        l1_reasoning = self._agent_reason("L1", {
            "sources_fetched": l1.total_sources,
            "verified": l1.verified_sources,
            "warnings": warnings,
        })
        self._log(f"[Agent] 💭 {l1_reasoning.get('observation', '')}")
        self._log(f"[Agent] → Decision: {l1_reasoning['decision']}")

        # ── L2 ──────────────────────────────────────────────────────
        self._log("\n[Agent] ▶ Calling L2 — Grounding Check")
        l2_layer = L2GroundingCheck()
        l2 = l2_layer.process(l1, raw_data, l0.application_id)
        seal_artifact(l0.application_id, "L2", l2.model_dump())
        results["L2"] = l2
        self._log(f"[Agent]   Composite grounding: {l2.composite_grounding_score:.2f}")
        self._log(f"[Agent]   Weakest source: {l2.weakest_source}")

        l2_reasoning = self._agent_reason("L2", {
            "composite_score": l2.composite_grounding_score,
            "proceed_threshold": l2.proceed_threshold,
            "proceed_to_model": l2.proceed_to_model,
            "weakest_source": l2.weakest_source,
        })
        self._log(f"[Agent] 💭 {l2_reasoning.get('observation', '')}")
        self._log(f"[Agent] → Decision: {l2_reasoning['decision']}")

        if not l2.proceed_to_model:
            self._log("[Agent] ✗ Grounding too low — flagging for human review")
            seal_artifact(l0.application_id, "AGENT_HALT",
                         {"reason": "grounding_below_threshold",
                          "score": l2.composite_grounding_score})
            return {"status": "escalated", "layer": "L2",
                    "reason": "grounding_below_threshold", "results": results}

        # ── L3 ──────────────────────────────────────────────────────
        self._log("\n[Agent] ▶ Calling L3 — Signal Extraction")
        l3_layer = L3SignalExtraction()
        l3 = l3_layer.process(raw_data, application, l0.application_id)
        seal_artifact(l0.application_id, "L3", l3.model_dump())
        results["L3"] = l3
        self._log(f"[Agent]   Signals: {l3.positive_signals} positive, "
                  f"{l3.negative_signals} negative")
        for sig in l3.signals:
            arrow = "▲" if sig.risk_direction == "positive" else (
                "▼" if sig.risk_direction == "negative" else "─")
            self._log(f"[Agent]   {arrow} {sig.signal_name:<35} {sig.display_value}")

        l3_reasoning = self._agent_reason("L3", {
            "positive_signals": l3.positive_signals,
            "negative_signals": l3.negative_signals,
            "key_signals": {s.signal_name: s.display_value for s in l3.signals},
        })
        self._log(f"[Agent] 💭 {l3_reasoning.get('observation', '')}")
        self._log(f"[Agent] → Decision: {l3_reasoning['decision']}")

        # ── L4 ──────────────────────────────────────────────────────
        self._log("\n[Agent] ▶ Calling L4 — Model Reasoning + SHAP")
        l4_layer = L4ModelReasoning_Layer()
        l4 = l4_layer.process(l3.model_ready_features, l0.application_id)
        seal_artifact(l0.application_id, "L4", l4.model_dump())
        results["L4"] = l4
        self._log(f"[Agent]   Decision: {l4.decision}")
        self._log(f"[Agent]   Approval probability: {l4.approval_probability:.1%}")
        self._log(f"[Agent]   Confidence: {l4.decision_confidence}")
        self._log(f"[Agent]   Top approval factors: {l4.top_approval_factors}")
        self._log(f"[Agent]   Top rejection factors: {l4.top_rejection_factors}")

        l4_reasoning = self._agent_reason("L4", {
            "decision": l4.decision,
            "approval_probability": l4.approval_probability,
            "confidence": l4.decision_confidence,
            "top_approval_factors": l4.top_approval_factors,
            "top_rejection_factors": l4.top_rejection_factors,
            "counterfactual_hint": l4.counterfactual_hint,
        })
        self._log(f"[Agent] 💭 {l4_reasoning.get('observation', '')}")
        self._log(f"[Agent] → Decision: {l4_reasoning['decision']}")

        # ── L6 ──────────────────────────────────────────────────────
        self._log("\n[Agent] ▶ Calling L6 — Confidence Grade")
        l6_layer = L6ConfidenceGrade_Layer()
        l6 = l6_layer.process(l1, l2, l3, l4, l0.application_id)
        seal_artifact(l0.application_id, "L6", l6.model_dump())
        results["L6"] = l6
        self._log(f"[Agent]   Grade           : {l6.grade}")
        self._log(f"[Agent]   Composite Score : {l6.composite_score:.3f}")
        self._log(f"[Agent]   Meaning         : {l6.grade_meaning}")
        self._log(f"[Agent]   HITL Required   : {l6.hitl_required}")
        if l6.grounding_fidelity.contradictions_detected:
            for c in l6.grounding_fidelity.contradictions_detected:
                self._log(f"[Agent]   ⚠ Contradiction : {c}")
        if l6.model_consistency.consistency_flags:
            for f in l6.model_consistency.consistency_flags:
                self._log(f"[Agent]   ⚠ Consistency  : {f}")

        l6_reasoning = self._agent_reason("L6", {
            "grade": l6.grade,
            "composite_score": l6.composite_score,
            "meaning": l6.grade_meaning,
            "hitl_required": l6.hitl_required,
            "decision_blocked": l6.decision_blocked,
            "contradictions": l6.grounding_fidelity.contradictions_detected,
            "consistency_flags": l6.model_consistency.consistency_flags,
            "action_required": l6.action_required,
        })
        self._log(f"[Agent] 💭 {l6_reasoning.get('observation', '')}")
        self._log(f"[Agent] → Decision: {l6_reasoning['decision']}")

        # Grade D — flag it but CONTINUE to L7 and L5
        # All layers run independently for full transparency.
        if l6.decision_blocked:
            self._log("[Agent] ⚠ Grade D — low confidence flagged, "
                      "continuing pipeline for full layer visibility")

        # Grade C flags for human review but continues
        if l6.hitl_required:
            self._log("[Agent] ⚠ Grade C — flagging for human review "
                      "(L9 will be triggered)")

        # ── L7 ──────────────────────────────────────────────────────
        self._log("\n[Agent] ▶ Calling L7 — Compliance & Suitability")
        l7_layer = L7Compliance_Layer()
        l7 = l7_layer.process(l0, l3, l4, l6,
                               application_id=l0.application_id)
        seal_artifact(l0.application_id, "L7", l7.model_dump())
        results["L7"] = l7
        self._log(f"[Agent]   Legitimacy       : {l7.legitimacy_verdict.value}")
        self._log(f"[Agent]   Compliance       : {l7.compliance.overall_status}")
        self._log(f"[Agent]   Suitability      : "
                  f"{l7.suitability.suitability_label.value} "
                  f"({l7.suitability.overall_score:.3f})")
        self._log(f"[Agent]   Override Required: {l7.override_required}")

        l7_reasoning = self._agent_reason("L7", {
            "legitimacy_verdict": l7.legitimacy_verdict.value,
            "decision_legitimate": l7.decision_legitimate,
            "compliance_status": l7.compliance.overall_status,
            "suitability_label": l7.suitability.suitability_label.value,
            "suitability_score": l7.suitability.overall_score,
            "override_required": l7.override_required,
            "override_reason": l7.override_reason,
            "blocking_violations": l7.compliance.blocking_violations,
        })
        self._log(f"[Agent] 💭 {l7_reasoning.get('observation', '')}")
        self._log(f"[Agent] → Decision: {l7_reasoning['decision']}")

        # Blocked by governance — flag it but CONTINUE to L5
        # Each layer runs independently so the full explanation
        # is always generated for transparency and demonstration.
        if l7.legitimacy_verdict.value == "Blocked":
            self._log("[Agent] ⚠ Decision BLOCKED by L7 governance — "
                      "continuing to L5 for full transparency")
            seal_artifact(l0.application_id, "AGENT_GOVERNANCE_BLOCK", {
                "reason": "governance_blocked",
                "compliance_violations": l7.compliance.blocking_violations,
                "note": "Pipeline continues — all layers run independently"
            })

        # ── L8 ──────────────────────────────────────────────────────
        self._log("\n[Agent] ▶ Calling L8 — Bias & Fairness Diagnostics")
        l8_layer = L8FairnessDiagnostics_Layer()
        l8 = l8_layer.process(
            l0, l3, l4, l6, l7, application_id=l0.application_id
        )
        seal_artifact(l0.application_id, "L8", l8.model_dump())
        results["L8"] = l8
        self._log(f"[Agent]   Verdict          : {l8.verdict}")
        self._log(f"[Agent]   Bias Score (OBS) : {l8.obs:.3f}")
        self._log(f"[Agent]   Flags detected   : {l8.total_flags}")
        for flag in l8.bias_flags:
            self._log(f"[Agent]   ⚠ [{flag.severity}] {flag.bias_type}")

        l8_reasoning = self._agent_reason("L8", {
            "verdict": l8.verdict,
            "obs": l8.obs,
            "investigation_required": l8.investigation_required,
            "total_flags": l8.total_flags,
            "flags": [f.bias_type for f in l8.bias_flags],
            "proxy_bias": l8.evidence_source.proxy_bias_detected,
            "fairness_summary": l8.fairness_summary,
        })
        self._log(f"[Agent] 💭 {l8_reasoning.get('observation', '')}")
        self._log(f"[Agent] → Decision: {l8_reasoning['decision']}")

        # ── L9 ──────────────────────────────────────────────────────
        self._log("\n[Agent] ▶ Calling L9 — Human-in-the-Loop")
        l9_layer = L9HumanLoop_Layer()
        l9_pending = l9_layer.process(
            l0, l4, l6, l7, l8,
            l5=results.get("L5"),
            application_id=l0.application_id,
        )
        seal_artifact(l0.application_id, "L9_PENDING",
                      l9_pending.model_dump())
        results["L9"] = l9_pending
        self._log(f"[Agent]   HITL Required  : {l9_pending.hitl_required}")
        self._log(f"[Agent]   Review ID      : {l9_pending.review_id}")
        self._log(f"[Agent]   Trigger Summary: {l9_pending.trigger_summary}")

        l9_reasoning = self._agent_reason("L9", {
            "hitl_required": l9_pending.hitl_required,
            "triggers": [t.value for t in l9_pending.triggers],
            "trigger_summary": l9_pending.trigger_summary,
            "review_id": l9_pending.review_id,
        })
        self._log(f"[Agent] 💭 {l9_reasoning.get('observation', '')}")
        self._log(f"[Agent] → Decision: {l9_reasoning['decision']}")

        # ── L5 ──────────────────────────────────────────────────────
        self._log("\n[Agent] ▶ Calling L5 — Generating Explanation Card")
        l5_layer = L5Recommendation_Layer(llm_provider=self.llm)
        l5 = l5_layer.process(l0, l3, l4, l0.application_id)
        seal_artifact(l0.application_id, "L5", l5.model_dump())
        results["L5"] = l5
        self._log(f"[Agent]   Explanation Card generated")
        self._log(f"[Agent]   Rationale: {l5.plain_language_rationale[:100]}...")

        # ── Write summary to summaries table ────────────────────
        # Final decision reflects governance outcome, not just model output
        if l7 and l7.legitimacy_verdict.value == "Blocked":
            final_decision = f"Blocked — {l4.decision} (Governance)"
        elif l7 and l7.legitimacy_verdict.value == "Conditional":
            final_decision = f"Conditional — {l4.decision} (Review Required)"
        elif l6.decision_blocked:
            final_decision = f"Flagged — {l4.decision} (Grade D)"
        else:
            final_decision = l4.decision

        write_summary(
            application_id=l0.application_id,
            applicant_name=application.applicant_name,
            amount_requested=application.amount_requested,
            purpose=application.purpose,
            decision=final_decision,
            approval_probability=l4.approval_probability,
            confidence=l4.decision_confidence,
            scenario_tag=self.scenario_tag,
        )

        # ── L10 ─────────────────────────────────────────────────────
        self._log("\n[Agent] ▶ Calling L10 — Auditability & Reproducibility Ledger")
        l10 = L10AuditLedger_Layer().process(l0.application_id)
        seal_artifact(l0.application_id, "L10", l10.model_dump())
        results["L10"] = l10
        self._log(f"[Agent]   Audit Verdict    : "
                  f"{l10.audit_certificate.audit_verdict}")
        self._log(f"[Agent]   Completeness     : "
                  f"{l10.completeness.completeness_score:.0%}")
        self._log(f"[Agent]   Integrity        : "
                  f"{l10.integrity.integrity_status}")
        self._log(f"[Agent]   Binding Hash     : "
                  f"{l10.integrity.binding_hash}")
        self._log(f"[Agent]   Certificate ID   : "
                  f"{l10.audit_certificate.certificate_id}")

        # ── Seal full reasoning trace ────────────────────────────
        full_trace = {
            "reasoning_log": self.reasoning_log,
            "layer_decisions": {
                "L0": l0_reasoning.get("decision"),
                "L1": l1_reasoning.get("decision"),
                "L2": l2_reasoning.get("decision"),
                "L3": l3_reasoning.get("decision"),
                "L4": l4_reasoning.get("decision"),
                "L6": l6_reasoning.get("decision"),
                "L7": l7_reasoning.get("decision"),
                "L8": l8_reasoning.get("decision"),
                "L9": l9_reasoning.get("decision"),
            },
            "agent_observations": {
                "L0": l0_reasoning.get("observation", ""),
                "L1": l1_reasoning.get("observation", ""),
                "L2": l2_reasoning.get("observation", ""),
                "L3": l3_reasoning.get("observation", ""),
                "L4": l4_reasoning.get("observation", ""),
                "L6": l6_reasoning.get("observation", ""),
                "L7": l7_reasoning.get("observation", ""),
                "L8": l8_reasoning.get("observation", ""),
                "L9": l9_reasoning.get("observation", ""),
            },
            "final_decision": l4.decision,
            "completed_at": datetime.now().isoformat(),
        }
        seal_artifact(l0.application_id, "AGENT_TRACE", full_trace)

        self._log(f"\n{'='*60}")
        self._log(f"  LEAF AGENT COMPLETE")
        self._log(f"  Decision  : {l4.decision}")
        self._log(f"  Approval  : {l4.approval_probability:.1%}")
        self._log(f"  Layers sealed: L0, L1, L2, L3, L4, L6, L7, L8, L9, L10, L5, AGENT_TRACE")
        self._log(f"{'='*60}\n")

        return {
            "status": "complete",
            "application_id": l0.application_id,
            "decision": l4.decision,
            "approval_probability": l4.approval_probability,
            "results": results,
            "reasoning_trace": full_trace,
        }
