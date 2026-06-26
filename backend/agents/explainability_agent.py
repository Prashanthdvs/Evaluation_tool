"""
Agent 5: Explainability Agent
Generates human-readable explanations for every model decision.
Satisfies the "Explainable" requirement of the problem statement.
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def _explain_model(model: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    m = model["metrics"]
    sc = model.get("score_components", {})
    gov = model["governance_check"]
    rank = model["rank"]
    name = model["model_name"]
    ws = model["weighted_score"]

    if model.get("selected"):
        status = "selected"
        reason = (
            f"{name} was selected as the optimal translation provider. "
            f"It achieved the highest weighted score of {ws:.1f} "
            f"(Quality: {m['quality_score']:.1f}, "
            f"Latency: {m['avg_latency']:.2f}s, "
            f"Cost: ${m['total_cost']:.4f}, "
            f"Terminology: {m['terminology_accuracy']:.1f}%, "
            f"Hallucination risk: {m['hallucination_risk']:.1%})."
        )
        reasons_list = [
            f"✓ Highest weighted score: {ws:.1f}",
            f"✓ Quality score: {m['quality_score']:.1f}/100",
            f"✓ Terminology accuracy: {m['terminology_accuracy']:.1f}%",
            f"✓ Fluency score: {m['fluency_score']:.1f}/100",
            f"✓ Latency: {m['avg_latency']:.2f}s",
            f"✓ Cost per run: ${m['total_cost']:.4f}",
            f"✓ Hallucination risk: {m['hallucination_risk']:.1%}",
        ]

    elif not gov["passed"]:
        status = "rejected"
        viol_text = "; ".join(gov["violations"])
        reason = (
            f"{name} was rejected because it failed {len(gov['violations'])} "
            f"governance check(s): {viol_text}."
        )
        reasons_list = [f"✗ {v}" for v in gov["violations"]]

    else:
        status = "qualified"
        reason = (
            f"{name} qualified (rank #{rank}) but was outperformed by "
            f"{rank - 1} other model(s). "
            f"Weighted score: {ws:.1f}."
        )
        reasons_list = [
            f"• Ranked #{rank} with weighted score {ws:.1f}",
            f"• Quality: {m['quality_score']:.1f}/100",
            f"• Outscored by {rank - 1} model(s)",
        ]

    return {
        "model_id": model["model_id"],
        "model_name": name,
        "provider": model["provider"],
        "status": status,
        "rank": rank,
        "reason": reason,
        "reasons_list": reasons_list,
        "score_breakdown": {
            "quality_score": m["quality_score"],
            "quality_contribution": sc.get("quality_contribution", 0),
            "quality_weight_pct": f"{weights['quality_weight'] * 100:.0f}%",
            "cost_score": sc.get("cost_score", 0),
            "cost_contribution": sc.get("cost_contribution", 0),
            "cost_weight_pct": f"{weights['cost_weight'] * 100:.0f}%",
            "latency_score": sc.get("latency_score", 0),
            "latency_contribution": sc.get("latency_contribution", 0),
            "latency_weight_pct": f"{weights['latency_weight'] * 100:.0f}%",
            "final_weighted_score": ws,
            "terminology_accuracy": m["terminology_accuracy"],
            "fluency_score": m["fluency_score"],
            "hallucination_risk": m["hallucination_risk"],
            "toxicity_risk": m.get("toxicity_risk", 0),
            "avg_latency_sec": m["avg_latency"],
            "total_cost_usd": m["total_cost"],
        },
    }


def _generate_summary(models: List[Dict[str, Any]], domain: str, weights: Dict[str, float]) -> str:
    selected = next((m for m in models if m.get("selected")), None)
    total = len(models)
    passed = sum(1 for m in models if m["governance_check"]["passed"])
    rejected = total - passed

    if not selected:
        return (
            f"No model passed all governance requirements for the '{domain}' domain. "
            f"Consider relaxing business rules or selecting different models."
        )

    m = selected["metrics"]
    runner_up = models[1] if len(models) > 1 else None

    q_pct = weights["quality_weight"] * 100
    c_pct = weights["cost_weight"] * 100
    l_pct = weights["latency_weight"] * 100

    summary = (
        f"{selected['model_name']} was selected as the best MT provider for the "
        f"'{domain}' domain. "
        f"Evaluated {total} model(s): {passed} passed governance checks"
    )
    if rejected:
        summary += f", {rejected} rejected"
    summary += (
        f". Decision weights — Quality: {q_pct:.0f}%, Cost: {c_pct:.0f}%, "
        f"Latency: {l_pct:.0f}%. "
        f"Key metrics: quality {m['quality_score']:.1f}/100, "
        f"latency {m['avg_latency']:.2f}s, cost ${m['total_cost']:.4f}/run, "
        f"terminology {m['terminology_accuracy']:.1f}%."
    )

    if runner_up:
        diff = round(selected["weighted_score"] - runner_up["weighted_score"], 1)
        summary += (
            f" Outperformed runner-up {runner_up['model_name']} by {diff} weighted points."
        )

    return summary


def run(governance_result: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Explainability Agent: generating explanations")

    models = governance_result["models"]
    weights = governance_result["weights"]
    domain = governance_result["domain"]
    rules_applied = governance_result["rules_applied"]

    explanations = [_explain_model(m, weights) for m in models]
    summary = _generate_summary(models, domain, weights)

    return {
        "summary": summary,
        "model_explanations": explanations,
        "business_rules_applied": rules_applied,
    }
