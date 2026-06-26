"""
Agent 4: Governance Agent
Applies domain-specific business rules from business_rules.yaml.
Computes weighted scores and ranks models.
"""
import os
import yaml
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

_RULES_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "business_rules.yaml")


def _load_rules() -> Dict[str, Any]:
    with open(_RULES_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _normalize_cost(cost: float, max_cost: float = 0.02) -> float:
    """Map cost → 0-100 score (cheaper = higher score)."""
    return max(0.0, (1.0 - cost / max_cost) * 100.0)


def _normalize_latency(latency: float, max_latency: float = 5.0) -> float:
    """Map latency → 0-100 score (faster = higher score)."""
    return max(0.0, (1.0 - latency / max_latency) * 100.0)


def _adjust_weights(
    domain_rule: Dict[str, Any],
    latency_priority: str,
    cost_priority: str,
) -> Dict[str, float]:
    """
    Start from domain base weights and nudge based on user-selected priorities.
    Priority levels: low → -0.10, good → 0, high → +0.15
    Weights are re-normalised to sum to 1.0.
    """
    adj = {"low": -0.10, "good": 0.0, "high": 0.15}

    q_w = float(domain_rule.get("quality_weight", 0.5))
    c_w = float(domain_rule.get("cost_weight", 0.3)) + adj.get(cost_priority, 0.0)
    l_w = float(domain_rule.get("latency_weight", 0.2)) + adj.get(latency_priority, 0.0)

    # Keep non-negative
    c_w = max(0.05, c_w)
    l_w = max(0.05, l_w)

    total = q_w + c_w + l_w
    return {
        "quality_weight": round(q_w / total, 3),
        "cost_weight": round(c_w / total, 3),
        "latency_weight": round(l_w / total, 3),
    }


def _check_governance(
    metrics: Dict[str, Any],
    domain_rule: Dict[str, Any],
) -> Dict[str, Any]:
    violations: List[str] = []

    min_q = domain_rule.get("min_quality", 0.0)
    if metrics["quality_score"] < min_q:
        violations.append(
            f"Quality {metrics['quality_score']:.1f} < required {min_q}"
        )

    max_cost = domain_rule.get("max_cost", float("inf"))
    if metrics["total_cost"] > max_cost:
        violations.append(
            f"Cost ${metrics['total_cost']:.4f} > max ${max_cost:.4f}"
        )

    max_lat = domain_rule.get("max_latency_sec", float("inf"))
    if metrics["avg_latency"] > max_lat:
        violations.append(
            f"Latency {metrics['avg_latency']:.2f}s > max {max_lat}s"
        )

    max_hall = domain_rule.get("max_hallucination_risk", 1.0)
    if metrics["hallucination_risk"] > max_hall:
        violations.append(
            f"Hallucination risk {metrics['hallucination_risk']:.1%} > max {max_hall:.0%}"
        )

    min_term = domain_rule.get("min_terminology_accuracy", 0.0)
    if metrics["terminology_accuracy"] < min_term:
        violations.append(
            f"Terminology {metrics['terminology_accuracy']:.1f}% < required {min_term}%"
        )

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "details": "All governance rules satisfied" if not violations else f"{len(violations)} violation(s)",
    }


def run(
    evaluated_results: List[Dict[str, Any]],
    domain: str,
    latency_priority: str,
    cost_priority: str,
) -> Dict[str, Any]:
    logger.info(f"Governance Agent: domain={domain}, latency={latency_priority}, cost={cost_priority}")

    rules = _load_rules()
    domain_rules = rules.get("domains", {})
    dom_rule = domain_rules.get(domain, domain_rules.get("general", {}))

    weights = _adjust_weights(dom_rule, latency_priority, cost_priority)

    rules_applied: List[str] = [
        f"Domain '{domain}' rules applied",
        f"Weights → Quality {weights['quality_weight']*100:.0f}% | "
        f"Cost {weights['cost_weight']*100:.0f}% | "
        f"Latency {weights['latency_weight']*100:.0f}%",
    ]
    if "min_quality" in dom_rule:
        rules_applied.append(f"Min quality threshold: {dom_rule['min_quality']}")
    if "max_cost" in dom_rule:
        rules_applied.append(f"Max cost per run: ${dom_rule['max_cost']}")
    if "max_latency_sec" in dom_rule:
        rules_applied.append(f"Max latency: {dom_rule['max_latency_sec']}s")
    if "max_hallucination_risk" in dom_rule:
        rules_applied.append(f"Max hallucination risk: {dom_rule['max_hallucination_risk']:.0%}")

    scored: List[Dict[str, Any]] = []
    for res in evaluated_results:
        m = res["metrics"]
        governance = _check_governance(m, dom_rule)

        # Weighted score (only meaningful for passing models)
        q_score = m["quality_score"]
        c_score = _normalize_cost(m["total_cost"])
        l_score = _normalize_latency(m["avg_latency"])

        weighted_score = round(
            q_score * weights["quality_weight"]
            + c_score * weights["cost_weight"]
            + l_score * weights["latency_weight"],
            2,
        )

        scored.append(
            {
                **res,
                "governance_check": governance,
                "weighted_score": weighted_score,
                "score_components": {
                    "quality_score": round(q_score, 1),
                    "quality_contribution": round(q_score * weights["quality_weight"], 1),
                    "cost_score": round(c_score, 1),
                    "cost_contribution": round(c_score * weights["cost_weight"], 1),
                    "latency_score": round(l_score, 1),
                    "latency_contribution": round(l_score * weights["latency_weight"], 1),
                },
            }
        )

    # Sort: passing models by score desc, failing models last
    scored.sort(
        key=lambda x: (x["governance_check"]["passed"], x["weighted_score"]),
        reverse=True,
    )

    for rank, model in enumerate(scored, start=1):
        model["rank"] = rank
        model["selected"] = rank == 1 and model["governance_check"]["passed"]

    return {
        "models": scored,
        "weights": weights,
        "rules_applied": rules_applied,
        "domain": domain,
    }
