"""
Orchestration pipeline — runs all five agents in sequence and writes
progress updates back into the shared `jobs` dict.
"""
import asyncio
import logging
from typing import Any, Dict, List

from agents import (
    dataset_agent,
    translation_agent,
    evaluation_agent,
    governance_agent,
    explainability_agent,
)

logger = logging.getLogger(__name__)


async def run_pipeline(
    job_id: str,
    csv_content: bytes,
    filename: str,
    target_language: str,
    models: List[str],
    latency_priority: str,
    cost_priority: str,
    jobs: Dict[str, Any],
) -> None:
    def tick(pct: int, stage: str) -> None:
        jobs[job_id]["progress"] = pct
        jobs[job_id]["stage"] = stage
        logger.info(f"[{job_id}] {pct}% — {stage}")

    try:
        # ── Stage 1: Dataset analysis ──────────────────────────────────────
        tick(10, "Agent 1 · Analyzing dataset…")
        dataset_info = dataset_agent.run(csv_content, target_language, filename)
        jobs[job_id]["dataset_analysis"] = dataset_info

        # ── Stage 2: Translation ───────────────────────────────────────────
        tick(25, "Agent 2 · Running translations…")
        samples = dataset_info["sample_texts"]
        domain = dataset_info["detected_domain"]

        translation_results = await translation_agent.run(
            models, samples, target_language, domain
        )

        # ── Stage 3: Evaluation ────────────────────────────────────────────
        tick(55, "Agent 3 · Evaluating quality…")
        reference_map = dataset_info.get("reference_map", {})
        if reference_map:
            tick(55, f"Agent 3 · Evaluating quality with ground truth ({len(reference_map)} references)…")
        evaluated_results = await evaluation_agent.run(
            translation_results, domain, target_language, reference_map=reference_map
        )

        # ── Stage 4: Governance ────────────────────────────────────────────
        tick(75, "Agent 4 · Applying business rules…")
        governance_result = governance_agent.run(
            evaluated_results, domain, latency_priority, cost_priority
        )

        # ── Stage 5: Explainability ────────────────────────────────────────
        tick(90, "Agent 5 · Generating explanations…")
        explain_report = explainability_agent.run(governance_result)

        # ── Build decision summary ─────────────────────────────────────────
        scored_models = governance_result["models"]
        selected = next((m for m in scored_models if m.get("selected")), None)
        runner_up = scored_models[1] if len(scored_models) > 1 else None

        decision = {
            "selected_model": selected["model_id"] if selected else None,
            "selected_model_name": selected["model_name"] if selected else None,
            "selected_provider": selected["provider"] if selected else None,
            "runner_up": runner_up["model_id"] if runner_up else None,
            "runner_up_name": runner_up["model_name"] if runner_up else None,
            "reason": explain_report["summary"],
            "weights_used": governance_result["weights"],
        }

        # ── Done ───────────────────────────────────────────────────────────
        tick(100, "Completed")
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["results"] = {
            "dataset_analysis": dataset_info,
            "model_results": scored_models,
            "decision": decision,
            "explainability": explain_report,
        }

    except Exception as exc:
        logger.exception(f"Pipeline failed for job {job_id}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)
