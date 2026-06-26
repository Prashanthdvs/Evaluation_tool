"""
FastAPI entry-point for the MT Evaluation Engine.
"""
import json
import logging
import os
import uuid
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

from pipeline import run_pipeline
from agents.translation_agent import MODEL_PROFILES

app = FastAPI(title="MT Evaluation Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ────────────────────────────────────────────────────────
jobs: Dict[str, Any] = {}

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(__file__)
_RULES_PATH = os.path.join(_BASE, "config", "business_rules.yaml")
_MODELS_PATH = os.path.join(_BASE, "config", "registered_models.json")


def _load_custom_models() -> List[Dict]:
    if os.path.exists(_MODELS_PATH):
        with open(_MODELS_PATH, "r") as fh:
            return json.load(fh)
    return []


def _save_custom_models(models: List[Dict]) -> None:
    with open(_MODELS_PATH, "w") as fh:
        json.dump(models, fh, indent=2)


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/evaluate")
async def evaluate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_language: str = Form("de"),
    models: str = Form('["llama-3.3-70b","gemma-4-31b"]'),
    latency_priority: str = Form("good"),
    cost_priority: str = Form("good"),
):
    try:
        model_list: List[str] = json.loads(models)
    except Exception:
        model_list = [models]

    if not model_list:
        raise HTTPException(status_code=400, detail="At least one model must be selected.")

    content = await file.read()
    job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "stage": "Queued…",
        "dataset_analysis": None,
        "results": None,
        "error": None,
    }

    background_tasks.add_task(
        run_pipeline,
        job_id,
        content,
        file.filename or "upload.csv",
        target_language,
        model_list,
        latency_priority,
        cost_priority,
        jobs,
    )

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] == "processing":
        return {"status": "processing", "progress": job["progress"], "stage": job["stage"]}
    if job["status"] == "failed":
        return {"status": "failed", "error": job.get("error", "Unknown error")}
    return {"status": "completed", **job["results"]}


@app.get("/api/models")
async def list_models():
    builtin = [
        {
            "model_id":           mid,
            "model_name":         p["name"],
            "provider":           p["provider"],
            "api_type":           p.get("api_type", "simulated"),
            "cost_per_1k_tokens": p["cost_per_1k_tokens"],
            "base_latency":       p["base_latency"],
            "color":              p.get("color", "#6b7280"),
            "is_builtin":         True,
        }
        for mid, p in MODEL_PROFILES.items()
    ]
    custom = _load_custom_models()
    return {"builtin": builtin, "custom": custom}


@app.post("/api/models")
async def add_model(model: Dict[str, Any]):
    if not model.get("model_id") or not model.get("model_name"):
        raise HTTPException(status_code=400, detail="model_id and model_name are required.")
    models = _load_custom_models()
    if any(m["model_id"] == model["model_id"] for m in models):
        raise HTTPException(status_code=409, detail="Model ID already exists.")
    model["is_builtin"] = False
    models.append(model)
    _save_custom_models(models)
    return {"success": True, "model": model}


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    models = _load_custom_models()
    updated = [m for m in models if m.get("model_id") != model_id]
    if len(updated) == len(models):
        raise HTTPException(status_code=404, detail="Custom model not found.")
    _save_custom_models(updated)
    return {"success": True}


@app.get("/api/rules")
async def get_rules():
    with open(_RULES_PATH, "r") as fh:
        return yaml.safe_load(fh)


@app.put("/api/rules")
async def update_rules(rules: Dict[str, Any]):
    with open(_RULES_PATH, "w") as fh:
        yaml.dump(rules, fh, default_flow_style=False, allow_unicode=True)
    return {"success": True}


# ── Benchmark results (Auto Model Router) ─────────────────────────────────────
_BENCHMARK_PATH = os.path.join(_BASE, "config", "benchmark_results.json")


def _load_benchmark() -> List[Dict]:
    if os.path.exists(_BENCHMARK_PATH):
        with open(_BENCHMARK_PATH, "r") as fh:
            return json.load(fh)
    return []


@app.get("/api/benchmark")
async def get_benchmark(
    language: str = None,
    content_type: str = None,
    domain: str = None,
    latency: str = None,
    cost: str = None,
):
    """
    Query pre-computed benchmark results, filtered by any combination of:
    language, content_type, domain, latency tier, cost tier.
    Returns all matching rows sorted by weighted_score desc.
    """
    rows = _load_benchmark()
    if language:
        rows = [r for r in rows if r.get("language","").lower() == language.lower()]
    if content_type:
        rows = [r for r in rows if r.get("content_type","").lower() == content_type.lower()]
    if domain:
        rows = [r for r in rows if r.get("domain","").lower() == domain.lower()]
    if latency:
        rows = [r for r in rows if r.get("latency_tier","").lower() == latency.lower()]
    if cost:
        rows = [r for r in rows if r.get("cost_tier","").lower() == cost.lower()]
    rows.sort(key=lambda r: r.get("weighted_score", 0), reverse=True)
    return {"results": rows, "total": len(rows)}


@app.get("/api/benchmark/languages")
async def benchmark_languages():
    rows = _load_benchmark()
    return sorted({r["language"] for r in rows})


@app.get("/api/benchmark/domains")
async def benchmark_domains():
    rows = _load_benchmark()
    return sorted({r["domain"] for r in rows})


@app.get("/api/benchmark/content_types")
async def benchmark_content_types():
    rows = _load_benchmark()
    return sorted({r["content_type"] for r in rows})


# ── Benchmark data files (for onboarding run-benchmark) ───────────────────────
_BENCH_DATA_MAP = {
    "IT Software":        os.path.join(_BASE, "config", "benchmark_data_IT_Software.csv"),
    "E-commerce Product": os.path.join(_BASE, "config", "benchmark_data_Ecommerce.csv"),
    "Pharma/Healthcare":  os.path.join(_BASE, "config", "benchmark_data_Pharma.csv"),
    "Legal Compliance":   os.path.join(_BASE, "config", "benchmark_data_Legal.csv"),
    "Finance/Banking":    os.path.join(_BASE, "config", "benchmark_data_Finance.csv"),
    "Multimedia Streaming": os.path.join(_BASE, "config", "benchmark_data_Multimedia.csv"),
    "Journals Publishing":  os.path.join(_BASE, "config", "benchmark_data_Journals.csv"),
}


@app.get("/api/benchmark/data/{domain_key}")
async def get_benchmark_data(domain_key: str):
    """Return the benchmark CSV content for a given domain."""
    from urllib.parse import unquote
    domain_key = unquote(domain_key)
    path = _BENCH_DATA_MAP.get(domain_key)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No benchmark data for domain: {domain_key}")
    with open(path, "r", encoding="utf-8") as fh:
        return {"domain": domain_key, "csv": fh.read()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
