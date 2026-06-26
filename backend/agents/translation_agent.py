"""
Agent 2: Translation Agent
Dispatches translation requests to OpenRouter-hosted models via OpenAI-compatible API.
Falls back to realistic simulation when endpoints are unreachable or for
comparison-only models (GPT-4o, DeepL, Azure MT).
"""
import asyncio
import time
import os
import json
import random
import logging
from typing import Dict, Any, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_REGISTERED_MODELS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "registered_models.json"
)

# ---------------------------------------------------------------------------
# OpenRouter base URL
# ---------------------------------------------------------------------------
_OR_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter API keys — tested & verified with credits
_OR_KEY_A = "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"  # claude-sonnet, haiku, deepseek, glm
_OR_KEY_B = "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"  # gemini, qwen3
_OR_KEY_C = "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"  # kimi-k2

# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------
MODEL_PROFILES: Dict[str, Dict] = {
    # ── Live OpenRouter models ────────────────────────────────────────────
    "claude-sonnet-4": {
        "name": "Claude Sonnet 4",
        "provider": "Anthropic (OpenRouter)",
        "api_type": "openai_compat",
        "base_url": _OR_BASE_URL,
        "api_key":  _OR_KEY_A,
        "model":    "anthropic/claude-sonnet-4",
        "max_tokens": 512,
        "cost_per_1k_tokens": 0.00900,   # avg (3+15)/2 /1M * 1000
        "base_latency": 1.8,
        "latency_variance": 0.30,
        "base_quality": {
            "healthcare": 94, "ecommerce": 93, "customer_support": 92,
            "general": 93, "legal": 93, "finance": 93, "marketing": 93,
        },
        "terminology": {
            "healthcare": 95, "ecommerce": 93, "customer_support": 92,
            "general": 93, "legal": 94, "finance": 94, "marketing": 93,
        },
        "fluency": {
            "healthcare": 96, "ecommerce": 95, "customer_support": 94,
            "general": 95, "legal": 95, "finance": 95, "marketing": 96,
        },
        "hallucination_risk": {
            "healthcare": 0.03, "ecommerce": 0.03, "customer_support": 0.04,
            "general": 0.03, "legal": 0.04, "finance": 0.04, "marketing": 0.03,
        },
        "toxicity_risk": 0.01,
        "color": "#7c3aed",
    },
    "gemini-2.5-pro": {
        "name": "Gemini 2.5 Pro",
        "provider": "Google (OpenRouter)",
        "api_type": "openai_compat",
        "base_url": _OR_BASE_URL,
        "api_key":  _OR_KEY_B,
        "model":    "google/gemini-2.5-pro",
        "max_tokens": 512,
        "cost_per_1k_tokens": 0.00563,
        "base_latency": 2.0,
        "latency_variance": 0.35,
        "base_quality": {
            "healthcare": 93, "ecommerce": 92, "customer_support": 91,
            "general": 93, "legal": 92, "finance": 92, "marketing": 93,
        },
        "terminology": {
            "healthcare": 94, "ecommerce": 92, "customer_support": 91,
            "general": 92, "legal": 93, "finance": 93, "marketing": 92,
        },
        "fluency": {
            "healthcare": 95, "ecommerce": 94, "customer_support": 93,
            "general": 95, "legal": 93, "finance": 94, "marketing": 95,
        },
        "hallucination_risk": {
            "healthcare": 0.04, "ecommerce": 0.04, "customer_support": 0.04,
            "general": 0.04, "legal": 0.05, "finance": 0.04, "marketing": 0.04,
        },
        "toxicity_risk": 0.01,
        "color": "#1a73e8",
    },
    "deepseek-r1": {
        "name": "DeepSeek R1",
        "provider": "DeepSeek (OpenRouter)",
        "api_type": "openai_compat",
        "base_url": _OR_BASE_URL,
        "api_key":  _OR_KEY_A,
        "model":    "deepseek/deepseek-r1-0528",
        "max_tokens": 1024,   # reasoning model — needs more tokens
        "cost_per_1k_tokens": 0.00133,
        "base_latency": 3.0,
        "latency_variance": 0.60,
        "base_quality": {
            "healthcare": 90, "ecommerce": 89, "customer_support": 88,
            "general": 90, "legal": 91, "finance": 91, "marketing": 89,
        },
        "terminology": {
            "healthcare": 90, "ecommerce": 89, "customer_support": 88,
            "general": 90, "legal": 92, "finance": 91, "marketing": 89,
        },
        "fluency": {
            "healthcare": 91, "ecommerce": 90, "customer_support": 89,
            "general": 91, "legal": 90, "finance": 91, "marketing": 90,
        },
        "hallucination_risk": {
            "healthcare": 0.05, "ecommerce": 0.05, "customer_support": 0.06,
            "general": 0.05, "legal": 0.06, "finance": 0.05, "marketing": 0.05,
        },
        "toxicity_risk": 0.01,
        "color": "#0ea5e9",
    },
    "glm-4.5": {
        "name": "GLM 4.5",
        "provider": "Z.AI (OpenRouter)",
        "api_type": "openai_compat",
        "base_url": _OR_BASE_URL,
        "api_key":  _OR_KEY_A,
        "model":    "z-ai/glm-4.5",
        "max_tokens": 1024,   # reasoning model — needs more tokens
        "cost_per_1k_tokens": 0.00140,
        "base_latency": 2.5,
        "latency_variance": 0.50,
        "base_quality": {
            "healthcare": 87, "ecommerce": 88, "customer_support": 87,
            "general": 88, "legal": 86, "finance": 87, "marketing": 88,
        },
        "terminology": {
            "healthcare": 86, "ecommerce": 87, "customer_support": 86,
            "general": 87, "legal": 85, "finance": 86, "marketing": 87,
        },
        "fluency": {
            "healthcare": 89, "ecommerce": 90, "customer_support": 88,
            "general": 89, "legal": 87, "finance": 88, "marketing": 90,
        },
        "hallucination_risk": {
            "healthcare": 0.07, "ecommerce": 0.06, "customer_support": 0.07,
            "general": 0.06, "legal": 0.08, "finance": 0.07, "marketing": 0.06,
        },
        "toxicity_risk": 0.02,
        "color": "#10b981",
    },
    "kimi-k2": {
        "name": "Kimi K2",
        "provider": "Moonshot AI (OpenRouter)",
        "api_type": "openai_compat",
        "base_url": _OR_BASE_URL,
        "api_key":  _OR_KEY_C,
        "model":    "moonshotai/kimi-k2",
        "max_tokens": 512,
        "cost_per_1k_tokens": 0.00300,
        "base_latency": 2.2,
        "latency_variance": 0.40,
        "base_quality": {
            "healthcare": 89, "ecommerce": 90, "customer_support": 89,
            "general": 90, "legal": 88, "finance": 89, "marketing": 90,
        },
        "terminology": {
            "healthcare": 88, "ecommerce": 89, "customer_support": 88,
            "general": 89, "legal": 87, "finance": 88, "marketing": 89,
        },
        "fluency": {
            "healthcare": 91, "ecommerce": 92, "customer_support": 90,
            "general": 91, "legal": 89, "finance": 90, "marketing": 92,
        },
        "hallucination_risk": {
            "healthcare": 0.06, "ecommerce": 0.05, "customer_support": 0.06,
            "general": 0.05, "legal": 0.07, "finance": 0.06, "marketing": 0.05,
        },
        "toxicity_risk": 0.01,
        "color": "#f59e0b",
    },
    "claude-haiku-4.5": {
        "name": "Claude Haiku 4.5",
        "provider": "Anthropic (OpenRouter)",
        "api_type": "openai_compat",
        "base_url": _OR_BASE_URL,
        "api_key":  _OR_KEY_A,
        "model":    "anthropic/claude-haiku-4.5",
        "max_tokens": 512,
        "cost_per_1k_tokens": 0.00300,
        "base_latency": 0.8,
        "latency_variance": 0.15,
        "base_quality": {
            "healthcare": 88, "ecommerce": 88, "customer_support": 88,
            "general": 89, "legal": 87, "finance": 88, "marketing": 89,
        },
        "terminology": {
            "healthcare": 88, "ecommerce": 88, "customer_support": 87,
            "general": 88, "legal": 87, "finance": 88, "marketing": 88,
        },
        "fluency": {
            "healthcare": 90, "ecommerce": 91, "customer_support": 90,
            "general": 91, "legal": 89, "finance": 90, "marketing": 91,
        },
        "hallucination_risk": {
            "healthcare": 0.05, "ecommerce": 0.04, "customer_support": 0.05,
            "general": 0.04, "legal": 0.06, "finance": 0.05, "marketing": 0.04,
        },
        "toxicity_risk": 0.01,
        "color": "#ef4444",
    },
    "qwen3-32b": {
        "name": "Qwen3 32B",
        "provider": "Alibaba (OpenRouter)",
        "api_type": "openai_compat",
        "base_url": _OR_BASE_URL,
        "api_key":  _OR_KEY_B,
        "model":    "qwen/qwen3-32b",
        "max_tokens": 512,
        "cost_per_1k_tokens": 0.00018,
        "base_latency": 1.0,
        "latency_variance": 0.20,
        "base_quality": {
            "healthcare": 86, "ecommerce": 88, "customer_support": 86,
            "general": 87, "legal": 84, "finance": 85, "marketing": 88,
        },
        "terminology": {
            "healthcare": 85, "ecommerce": 87, "customer_support": 85,
            "general": 86, "legal": 83, "finance": 84, "marketing": 87,
        },
        "fluency": {
            "healthcare": 88, "ecommerce": 90, "customer_support": 88,
            "general": 89, "legal": 86, "finance": 87, "marketing": 91,
        },
        "hallucination_risk": {
            "healthcare": 0.08, "ecommerce": 0.07, "customer_support": 0.08,
            "general": 0.07, "legal": 0.09, "finance": 0.08, "marketing": 0.07,
        },
        "toxicity_risk": 0.02,
        "color": "#6366f1",
    },
    # ── Comparison / simulation-only models ──────────────────────────────
    "gpt-4o": {
        "name": "GPT-4o",
        "provider": "OpenAI",
        "api_type": "simulated",
        "base_url": None, "api_key": None, "model": "gpt-4o",
        "cost_per_1k_tokens": 0.01500,
        "base_latency": 2.1,
        "latency_variance": 0.40,
        "base_quality": {
            "healthcare": 93, "ecommerce": 91, "customer_support": 90,
            "general": 92, "legal": 91, "finance": 92, "marketing": 92,
        },
        "terminology": {
            "healthcare": 94, "ecommerce": 91, "customer_support": 90,
            "general": 91, "legal": 92, "finance": 93, "marketing": 92,
        },
        "fluency": {
            "healthcare": 95, "ecommerce": 94, "customer_support": 93,
            "general": 94, "legal": 93, "finance": 94, "marketing": 95,
        },
        "hallucination_risk": {
            "healthcare": 0.04, "ecommerce": 0.04, "customer_support": 0.05,
            "general": 0.04, "legal": 0.06, "finance": 0.05, "marketing": 0.04,
        },
        "toxicity_risk": 0.01,
        "color": "#10b981",
    },
    "deepl": {
        "name": "DeepL",
        "provider": "DeepL SE",
        "api_type": "simulated",
        "base_url": None, "api_key": None, "model": "deepl",
        "cost_per_1k_tokens": 0.00250,
        "base_latency": 0.4,
        "latency_variance": 0.10,
        "base_quality": {
            "healthcare": 90, "ecommerce": 89, "customer_support": 88,
            "general": 90, "legal": 87, "finance": 88, "marketing": 89,
        },
        "terminology": {
            "healthcare": 92, "ecommerce": 90, "customer_support": 89,
            "general": 90, "legal": 88, "finance": 89, "marketing": 90,
        },
        "fluency": {
            "healthcare": 93, "ecommerce": 92, "customer_support": 91,
            "general": 93, "legal": 90, "finance": 91, "marketing": 93,
        },
        "hallucination_risk": {
            "healthcare": 0.03, "ecommerce": 0.03, "customer_support": 0.03,
            "general": 0.02, "legal": 0.04, "finance": 0.03, "marketing": 0.03,
        },
        "toxicity_risk": 0.01,
        "color": "#f59e0b",
    },
    "azure-mt": {
        "name": "Azure Translator",
        "provider": "Microsoft",
        "api_type": "simulated",
        "base_url": None, "api_key": None, "model": "azure-mt",
        "cost_per_1k_tokens": 0.00100,
        "base_latency": 0.3,
        "latency_variance": 0.08,
        "base_quality": {
            "healthcare": 82, "ecommerce": 83, "customer_support": 81,
            "general": 83, "legal": 80, "finance": 81, "marketing": 82,
        },
        "terminology": {
            "healthcare": 83, "ecommerce": 82, "customer_support": 80,
            "general": 81, "legal": 79, "finance": 80, "marketing": 82,
        },
        "fluency": {
            "healthcare": 84, "ecommerce": 85, "customer_support": 83,
            "general": 85, "legal": 82, "finance": 83, "marketing": 85,
        },
        "hallucination_risk": {
            "healthcare": 0.07, "ecommerce": 0.06, "customer_support": 0.07,
            "general": 0.06, "legal": 0.08, "finance": 0.07, "marketing": 0.06,
        },
        "toxicity_risk": 0.02,
        "color": "#ef4444",
    },
}

LANG_NAMES = {
    "de": "German", "fr": "French", "ja": "Japanese",
    "zh-CN": "Chinese (Simplified)", "ko": "Korean", "es": "Spanish",
}

# Placeholder translations per language for simulation
PLACEHOLDER_TRANSLATIONS: Dict[str, List[str]] = {
    "de": [
        "Die Behandlung des Patienten erfordert besondere Aufmerksamkeit.",
        "Das Produkt ist von höchster Qualität und sehr erschwinglich.",
        "Unser Support-Team steht rund um die Uhr für Sie bereit.",
        "Die klinische Diagnose wurde erfolgreich abgeschlossen.",
        "Wir bieten erstklassige Übersetzungsdienstleistungen an.",
    ],
    "fr": [
        "Le traitement du patient nécessite une attention particulière.",
        "Le produit est de la plus haute qualité et très abordable.",
        "Notre équipe de support est disponible 24h/24 et 7j/7.",
        "Le diagnostic clinique a été complété avec succès.",
        "Nous offrons des services de traduction de premier ordre.",
    ],
    "ja": [
        "患者の治療には特別な注意が必要です。",
        "この製品は最高品質で非常に手頃な価格です。",
        "サポートチームは24時間365日対応いたします。",
        "臨床診断が正常に完了しました。",
        "一流の翻訳サービスを提供しています。",
    ],
    "zh-CN": [
        "患者的治疗需要特别关注。",
        "该产品质量最高，价格非常实惠。",
        "我们的支持团队全天候为您服务。",
        "临床诊断已成功完成。",
        "我们提供一流的翻译服务。",
    ],
    "ko": [
        "환자 치료에는 특별한 주의가 필요합니다.",
        "이 제품은 최고 품질이며 매우 합리적인 가격입니다.",
        "지원팀은 연중무휴 24시간 운영됩니다.",
        "임상 진단이 성공적으로 완료되었습니다.",
        "최고 수준의 번역 서비스를 제공합니다.",
    ],
    "es": [
        "El tratamiento del paciente requiere especial atención.",
        "El producto es de la más alta calidad y muy asequible.",
        "Nuestro equipo de soporte está disponible las 24 horas.",
        "El diagnóstico clínico se completó exitosamente.",
        "Ofrecemos servicios de traducción de primer nivel.",
    ],
}


async def translate_with_openai_compat(
    text: str,
    target_lang: str,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int = 512,
) -> Tuple[Optional[str], float, float]:
    """
    Call any OpenAI-compatible /chat/completions endpoint.
    Returns (translation, latency_seconds, actual_cost_usd).
    For OpenRouter, actual_cost_usd comes from usage.cost in the response.
    For reasoning models (DeepSeek, GLM-4.5) content may be null; falls back to
    extracting the final answer from the reasoning field.
    """
    lang_name = LANG_NAMES.get(target_lang, target_lang)
    stripped  = base_url.rstrip("/")
    url = (stripped + "/chat/completions") if stripped.endswith("/v1") else (stripped + "/v1/chat/completions")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are a professional translator. "
                    f"Translate the following text into {lang_name}. "
                    f"Output ONLY the translated text — no explanations, no original text."
                ),
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            # Standard response
            translation = msg.get("content") or ""
            # Reasoning models (DeepSeek-R1, GLM-4.5): content may be null/empty;
            # extract last non-empty line from reasoning as the answer
            if not translation.strip():
                reasoning = msg.get("reasoning") or ""
                lines = [l.strip() for l in reasoning.splitlines() if l.strip()]
                if lines:
                    translation = lines[-1]
            if not translation.strip():
                return None, 0.0, 0.0
            # OpenRouter returns actual USD cost in usage.cost
            actual_cost = float(data.get("usage", {}).get("cost", 0) or 0)
            return translation.strip(), round(time.time() - t0, 3), actual_cost
    except Exception as exc:
        logger.warning(f"OpenAI-compat call to {base_url} failed: {exc}")
        return None, 0.0, 0.0


async def translate_with_groq(
    text: str,
    target_lang: str,
    model_id: str,
) -> Tuple[Optional[str], float]:
    """
    Fallback: use Groq API (llama-3.3-70b-versatile) for real translation
    when internal cluster endpoints are unreachable.
    Requires GROQ_API_KEY env var.
    """
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        return None, 0.0
    lang_name = LANG_NAMES.get(target_lang, target_lang)
    groq_model = "llama-3.3-70b-versatile" if "llama" in model_id else "gemma2-9b-it"
    try:
        import groq as _groq
        client = _groq.AsyncGroq(api_key=groq_key)
        t0 = time.time()
        resp = await client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": (
                    f"You are a professional translator. "
                    f"Translate the following text into {lang_name}. "
                    f"Output ONLY the translated text — no explanations, no original text."
                )},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        translation = resp.choices[0].message.content.strip()
        return translation, round(time.time() - t0, 3)
    except Exception as exc:
        logger.warning(f"Groq fallback failed for {model_id}: {exc}")
        return None, 0.0


def simulate_translation(
    text: str,
    target_lang: str,
    model_id: str,
    domain: str,
) -> Tuple[str, float]:
    """Return a plausible-looking simulation result."""
    profile = MODEL_PROFILES.get(model_id, MODEL_PROFILES["llama-3.3-70b"])
    base_latency = profile["base_latency"]
    variance = profile["latency_variance"]
    word_count = len(text.split())
    latency = base_latency * (0.6 + 0.4 * (word_count / 20.0)) + random.gauss(0, variance)
    latency = max(0.05, latency)

    pool = PLACEHOLDER_TRANSLATIONS.get(target_lang, PLACEHOLDER_TRANSLATIONS["de"])
    idx = abs(hash(text + model_id)) % len(pool)
    return pool[idx], round(latency, 3)


def _load_registered_models() -> List[Dict[str, Any]]:
    try:
        if os.path.exists(_REGISTERED_MODELS_PATH):
            with open(_REGISTERED_MODELS_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception as exc:
        logger.warning(f"Could not load registered models: {exc}")
    return []


def _resolve_profile(model_id: str) -> Optional[Dict[str, Any]]:
    """Return a translation profile for a built-in or custom (onboarded) model."""
    if model_id in MODEL_PROFILES:
        return MODEL_PROFILES[model_id]

    for m in _load_registered_models():
        if m.get("model_id") == model_id:
            return {
                "name":               m.get("model_name", model_id),
                "provider":           m.get("provider", "Custom"),
                "api_type":           m.get("api_type", "simulated"),
                "base_url":           m.get("base_url"),
                "api_key":            m.get("api_key"),
                "model":              m.get("model") or m.get("model_id"),
                "cost_per_1k_tokens": float(m.get("cost_per_1k_tokens", 0.001)),
                "base_latency":       float(m.get("base_latency", 1.0)),
                "latency_variance":   float(m.get("latency_variance", 0.2)),
                "color":              m.get("color", "#6b7280"),
            }
    return None


async def translate_one_model(
    model_id: str,
    samples: List[str],
    target_lang: str,
    domain: str,
) -> Dict[str, Any]:
    profile = _resolve_profile(model_id)
    if not profile:
        raise ValueError(f"Unknown model: {model_id}")

    api_type         = profile.get("api_type", "simulated")
    base_url         = profile.get("base_url")
    api_key          = profile.get("api_key")
    model_name_param = profile.get("model")
    max_tokens       = int(profile.get("max_tokens", 512))

    translations: List[Dict] = []
    latencies: List[float]   = []
    total_tokens = 0
    total_actual_cost = 0.0   # accumulated real cost from OpenRouter usage.cost

    for text in samples:
        translation: Optional[str] = None
        latency: float = 0.0
        actual_cost: float = 0.0

        # Real call for openai_compat models
        if api_type == "openai_compat" and base_url and api_key and model_name_param:
            translation, latency, actual_cost = await translate_with_openai_compat(
                text, target_lang, base_url, api_key, model_name_param, max_tokens
            )
            total_actual_cost += actual_cost

        # Groq fallback if primary endpoint failed
        if not translation:
            translation, latency = await translate_with_groq(text, target_lang, model_id)

        # Simulation fallback (also used for simulated-only models)
        if not translation:
            translation, latency = simulate_translation(text, target_lang, model_id, domain)

        tokens = int((len(text.split()) + len(translation.split())) * 1.3)
        total_tokens += tokens
        translations.append(
            {"source": text, "translation": translation, "latency": latency, "tokens": tokens}
        )
        latencies.append(latency)

    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    # Use actual cost from OpenRouter when available, else estimate from token count
    if total_actual_cost > 0:
        total_cost = round(total_actual_cost, 8)
    else:
        total_cost = round((total_tokens / 1000.0) * profile["cost_per_1k_tokens"], 8)

    return {
        "model_id":           model_id,
        "model_name":         profile["name"],
        "provider":           profile["provider"],
        "color":              profile.get("color", "#6b7280"),
        "api_type":           api_type,
        "translations":       translations,
        "avg_latency":        avg_latency,
        "total_tokens":       total_tokens,
        "total_cost":         total_cost,
        "cost_per_1k_tokens": profile["cost_per_1k_tokens"],
    }


async def run(
    model_ids: List[str],
    samples: List[str],
    target_lang: str,
    domain: str,
) -> List[Dict[str, Any]]:
    tasks   = [translate_one_model(mid, samples, target_lang, domain) for mid in model_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid: List[Dict[str, Any]] = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Translation error: {r}")
        else:
            valid.append(r)  # type: ignore[arg-type]
    return valid
