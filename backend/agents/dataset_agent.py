"""
Agent 1: Dataset Agent — Enhanced
Supports CSV, JSON, TSV, TXT and any delimiter-separated format.
Dynamically detects source column, language, domain, and content type.
Uses adaptive intelligent sampling based on corpus size and diversity.
"""
import pandas as pd
import chardet
import io
import json
import hashlib
import logging
import re
import random
from typing import Dict, Any, List, Optional

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN KEYWORD REGISTRY — all 7 supported domains
# ══════════════════════════════════════════════════════════════════════════════
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "pharma_healthcare": [
        "patient", "medical", "diagnosis", "treatment", "doctor", "hospital",
        "medication", "clinical", "symptom", "disease", "health", "pharmacy",
        "prescription", "therapy", "surgery", "nurse", "dosage", "chronic",
        "condition", "specialist", "laboratory", "pathology", "drug", "dose",
        "mg", "ml", "contraindication", "adverse", "pharmacist", "clinical trial",
        "informed consent", "investigational", "placebo", "cohort",
    ],
    "ecommerce": [
        "product", "price", "shipping", "order", "cart", "buy", "purchase",
        "discount", "delivery", "checkout", "payment", "store", "shop",
        "customer", "review", "return", "refund", "tracking", "catalogue",
        "inventory", "warehouse", "sku", "seller", "marketplace", "brand",
        "stock", "out of stock", "add to cart", "wishlist", "promo", "coupon",
    ],
    "it_software": [
        "click", "button", "menu", "settings", "error", "warning", "loading",
        "install", "download", "upload", "login", "logout", "password", "api",
        "server", "client", "database", "config", "file", "folder", "path",
        "version", "update", "patch", "debug", "deploy", "build", "sdk",
        "json", "xml", "csv", "http", "url", "endpoint", "token", "cache",
        "placeholder", "variable", "string", "integer", "boolean", "null",
    ],
    "legal_compliance": [
        "contract", "clause", "agreement", "terms", "liability", "compliance",
        "regulation", "law", "court", "legal", "rights", "obligation",
        "jurisdiction", "arbitration", "indemnify", "warrant", "breach",
        "statute", "legislation", "intellectual property", "confidential",
        "data retention", "gdpr", "privacy", "consent", "audit", "penalty",
        "force majeure", "termination", "notice", "binding", "parties",
    ],
    "finance_banking": [
        "payment", "invoice", "account", "balance", "transaction", "bank",
        "credit", "debit", "interest", "loan", "investment", "financial",
        "revenue", "budget", "audit", "quarterly", "equity", "dividend",
        "swift", "iban", "otp", "kyc", "aml", "transfer", "wire", "deposit",
        "withdrawal", "statement", "currency", "exchange", "portfolio",
        "risk", "margin", "collateral", "amortization",
    ],
    "multimedia_streaming": [
        "stream", "play", "pause", "video", "audio", "subtitle", "caption",
        "resolution", "bitrate", "buffer", "download", "offline", "episode",
        "series", "season", "channel", "playlist", "quality", "drm", "hd",
        "4k", "codec", "format", "broadcast", "live", "on demand", "player",
        "accessibility", "closed caption", "dubbing", "sync",
    ],
    "journals_publishing": [
        "manuscript", "submission", "peer review", "editor", "author",
        "journal", "publication", "article", "doi", "orcid", "citation",
        "reference", "abstract", "bibliography", "issn", "volume", "issue",
        "retraction", "revision", "accepted", "rejected", "proofs", "embargo",
        "open access", "copyright", "plagiarism", "figure", "table",
    ],
    "customer_support": [
        "issue", "problem", "help", "support", "ticket", "resolve", "complaint",
        "service", "assist", "solution", "request", "feedback",
        "account", "reset", "contact", "representative", "escalate",
        "status", "pending", "investigation", "follow-up",
    ],
}

# Map internal domain IDs → display names
DOMAIN_DISPLAY: Dict[str, str] = {
    "pharma_healthcare":    "Pharma/Healthcare",
    "ecommerce":            "E-commerce Product",
    "it_software":          "IT Software",
    "legal_compliance":     "Legal Compliance",
    "finance_banking":      "Finance/Banking",
    "multimedia_streaming": "Multimedia Streaming",
    "journals_publishing":  "Journals Publishing",
    "customer_support":     "Customer Support",
    "general":              "General",
}

# ── Content-type heuristics ───────────────────────────────────────────────────
_UI_TERMS = re.compile(
    r'\b(click|tap|press|select|choose|open|close|save|cancel|ok|yes|no|next|back|'
    r'submit|confirm|delete|edit|add|remove|update|search|filter|sort|view|'
    r'login|logout|sign in|sign out|continue|retry|dismiss)\b', re.I)
_WARN_TERMS = re.compile(r'\b(warning|caution|alert|attention|notice|important)\b', re.I)
_ERROR_TERMS = re.compile(r'\b(error|failed|invalid|not found|unable|cannot|denied|timeout)\b', re.I)
_FORMAT_TOKEN = re.compile(
    r'\{[^}]+\}|%\([^)]+\)[sdifxg]|%[sdifxg]|\[\[[^\]]+\]\]'
    r'|</?[a-zA-Z][^>]*>|&[a-zA-Z]+;|\$\{[^}]+\}|\$[A-Z_]{2,}|__[A-Z_]{2,}__'
)

CONTENT_TYPE_PATTERNS = {
    "question":   lambda t: t.strip().endswith("?"),
    "document":   lambda t: len(t.split("\n")) >= 3,
    "paragraph":  lambda t: len(t.split()) >= 30,
    "sentence":   lambda t: True,  # fallback
}

LANGUAGE_NAMES = {
    "zh-CN": "Chinese Simplified", "zh": "Chinese", "ja": "Japanese",
    "fr": "French", "ko": "Korean", "de": "German", "es": "Spanish",
    "en": "English", "it": "Italian", "pt": "Portuguese", "nl": "Dutch",
    "ar": "Arabic", "ru": "Russian",
}


def detect_encoding(content: bytes) -> str:
    # Check for BOM first — most reliable signal
    if content[:2] == b"\xff\xfe":
        return "utf-16-le"
    if content[:2] == b"\xfe\xff":
        return "utf-16-be"
    if content[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"
    result = chardet.detect(content)
    return result.get("encoding") or "utf-8"


def _strip_bom(content: bytes) -> bytes:
    """Remove any BOM so pandas sees clean bytes."""
    for bom in (b"\xff\xfe", b"\xfe\xff", b"\xef\xbb\xbf"):
        if content.startswith(bom):
            return content[len(bom):]
    return content


def _detect_delimiter(sample: str) -> str:
    """Sniff the delimiter from the first line."""
    first_line = sample.split("\n")[0]
    counts = {d: first_line.count(d) for d in (",", "\t", ";", "|")}
    return max(counts, key=counts.get)


def _flatten_json(obj: Any, prefix: str = "") -> List[str]:
    """Recursively extract all string leaf values from a JSON object/array."""
    texts: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            texts.extend(_flatten_json(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            texts.extend(_flatten_json(v, f"{prefix}[{i}]"))
    elif isinstance(obj, str) and len(obj.strip()) > 2:
        texts.append(obj.strip())
    return texts


def parse_json(content: bytes) -> pd.DataFrame:
    """
    Parse JSON content → DataFrame.
    Handles: list-of-dicts (tabular), dict-with-list, flat dict, deeply nested.
    """
    enc = detect_encoding(content)
    for try_enc in [enc, "utf-8", "latin-1"]:
        try:
            text = content.decode(try_enc, errors="replace")
            break
        except Exception:
            text = content.decode("latin-1", errors="replace")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    # Case 1: standard list of dicts → tabular JSON
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return pd.DataFrame(data)

    # Case 2: dict with a key whose value is a list of dicts
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return pd.DataFrame(val)

    # Case 3: flat string dict — keys as IDs, values as source text
    if isinstance(data, dict) and all(isinstance(v, str) for v in data.values()):
        return pd.DataFrame([{"key": k, "source": v} for k, v in data.items()])

    # Case 4: deeply nested — flatten all string values
    all_strings = _flatten_json(data)
    if all_strings:
        return pd.DataFrame({"source": all_strings})

    raise ValueError("JSON structure not recognized as a translatable dataset.")


def parse_csv(content: bytes) -> pd.DataFrame:
    encoding = detect_encoding(content)
    clean    = _strip_bom(content)

    # Decode to string to sniff delimiter
    for enc in [encoding, "utf-8", "utf-16", "utf-16-le", "latin-1", "cp1252"]:
        try:
            sample = clean.decode(enc, errors="replace")
            break
        except Exception:
            continue
    else:
        sample = clean.decode("latin-1", errors="replace")

    delimiter = _detect_delimiter(sample)

    # Try reading with detected encoding + delimiter
    for enc in [encoding, "utf-8", "utf-16", "utf-16-le", "latin-1", "cp1252"]:
        for delim in [delimiter, ",", "\t", ";", "|"]:
            try:
                df = pd.read_csv(
                    io.BytesIO(clean),
                    encoding=enc,
                    sep=delim,
                    engine="python",
                    on_bad_lines="skip",
                )
                if not df.empty and len(df.columns) >= 1:
                    return df
            except Exception:
                continue

    raise ValueError("Could not parse CSV with any known encoding or delimiter.")


def parse_any_format(content: bytes, filename: str = "") -> pd.DataFrame:
    """
    Dispatch to the right parser based on file extension or content sniffing.
    Falls back: JSON → CSV/TSV → plain-text lines.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    # JSON: try when extension says so or when content starts with { or [
    if ext == "json" or content.lstrip()[:1] in (b"{", b"["):
        try:
            return parse_json(content)
        except Exception as e:
            logger.debug(f"JSON parse failed ({e}), falling back to CSV")

    # CSV / TSV (default)
    try:
        return parse_csv(content)
    except Exception as e:
        logger.debug(f"CSV parse failed ({e}), falling back to plain-text")

    # Plain text: one segment per line
    enc = detect_encoding(content)
    text = content.decode(enc, errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("File appears to be empty or unreadable.")
    return pd.DataFrame({"source": lines})


def find_source_column(df: pd.DataFrame) -> str:
    # 1. Exact priority match (case-insensitive)
    priority = [
        "source", "src", "text", "source_text", "source_segment",
        "segment", "en", "english", "input", "original", "raw",
        "sentence", "content", "src_text", "text_en", "q", "query",
    ]
    col_lower = {c.lower().strip(): c for c in df.columns}
    for name in priority:
        if name in col_lower:
            return col_lower[name]

    # 2. Column name contains a priority keyword
    for col in df.columns:
        for name in priority:
            if name in col.lower():
                return col

    # 3. First string column with meaningful average length (> 10 chars)
    for col in df.columns:
        if df[col].dtype == object:
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if avg_len and avg_len > 10:
                return col

    # 4. Any string column
    for col in df.columns:
        if df[col].dtype == object:
            return col

    # 5. Column with most unique string-like values (last resort)
    best_col, best_unique = None, 0
    for col in df.columns:
        try:
            n = df[col].astype(str).nunique()
            if n > best_unique:
                best_unique, best_col = n, col
        except Exception:
            pass
    if best_col:
        return best_col

    raise ValueError(
        f"Could not find a source-text column. Columns found: {list(df.columns)}"
    )


def detect_language(texts: List[str]) -> str:
    if not LANGDETECT_AVAILABLE:
        return "en"
    sample = " ".join(texts[:5])[:2000]
    try:
        return detect(sample)
    except Exception:
        return "en"


def classify_domain(texts: List[str]) -> str:
    joined = " ".join(texts[:50]).lower()
    scores = {d: sum(1 for kw in kws if kw in joined) for d, kws in DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def classify_content_type(text: str) -> str:
    """Classify a string into UI / Warning / Error / Info."""
    if _WARN_TERMS.search(text):
        return "Warning"
    if _ERROR_TERMS.search(text):
        return "Error"
    if _UI_TERMS.search(text) or len(text.split()) <= 6:
        return "UI"
    return "Info"


def get_text_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode("utf-8", errors="ignore")).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE INTELLIGENT SAMPLING
#
# Sample budget based on corpus size:
#   ≤ 15 unique   → use all
#   16–50         → up to 10
#   51–200        → up to 15
#   201–1000      → up to 20
#   > 1000        → up to 30
#
# Within budget: format-tokens → high-risk warnings/errors → length extremes
#   → one question → length strata → lexical diversity fill
# ══════════════════════════════════════════════════════════════════════════════
def _adaptive_budget(n_unique: int) -> int:
    if n_unique <= 15:  return n_unique
    if n_unique <= 50:  return 10
    if n_unique <= 200: return 15
    if n_unique <= 1000: return 20
    return 30


def select_samples(unique_texts: List[str], n: Optional[int] = None) -> Dict[str, Any]:
    """
    Adaptive stratified sampling. When corpus is small, uses all segments.
    When corpus is large, samples representatively across multiple strata.
    """
    if not unique_texts:
        return {"samples": [], "coverage": {}, "selection_reasons": {}}

    budget = n if n is not None else _adaptive_budget(len(unique_texts))
    budget = max(1, min(budget, len(unique_texts)))

    # Entire corpus fits within budget
    if len(unique_texts) <= budget:
        wc = [len(t.split()) for t in unique_texts]
        cov = {
            "total_unique": len(unique_texts),
            "selected": len(unique_texts),
            "budget": budget,
            "with_format_tokens": sum(1 for t in unique_texts if _FORMAT_TOKEN.search(t)),
            "avg_word_count": round(sum(wc) / len(wc), 1) if wc else 0,
            "strategy": f"all segments included (corpus ≤ budget of {budget})",
        }
        return {
            "samples": unique_texts[:],
            "coverage": cov,
            "selection_reasons": {t: "included (full corpus)" for t in unique_texts},
        }

    chosen: List[str] = []
    reasons: Dict[str, str] = {}

    def _add(text: str, reason: str) -> bool:
        if text not in reasons and len(chosen) < budget:
            chosen.append(text)
            reasons[text] = reason
            return True
        return False

    # 1 ── Format-token segments ──────────────────────────────────────────────
    token_texts = [t for t in unique_texts if _FORMAT_TOKEN.search(t)]
    for t in token_texts[:max(1, budget // 3)]:
        _add(t, "contains format tokens — placeholder/tag/variable preservation test")

    # 2 ── High-risk: warning and error strings ───────────────────────────────
    for t in unique_texts:
        if len(chosen) >= budget: break
        if _WARN_TERMS.search(t):
            _add(t, "warning string — meaning accuracy and safety critical")
    for t in unique_texts:
        if len(chosen) >= budget: break
        if _ERROR_TERMS.search(t):
            _add(t, "error string — precision and user-impact test")

    # 3 ── Length extremes ────────────────────────────────────────────────────
    by_len = sorted(unique_texts, key=lambda t: len(t))
    _add(by_len[0],  "shortest segment — UI label / terse string")
    _add(by_len[-1], "longest segment — paragraph / complex structure")

    # 4 ── One question ───────────────────────────────────────────────────────
    for t in unique_texts:
        if t.strip().endswith("?"):
            _add(t, "interrogative — fluency and structure stress test")
            break

    # 5 ── Length strata ──────────────────────────────────────────────────────
    n_strata = max(3, budget // 5)
    stratum_size = max(1, len(by_len) // n_strata)
    for i in range(n_strata):
        bucket = by_len[i * stratum_size: (i + 1) * stratum_size]
        if bucket and len(chosen) < budget:
            _add(bucket[len(bucket) // 2], f"stratum {i+1}/{n_strata} length representative")

    # 6 ── Lexical diversity fill ─────────────────────────────────────────────
    def _tokset(t: str) -> set:
        return set(re.sub(r'[^\w\s]', '', t.lower()).split())

    chosen_toks = [_tokset(t) for t in chosen]
    remaining = [t for t in unique_texts if t not in reasons]

    while len(chosen) < budget and remaining:
        best, best_score = None, 1e9
        for t in remaining:
            ts = _tokset(t)
            if not ts:
                continue
            overlap = max(
                (len(ts & cs) / max(1, len(ts | cs)) for cs in chosen_toks),
                default=0.0,
            )
            if overlap < best_score:
                best_score, best = overlap, t
        if best is None:
            break
        _add(best, "lexically diverse — distinct vocabulary from other samples")
        chosen_toks.append(_tokset(best))
        remaining.remove(best)

    # Sequential fallback
    for t in unique_texts:
        if len(chosen) >= budget: break
        _add(t, "sequential fill")

    wc = [len(t.split()) for t in chosen]
    coverage = {
        "total_unique":       len(unique_texts),
        "selected":           len(chosen),
        "budget":             budget,
        "with_format_tokens": sum(1 for t in chosen if _FORMAT_TOKEN.search(t)),
        "warnings":           sum(1 for t in chosen if _WARN_TERMS.search(t)),
        "errors":             sum(1 for t in chosen if _ERROR_TERMS.search(t)),
        "questions":          sum(1 for t in chosen if t.strip().endswith("?")),
        "avg_word_count":     round(sum(wc) / len(wc), 1) if wc else 0,
        "min_word_count":     min(wc) if wc else 0,
        "max_word_count":     max(wc) if wc else 0,
        "strategy": (
            f"adaptive stratified ({len(chosen)}/{len(unique_texts)} unique): "
            "format-tokens + high-risk + length-extremes + strata + diversity"
        ),
    }
    return {"samples": chosen, "coverage": coverage, "selection_reasons": reasons}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def _extract_reference_map(df: pd.DataFrame, source_col: str, target_language: str) -> Dict[str, str]:
    """
    Detect ground-truth reference columns (ref_de, ref_fr, ref_ja, etc.) and
    return a dict mapping source text → reference translation for the given target_language.

    Column naming conventions supported:
        ref_de / ref_deu / reference_de / ground_truth_de / gt_de
    """
    lang_variants = {
        "de":  ["de", "deu", "ger", "german", "deutsch"],
        "fr":  ["fr", "fra", "fre", "french", "francais"],
        "es":  ["es", "esp", "spa", "spanish", "espanol"],
        "ja":  ["ja", "jpa", "jpn", "japanese"],
        "ko":  ["ko", "kor", "korean"],
        "zh":  ["zh", "chs", "chi", "zho", "chinese"],
        "it":  ["it", "ita", "italian"],
        "pt":  ["pt", "por", "portuguese"],
        "nl":  ["nl", "nld", "dutch"],
        "ar":  ["ar", "ara", "arabic"],
        "ru":  ["ru", "rus", "russian"],
    }
    prefixes = ["ref_", "reference_", "ground_truth_", "gt_", "target_"]
    col_lower = {c.lower(): c for c in df.columns}
    variants = lang_variants.get(target_language, [target_language])

    ref_col = None
    for prefix in prefixes:
        for var in variants:
            candidate = prefix + var
            if candidate in col_lower:
                ref_col = col_lower[candidate]
                break
        if ref_col:
            break

    if ref_col is None:
        return {}

    logger.info(f"Dataset Agent: Found reference column '{ref_col}' for language '{target_language}'")
    ref_map: Dict[str, str] = {}
    for _, row in df.iterrows():
        src = str(row.get(source_col, "")).strip()
        ref = str(row.get(ref_col, "")).strip()
        if src and ref and ref not in ("nan", "None", ""):
            ref_map[src] = ref
    return ref_map


def run(content: bytes, target_language: str, filename: str = "upload") -> Dict[str, Any]:
    logger.info("Dataset Agent: Starting analysis")

    # Parse any supported format (CSV, JSON, TSV, TXT)
    df = parse_any_format(content, filename)
    source_col = find_source_column(df)

    texts: List[str] = df[source_col].dropna().astype(str).str.strip().tolist()
    texts = [t for t in texts if t and len(t) > 1]
    total_rows = len(texts)

    if total_rows == 0:
        raise ValueError(
            f"No usable text found in column '{source_col}'. "
            "Please check your file format and ensure it contains translatable strings."
        )

    # Deduplicate preserving order
    seen: set = set()
    unique_texts: List[str] = []
    for t in texts:
        h = get_text_hash(t)
        if h not in seen:
            seen.add(h)
            unique_texts.append(t)

    duplicate_count = total_rows - len(unique_texts)

    src_lang = detect_language(unique_texts)
    domain = classify_domain(unique_texts)
    content_types = sorted({classify_content_type(t) for t in unique_texts[:30]})

    # Adaptive sampling (budget auto-scales with corpus size)
    sample_result = select_samples(unique_texts)
    samples = sample_result["samples"]

    # Ground truth reference map: source_text → reference_translation (if available)
    reference_map = _extract_reference_map(df, source_col, target_language)
    has_ground_truth = bool(reference_map)

    # Per-sample context
    sample_contexts = [
        {
            "text": s,
            "content_type": classify_content_type(s),
            "has_tokens": bool(_FORMAT_TOKEN.search(s)),
            "word_count": len(s.split()),
            "has_reference": s in reference_map,
        }
        for s in samples
    ]

    logger.info(
        f"Dataset Agent: total={total_rows}, unique={len(unique_texts)}, "
        f"lang={src_lang}, domain={domain}, samples={len(samples)}"
    )

    return {
        "total_rows":                total_rows,
        "unique_rows":               len(unique_texts),
        "duplicate_count":           duplicate_count,
        "detected_source_language":  src_lang,
        "source_language_name":      LANGUAGE_NAMES.get(src_lang, src_lang.upper()),
        "target_language":           target_language,
        "target_language_name":      LANGUAGE_NAMES.get(target_language, target_language.upper()),
        "detected_domain":           domain,
        "domain_display":            DOMAIN_DISPLAY.get(domain, domain),
        "content_types":             content_types,
        "samples_selected":          len(samples),
        "sample_texts":              samples,
        "sample_contexts":           sample_contexts,
        "sample_coverage":           sample_result["coverage"],
        "sample_selection_reasons":  sample_result["selection_reasons"],
        "column_used":               source_col,
        "total_columns":             len(df.columns),
        "columns":                   list(df.columns),
        "file_format":               filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown",
        "has_ground_truth":          has_ground_truth,
        "reference_map":             reference_map,
        "ground_truth_coverage":     f"{len(reference_map)}/{total_rows}" if has_ground_truth else "none",
    }
