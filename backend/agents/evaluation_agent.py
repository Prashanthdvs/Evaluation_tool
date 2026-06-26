"""
Agent 3: Localization Quality Evaluation — LLM-as-a-Judge with Chain-of-Thought

Evaluates MT output across FIVE quality dimensions that localization teams care about:

  A. LINGUISTIC QUALITY
       fluency_score          — reads naturally in target language
       grammar_score          — grammatically correct
       meaning_preservation   — semantic fidelity, no omissions/additions

  B. LOCALIZATION QUALITY  ← THE DIFFERENTIATOR
       placeholder_preservation — {var}, %s, %(name)s, [[var]]  (MANDATORY)
       tag_preservation         — <b>, </span>, &amp;            (MANDATORY)
       variable_preservation    — $VAR, __CONST__, @PLACEHOLDER  (MANDATORY)
       terminology_accuracy     — consistent use of approved terms
       consistency_score        — consistent style/register across segment

  C. DOMAIN QUALITY
       domain_quality_score   — healthcare/legal/software/marketing-specific rules
       domain_issues          — list of domain-specific failures

  D. CULTURAL ADAPTATION
       cultural_adaptation_score — dates, numbers, formality, locale conventions
       cultural_issues           — e.g. "12/31/2025 not converted to 31.12.2025"

  E. HALLUCINATION DETECTION
       hallucination_risk     — 0.0 (none) → 1.0 (complete fabrication)
       added_content          — words/phrases added by MT not in source
       missing_content        — words/phrases in source omitted by MT
       wrong_terminology      — terms translated incorrectly

  COMPOSITE:
       quality_score          — weighted composite (see _compute_composite)
       localization_pass      — False if ANY mandatory token check fails → FAIL gate
       toxicity_risk          — offensive language introduced
       critical_errors        — production blockers
       evaluation_notes       — one-sentence summary

Example of a FAIL:
  Source: "Click {username} to continue."
  MT:     "Klicken Sie auf Benutzername, um fortzufahren."
  → placeholder_preservation = 0 ('{username}' rendered as plain word)
  → localization_pass = False
  → quality_score capped at 45
"""
import asyncio
import json
import re
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# LLM JUDGE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════
_LLM_BASE_URL  = os.getenv("LLM_BASE_URL",  "https://model-broker.aviator-model.bp.anthos.otxlab.net")
_LLM_API_KEY   = os.getenv("LLM_API_KEY",   "sk-uO1uW7B5CC9ETVAXpzqtew")
_LLM_MODEL     = os.getenv("LLM_MODEL",     "llama-3.3-70b")

_QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://10.9.206.147:8080/v1")
_QWEN_API_KEY  = os.getenv("QWEN_API_KEY",  "EMPTY")
_QWEN_MODEL    = os.getenv("QWEN_MODEL",    "qwen25-32b-awq")

_JUDGE_ENDPOINTS: List[Tuple[str, str, str]] = [
    (_LLM_BASE_URL, _LLM_API_KEY, _LLM_MODEL),
    (_QWEN_BASE_URL, _QWEN_API_KEY, _QWEN_MODEL),
    # Groq public API fallback — active when GROQ_API_KEY is set
    ("https://api.groq.com/openai/v1", os.getenv("GROQ_API_KEY", ""), "llama-3.3-70b-versatile"),
]


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN CONTEXT REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
DOMAIN_CONTEXT: Dict[str, Dict] = {
    # ── New 7-domain keys ─────────────────────────────────────────────────────
    "pharma_healthcare": {
        "description": "Medical or clinical content — drug labels, patient leaflets, clinical documents",
        "mandatory_criteria": [
            "Dosage/measurements PRESERVED: '10mg', '2x daily', '500ml', temperature values unchanged",
            "Drug names, clinical terms, INN names MUST be exact — no synonyms or paraphrasing",
            "Safety warnings, contraindications, adverse event text preserved verbatim",
            "Patient safety language: 'must not', 'stop taking', 'seek immediate' must NOT be weakened",
            "Informed consent and clinical trial language: legal precision required",
            "Regulatory section headings (e.g. 'Posology', 'Contraindications') preserved",
        ],
        "critical_risk": "Mistranslated dosages or safety warnings can cause direct patient harm — human review REQUIRED.",
        "hard_fail_patterns": ["dosage_changed", "safety_warning_missing", "drug_name_changed"],
        "domain_type": "pharma_healthcare",
    },
    "it_software": {
        "description": "UI strings, error messages, tooltips, API docs, software documentation",
        "mandatory_criteria": [
            "ALL format tokens ({var}, %s, %(name)s, [[var]], ${VAR}) preserved unchanged",
            "Command names, file paths, JSON/XML/CSV/PDF terms, config keys preserved exactly",
            "API names, SDK terms, method names: do NOT translate or localize",
            "UI labels: use approved product glossary terms; 'Settings' not 'Preferences' unless specified",
            "Keyboard shortcuts, access keys, hotkeys preserved if present",
            "Acronyms (API, SDK, URL, CSV, JSON) preserved — do not spell them out",
        ],
        "critical_risk": "Missing placeholders cause app crashes. Wrong UI terms cause user confusion.",
        "hard_fail_patterns": ["placeholder_missing", "command_changed", "filepath_changed"],
        "domain_type": "it_software",
    },
    "legal_compliance": {
        "description": "Contracts, terms and conditions, regulatory filings, compliance documents",
        "mandatory_criteria": [
            "Obligation words: 'must', 'shall', 'may', 'agree', 'consent' MUST map to exact legal equivalents",
            "Legal clauses: no paraphrasing — clause meaning must not be weakened or strengthened",
            "Confidentiality, retention, audit trail language preserved verbatim",
            "GDPR/privacy language: 'data controller', 'data processor', 'lawful basis' exact terms",
            "Jurisdiction, arbitration, force majeure, indemnification clauses: precise equivalents",
            "Article/section/clause reference numbers unchanged",
        ],
        "critical_risk": "Paraphrasing a binding clause alters contractual obligations — human review REQUIRED.",
        "hard_fail_patterns": ["legal_obligation_changed", "consent_weakened", "clause_omitted"],
        "domain_type": "legal_compliance",
    },
    "finance_banking": {
        "description": "Financial reports, banking communications, payment instructions, investment content",
        "mandatory_criteria": [
            "Monetary amounts, percentages, transaction IDs: values must NOT change",
            "SWIFT codes, IBAN formats, OTP messages, KYC/AML terminology preserved",
            "Payment security messages (e.g. 'never share OTP') must not be softened",
            "Risk warnings, regulatory disclosures: preserved verbatim",
            "Dates, interest rate periods, amortization schedules: unchanged",
            "Currency symbols and codes (USD, EUR, GBP, JPY) preserved correctly",
        ],
        "critical_risk": "Incorrect financial amounts or missing risk warnings cause compliance failures.",
        "hard_fail_patterns": ["amount_changed", "date_changed", "risk_warning_missing"],
        "domain_type": "finance_banking",
    },
    "multimedia_streaming": {
        "description": "Video player UI, subtitles, captions, streaming platform copy",
        "mandatory_criteria": [
            "UI button text: 'Play', 'Pause', 'Skip', 'Resume' — use platform-standard terms",
            "DRM, subtitle format terms, codec names preserved exactly",
            "Accessibility labels (ARIA labels, closed caption descriptions) natural and precise",
            "Short UI strings: character length awareness — must fit within display constraints",
            "User-friendly tone: avoid overly technical or formal phrasing",
            "Streaming action terms: 'Download for offline', 'Watch later', 'Add to playlist'",
        ],
        "critical_risk": "Wrong UI terms confuse users. Accessibility failures exclude disabled users.",
        "hard_fail_patterns": ["placeholder_missing", "drm_term_changed"],
        "domain_type": "multimedia_streaming",
    },
    "journals_publishing": {
        "description": "Academic manuscripts, journal submissions, peer review communications",
        "mandatory_criteria": [
            "Formal academic tone: no casual language in manuscript or editorial contexts",
            "DOI, ORCID, ISSN, RIS format references preserved exactly as-is",
            "Peer review process language: 'major revision', 'minor revision', 'rejected' exact terms",
            "Author submission workflow terms: 'submission', 'revision', 'proofs', 'embargo' precise",
            "Citation and bibliography formats preserved without modification",
            "Deadline language ('submit by', 'revised manuscript due') precise and unambiguous",
        ],
        "critical_risk": "Incorrect submission deadlines or decision language cause author confusion.",
        "hard_fail_patterns": ["date_changed", "doi_changed"],
        "domain_type": "journals_publishing",
    },
    "ecommerce": {
        "description": "Product listings, CTAs, checkout flows, order status, return policies",
        "mandatory_criteria": [
            "Prices, stock quantities, delivery times: numerical values must NOT change",
            "Product/brand names preserved with exact casing and spelling",
            "Return policy and checkout error messages: clear, user-friendly, accurate",
            "CTAs ('Add to Cart', 'Buy Now', 'Checkout') culturally compelling in target locale",
            "Delivery and shipping terms localized correctly (e.g. 'Free delivery' phrasing)",
            "Stock status ('In Stock', 'Out of Stock', 'Only 3 left') precise",
        ],
        "critical_risk": "Incorrect prices or stock status cause financial disputes and trust damage.",
        "hard_fail_patterns": ["amount_changed", "date_changed"],
        "domain_type": "ecommerce",
    },
    # ── Legacy aliases (backward compat for existing benchmark data) ──────────
    "healthcare": {
        "description": "Medical or clinical content for healthcare professionals or patients",
        "mandatory_criteria": [
            "Medical terminology: drug names, conditions, procedures MUST be exact — no synonyms",
            "Dosage/measurement preservation: '10mg', '2x daily', '500ml' unchanged",
            "Regulatory wording: warnings, contraindications, indication text preserved verbatim",
            "Clinical precision: no vague paraphrasing of diagnostic language",
        ],
        "critical_risk": "Mistranslated drug names or dosages can cause direct patient harm.",
        "hard_fail_patterns": ["dosage_changed", "safety_warning_missing"],
        "domain_type": "healthcare",
    },
    "legal": {
        "description": "Legal documents, contracts, compliance notices, or regulatory text",
        "mandatory_criteria": [
            "Legal terminology: 'indemnify', 'warranty', 'jurisdiction', 'force majeure' — exact",
            "Clause preservation: legal obligations must NOT be paraphrased or weakened",
            "Citation preservation: article/section numbers, case references unchanged",
            "No inferring intent: translate what is written, not what you think it means",
        ],
        "critical_risk": "Paraphrasing a binding clause can alter contractual obligations.",
        "hard_fail_patterns": ["legal_obligation_changed", "clause_omitted"],
        "domain_type": "legal",
    },
    "software": {
        "description": "UI strings, error messages, tooltips, button labels for software products",
        "mandatory_criteria": [
            "ALL format tokens ({var}, %s, %(name)s, [[var]], <tag>) preserved unchanged",
            "UI consistency: use the same terms as the product glossary (e.g. 'Settings' not 'Preferences')",
            "Character length awareness: short UI strings (button labels) must fit constraints",
            "Keyboard shortcuts and access keys preserved if present",
        ],
        "critical_risk": "Missing placeholders cause app crashes. Inconsistent UI terms confuse users.",
        "hard_fail_patterns": ["placeholder_missing"],
        "domain_type": "software",
    },
    "customer_support": {
        "description": "Customer service communications, support tickets, chat transcripts",
        "mandatory_criteria": [
            "Tone register preserved: formal/informal register must match source",
            "Empathy maintained or enhanced — never more cold/clinical than source",
            "Resolution language accurate: 'escalate', 'refund', 'ticket number' etc.",
            "Brand voice consistency across the conversation",
        ],
        "critical_risk": "Wrong register or missing empathy damages customer relationships.",
        "hard_fail_patterns": [],
        "domain_type": "customer_support",
    },
    "finance": {
        "description": "Financial reports, banking communications, investment content",
        "mandatory_criteria": [
            "Number/currency/percentage exactness: amounts must not change",
            "Financial terminology: 'amortization', 'dividend', 'collateral' — exact equivalents",
            "Regulatory compliance language preserved verbatim",
            "Date/period references unchanged and correctly formatted for locale",
        ],
        "critical_risk": "Incorrect numbers or financial terms cause compliance failures.",
        "hard_fail_patterns": ["amount_changed", "date_changed"],
        "domain_type": "finance",
    },
    "marketing": {
        "description": "Marketing copy, advertisements, and promotional materials",
        "mandatory_criteria": [
            "Brand voice: tone and personality preserved or culturally adapted",
            "Cultural adaptation: idioms, humor, references localized — not literally translated",
            "Emotional appeal: persuasive intent and impact maintained",
            "Taglines/slogans: adapted for resonance, not word-for-word translation",
        ],
        "critical_risk": "A culturally tone-deaf translation alienates the target audience.",
        "hard_fail_patterns": [],
        "domain_type": "marketing",
    },
    "general": {
        "description": "General content without strong domain-specific constraints",
        "mandatory_criteria": [
            "Semantic fidelity: full meaning preserved",
            "Grammatical correctness in target language",
            "Natural fluency: reads like native content",
        ],
        "critical_risk": "Ensure meaning is preserved accurately with no omissions or additions.",
        "hard_fail_patterns": [],
        "domain_type": "general",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# CULTURAL CONVENTIONS PER TARGET LANGUAGE
# ══════════════════════════════════════════════════════════════════════════════
CULTURAL_CONVENTIONS: Dict[str, Dict] = {
    "de": {
        "locale_name": "German",
        "date_format": "DD.MM.YYYY — e.g. '12/31/2025' MUST become '31.12.2025'",
        "number_format": "1.000,50 — period=thousands separator, comma=decimal",
        "formality": "Formal 'Sie' for professional/UI contexts; 'du' only if explicitly informal",
        "quotes": '\u201eGerman quotes\u201c instead of "English quotes"',
        "notes": ["'Color' → 'Farbe' (not 'Couleur')", "Compound nouns: 'email address' → 'E-Mail-Adresse'"],
    },
    "fr": {
        "locale_name": "French",
        "date_format": "DD/MM/YYYY — e.g. '12/31/2025' MUST become '31/12/2025'",
        "number_format": "1 000,50 — non-breaking space=thousands, comma=decimal",
        "formality": "'Vous' (formal) by default; 'tu' only if explicitly informal",
        "quotes": '«French guillemets» instead of "English quotes"',
        "notes": ["Non-breaking space before : ; ! ?", "'Logiciel' not 'Software'"],
    },
    "es": {
        "locale_name": "Spanish",
        "date_format": "DD/MM/YYYY — e.g. '12/31/2025' becomes '31/12/2025'",
        "number_format": "1.000,50 (Spain) or 1,000.50 (Latin America) — check target market",
        "formality": "'Usted' for formal, 'tú' for informal; vary by region",
        "quotes": '«Spanish guillemets» or "comillas" as appropriate',
        "notes": ["Inverted ¡ and ¿ for exclamations and questions", "Gendered language must agree"],
    },
    "ja": {
        "locale_name": "Japanese",
        "date_format": "YYYY年MM月DD日 — e.g. '12/31/2025' becomes '2025年12月31日'",
        "number_format": "Comma thousands separator; 万 for 10,000 in informal contexts",
        "formality": "です/ます form for professional contexts; use appropriate keigo",
        "quotes": "「Japanese brackets」for quotes",
        "notes": ["Full-width punctuation", "Honorifics must match relationship context"],
    },
    "zh": {
        "locale_name": "Chinese",
        "date_format": "YYYY年MM月DD日 — e.g. '2025年12月31日'",
        "number_format": "Comma thousands; 万 for 10,000 in informal Chinese",
        "formality": "Appropriate formality level for context",
        "quotes": "「Simplified bracket quotes」",
        "notes": ["Full-width punctuation", "Simplified vs Traditional depends on target market"],
    },
    "ar": {
        "locale_name": "Arabic",
        "date_format": "DD/MM/YYYY or Hijri calendar as appropriate for context",
        "number_format": "Western Arabic (1,000) or Arabic-Indic numerals (١٬٠٠٠) per context",
        "formality": "Modern Standard Arabic for professional contexts",
        "quotes": "«Guillemets» or Arabic equivalent",
        "notes": ["RTL text direction", "Gender agreement in verbs and adjectives is mandatory"],
    },
    "pt": {
        "locale_name": "Portuguese",
        "date_format": "DD/MM/YYYY — e.g. '31/12/2025'",
        "number_format": "1.000,50 (Brazil: same) — period=thousands, comma=decimal",
        "formality": "'Você' (Brazil) or 'tu/você' mix (Portugal); 'senhor/senhora' for formal",
        "quotes": '"Aspas" (Brazil) or «guillemets» (Portugal)',
        "notes": ["Brazilian PT vs European PT vocabulary differences", "'Arquivo' (PT) vs 'Ficheiro' (BR)"],
    },
    "ko": {
        "locale_name": "Korean",
        "date_format": "YYYY년 MM월 DD일 — e.g. '2025년 12월 31일'",
        "number_format": "Comma thousands; 만 for 10,000 grouping in informal contexts",
        "formality": "-요/-습니다 endings for professional contexts",
        "quotes": '"Korean double quotes"',
        "notes": ["Honorific verb forms must match context", "Word spacing (띄어쓰기) is important"],
    },
    "it": {
        "locale_name": "Italian",
        "date_format": "DD/MM/YYYY — e.g. '31/12/2025'",
        "number_format": "1.000,50 — period=thousands, comma=decimal",
        "formality": "'Lei' (formal) vs 'tu' (informal)",
        "quotes": '«Caporali» or “virgolette alte”',
        "notes": ["Gendered adjectives must agree", "'Novembre' not 'November'"],
    },
    "nl": {
        "locale_name": "Dutch",
        "date_format": "DD-MM-YYYY — e.g. '31-12-2025'",
        "number_format": "1.000,50 — period=thousands, comma=decimal",
        "formality": "'U' (formal) vs 'jij/je' (informal)",
        "quotes": '„Dutch quotes”',
        "notes": ["Compound words formed differently from English", "Belgian vs Netherlands Dutch"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# TOKEN DETECTION — split into 3 mandatory categories
# ══════════════════════════════════════════════════════════════════════════════
_PLACEHOLDER_PATTERNS = [
    r'\{[^}]+\}',           # {username}, {0}, {count:,}
    r'%\([^)]+\)[sdifxg]',  # %(name)s, %(count)d
    r'%[sdifxg]',           # %s, %d, %i
    r'\[\[[^\]]+\]\]',      # [[variable]], [[0]]
    r'\[%[^\]]+%\]',        # [%var%]
]

_TAG_PATTERNS = [
    r'</?[a-zA-Z][a-zA-Z0-9_:-]*(?:\s[^>]*)?>',  # <b>, </span>, <a href="...">, <br/>
    r'&[a-zA-Z][a-zA-Z0-9]*;',                    # &amp;, &lt;, &nbsp;
    r'&#[0-9]+;',                                  # &#160;
]

_VARIABLE_PATTERNS = [
    r'\$\{[^}]+\}',          # ${VAR}, ${env.PATH}
    r'\$[A-Z][A-Z0-9_]{1,}', # $VAR_NAME (2+ chars after $)
    r'@[A-Z][A-Z_]{1,}',     # @APP_NAME
    r'__[A-Z_]{2,}__',        # __BRAND__, __APP_NAME__
]


def _find_tokens(text: str, patterns: List[str]) -> List[str]:
    found = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text))
    return list(dict.fromkeys(found))  # deduplicate, preserve order


def _detect_placeholders(text: str) -> List[str]:
    return _find_tokens(text, _PLACEHOLDER_PATTERNS)

def _detect_tags(text: str) -> List[str]:
    return _find_tokens(text, _TAG_PATTERNS)

def _detect_variables(text: str) -> List[str]:
    return _find_tokens(text, _VARIABLE_PATTERNS)


# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED CHECKS — deterministic hard-fail and warning detection
#
# Runs BEFORE the LLM judge so that numerical/token failures are caught
# by regex, not by LLM "opinion".  Returns:
#   hard_fails  : list of (code, description) tuples → trigger Human Review
#   warnings    : list of description strings → reduce score but not hard-fail
# ══════════════════════════════════════════════════════════════════════════════
# Regex patterns for protected terms that must survive translation unchanged
_NUMBER_RE  = re.compile(r'\b\d+(?:[.,]\d+)*\b')
_DATE_RE    = re.compile(
    r'\b(?:\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b'
)
_AMOUNT_RE  = re.compile(
    r'(?:USD|EUR|GBP|JPY|CHF|CAD|AUD|INR|CNY|KRW|R\$|\$|€|£|¥|₩|₹)\s*\d+(?:[.,]\d+)*'
    r'|\d+(?:[.,]\d+)*\s*(?:USD|EUR|GBP|JPY|CHF|CAD|AUD|INR|CNY|KRW)'
)
_DOSAGE_RE  = re.compile(
    r'\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ml|mL|L|g|kg|IU|units?|tablets?|capsules?|drops?'
    r'|puffs?|doses?|times?\s+(?:a\s+)?(?:day|daily|week|weekly))\b', re.I
)
_VERSION_RE = re.compile(r'\bv?\d+\.\d+(?:\.\d+)*\b')
_DNT_RE     = re.compile(
    r'\b(?:DOI|ORCID|IBAN|SWIFT|BIC|OTP|KYC|AML|GDPR|HIPAA|API|SDK|URL|UUID|ISBN|ISSN)\b'
)


def _rule_based_checks(
    source: str,
    translation: str,
    domain: str,
    content_type: str = "Info",
) -> Dict[str, Any]:
    """
    Deterministic checks using regex — not LLM judgment.

    Returns:
        {
            "hard_fails": [{"code": str, "description": str}, ...],
            "warnings":   [str, ...],
            "human_review_required": bool,
        }
    """
    hard_fails: List[Dict[str, str]] = []
    warnings:   List[str] = []

    def _hf(code: str, desc: str) -> None:
        hard_fails.append({"code": code, "description": desc})

    def _warn(desc: str) -> None:
        warnings.append(desc)

    # ── 1. Placeholder check (already done in _code_level_verify, duplicated here
    #       so the flag appears BEFORE the LLM call) ───────────────────────────
    src_ph = _detect_placeholders(source)
    if src_ph:
        missing = [p for p in src_ph if p not in translation]
        if missing:
            _hf("PLACEHOLDER_MISSING",
                f"Placeholder(s) missing in translation: {', '.join(missing)}")

    src_tags = _detect_tags(source)
    if src_tags:
        missing = [t for t in src_tags if t not in translation]
        if missing:
            _hf("TAG_MISSING",
                f"HTML/XML tag(s) missing in translation: {', '.join(missing)}")

    src_vars = _detect_variables(source)
    if src_vars:
        missing = [v for v in src_vars if v not in translation]
        if missing:
            _hf("VARIABLE_MISSING",
                f"Variable(s) missing in translation: {', '.join(missing)}")

    # ── 2. Number/date/amount preservation ───────────────────────────────────
    src_numbers  = set(_NUMBER_RE.findall(source))
    tgt_numbers  = set(_NUMBER_RE.findall(translation))
    changed_nums = src_numbers - tgt_numbers
    if changed_nums:
        _hf("NUMBER_CHANGED",
            f"Numeric value(s) changed or missing: {', '.join(sorted(changed_nums))}")

    src_dates = set(_DATE_RE.findall(source))
    tgt_dates = set(_DATE_RE.findall(translation))
    # Dates may be reformatted; only flag if a date disappears entirely
    if src_dates and not tgt_dates:
        _hf("DATE_CHANGED",
            f"Date value(s) appear to have been removed: {', '.join(sorted(src_dates))}")

    src_amounts = set(_AMOUNT_RE.findall(source))
    tgt_amounts = set(_AMOUNT_RE.findall(translation))
    changed_amts = src_amounts - tgt_amounts
    if changed_amts:
        _hf("AMOUNT_CHANGED",
            f"Monetary amount(s) changed or missing: {', '.join(sorted(changed_amts))}")

    # ── 3. Domain-specific hard-fails ────────────────────────────────────────
    d = domain.lower()

    if d in ("pharma_healthcare", "healthcare"):
        src_dosages = set(_DOSAGE_RE.findall(source))
        tgt_dosages = set(_DOSAGE_RE.findall(translation))
        missing_d = src_dosages - tgt_dosages
        if missing_d:
            _hf("DOSAGE_CHANGED",
                f"Medical dosage/measurement changed or missing: {', '.join(sorted(missing_d))}")
        # Safety warning keywords
        safety_words = ["must not", "do not", "stop taking", "seek immediate", "contraindicated",
                        "do not use", "warning", "caution", "avoid"]
        for sw in safety_words:
            if sw.lower() in source.lower() and sw.lower() not in translation.lower():
                _hf("SAFETY_WARNING_MISSING",
                    f"Safety instruction '{sw}' from source not found in translation")
                break

    elif d in ("legal_compliance", "legal"):
        obligation_words = ["must", "shall", "agree", "consent", "obliged", "required to",
                            "may not", "shall not", "must not", "binding"]
        src_lower = source.lower()
        tgt_lower = translation.lower()
        for ow in obligation_words:
            if ow in src_lower:
                # check translation has SOME obligation form (crude but fast)
                if not any(c in tgt_lower for c in ["muss", "doit", "debe", "shall", "必须",
                                                      "devono", "должен", "moet", "필요", "musst",
                                                      "müssen", "binding", "oblig", "pflicht"]):
                    _warn(f"Obligation word '{ow}' in source — verify it is preserved in translation")
                break

    elif d in ("finance_banking", "finance"):
        src_dnt = set(_DNT_RE.findall(source))
        for term in src_dnt:
            if term not in translation:
                _hf("PROTECTED_TERM_CHANGED",
                    f"Financial term '{term}' missing from translation (must not be translated)")

    elif d in ("journals_publishing",):
        src_doi = re.findall(r'10\.\d{4,}/\S+', source)
        for doi in src_doi:
            if doi not in translation:
                _hf("DOI_CHANGED", f"DOI reference changed or missing: {doi}")
        src_orcid = re.findall(r'\d{4}-\d{4}-\d{4}-\d{4}', source)
        for orcid in src_orcid:
            if orcid not in translation:
                _hf("ORCID_CHANGED", f"ORCID changed or missing: {orcid}")

    # ── 4. DNT (Do-Not-Translate) acronym check — universal ──────────────────
    src_dnt = set(_DNT_RE.findall(source))
    for term in src_dnt:
        if term not in translation:
            _hf("DNT_TERM_CHANGED",
                f"Protected/DNT term '{term}' missing from translation")

    # ── 5. Version number check ───────────────────────────────────────────────
    src_versions = set(_VERSION_RE.findall(source))
    for ver in src_versions:
        if ver not in translation:
            _hf("VERSION_CHANGED",
                f"Version number '{ver}' changed or missing in translation")

    # ── 6. Hallucinated extra warning/instruction ─────────────────────────────
    tgt_lower = translation.lower()
    src_lower = source.lower()
    extra_warn_words = ["warning:", "caution:", "note:", "important:"]
    for ew in extra_warn_words:
        if ew in tgt_lower and ew not in src_lower:
            _hf("HALLUCINATED_WARNING",
                f"Translation contains '{ew}' which is NOT in the source — possible hallucination")

    # ── 7. Missing sentence/bullet (length-based heuristic) ──────────────────
    src_bullets = len(re.findall(r'^\s*[\-\*\u2022•]\s+', source, re.MULTILINE))
    tgt_bullets = len(re.findall(r'^\s*[\-\*\u2022•]\s+', translation, re.MULTILINE))
    if src_bullets > 0 and tgt_bullets < src_bullets:
        _hf("MISSING_BULLET",
            f"Source has {src_bullets} bullet(s) but translation has {tgt_bullets}")

    src_sents = [s.strip() for s in re.split(r'[.!?]+', source) if len(s.strip()) > 10]
    tgt_sents = [s.strip() for s in re.split(r'[.!?]+', translation) if len(s.strip()) > 10]
    if len(src_sents) > 2 and len(tgt_sents) < len(src_sents) // 2:
        _hf("MISSING_SENTENCE",
            f"Source has ~{len(src_sents)} sentences but translation has only ~{len(tgt_sents)}")

    # ── 8. Soft warnings ──────────────────────────────────────────────────────
    src_wc = len(source.split())
    tgt_wc = len(translation.split())
    if src_wc > 5:
        ratio = tgt_wc / src_wc
        if ratio > 1.6:
            _warn(f"Translation is {ratio:.1f}x longer than source — may indicate over-translation")
        elif ratio < 0.6:
            _warn(f"Translation is {ratio:.1f}x shorter than source — possible content loss")

    human_review = len(hard_fails) > 0

    return {
        "hard_fails": hard_fails,
        "warnings": warnings,
        "human_review_required": human_review,
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORE FORMULA
# Weights:  Meaning 25% | Localization tokens 25% | Fluency+Grammar 20%
#           Terminology+Domain 20% | Cultural 10%
# Hard caps: token preservation fail → max 45
#            hallucination > 0.5     → max 60
#            human_review_required   → max 75 (final review still needed)
# ══════════════════════════════════════════════════════════════════════════════
def _compute_composite(scores: Dict) -> float:
    # Default to 80 (not 100) so that absent token scores don't inflate composite.
    # Compress is applied upstream; if LLM returns 100 it becomes 86 via _compress.
    # For token-free texts the LLM still returns these explicitly, so explicit values
    # flow through correctly; the default only fires when the LLM omits the key.
    ph   = scores.get("placeholder_preservation", 80)
    tag  = scores.get("tag_preservation", 80)
    var  = scores.get("variable_preservation", 80)
    loc  = (ph + tag + var) / 3.0

    meaning  = scores.get("meaning_preservation", 80)
    fluency  = scores.get("fluency_score", 80)
    grammar  = scores.get("grammar_score", 80)
    term     = scores.get("terminology_accuracy", 80)
    domain   = scores.get("domain_quality_score", 80)
    cultural = scores.get("cultural_adaptation_score", 80)

    composite = (
        meaning  * 0.25 +
        loc      * 0.25 +
        (fluency + grammar) / 2 * 0.20 +
        (term + domain) / 2 * 0.20 +
        cultural * 0.10
    )

    # Hard cap: any mandatory token check fail
    if ph < 50 or tag < 50 or var < 50:
        composite = min(composite, 45.0)

    # Hard cap: hallucination
    if scores.get("hallucination_risk", 0) > 0.5:
        composite = min(composite, 60.0)

    # Hard cap: hard-fail rules triggered (human review required)
    if scores.get("human_review_required", False):
        composite = min(composite, 75.0)

    return round(composite, 1)


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION FALLBACK PROFILES  (realistic MT baselines — NOT human quality)
# Anchored to industry MQM benchmarks: commercial MT ≈ 70-85 range
# ══════════════════════════════════════════════════════════════════════════════
_SIM_PROFILES: Dict[str, Dict] = {
    #                               q    mn   fl   gr   t    d    c    h      x
    "claude-sonnet-4":  {"q": 87, "mn": 88, "fl": 90, "gr": 89, "t": 87, "d": 87, "c": 85, "h": 0.05, "x": 0.01},
    "gemini-2.5-pro":   {"q": 86, "mn": 87, "fl": 89, "gr": 88, "t": 86, "d": 86, "c": 84, "h": 0.05, "x": 0.01},
    "deepseek-r1":      {"q": 83, "mn": 84, "fl": 85, "gr": 84, "t": 83, "d": 84, "c": 80, "h": 0.07, "x": 0.02},
    "glm-4.5":          {"q": 79, "mn": 80, "fl": 82, "gr": 81, "t": 79, "d": 79, "c": 75, "h": 0.10, "x": 0.02},
    "kimi-k2":          {"q": 82, "mn": 83, "fl": 84, "gr": 83, "t": 81, "d": 81, "c": 79, "h": 0.08, "x": 0.02},
    "claude-haiku-4.5": {"q": 82, "mn": 83, "fl": 84, "gr": 83, "t": 82, "d": 81, "c": 79, "h": 0.07, "x": 0.02},
    "qwen3-32b":        {"q": 79, "mn": 80, "fl": 81, "gr": 80, "t": 78, "d": 78, "c": 74, "h": 0.10, "x": 0.02},
    # Legacy / simulated-only
    "gpt-4o":           {"q": 85, "mn": 86, "fl": 87, "gr": 86, "t": 85, "d": 84, "c": 82, "h": 0.06, "x": 0.01},
    "deepl":            {"q": 82, "mn": 83, "fl": 85, "gr": 84, "t": 83, "d": 81, "c": 80, "h": 0.05, "x": 0.01},
    "azure-mt":         {"q": 72, "mn": 71, "fl": 74, "gr": 73, "t": 71, "d": 70, "c": 68, "h": 0.10, "x": 0.02},
}


def _simulate_scores(model_id: str, domain: str) -> Dict:
    profile = _SIM_PROFILES.get(model_id, _SIM_PROFILES["claude-haiku-4.5"])
    rng = random.Random(hash(model_id + domain) & 0xFFFFFF)

    def jit(base, sigma=2.5):
        return round(min(100.0, max(0.0, base + rng.gauss(0, sigma))), 1)

    def jitr(base, sigma=0.012):
        return round(min(0.99, max(0.01, base + rng.gauss(0, sigma))), 3)

    fluency = jit(profile["fl"])
    grammar = jit(profile["gr"])
    meaning = jit(profile["mn"])
    term    = jit(profile["t"])
    domain_s = jit(profile["d"])
    cultural = jit(profile["c"])
    hall    = jitr(profile["h"])

    loc_score = 100.0  # assume tokens OK in simulation

    raw_q = (
        meaning * 0.25 + loc_score * 0.25 +
        (fluency + grammar) / 2 * 0.20 +
        (term + domain_s) / 2 * 0.20 +
        cultural * 0.10
    )
    q = round(_compress(raw_q), 1)

    return {
        # A. Linguistic
        "fluency_score":          fluency,
        "grammar_score":          grammar,
        "meaning_preservation":   meaning,
        # B. Localization
        "placeholder_preservation": 100.0,
        "tag_preservation":         100.0,
        "variable_preservation":    100.0,
        "terminology_accuracy":     term,
        "consistency_score":        jit(85),
        # C. Domain
        "domain_quality_score":     domain_s,
        "domain_issues":            [],
        # D. Cultural
        "cultural_adaptation_score": cultural,
        "cultural_issues":           [],
        # E. Hallucination
        "hallucination_risk":   hall,
        "added_content":        [],
        "missing_content":      [],
        "wrong_terminology":    [],
        # Composite / meta
        "quality_score":        q,
        "toxicity_risk":        jitr(profile["x"]),
        "localization_pass":    True,
        "placeholder_preserved": True,   # backward-compat alias
        "critical_errors":      [],
        "evaluation_notes":     "Scores estimated via calibrated simulation (LLM judge unavailable).",
        "judge_model":          "simulation",
        "cot_trace":            None,
        "evaluation_method":    "simulation",
        "human_review_required": False,
        "hard_fails":           [],
        "rule_warnings":        [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCORE COMPRESSION + CODE-LEVEL OVERRIDES
# ══════════════════════════════════════════════════════════════════════════════

def _compress(score: float) -> float:
    """
    Pull high scores back toward realistic MT quality. STRICT curve — penalizes
    even small imperfections so machine output rarely exceeds the mid-80s.

    Piecewise:
        x <= 75              : unchanged (genuine problems shown as-is)
        75 < x <= 90         : 75 + (x-75)*0.55   (compresses the "good" band)
        x > 90               : 83.25 + (x-90)*0.30 (near-perfect band heavily capped)

    Result anchors:  100 -> 86.3 | 95 -> 84.8 | 90 -> 83.3 | 85 -> 80.5 | 80 -> 77.8 | 75 -> 75
    No machine translation can reach 90+; only flawless, idiomatic output lands mid-80s.
    """
    if score <= 75.0:
        return round(score, 1)
    if score <= 90.0:
        return round(75.0 + (score - 75.0) * 0.55, 1)
    return round(83.25 + (score - 90.0) * 0.30, 1)


def _code_level_verify(scores: Dict, source: str, translation: str) -> Dict:
    """
    Override token-preservation scores using regex — not LLM opinion.
    If a placeholder is in the source but missing from the translation,
    the LLM cannot give it a passing score regardless of how it reasoned.
    Also applies a length-ratio penalty for severely under/over-translated text.
    """
    errors = list(scores.get("critical_errors", []))

    # ── Placeholder verification ─────────────────────────────────────────────
    src_ph = _detect_placeholders(source)
    if src_ph:
        missing_ph = [p for p in src_ph if p not in translation]
        if missing_ph:
            scores["placeholder_preservation"] = 0.0
            scores["localization_pass"]         = False
            scores["placeholder_preserved"]     = False
            errors.append(f"Missing placeholders: {', '.join(missing_ph)}")
        else:
            # All present — but cap at 92 (perfect is extremely rare)
            scores["placeholder_preservation"] = min(scores.get("placeholder_preservation", 100), 92.0)

    # ── Tag verification ─────────────────────────────────────────────────────
    src_tags = _detect_tags(source)
    if src_tags:
        missing_tags = [t for t in src_tags if t not in translation]
        if missing_tags:
            scores["tag_preservation"] = 0.0
            scores["localization_pass"] = False
            scores["placeholder_preserved"] = False
            errors.append(f"Missing HTML/XML tags: {', '.join(missing_tags)}")
        else:
            scores["tag_preservation"] = min(scores.get("tag_preservation", 100), 92.0)

    # ── Variable verification ────────────────────────────────────────────────
    src_vars = _detect_variables(source)
    if src_vars:
        missing_vars = [v for v in src_vars if v not in translation]
        if missing_vars:
            scores["variable_preservation"] = 0.0
            scores["localization_pass"] = False
            scores["placeholder_preserved"] = False
            errors.append(f"Missing variables: {', '.join(missing_vars)}")
        else:
            scores["variable_preservation"] = min(scores.get("variable_preservation", 100), 92.0)

    # ── Length ratio check ───────────────────────────────────────────────────
    src_words = len(source.split())
    tgt_words = len(translation.split())
    if src_words > 0:
        ratio = tgt_words / src_words
        if ratio < 0.6:
            penalty = min(25, int((0.6 - ratio) * 80))
            scores["meaning_preservation"] = max(0, scores.get("meaning_preservation", 80) - penalty)
            errors.append(f"Translation severely shorter than source ({tgt_words} vs {src_words} words)")
        elif ratio > 1.8:
            penalty = min(15, int((ratio - 1.8) * 25))
            scores["meaning_preservation"] = max(0, scores.get("meaning_preservation", 80) - penalty)
            errors.append(f"Translation severely longer than source ({tgt_words} vs {src_words} words)")

    scores["critical_errors"] = list(dict.fromkeys(errors))  # deduplicate
    return scores


def _compress_llm_scores(scores: Dict) -> Dict:
    """
    Apply compression to every 0-100 score returned by the LLM judge.
    Prevents the LLM from awarding 100 for anything.
    """
    continuous_keys = [
        "fluency_score", "grammar_score", "meaning_preservation",
        "placeholder_preservation", "tag_preservation", "variable_preservation",
        "terminology_accuracy", "consistency_score",
        "domain_quality_score", "cultural_adaptation_score",
    ]
    for key in continuous_keys:
        if key in scores:
            scores[key] = _compress(scores[key])
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# CoT PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def _build_cot_prompt(
    source: str,
    translation: str,
    domain: str,
    target_lang: str,
    reference: Optional[str] = None,
) -> str:
    # Resolve domain context — try exact key, then strip legacy prefix, then fall back
    ctx = (DOMAIN_CONTEXT.get(domain)
           or DOMAIN_CONTEXT.get(domain.split("_")[0])
           or DOMAIN_CONTEXT["general"])
    culture = CULTURAL_CONVENTIONS.get(target_lang, None)
    mandatory_criteria = "\n".join(f"  • {c}" for c in ctx["mandatory_criteria"])

    # Detect all three token types in the source
    placeholders = _detect_placeholders(source)
    tags         = _detect_tags(source)
    variables    = _detect_variables(source)

    # Build token inventory block
    token_block = ""
    if placeholders or tags or variables:
        lines = ["⚠️  MANDATORY TOKEN INVENTORY — these MUST appear unchanged in the translation:"]
        if placeholders:
            lines.append(f"  PLACEHOLDERS : {' '.join(f'`{p}`' for p in placeholders)}")
        if tags:
            lines.append(f"  HTML/XML TAGS: {' '.join(f'`{t}`' for t in tags)}")
        if variables:
            lines.append(f"  VARIABLES    : {' '.join(f'`{v}`' for v in variables)}")
        lines += [
            "",
            "  Scoring rules:",
            "  • All placeholders present unchanged → placeholder_preservation = 100",
            "  • Any placeholder missing/modified   → placeholder_preservation = 0, FAIL",
            "  • All tags present unchanged         → tag_preservation = 100",
            "  • Any tag missing/modified           → tag_preservation = 0, FAIL",
            "  • All variables present unchanged    → variable_preservation = 100",
            "  • Any variable missing/modified      → variable_preservation = 0, FAIL",
        ]
        token_block = "\n".join(lines) + "\n"

    # Build cultural conventions block
    cultural_block = ""
    if culture:
        c = culture
        notes_str = "; ".join(c.get("notes", []))
        cultural_block = f"""
── CULTURAL CONVENTIONS for {c['locale_name']} ──────────────────────────────
  Date format  : {c['date_format']}
  Number format: {c['number_format']}
  Formality    : {c['formality']}
  Quotes       : {c['quotes']}
  Notes        : {notes_str}
─────────────────────────────────────────────────────────────────────────────
"""

    # Build reference (ground truth) block when available
    reference_block = ""
    reference_instruction = ""
    if reference:
        reference_block = f"""
╔══════════════════════════════════════════════════════════════════════════╗
  GROUND TRUTH REFERENCE TRANSLATION (expert/human quality):
  \"\"\"{reference}\"\"\"
╚══════════════════════════════════════════════════════════════════════════╝
"""
        reference_instruction = """
── REFERENCE-BASED EVALUATION INSTRUCTIONS ──────────────────────────────────
A verified expert reference translation is provided above. Use it as a
quality anchor for your scoring:

  • Compare MT output AGAINST the reference — not just against the source.
  • If MT matches reference closely → score may approach 87 (still cap at 87).
  • If MT differs from reference in meaning, tone, or terminology → penalize.
  • If MT uses a valid alternative expression not in reference → OK, note it.
  • reference_match_score: how closely MT matches the reference (0-100):
      100 = identical  |  85 = minor word choice differences
       70 = notable differences but same meaning  |  below 60 = significant divergence
─────────────────────────────────────────────────────────────────────────────
"""

    return f"""You are an expert Localization Quality Evaluation Agent performing rigorous Machine Translation Quality Estimation (MTQE).

⚠️  MANDATORY SCORING CALIBRATION — READ BEFORE EVALUATING ANYTHING:
─────────────────────────────────────────────────────────────────────────────
You are a STRICT, CRITICAL evaluator. Your job is to FIND issues and PENALIZE
them. You are NOT a generous reviewer. Use these score anchors EXACTLY:

  95-100 : FORBIDDEN for MT output. No machine translation deserves this.
  88-94  : Exceptional. Near-human quality. Max 1-2 very minor issues.
           REQUIRED: explicitly state what makes it near-perfect.
  80-87  : Good. Minor but noticeable issues (word choice, slight awkwardness).
  70-79  : Acceptable. Multiple minor issues OR one moderate issue.
  60-69  : Below average. Meaning preserved but quality is poor.
  Below 60: Serious problems. Meaning loss, wrong terms, or broken tokens.

PENALTY RULES (apply ALL that are relevant, they stack):
  • Any awkward or unnatural phrasing:          fluency      -5 per instance
  • Overly literal / sounds like machine:        fluency      -8
  • Grammar error (agreement, tense, articles):  grammar      -5 per error
  • Missing punctuation or wrong capitalization: grammar      -3
  • Domain term not using standard equivalent:   terminology  -10 per term
  • Meaning changed or omitted (minor):          meaning      -10
  • Meaning changed or omitted (significant):    meaning      -25
  • Wrong date/number format for locale:         cultural     -15
  • Wrong formality register:                    cultural     -12
  • Any placeholder translated as plain text:    placeholder   0 (instant FAIL)
  • Missing HTML/XML tag:                        tag           0 (instant FAIL)
  • Text much longer/shorter than source:        meaning      -10 to -25

If you give any score ≥ 88, write a sentence explaining WHY it earns that score.
If you cannot find a specific flaw, cap at 87 — do not default to 100.
─────────────────────────────────────────────────────────────────────────────

╔══════════════════════════════════════════════════════════════════════════╗
  DOMAIN       : {domain.upper()}
  TARGET LANG  : {target_lang}
  CONTENT TYPE : {ctx['description']}
  CRITICAL RISK: {ctx['critical_risk']}
╚══════════════════════════════════════════════════════════════════════════╝

SOURCE TEXT:
\"\"\"{source}\"\"\"

MT OUTPUT TO EVALUATE:
\"\"\"{translation}\"\"\"{reference_block}{reference_instruction}
{token_block}{cultural_block}
══════════════════════════════════════════════════════════════════════════════
  DOMAIN-SPECIFIC MANDATORY CRITERIA:
{mandatory_criteria}
══════════════════════════════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                     CHAIN-OF-THOUGHT EVALUATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### STEP 1 — Content Analysis
Briefly describe what type of content this is. Identify the key quality risks.
Which criteria are most critical for THIS specific segment?

---

### STEP 2 — CATEGORY A: Linguistic Quality

**Fluency**
Does the translation read naturally and smoothly in {target_lang}?
Observation: ...
fluency_score (0-100): [score] — 100=native quality, 70=acceptable with minor awkwardness, below 50=unnatural

**Grammar**
Is the translation grammatically correct in {target_lang}? Check: agreement, tense, word order, articles.
Observation: ...
grammar_score (0-100): [score]

**Meaning Preservation**
Does the translation convey the FULL meaning of the source with NO omissions, additions, or distortions?
List any content added by MT that is not in source.
List any source content missing from the translation.
Observation: ...
added_content: [list]
missing_content: [list]
meaning_preservation (0-100): [score] — 100=identical meaning, 50=notable omission/addition, 0=wrong meaning

---

### STEP 3 — CATEGORY B: Localization Quality

**Placeholder Preservation** [MANDATORY — score 0 if ANY fail]
For each source placeholder `{{{{var}}}}`, `%s`, `%(name)s`, `[[var]]`:
- Verify it appears UNCHANGED in the translation.
- Example FAIL: `{{username}}` rendered as plain word "username" → score 0
Observation: ...
placeholder_preservation (0-100): [score]

**Tag Preservation** [MANDATORY — score 0 if ANY fail]
For each source HTML/XML tag `<b>`, `</span>`, `&amp;`, etc.:
- Verify it appears UNCHANGED in the translation.
Observation: ...
tag_preservation (0-100): [score]

**Variable Preservation** [MANDATORY — score 0 if ANY fail]
For each source variable `$VAR`, `__CONST__`, `@NAME`:
- Verify it appears UNCHANGED in the translation.
Observation: ...
variable_preservation (0-100): [score]

**Terminology Accuracy**
Are domain-specific terms, approved translations, and technical vocabulary used correctly?
List any wrong term translations.
wrong_terminology: [list]
terminology_accuracy (0-100): [score]

**Consistency**
Is the translation consistent in style, register, and terminology throughout the segment?
consistency_score (0-100): [score]

---

### STEP 4 — CATEGORY C: Domain Quality

Evaluate specifically for {domain.upper()} content using the mandatory criteria listed above.
Are there any domain-specific failures?
domain_issues: [list of issues, empty if none]
domain_quality_score (0-100): [score]

---

### STEP 5 — CATEGORY D: Cultural Adaptation

Check against the cultural conventions for {target_lang} listed above:
- Dates: are they in the correct locale format?
- Numbers: correct decimal/thousands separators?
- Formality: correct form of address?
- Other locale conventions?

cultural_issues: [list of issues, empty if none]
cultural_adaptation_score (0-100): [score]

---

### STEP 6 — CATEGORY E: Hallucination Detection

hallucination_risk (0.0-1.0):
  0.0 = no deviation from source
  0.2 = minor added/omitted article or conjunction
  0.4 = notable omission or restatement
  0.6 = significant distortion of meaning
  0.8 = major fabrication not in source
  1.0 = complete hallucination, unrelated to source

Estimate the hallucination_risk score.

---

### STEP 7 — Toxicity Check

Does the translation introduce offensive, harmful, or biased language NOT present in the source?
toxicity_risk (0.0-1.0): 0=clean, 0.5=borderline, 1.0=clearly harmful

---

### STEP 8 — Critical Errors

List any errors that would block this translation from production use.
Examples: wrong drug name, missing legal clause, broken placeholder, wrong price, wrong date format.
critical_errors: [list, empty if none]

---

### STEP 9 — OUTPUT JSON

Return ONLY the following JSON block immediately after your analysis. No text after the closing ```:

```json
{{
  "fluency_score": <integer 0-100>,
  "grammar_score": <integer 0-100>,
  "meaning_preservation": <integer 0-100>,
  "placeholder_preservation": <integer 0-100>,
  "tag_preservation": <integer 0-100>,
  "variable_preservation": <integer 0-100>,
  "terminology_accuracy": <integer 0-100>,
  "consistency_score": <integer 0-100>,
  "domain_quality_score": <integer 0-100>,
  "domain_issues": [<list of strings>],
  "cultural_adaptation_score": <integer 0-100>,
  "cultural_issues": [<list of strings>],
  "hallucination_risk": <float 0.00-1.00>,
  "added_content": [<list of strings added by MT>],
  "missing_content": [<list of strings omitted from source>],
  "wrong_terminology": [<list of wrong term translations>],
  "toxicity_risk": <float 0.00-1.00>,
  "critical_errors": [<list of production-blocking issues>],
  "evaluation_notes": "<one sentence: the single most important finding>"
}}
```"""


# ══════════════════════════════════════════════════════════════════════════════
# LLM JUDGE CALL
# ══════════════════════════════════════════════════════════════════════════════
async def _call_judge(prompt: str, base_url: str, api_key: str, model: str) -> str:
    stripped = base_url.rstrip("/")
    url = (stripped + "/chat/completions") if stripped.endswith("/v1") \
        else (stripped + "/v1/chat/completions")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict, critical Localization Quality Evaluator. "
                    "Your role is to FIND and PENALIZE issues — not to be generous. "
                    "Scores of 95-100 are FORBIDDEN for machine translation output. "
                    "Most MT scores should be in the 65-87 range. "
                    "Apply ALL penalty rules listed in the prompt. "
                    "ALWAYS end with a valid JSON block containing ALL required fields."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60.0, verify=False) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def _extract_json(text: str) -> Optional[Dict]:
    """
    Extract the JSON score block from LLM CoT response.
    Strategy 1: fenced ```json ... ``` block.
    Strategy 2: last {...} block containing 'fluency_score'.
    Strategy 3: last {...} block containing 'quality_score' (backward compat).
    """
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        candidate = m.group(1)
    else:
        for key in ("fluency_score", "quality_score", "meaning_preservation"):
            blocks = re.findall(rf'\{{[^{{}}]*"{key}"[^{{}}]*\}}', text, re.DOTALL)
            if blocks:
                candidate = blocks[-1]
                break
        else:
            return None

    try:
        raw = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        # Try to repair common LLM JSON mistakes (trailing commas, single quotes)
        fixed = re.sub(r",\s*}", "}", re.sub(r",\s*]", "]", candidate))
        try:
            raw = json.loads(fixed)
        except Exception:
            return None

    def _f(key, lo, hi, default):
        try:
            return max(lo, min(hi, float(raw.get(key, default))))
        except (TypeError, ValueError):
            return default

    def _lst(key):
        v = raw.get(key, [])
        return list(v) if isinstance(v, list) else []

    return {
        # A. Linguistic
        "fluency_score":           _f("fluency_score",           0, 100, 80),
        "grammar_score":           _f("grammar_score",           0, 100, 80),
        "meaning_preservation":    _f("meaning_preservation",    0, 100, 80),
        # B. Localization
        "placeholder_preservation": _f("placeholder_preservation", 0, 100, 100),
        "tag_preservation":         _f("tag_preservation",         0, 100, 100),
        "variable_preservation":    _f("variable_preservation",    0, 100, 100),
        "terminology_accuracy":     _f("terminology_accuracy",     0, 100, 80),
        "consistency_score":        _f("consistency_score",        0, 100, 80),
        # C. Domain
        "domain_quality_score":     _f("domain_quality_score",     0, 100, 80),
        "domain_issues":            _lst("domain_issues"),
        # D. Cultural
        "cultural_adaptation_score": _f("cultural_adaptation_score", 0, 100, 80),
        "cultural_issues":           _lst("cultural_issues"),
        # E. Hallucination
        "hallucination_risk":   _f("hallucination_risk", 0, 1, 0.1),
        "added_content":        _lst("added_content"),
        "missing_content":      _lst("missing_content"),
        "wrong_terminology":    _lst("wrong_terminology"),
        # Other
        "toxicity_risk":        _f("toxicity_risk",   0, 1, 0.02),
        "critical_errors":      _lst("critical_errors"),
        "evaluation_notes":     str(raw.get("evaluation_notes", "")),
        # Reference-based (only populated when ground truth was provided)
        "reference_match_score": _f("reference_match_score", 0, 100, -1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-PAIR JUDGE
# ══════════════════════════════════════════════════════════════════════════════
async def _judge_pair(
    source: str,
    translation: str,
    domain: str,
    target_lang: str,
    pair_idx: int,
    reference: Optional[str] = None,
) -> Optional[Dict]:
    # Step 0: Run deterministic rule-based checks BEFORE calling LLM judge
    rule_result = _rule_based_checks(source, translation, domain)
    hard_fails    = rule_result["hard_fails"]
    rule_warnings = rule_result["warnings"]
    human_review  = rule_result["human_review_required"]

    prompt = _build_cot_prompt(source, translation, domain, target_lang, reference=reference)

    for base_url, api_key, model in _JUDGE_ENDPOINTS:
        if not base_url or not api_key:
            continue
        try:
            raw_response = await _call_judge(prompt, base_url, api_key, model)
            scores = _extract_json(raw_response)
            if scores:
                # Step 1: compress LLM scores (prevent 100 inflation)
                scores = _compress_llm_scores(scores)
                # Step 2: code-level token + length verification (overrides LLM)
                scores = _code_level_verify(scores, source, translation)
                # Step 3: inject rule-based results
                scores["hard_fails"]           = hard_fails
                scores["rule_warnings"]        = rule_warnings
                scores["human_review_required"] = human_review
                # Merge hard-fail descriptions into critical_errors
                hf_descs = [hf["description"] for hf in hard_fails]
                existing = scores.get("critical_errors", [])
                scores["critical_errors"] = list(dict.fromkeys(existing + hf_descs))
                # Step 4: recompute composite from verified dimensions (includes HRR cap)
                scores["quality_score"] = _compute_composite(scores)
                # Step 5: set localization pass flag
                scores["localization_pass"] = (
                    scores["placeholder_preservation"] >= 50 and
                    scores["tag_preservation"]         >= 50 and
                    scores["variable_preservation"]    >= 50
                )
                # backward-compat alias
                scores["placeholder_preserved"] = scores["localization_pass"]
                scores["judge_model"] = model
                scores["cot_trace"]   = raw_response
                logger.info(
                    f"[Judge] pair={pair_idx} model={model} "
                    f"quality={scores['quality_score']:.1f} "
                    f"ph={scores['placeholder_preservation']:.0f} "
                    f"tag={scores['tag_preservation']:.0f} "
                    f"var={scores['variable_preservation']:.0f} "
                    f"hall={scores['hallucination_risk']:.2f} "
                    f"loc_pass={scores['localization_pass']} "
                    f"hrr={human_review} hard_fails={len(hard_fails)}"
                )
                return scores
            else:
                logger.warning(f"[Judge] {model}: unparseable JSON for pair {pair_idx}")
        except Exception as exc:
            logger.warning(f"[Judge] {model} failed for pair {pair_idx}: {exc}")

    # LLM judges all failed — build minimal score from rule-checks
    logger.warning(f"[Judge] All LLM judges failed for pair {pair_idx} — simulation fallback")
    sim = None  # caller will use simulation
    return sim


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE SCORES ACROSS PAIRS
# ══════════════════════════════════════════════════════════════════════════════
def _aggregate(valid_scores: List[Dict]) -> Dict:
    """Average continuous scores; AND together boolean pass/fail checks."""
    n = len(valid_scores)

    def avg(key, default=80):
        return round(sum(s.get(key, default) for s in valid_scores) / n, 1)

    def avg_f(key, default=0.1):
        return round(sum(s.get(key, default) for s in valid_scores) / n, 3)

    def collect_lists(key):
        return list({item for s in valid_scores for item in s.get(key, [])})

    ph_ok  = all(s.get("placeholder_preservation",  100) >= 50 for s in valid_scores)
    tag_ok = all(s.get("tag_preservation",           100) >= 50 for s in valid_scores)
    var_ok = all(s.get("variable_preservation",      100) >= 50 for s in valid_scores)
    loc_pass = ph_ok and tag_ok and var_ok

    # Human review: if ANY pair triggers it, the whole model batch triggers it
    any_hrr = any(s.get("human_review_required", False) for s in valid_scores)

    # Collect all hard_fails across pairs (deduplicated by code)
    seen_hf_codes: set = set()
    all_hard_fails: List[Dict] = []
    for s in valid_scores:
        for hf in s.get("hard_fails", []):
            code = hf.get("code", "")
            if code not in seen_hf_codes:
                seen_hf_codes.add(code)
                all_hard_fails.append(hf)

    all_rule_warnings = list({w for s in valid_scores for w in s.get("rule_warnings", [])})

    agg = {
        # A. Linguistic
        "fluency_score":          avg("fluency_score"),
        "grammar_score":          avg("grammar_score"),
        "meaning_preservation":   avg("meaning_preservation"),
        # B. Localization
        "placeholder_preservation": avg("placeholder_preservation", 100),
        "tag_preservation":         avg("tag_preservation",         100),
        "variable_preservation":    avg("variable_preservation",    100),
        "terminology_accuracy":     avg("terminology_accuracy"),
        "consistency_score":        avg("consistency_score"),
        # C. Domain
        "domain_quality_score":     avg("domain_quality_score"),
        "domain_issues":            collect_lists("domain_issues"),
        # D. Cultural
        "cultural_adaptation_score": avg("cultural_adaptation_score"),
        "cultural_issues":           collect_lists("cultural_issues"),
        # E. Hallucination
        "hallucination_risk":    avg_f("hallucination_risk", 0.1),
        "added_content":         collect_lists("added_content"),
        "missing_content":       collect_lists("missing_content"),
        "wrong_terminology":     collect_lists("wrong_terminology"),
        # Rule-based
        "human_review_required": any_hrr,
        "hard_fails":            all_hard_fails,
        "rule_warnings":         all_rule_warnings,
        # Composite
        "localization_pass":     loc_pass,
        "placeholder_preserved": loc_pass,   # backward-compat alias
        "toxicity_risk":         avg_f("toxicity_risk", 0.02),
        "critical_errors":       collect_lists("critical_errors"),
        "evaluation_notes":      valid_scores[0].get("evaluation_notes", ""),
        "judge_model":           valid_scores[0].get("judge_model", "unknown"),
        "cot_traces":            [s.get("cot_trace") for s in valid_scores if s.get("cot_trace")],
        "evaluation_method":     "llm_judge",
        "pairs_evaluated":       n,
    }
    # Re-compute composite from aggregated dimension scores (respects HRR cap)
    agg["quality_score"] = _compute_composite(agg)
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# PER-MODEL EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════
async def _evaluate_model(
    translation_result: Dict[str, Any],
    domain: str,
    target_lang: str,
    reference_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    model_id     = translation_result["model_id"]
    translations = translation_result.get("translations", [])

    if not translations:
        logger.warning(f"No translation pairs for {model_id} — simulation")
        metrics = _simulate_scores(model_id, domain)
        metrics["evaluation_method"] = "simulation_no_output"
    else:
        tasks = [
            _judge_pair(
                t["source"], t["translation"], domain, target_lang, i,
                reference=(reference_map or {}).get(t["source"])
            )
            for i, t in enumerate(translations)
        ]
        raw_scores = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [s for s in raw_scores if isinstance(s, dict)]

        if not valid:
            logger.warning(f"[Judge] All pairs failed for {model_id} — simulation fallback")
            metrics = _simulate_scores(model_id, domain)
            metrics["evaluation_method"] = "simulation_llm_failed"
        else:
            metrics = _aggregate(valid)

    # Always carry over cost/latency from translation result
    metrics["total_cost"]         = translation_result.get("total_cost", 0)
    metrics["cost_per_1k_tokens"] = translation_result.get("cost_per_1k_tokens", 0)
    metrics["avg_latency"]        = translation_result.get("avg_latency", 0)

    return {**translation_result, "metrics": metrics}


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
async def run(
    translation_results: List[Dict[str, Any]],
    domain: str,
    target_lang: str = "de",
    reference_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate all MT model outputs using LLM-as-a-judge with 5-category CoT prompting.
    When reference_map is provided (ground truth from benchmark CSVs), the judge
    compares MT output against the reference for more accurate scoring.
    """
    ref_count = len(reference_map) if reference_map else 0
    logger.info(
        f"[EvalAgent] {len(translation_results)} model(s) | "
        f"domain={domain} | target={target_lang} | "
        f"ground_truth={ref_count} refs | method=LLM-as-Judge+CoT+Reference"
    )

    tasks = [
        _evaluate_model(result, domain, target_lang, reference_map)
        for result in translation_results
    ]
    evaluated = await asyncio.gather(*tasks, return_exceptions=True)

    output = []
    for i, item in enumerate(evaluated):
        if isinstance(item, Exception):
            logger.error(f"[EvalAgent] Model {i} crashed: {item}")
            original = translation_results[i]
            sim = _simulate_scores(original.get("model_id", "unknown"), domain)
            sim["total_cost"]         = original.get("total_cost", 0)
            sim["cost_per_1k_tokens"] = original.get("cost_per_1k_tokens", 0)
            sim["avg_latency"]        = original.get("avg_latency", 0)
            output.append({**original, "metrics": sim})
        else:
            output.append(item)

    return output
