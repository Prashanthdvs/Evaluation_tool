# MT Evaluation Engine
### Hackathon Project — Translation Provider Selection Engine

A fully-automated, explainable AI system that benchmarks multiple MT providers and recommends the optimal one based on domain-specific business rules.

---

## Architecture

```
CSV Upload
    │
    ▼
Agent 1 · Dataset Agent      — detects language, domain, deduplicates, samples
    │
    ▼
Agent 2 · Translation Agent  — runs all selected MT providers in parallel
    │
    ▼
Agent 3 · Evaluation Agent   — scores quality, terminology, fluency, hallucination
    │
    ▼
Agent 4 · Governance Agent   — applies YAML business rules, computes weighted score
    │
    ▼
Agent 5 · Explainability Agent — generates human-readable decision report
    │
    ▼
Dashboard — charts, comparison table, per-model explanations
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+

### Run (Windows)
```bat
start.bat
```

### Manual Setup

**Backend**
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # Add GROQ_API_KEY for real translations
python main.py
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**

---

## Features

### Evaluation Pipeline
| Agent | Role |
|-------|------|
| Dataset Agent | Auto-detects encoding (UTF-8/16), language, domain, removes duplicates |
| Translation Agent | Runs Llama 3.3 70B, Gemma 4 27B, GPT-4o, DeepL, Azure MT |
| Evaluation Agent | Scores quality, terminology accuracy, fluency, hallucination & toxicity risk |
| Governance Agent | Applies domain YAML rules, adjusts weights per user priority |
| Explainability Agent | Generates full decision narrative with score breakdown |

### Business Rules (YAML)
Edit `backend/config/business_rules.yaml`:
```yaml
healthcare:
  quality_weight: 0.70
  cost_weight:    0.10
  latency_weight: 0.20
  min_quality: 85.0
  max_hallucination_risk: 0.15

customer_support:
  quality_weight: 0.30
  cost_weight:    0.20
  latency_weight: 0.50   # latency is critical
```

### Weighted Scoring Formula
```
Final Score = (Quality × w_q) + (CostScore × w_c) + (LatencyScore × w_l)
```
- CostScore = (1 − cost/max_cost) × 100  (cheaper = higher)
- LatencyScore = (1 − latency/5s) × 100  (faster = higher)

### Supported Languages
| Code | Language |
|------|----------|
| de   | German (DEU) |
| fr   | French (FRA) |
| ja   | Japanese (JPN) |
| zh-CN| Chinese Simplified (CHS) |
| ko   | Korean (KOR) |
| es   | Spanish (ESP) |

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/evaluate` | POST | Upload CSV + config → returns `job_id` |
| `/api/jobs/{id}` | GET | Poll job status & progress |
| `/api/results/{id}` | GET | Get full evaluation results |
| `/api/models` | GET | List all registered models |
| `/api/models` | POST | Register new custom model |
| `/api/models/{id}` | DELETE | Remove custom model |
| `/api/rules` | GET | Get business rules YAML |
| `/api/rules` | PUT | Update business rules |

---

## Adding Real API Keys

Edit `backend/.env`:
```env
GROQ_API_KEY=gsk_...     # https://console.groq.com (free)
```

With `GROQ_API_KEY` set, **Llama 3.3 70B** and **Gemma** calls use the real Groq API.
All other models use calibrated simulation (realistic scores + latency).

---

## Project Structure
```
Hackathon_Evaluation/
├── backend/
│   ├── agents/
│   │   ├── dataset_agent.py        # Agent 1
│   │   ├── translation_agent.py    # Agent 2
│   │   ├── evaluation_agent.py     # Agent 3
│   │   ├── governance_agent.py     # Agent 4
│   │   └── explainability_agent.py # Agent 5
│   ├── config/
│   │   ├── business_rules.yaml     # Domain business rules
│   │   └── registered_models.json  # Custom model registry
│   ├── main.py                     # FastAPI server
│   ├── pipeline.py                 # Orchestrator
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Dashboard.tsx       # Results charts
│       │   ├── ExplainabilityPanel.tsx
│       │   ├── ConfigPanel.tsx
│       │   ├── ProcessingView.tsx
│       │   ├── UploadSection.tsx
│       │   └── ModelRegistry.tsx
│       └── pages/
│           ├── EvaluationPage.tsx
│           └── OnboardingPage.tsx
└── start.bat
```
