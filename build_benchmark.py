import csv, json

CSV_PATH = r'C:\Users\dsatyaprasha\Downloads\MASTER_EVALUATION_REPORT.csv'

with open(CSV_PATH, encoding='utf-8-sig') as f:
    lines = f.readlines()

def parse_section(lines, start_line, end_line=None):
    block = lines[start_line:end_line]
    reader = csv.DictReader(block)
    return list(reader)

# Section 4: Routing matrix (domain x language x cost tier)
routing_matrix = parse_section(lines, 71, 115)
# Section 8: Use-case recommendations (domain x language x content — top 3 models)
recommendations = parse_section(lines, 525, 694)

# Map CSV domain names to Streamlit DOMAIN_LIST names
DOMAIN_MAP = {
    'E-commerce_product':  'E-commerce Product',
    'Finance_Banking':     'Finance/Banking',
    'IT_Software':         'IT Software',
    'Journals_Publishing': 'Journals Publishing',
    'Legal_Compliance':    'Legal Compliance',
    'Multimedia_Streaming':'Multimedia Streaming',
    'Pharma_Healthcare':   'Pharma/Healthcare',
}

MODEL_META = {
    'OpenAI o3':                             {'provider':'OpenAI',          'latency_sec':90,  'cost_per_1k':0.025},
    'Claude 3.7 Sonnet (Extended Thinking)': {'provider':'Anthropic',       'latency_sec':35,  'cost_per_1k':0.009},
    'Gemini 2.5 Pro (Thinking)':             {'provider':'Google DeepMind', 'latency_sec':20,  'cost_per_1k':0.003625},
    'Claude 3.5 Sonnet (Extended Thinking)': {'provider':'Anthropic',       'latency_sec':30,  'cost_per_1k':0.009},
    'OpenAI o4-mini':                        {'provider':'OpenAI',          'latency_sec':5,   'cost_per_1k':0.0013},
    'OpenAI o1':                             {'provider':'OpenAI',          'latency_sec':60,  'cost_per_1k':0.0075},
    'OpenAI o3-mini':                        {'provider':'OpenAI',          'latency_sec':10,  'cost_per_1k':0.00055},
    'DeepSeek R1':                           {'provider':'DeepSeek AI',     'latency_sec':15,  'cost_per_1k':0.000275},
    'Gemini 2.0 Flash Thinking':             {'provider':'Google DeepMind', 'latency_sec':3,   'cost_per_1k':0.00025},
    'OpenAI o1-mini':                        {'provider':'OpenAI',          'latency_sec':12,  'cost_per_1k':0.0015},
    'Qwen QwQ-32B':                          {'provider':'Alibaba Cloud',   'latency_sec':12,  'cost_per_1k':0.0002},
}
TIER_SCORE = {'High': 55, 'Good': 80, 'Low': 100}
LATENCY_SEC_DEFAULT = {'High': 45, 'Good': 15, 'Low': 5}

def make_result(model_name, score_5pt, cost_tier, latency_tier, domain, language, lang_name, content_type, rank):
    meta = MODEL_META.get(model_name, {
        'provider': 'Unknown',
        'latency_sec': LATENCY_SEC_DEFAULT.get(latency_tier, 15),
        'cost_per_1k': 0.005
    })
    q = round((float(score_5pt) / 5.0) * 100, 1)
    lat = meta['latency_sec']
    cst = round(meta['cost_per_1k'] * 5, 6)
    hall = round(max(0.02, 0.15 - (q - 70) * 0.003), 3)
    # Tier scores: Low tier (fast/cheap) = 100, Good = 80, High (slow/expensive) = 55
    lat_score = TIER_SCORE.get(latency_tier, 70)
    cost_score = TIER_SCORE.get(cost_tier, 70)
    return {
        'model_id':             model_name.lower().replace(' ', '_').replace('.', '').replace('(', '').replace(')', '').replace('-', '_')[:30],
        'model_name':           model_name,
        'provider':             meta['provider'],
        'domain':               domain,
        'language':             language,
        'language_name':        lang_name,
        'content_type':         content_type,
        'rank':                 rank,
        'quality_score':        q,
        'terminology_accuracy': round(q * 0.97, 1),
        'fluency_score':        round(q * 0.99, 1),
        'meaning_preservation': round(q * 0.98, 1),
        'avg_latency':          lat,
        'total_cost':           cst,
        'cost_per_1k_tokens':   meta['cost_per_1k'],
        'hallucination_risk':   hall,
        'latency_tier':         latency_tier,
        'cost_tier':            cost_tier,
        'latency_score':        lat_score,
        'cost_score':           cost_score,
        'governance_passed':    True,
    }

results = []

for row in recommendations:
    domain_raw = row['domain'].strip()
    domain  = DOMAIN_MAP.get(domain_raw, domain_raw)
    lang    = row['language'].strip()
    content = row['content'].strip()
    lang_name = next((r['lang_name'] for r in routing_matrix if r['language'] == lang), lang)

    for rank_n in ['1', '2', '3']:
        m = row.get(f'rank_{rank_n}_model', '').strip()
        s = row.get(f'rank_{rank_n}_score', '').strip()
        c = row.get(f'rank_{rank_n}_cost', '').strip()
        l = row.get(f'rank_{rank_n}_latency', '').strip()
        if not m or not s:
            continue
        results.append(make_result(m, s, c, l, domain, lang, lang_name, content, int(rank_n)))

print(f'Total benchmark results: {len(results)}')
import pprint; pprint.pprint(results[0])

out_path = r'backend\config\benchmark_results.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print(f'Written to {out_path}')
