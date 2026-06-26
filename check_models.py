import sys, json
d = json.load(sys.stdin)
print("=== Built-in Models ===")
for m in d['builtin']:
    tag = "[LIVE API] " if m['api_type'] == 'openai_compat' else "[simulated]"
    print(f"  {tag}  {m['model_name']:<25} | {m['provider']:<20} | ${m['cost_per_1k_tokens']}/1K | {m['base_latency']}s latency")
print(f"Custom models registered: {len(d['custom'])}")
