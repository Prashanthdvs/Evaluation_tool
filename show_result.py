import sys, json

d = json.load(sys.stdin)
print("Status:", d["status"])
print()

da = d["dataset_analysis"]
print(f"Dataset  : {da['detected_domain']} | {da['source_language_name']} -> {da['target_language_name']} | {da['unique_rows']} unique rows")
print()

for m in d["model_results"]:
    gov = m["governance_check"]
    tag = "WINNER  " if m["selected"] else ("REJECTED" if not gov["passed"] else "OK      ")
    met = m["metrics"]
    api = m.get("api_type", "?")
    print(f"  [{tag}] {m['model_name']:<22} score={m['weighted_score']:.1f}  quality={met['quality_score']:.0f}  terminology={met['terminology_accuracy']:.0f}%  latency={met['avg_latency']:.2f}s  cost=${met['total_cost']:.4f}  [{api}]")
    if not gov["passed"]:
        for v in gov["violations"]:
            print(f"            ! {v}")

print()
dec = d["decision"]
w = dec["weights_used"]
print(f"Winner  : {dec['selected_model_name'] or 'None passed governance'}")
print(f"Weights : Quality {w['quality_weight']*100:.0f}%  Cost {w['cost_weight']*100:.0f}%  Latency {w['latency_weight']*100:.0f}%")
print()

exp = d["explainability"]
print("Summary :", exp["summary"][:300])
print()
print("Rules applied:")
for r in exp["business_rules_applied"]:
    print(f"  -> {r}")
