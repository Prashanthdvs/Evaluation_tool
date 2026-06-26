"""Quick end-to-end test that submits a job and polls for results."""
import asyncio, json, time
import httpx

CSV = b"source\nThe patient requires immediate medical treatment.\nPlease order the product online.\nContact our support team immediately."

async def main():
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=60) as c:
        # Submit
        resp = await c.post(
            "/api/evaluate",
            data={
                "target_language": "de",
                "models": json.dumps(["llama-3.3-70b", "qwen25-32b-awq"]),
                "latency_priority": "good",
                "cost_priority": "good",
            },
            files={"file": ("test.csv", CSV, "text/csv")},
        )
        job_id = resp.json()["job_id"]
        print(f"Job submitted: {job_id}")

        # Poll
        for _ in range(20):
            await asyncio.sleep(2)
            job = (await c.get(f"/api/jobs/{job_id}")).json()
            print(f"  [{job['progress']}%] {job['stage']}")
            if job["status"] in ("completed", "failed"):
                break

        if job["status"] == "failed":
            print("FAILED:", job.get("error"))
            return

        # Results
        r = (await c.get(f"/api/results/{job_id}")).json()
        da = r["dataset_analysis"]
        print(f"\nDataset: {da['detected_domain']} | {da['source_language_name']} -> {da['target_language_name']} | {da['unique_rows']} rows")
        print(f"Models evaluated: {len(r['model_results'])}\n")

        for m in r["model_results"]:
            gov  = m["governance_check"]
            met  = m["metrics"]
            tag  = "WINNER" if m["selected"] else ("REJECT" if not gov["passed"] else "OK    ")
            print(f"  [{tag}] {m['model_name']:<22}  score={m['weighted_score']:.1f}  quality={met['quality_score']:.0f}  latency={met['avg_latency']:.2f}s  cost=${met['total_cost']:.4f}  [{m['api_type']}]")
            if not gov["passed"]:
                for v in gov["violations"]:
                    print(f"           ! {v}")

        dec = r["decision"]
        w   = dec["weights_used"]
        print(f"\nWinner  : {dec['selected_model_name'] or 'None passed governance'}")
        print(f"Weights : Quality {w['quality_weight']*100:.0f}%  Cost {w['cost_weight']*100:.0f}%  Latency {w['latency_weight']*100:.0f}%")
        print(f"\nSummary : {r['explainability']['summary'][:250]}")

asyncio.run(main())
