import hashlib
import json
import platform
import statistics
import time
from collections import Counter

from .retrieval import Router


def evaluate(engine, corpus_path, tenant="demo", user="alice", repeats=20):
    raw = corpus_path.read_bytes()
    cases = json.loads(raw)
    router = Router()
    mistakes, confusion = [], Counter()
    for case in cases:
        actual = router.decide(case["query"]).route
        confusion[f"{case['route']} -> {actual}"] += 1
        if actual != case["route"]:
            mistakes.append({**case, "actual": actual})
    routes = {}
    for query in ["Who authored Linear ticket DEMO123?", "Search provenance in DEMO-123", "Search hybrid retrieval", "What is turbolay?"]:
        first = engine.query(tenant, user, query)
        samples = [engine.query(tenant, user, query)["latency_ms"] for _ in range(repeats)]
        routes[first["route"]] = {"first_query_ms": first["latency_ms"], "warm_median_ms": statistics.median(samples),
                                  "warm_p95_ms": sorted(samples)[int(len(samples)*.95)-1], "stores": first["stores"]}
    retrieval_cases = json.loads(corpus_path.with_name("retrieval-cases.json").read_text())
    retrieval_results = []
    for case in retrieval_cases:
        result = engine.query(case.get("tenant", tenant), case.get("user", user), case["query"], personalized=case.get("personalized", True))
        checks = {"route": result["route"] == case["route"]}
        if "status" in case:
            checks["status"] = result["status"] == case["status"]
        if "stores" in case:
            checks["stores"] = result["stores"] == case["stores"]
        if "contains" in case:
            checks["contains"] = any(r["node_id"] == case["contains"] for r in result["results"])
        if "scope_prefix" in case:
            checks["scope"] = bool(result["results"]) and all(r["node_id"].startswith(case["scope_prefix"]) for r in result["results"])
        preferences = [p for trace in result["trace"] for p in trace.get("preferences", [])]
        if case.get("no_preferences"):
            checks["no_preferences"] = not preferences
        if case.get("evidence"):
            checks["evidence"] = bool(preferences) and all(p["quote"] and p["turn_id"] for p in preferences)
        if case.get("no_preference_store"):
            checks["no_preference_store"] = "preferences" not in result["stores"]
        if "all_kind" in case:
            checks["kind"] = bool(result["results"]) and all(r["kind"] == case["all_kind"] for r in result["results"])
        retrieval_results.append({"case": case, "passed": all(checks.values()), "checks": checks})
    return {"backend": engine.store.encoder.name, "python": platform.python_version(), "platform": platform.platform(),
            "cases": len(cases), "correct": len(cases)-len(mistakes), "accuracy": (len(cases)-len(mistakes))/len(cases),
            "corpus_sha256": hashlib.sha256(raw).hexdigest(), "confusion": dict(confusion), "mistakes": mistakes,
            "route_latency": routes, "repeats": repeats, "network_calls": 0, "generative_calls": 0,
            "retrieval_cases": retrieval_results, "retrieval_passed": sum(r["passed"] for r in retrieval_results),
            "limits": "Hand-authored synthetic routing holdout, not independent production evidence. No learned router or tuning after evaluation. First query excludes imports, model loading, ingestion, OS cold cache; CLI reports initialization separately."}
