"""Reproducible fixture-only latency evidence; no model downloads or live sources."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--reranker")
    parser.add_argument("--output", default="docs/evaluation-lexical.json")
    args = parser.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with tempfile.TemporaryDirectory() as temp:
        db = str(Path(temp) / "benchmark.db")
        base = [sys.executable, "-m", "poc.cli", "--db", db]
        if args.model:
            base += ["--model", args.model]
        if args.reranker:
            base += ["--reranker", args.reranker]
        fixture = json.loads(subprocess.check_output(base + ["fixture"], env=env))
        evaluation = json.loads(subprocess.check_output(base + ["evaluate"], env=env))
        process_samples = {}
        for query in ["Who authored Linear ticket DEMO123?", "What is turbolay?"]:
            samples = []
            for _ in range(3):
                start = time.perf_counter()
                result = json.loads(subprocess.check_output(base + ["query", query], env=env))
                samples.append({"wall_ms": round((time.perf_counter()-start)*1000, 3),
                                "initialization_ms": result["initialization_ms"], "query_ms": result["latency_ms"]})
            process_samples[result["route"]] = samples
        evaluation["process_cold_samples"] = process_samples
        evaluation["fixture_setup"] = fixture
        evaluation["cpu_threads"] = 1
        evaluation["cold_definition"] = "Fresh interpreter and model objects, same host and existing downloaded files. OS filesystem/page caches not flushed; not machine-cold or network time. Authorship bypass skips model loading."
        Path(args.output).write_text(json.dumps(evaluation, indent=2) + "\n")
        print(args.output)


if __name__ == "__main__":
    main()
