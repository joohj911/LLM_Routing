"""
여러 seed(× embedding × pair)의 eval_results.json을 모아 headline 지표를
평균±표준편차로 집계한다. 단일 seed 노이즈가 큰 UniRoute/CSCR 결론을
확정하기 위한 스크립트.

집계 지표: per_category[method].operating_point.weak_pct_interp
  = "pass ≥ strong-only − 1%p 에서 보간한 weak 사용 비율(%)" (클수록 좋음).

사용법:
  python aggregate_seeds.py \
    --results-jsons results_seed1/pair_0.8B/eval_results.json \
                    results_seed2/pair_0.8B/eval_results.json \
                    results_cscr_seed1/pair_2B/eval_results.json ... \
    --output seed_summary.csv

embedding(e5/cscr)·seed·pair는 경로/JSON에서 자동 추론한다
(경로에 'cscr'이 있으면 cscr, 'seed<N>'에서 seed 번호). 필요하면
'label=path' 로 embedding 라벨을 직접 지정할 수 있다.
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd


def _infer(path: str, data: dict):
    emb = "cscr" if "cscr" in path.lower() else "e5"
    m = re.search(r"seed[_-]?(\d+)", path, re.I)
    seed = m.group(1) if m else "?"
    weak = data["weak_model"].split("/")[-1]
    strong = data["strong_model"].split("/")[-1]
    return emb, seed, f"{weak} vs {strong}"


def main():
    ap = argparse.ArgumentParser(description="Aggregate Weak@drop metric across seeds")
    ap.add_argument("--results-jsons", nargs="+", required=True,
                    help="eval_results.json 경로들. 'emb=path' 로 embedding 라벨 지정 가능.")
    ap.add_argument("--output", default="seed_summary.csv")
    args = ap.parse_args()

    # (pair, embedding, method) -> list of weak_pct_interp
    acc = defaultdict(list)
    for entry in args.results_jsons:
        emb_override, path = (entry.split("=", 1) if "=" in entry else (None, entry))
        with open(path) as f:
            data = json.load(f)
        emb, seed, pair = _infer(path, data)
        if emb_override:
            emb = emb_override
        pc = data.get("per_category", {})
        if not pc:
            print(f"[warn] no per_category in {path} (skipped)")
            continue
        for method, block in pc.items():
            wi = block.get("operating_point", {}).get("weak_pct_interp")
            if wi is not None:
                acc[(pair, emb, method)].append(float(wi))

    rows = []
    for (pair, emb, method), vals in acc.items():
        a = np.array(vals, dtype=float)
        rows.append({
            "Pair": pair, "Embedding": emb, "Method": method,
            "n_seeds": len(a),
            "Weak@drop mean": round(float(a.mean()), 2),
            "Weak@drop std": round(float(a.std(ddof=0)), 2),
            "min": round(float(a.min()), 2),
            "max": round(float(a.max()), 2),
        })
    df = pd.DataFrame(rows).sort_values(["Pair", "Embedding", "Weak@drop mean"],
                                        ascending=[True, True, False])
    df.to_csv(args.output, index=False)

    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df.to_string(index=False))
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
