"""
여러 seed의 deferral curve를 평균±표준편차 밴드로 그린다.

각 seed의 eval_results.json에서 (method별) deferral curve를 공통 grid(strong%)에
보간한 뒤, seed 축으로 평균/표준편차를 내어:
  - 평균 = 실선
  - ±1 std = 옅은 음영(fill) + 점선 테두리
로 그린다. weak-only / strong-only / strong−drop%p 기준선도 표시.

두 그림을 만든다:
  <prefix>_all.png   : 모든 method
  <prefix>_best.png  : 각 pair에서 최고 MF 변형 + 최고 UniRoute 변형만 (+ random)
                       (선택 기준 = pass ≥ strong−drop 에서 보간한 weak%의 seed 평균)

사용법:
  python plot_seed_curves.py --results-jsons \
      results_seed1/pair_0.8B/eval_results.json \
      results_seed2/pair_0.8B/eval_results.json ... \
      --output-prefix seed_curves

split seed를 고정(run_experiments --split-seed 42)하고 라우터 seed만 바꿔 돌리면
test set·기준선이 상수라 밴드가 '라우터 재현성'만 반영한다.
"""
import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# collect_results와 동일한 색/라벨 규약 재사용
from collect_results import METHOD_STYLE, METHOD_LABEL

GRID = np.linspace(0.0, 100.0, 101)  # 공통 strong% 격자


def _pair(data):
    w = data["weak_model"].split("/")[-1]
    s = data["strong_model"].split("/")[-1]
    return f"{w} vs {s}"


def load_runs(paths):
    """반환: pairs[pair] = {
        'methods': {method: [interp_acc(GRID) per seed]},
        'weak': [..], 'strong': [..], 'drop': float,
        'weak_at_drop': {method: [weak% per seed]} }"""
    pairs = defaultdict(lambda: {
        "methods": defaultdict(list), "weak": [], "strong": [],
        "drop": 1.0, "weak_at_drop": defaultdict(list),
    })
    for p in paths:
        data = json.load(open(p))
        pair = _pair(data)
        P = pairs[pair]
        P["weak"].append(data["weak_only_accuracy"])
        P["strong"].append(data["strong_only_accuracy"])
        P["drop"] = data.get("per_category_pass_drop", 1.0)
        # deferral curve → strong% 오름차순 보간
        df = defaultdict(list)
        for r in data["results"]:
            df[r["method"]].append((r["strong_percentage"], r["accuracy"]))
        for method, pts in df.items():
            pts = sorted(pts)
            xs = np.array([x for x, _ in pts]); ys = np.array([y for _, y in pts])
            P["methods"][method].append(np.interp(GRID, xs, ys))
        # per_category의 headline weak@drop (seed별)
        for method, blk in (data.get("per_category") or {}).items():
            wi = blk.get("operating_point", {}).get("weak_pct_interp")
            if wi is not None:
                P["weak_at_drop"][method].append(float(wi))
    return pairs


def _draw(ax, method, curves):
    arr = np.vstack(curves)              # (n_seeds, 101)
    mean = arr.mean(axis=0); std = arr.std(axis=0)
    st = METHOD_STYLE.get(method, {})
    color = st.get("color", "black")
    lbl = METHOD_LABEL.get(method, method)
    n = len(curves)
    ax.plot(GRID, mean, color=color, linewidth=st.get("linewidth", 2.0),
            linestyle=st.get("linestyle", "-"), marker=None,
            label=f"{lbl} (n={n})")
    ax.fill_between(GRID, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    for edge in (mean - std, mean + std):
        ax.plot(GRID, edge, color=color, linewidth=0.7, linestyle=(0, (1, 3)), alpha=0.6)


def _refs(ax, P):
    wk = float(np.mean(P["weak"])); sg = float(np.mean(P["strong"])); drop = P["drop"]
    ax.axhline(wk, color="#555555", linestyle=":", linewidth=1.0, label=f"Weak only ({wk:.1f}%)")
    ax.axhline(sg, color="#C62828", linestyle=":", linewidth=1.0, label=f"Strong only ({sg:.1f}%)")
    ax.axhline(sg - drop, color="#C62828", linestyle=(0, (1, 3)), linewidth=0.9,
               label=f"Strong − {drop:.0f}%p ({sg - drop:.1f}%)")


def _finish(ax, pair):
    ax.set_xlabel("Strong Model Calls (%)", fontsize=11)
    ax.set_ylabel("Pass Rate (%)", fontsize=11)
    ax.set_title(pair, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 102)


def _figure(pairs, methods_for, out_png, title):
    plist = list(pairs.keys())
    fig, axes = plt.subplots(1, len(plist), figsize=(7 * len(plist), 5.5))
    if len(plist) == 1:
        axes = [axes]
    for ax, pair in zip(axes, plist):
        P = pairs[pair]
        sel = methods_for(pair)
        for method in sel:
            if method == "random":
                continue  # 아래에서 대각선 기준선으로
            if method in P["methods"]:
                _draw(ax, method, P["methods"][method])
        # random은 기대 곡선이 weak→strong 직선이라 대각선 기준선으로 표시
        if "random" in sel:
            wk = float(np.mean(P["weak"])); sg = float(np.mean(P["strong"]))
            st = METHOD_STYLE.get("random", {})
            ax.plot([0, 100], [wk, sg], color=st.get("color", "#888888"),
                    linestyle="--", linewidth=1.3, label="Random (diagonal)")
        _refs(ax, P)
        _finish(ax, pair)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_png}")


# best 그래프에 고정으로 보여줄 (method, legend 라벨)
FIXED_BEST = [
    ("mf", "MF Router"),                 # tie→strong
    ("uniroute_train", "UniRoute (K-Means)"),  # honest K, Ψ=train
]


def _figure_best(pairs, out_png, title):
    """고정 선택(MF tie→strong / UniRoute honest·Ψ=train) 밴드 그래프.
    strong−drop%p 에서의 평균 weak% 를 마커+주석으로 표시하고 콘솔에도 출력."""
    plist = list(pairs.keys())
    fig, axes = plt.subplots(1, len(plist), figsize=(7 * len(plist), 5.5))
    if len(plist) == 1:
        axes = [axes]
    for ax, pair in zip(axes, plist):
        P = pairs[pair]
        wk = float(np.mean(P["weak"])); sg = float(np.mean(P["strong"])); drop = P["drop"]
        print(f"\n[{pair}]  weak@(strong−{drop:.0f}%p) average weak model %:")
        for method, label in FIXED_BEST:
            if method not in P["methods"]:
                continue
            arr = np.vstack(P["methods"][method]); mean = arr.mean(0); std = arr.std(0)
            st = METHOD_STYLE.get(method, {}); color = st.get("color", "black")
            wd = P["weak_at_drop"].get(method, [])
            wtxt = ""
            if wd:
                wmean = float(np.mean(wd)); wstd = float(np.std(wd))
                wtxt = f"  (weak@−{drop:.0f}%p: {wmean:.1f}±{wstd:.1f}%)"
                print(f"    {label:<22} {wmean:5.1f} ± {wstd:.1f} %   (n={len(wd)})")
            ax.plot(GRID, mean, color=color, linewidth=2.2, linestyle="-",
                    label=f"{label}{wtxt}")
            ax.fill_between(GRID, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
            for edge in (mean - std, mean + std):
                ax.plot(GRID, edge, color=color, linewidth=0.7, linestyle=(0, (1, 3)), alpha=0.6)
            # strong−drop%p 에서의 평균 weak% 지점 표시 (x = 100 − weak%)
            if wd:
                x = 100.0 - wmean
                ax.plot([x], [sg - drop], marker="o", color=color, markersize=7, zorder=5)
                ax.annotate(f"{wmean:.1f}%", (x, sg - drop),
                            textcoords="offset points", xytext=(5, 6),
                            fontsize=10, fontweight="bold", color=color)
        # random 대각선 기준선
        rs = METHOD_STYLE.get("random", {})
        ax.plot([0, 100], [wk, sg], color=rs.get("color", "#888888"),
                linestyle="--", linewidth=1.3, label="Random (diagonal)")
        _refs(ax, P)
        _finish(ax, pair)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Plot seed-averaged deferral curves (mean ± std bands)")
    ap.add_argument("--results-jsons", nargs="+", required=True,
                    help="여러 seed의 eval_results.json 경로들 (pair는 JSON에서 자동 그룹화)")
    ap.add_argument("--output-prefix", default="seed_curves")
    ap.add_argument("--graph-random", action="store_true",
                    help="전체 그래프에 random baseline도 포함")
    args = ap.parse_args()

    pairs = load_runs(args.results_jsons)

    # (1) 전체 method 밴드
    base = ["mf", "mf_tieweak", "uniroute", "uniroute_train", "uniroute_legacy"]
    all_methods = (["random"] if args.graph_random else []) + base
    _figure(pairs, lambda pair: all_methods, f"{args.output_prefix}_all.png",
            "Seed-averaged deferral curves (mean ± 1 std)")

    # (2) best: 고정 선택 (MF Router / UniRoute (K-Means)) + strong−drop 평균 weak% 표시
    _figure_best(pairs, f"{args.output_prefix}_best.png",
                 "MF Router vs UniRoute (K-Means) — mean ± 1 std")


if __name__ == "__main__":
    main()
