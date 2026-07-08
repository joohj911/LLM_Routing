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


def _parse_entry(entry):
    """'embedding=path' → (embedding, path). '=' 없으면 ('default', path).
    collect_results 와 동일 규약 → e5=... / cscr=... 로 임베딩을 구분한다."""
    if "=" in entry:
        emb, path = entry.split("=", 1)
        return emb, path
    return "default", entry


def load_runs(entries):
    """entries: 'emb=path' 또는 'path'. 임베딩별로 분리해서 반환.
    반환: by_emb[embedding][pair] = {
        'methods': {method: [interp_acc(GRID) per seed]},
        'weak': [..], 'strong': [..], 'drop': float,
        'weak_at_drop': {method: [weak% per seed]} }"""
    by_emb = defaultdict(lambda: defaultdict(lambda: {
        "methods": defaultdict(list), "weak": [], "strong": [],
        "drop": 1.0, "weak_at_drop": defaultdict(list),
    }))
    for entry in entries:
        emb, p = _parse_entry(entry)
        data = json.load(open(p))
        pair = _pair(data)
        P = by_emb[emb][pair]
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
    return by_emb


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


# best 그래프의 기본 선택(method 키). --best-methods 로 사용자가 자유롭게 바꾼다.
DEFAULT_BEST_METHODS = ["mf", "uniroute_train"]
# best 그래프에서만 쓰는 '보기 좋은' 라벨(없으면 collect_results.METHOD_LABEL 사용).
BEST_LABEL = {
    "mf": "MF Router",                       # tie→strong
    "uniroute_train": "UniRoute (K-Means)",  # honest K, Ψ=train (hard)
    "uni_r2": "Uni-R2 (soft Φ·Ψ)",           # R2-Router UniRoute fusion
}


def _best_label(method):
    return BEST_LABEL.get(method, METHOD_LABEL.get(method, method))


def _figure_best(pairs, out_png, title, best_methods):
    """사용자가 고른 best_methods 밴드 그래프(기본 MF tie→strong / UniRoute honest·Ψ=train).
    strong−drop%p 에서의 평균 weak% 를 마커+주석으로 표시하고 콘솔에도 출력."""
    fixed_best = [(m, _best_label(m)) for m in best_methods]
    plist = list(pairs.keys())
    fig, axes = plt.subplots(1, len(plist), figsize=(7 * len(plist), 5.5))
    if len(plist) == 1:
        axes = [axes]
    for ax, pair in zip(axes, plist):
        P = pairs[pair]
        wk = float(np.mean(P["weak"])); sg = float(np.mean(P["strong"])); drop = P["drop"]
        print(f"\n[{pair}]  weak@(strong−{drop:.0f}%p) average weak model %:")
        for method, label in fixed_best:
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
                    help="여러 seed의 eval_results.json 경로들. 'emb=path' 로 임베딩을 붙이면 "
                    "(예: e5=... cscr=...) 임베딩마다 별도 그림으로 그린다 (pair는 JSON에서 자동 그룹화).")
    ap.add_argument("--output-prefix", default="seed_curves")
    ap.add_argument("--graph-random", action="store_true",
                    help="전체 그래프에 random baseline도 포함")
    ap.add_argument("--figures", choices=["both", "all", "best"], default="both",
                    help="어떤 그림을 만들지: both(기본) / all(전체 method) / best(선택 method만)")
    ap.add_argument("--best-methods", nargs="+", default=DEFAULT_BEST_METHODS,
                    help="best 그래프에 그릴 method 키들 (기본: mf uniroute_train). "
                    "예: --best-methods mf uniroute_train uni_r2 r2_router")
    ap.add_argument("--all-methods", nargs="+", default=None,
                    help="all 그래프에 그릴 method 키들 (기본: 등록된 전체). 필터링용.")
    args = ap.parse_args()

    by_emb = load_runs(args.results_jsons)
    multi_emb = len(by_emb) > 1 or "default" not in by_emb

    def _suffix(emb):
        # 임베딩이 하나뿐이고 라벨이 default면 접미사 없음. 아니면 _<emb> 로 파일 분리.
        return "" if (emb == "default" and not multi_emb) else f"_{emb}"

    def _emb_tag(emb):
        return "" if emb == "default" else f" [{emb}]"

    for emb, pairs in by_emb.items():
        sfx = _suffix(emb); tag = _emb_tag(emb)

        # (1) 전체 method 밴드 (임베딩마다 별도 그림 → e5 / cscr 분리)
        if args.figures in ("both", "all"):
            base = args.all_methods or [
                "mf", "uniroute", "uniroute_train", "uni_r2", "uniroute_legacy",
                "r2_router", "cscr",
            ]
            all_methods = (["random"] if args.graph_random else []) + base
            _figure(pairs, lambda pair: all_methods, f"{args.output_prefix}_all{sfx}.png",
                    f"Seed-averaged deferral curves (mean ± 1 std){tag}")

        # (2) best: 사용자 선택 method + strong−drop 평균 weak% 표시
        if args.figures in ("both", "best"):
            title = " vs ".join(_best_label(m) for m in args.best_methods) + f" — mean ± 1 std{tag}"
            _figure_best(pairs, f"{args.output_prefix}_best{sfx}.png", title, args.best_methods)


if __name__ == "__main__":
    main()
