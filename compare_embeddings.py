"""
seed 평균(41/42/43) 기반, 임베딩별(e5 / Qwen3-Embedding-0.6B) 최종 리포트 + 비교.

최종 method 5개만: MF / UniRoute(=uniroute_train) / Uni-R2 / R2-Router / CSCR.
모든 headline 수치는 **Per-Category 시트의 선형보간 weak%** — 즉 각 seed eval_results.json
의 per_category[method].operating_point.weak_pct_interp (Strong acc−drop%p 에서 보간한
weak 사용률) 을 seed 평균한 값이다(측정 point 원값이 아님).

추가로 deferral curve AUC(정확도 vs strong%, 면적)를 seed 평균으로 계산하고,
e5 → Qwen 교체 시 (a) 보간 weak% 증가량, (b) 두 임베딩 weak% 차이, (c) AUC 개선을 낸다.

입력은 plot_seed_curves 와 동일한 'emb=path' 규약:
  python compare_embeddings.py --results-jsons \
     e5=results_e5_seed41/pair_0.8B/eval_results.json e5=... \
     qwen=results_qwen_seed41/pair_0.8B/eval_results.json qwen=... \
     --output-prefix final --output-excel final_report.xlsx
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# numpy 2.x 는 np.trapz 를 제거(→ trapezoid). 서버 numpy(구버전)엔 trapz 존재.
_TRAPZ = getattr(np, "trapz", None) or np.trapezoid

GRID = np.linspace(0.0, 100.0, 101)  # 공통 strong% 격자

# 최종 5개 method (키, 표시라벨, 색)
FINAL = [
    ("mf",             "MF",        "#2196F3"),
    ("uniroute_train", "UniRoute",  "#E91E63"),
    ("uni_r2",         "Uni-R2",    "#00ACC1"),
    ("r2_router",      "R2-Router", "#4CAF50"),
    ("cscr",           "CSCR",      "#795548"),
]
LABEL = {k: lbl for k, lbl, _ in FINAL}
COLOR = {k: c for k, _, c in FINAL}
KEYS = [k for k, _, _ in FINAL]


def _parse_entry(entry):
    if "=" in entry:
        emb, path = entry.split("=", 1)
        return emb, path
    return "default", entry


def _pair(data):
    w = data["weak_model"].split("/")[-1]
    s = data["strong_model"].split("/")[-1]
    return f"{w} vs {s}"


def _auc(pts):
    """deferral curve 면적 / 100 = strong%∈[0,100] 스윕 평균 정확도(정확도 단위)."""
    pts = sorted(pts)
    xs = np.array([x for x, _ in pts], dtype=float)
    ys = np.array([y for _, y in pts], dtype=float)
    if len(xs) < 2:
        return float("nan")
    return float(_TRAPZ(ys, xs) / (xs.max() - xs.min()))


def load(entries):
    """by_emb[emb][pair] = {curves, weak, strong, drop, weak_interp{m:[..]}, auc{m:[..]}}."""
    by_emb = defaultdict(lambda: defaultdict(lambda: {
        "curves": defaultdict(list), "weak": [], "strong": [], "drop": 1.0,
        "weak_interp": defaultdict(list), "auc": defaultdict(list),
    }))
    for entry in entries:
        emb, path = _parse_entry(entry)
        data = json.load(open(path))
        P = by_emb[emb][_pair(data)]
        P["weak"].append(data["weak_only_accuracy"])
        P["strong"].append(data["strong_only_accuracy"])
        P["drop"] = data.get("per_category_pass_drop", 1.0)
        # deferral curve → method별 (strong%, acc)
        curve = defaultdict(list)
        for r in data["results"]:
            curve[r["method"]].append((r["strong_percentage"], r["accuracy"]))
        for m in KEYS:
            if m in curve:
                pts = sorted(curve[m])
                xs = np.array([x for x, _ in pts]); ys = np.array([y for _, y in pts])
                P["curves"][m].append(np.interp(GRID, xs, ys))
                P["auc"][m].append(_auc(pts))
        # 선형보간 weak% (headline)
        for m, blk in (data.get("per_category") or {}).items():
            if m in KEYS:
                wi = blk.get("operating_point", {}).get("weak_pct_interp")
                if wi is not None:
                    P["weak_interp"][m].append(float(wi))
    return by_emb


def _ms(xs):
    a = np.array([x for x in xs if x is not None and not np.isnan(x)], dtype=float)
    return (float(a.mean()), float(a.std())) if len(a) else (float("nan"), float("nan"))


def op_weak_from_curve(mean, target):
    """seed평균 deferral 곡선(GRID strong% 축의 pass%)이 pass=target 에 처음 도달하는
    strong% 를 선형보간으로 찾아 weak% = 100 − strong% 반환. 이 값을 쓰면 그래프의 점이
    '곡선 ∩ (strong−drop) 수평선' 에 정확히 얹히고, 표/legend 수치와도 완전히 일치한다."""
    mean = np.asarray(mean, dtype=float)
    idx = np.where(mean >= target)[0]
    if len(idx) == 0:
        return 0.0            # target 도달 못함 → 전부 strong 필요 → weak%=0
    j = int(idx[0])
    if j == 0:
        return 100.0          # 0% strong(전부 weak)도 이미 target 이상 → weak%=100
    x0, y0, x1, y1 = GRID[j - 1], mean[j - 1], GRID[j], mean[j]
    xs = x0 if y1 == y0 else x0 + (target - y0) / (y1 - y0) * (x1 - x0)
    return float(100.0 - xs)


def op_weak(P, m):
    """(emb,pair,method) 의 operating-point weak% + 표시용 seed std. 없으면 (nan,nan)."""
    if m not in P["curves"] or not P["curves"][m]:
        return float("nan"), float("nan")
    mean = np.vstack(P["curves"][m]).mean(0)
    target = float(np.mean(P["strong"])) - P["drop"]
    _, wstd = _ms(P["weak_interp"][m])   # seed 변동(밴드/±) 지표
    return op_weak_from_curve(mean, target), wstd


def make_graph(emb, pairs, drop, out_png):
    plist = list(pairs.keys())
    fig, axes = plt.subplots(1, len(plist), figsize=(7 * len(plist), 5.5))
    if len(plist) == 1:
        axes = [axes]
    for ax, pair in zip(axes, plist):
        P = pairs[pair]
        wk = float(np.mean(P["weak"])); sg = float(np.mean(P["strong"])); dr = P["drop"]
        target = sg - dr
        for m in KEYS:
            if m not in P["curves"]:
                continue
            arr = np.vstack(P["curves"][m]); mean = arr.mean(0); std = arr.std(0)
            ow, wstd = op_weak(P, m)   # 평균 곡선 ∩ (strong−drop) 의 weak% + seed std
            wtxt = f"  ({ow:.1f}±{wstd:.1f}% @−{dr:.0f}%p)" if not np.isnan(ow) else ""
            ax.plot(GRID, mean, color=COLOR[m], lw=2.2, label=f"{LABEL[m]}{wtxt}")
            ax.fill_between(GRID, mean - std, mean + std, color=COLOR[m], alpha=0.13, lw=0)
            if not np.isnan(ow):
                # 점 = 곡선 ∩ (strong−drop): x=100−weak%(=strong%), y=target. 정확히 선 위·수평선 위.
                x = 100.0 - ow
                ax.plot([x], [target], "o", color=COLOR[m], ms=7, zorder=5)
                ax.annotate(f"{ow:.1f}%", (x, target), textcoords="offset points",
                            xytext=(4, 5), fontsize=8, fontweight="bold", color=COLOR[m])
        ax.axhline(wk, color="#555", ls=":", lw=1.0, label=f"Weak only ({wk:.1f}%)")
        ax.axhline(sg, color="#C62828", ls=":", lw=1.0, label=f"Strong only ({sg:.1f}%)")
        ax.axhline(sg - dr, color="#C62828", ls=(0, (1, 3)), lw=0.9, label=f"Strong −{dr:.0f}%p")
        ax.plot([0, 100], [wk, sg], color="#888", ls="--", lw=1.2, label="Random (diagonal)")
        ax.set_xlabel("Strong Model Calls (%)"); ax.set_ylabel("Pass Rate (%)")
        ax.set_title(pair, fontweight="bold"); ax.grid(True, alpha=0.3); ax.set_xlim(-2, 102)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Seed-averaged deferral curves — embedding: {emb}", fontsize=13)
    plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Seed-averaged per-embedding report + e5↔Qwen comparison")
    ap.add_argument("--results-jsons", nargs="+", required=True, help="emb=path ... (e5= / qwen=)")
    ap.add_argument("--output-prefix", default="final")
    ap.add_argument("--output-excel", default="final_report.xlsx")
    args = ap.parse_args()

    by_emb = load(args.results_jsons)
    embs = list(by_emb.keys())

    # (1) 임베딩별 그래프
    for emb, pairs in by_emb.items():
        make_graph(emb, pairs, None, f"{args.output_prefix}_{emb}.png")

    # (2) 임베딩별 표 + 비교표. weak% = 그래프 점과 동일한 '평균 곡선 ∩ strong−drop' 보간값.
    rows = []          # 임베딩별 상세
    for emb, pairs in by_emb.items():
        for pair, P in pairs.items():
            dr = P["drop"]
            for m in KEYS:
                if m not in P["curves"]:
                    continue
                ow, wstd = op_weak(P, m)
                amean, astd = _ms(P["auc"][m])
                rows.append({
                    "Embedding": emb, "Pair": pair, "Method": LABEL[m],
                    "Weak@-{:.0f}%p (interp) %".format(dr): round(ow, 2),
                    "Weak seed-std": round(wstd, 2),
                    "AUC mean": round(amean, 3), "AUC std": round(astd, 3),
                    "n_seeds": len(P["curves"][m]),
                })
    detail = pd.DataFrame(rows)

    # 비교: 두 임베딩(가정: 정확히 2개, 첫=baseline, 둘째=new)
    comp_rows = []
    if len(embs) == 2:
        base, new = embs
        pairs_all = set(by_emb[base]) & set(by_emb[new])
        for pair in sorted(pairs_all):
            Pb, Pn = by_emb[base][pair], by_emb[new][pair]
            for m in KEYS:
                if m not in Pb["curves"] or m not in Pn["curves"]:
                    continue
                wb, _ = op_weak(Pb, m); wn, _ = op_weak(Pn, m)
                ab, _ = _ms(Pb["auc"][m]); an, _ = _ms(Pn["auc"][m])
                comp_rows.append({
                    "Pair": pair, "Method": LABEL[m],
                    f"Weak% ({base})": round(wb, 2), f"Weak% ({new})": round(wn, 2),
                    "Δ Weak% (new−base)": round(wn - wb, 2),
                    f"AUC ({base})": round(ab, 3), f"AUC ({new})": round(an, 3),
                    "Δ AUC (new−base)": round(an - ab, 3),
                })
    comp = pd.DataFrame(comp_rows)

    with pd.ExcelWriter(args.output_excel) as xw:
        detail.to_excel(xw, sheet_name="Per-Embedding (interp)", index=False)
        if not comp.empty:
            comp.to_excel(xw, sheet_name="Embedding Comparison", index=False)
    print(f"Saved → {args.output_excel}")

    # (3) 콘솔 요약
    print("\n=== e5 → Qwen 비교 (선형보간 weak% @ Strong−drop, seed 평균) ===")
    if not comp.empty:
        for _, r in comp.iterrows():
            print(f"  [{r['Pair']}] {r['Method']:<10} "
                  f"weak% {r[[c for c in comp.columns if c.startswith('Weak% (') and base in c][0]]:5.1f} "
                  f"→ {r[[c for c in comp.columns if c.startswith('Weak% (') and new in c][0]]:5.1f} "
                  f"(Δ{r['Δ Weak% (new−base)']:+.1f}p)   "
                  f"AUC Δ{r['Δ AUC (new−base)']:+.3f}")
    else:
        print("  (비교하려면 정확히 두 임베딩을 emb= 접두사로 넣으세요)")


if __name__ == "__main__":
    main()
