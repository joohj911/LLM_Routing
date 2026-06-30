"""
라우터 점수 분포·판별력 분석 스크립트.

학습된 라우터(MF / UniRoute)가 프롬프트마다 실제로 점수를 "차별화"하는지,
아니면 상수에 가깝게 붕괴했는지를 진단한다. evaluate.py의 deferral curve는
qcut으로 분위를 자르기 때문에 점수가 거의 상수여도 곡선은 그려지므로, 붕괴를
직접 보려면 점수 분포 자체를 봐야 한다.

사용법:
  python analyze_routers.py \\
    --test-data ./bfcl_data_0.8B/test_data.json \\
    --mf-checkpoint ./bfcl_data_0.8B/mf_model.pt \\
    --uniroute-checkpoint ./bfcl_data_0.8B/uniroute_model.pt \\
    --strong-model Qwen/Qwen3.5-9B \\
    --weak-model   Qwen/Qwen3.5-0.8B

보는 법:
  - std / n_unique 가 매우 작으면 → 라우터가 상수로 붕괴 (프롬프트 무시)
  - AUC(route-strong) ≈ 0.5 → 점수에 라우팅 신호가 사실상 없음 (= random)
  - AUC > 0.6 → 점수가 weak 실패를 유의미하게 예측
"""

import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def score_all(router, prompts: list[str]) -> np.ndarray:
    return np.array(
        [float(router.calculate_strong_win_rate(p)) for p in tqdm(prompts, desc="scoring")],
        dtype=np.float64,
    )


def text_histogram(scores: np.ndarray, bins: int = 20, width: int = 50) -> str:
    lo = min(scores.min(), 0.0)
    hi = max(scores.max(), 1.0)
    counts, edges = np.histogram(scores, bins=bins, range=(lo, hi))
    mx = counts.max() or 1
    rows = []
    for c, l, h in zip(counts, edges[:-1], edges[1:]):
        bar = "█" * int(round(width * c / mx))
        rows.append(f"    [{l:5.3f}, {h:5.3f})  {c:5d}  {bar}")
    return "\n".join(rows)


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if not _HAS_SKLEARN or len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def deferral_curve(scores, weak_pass, strong_pass, points=11):
    """
    점수 내림차순으로 상위 f 비율을 strong에 보냈을 때의 (strong%, pass_rate%) 곡선.
    곡선의 '모양'을 봐서 strong%를 줄일 때 성능이 얼마나 빨리 꺾이는지 직접 확인한다.
    """
    order = np.argsort(-scores)              # 높은 점수 = strong 우선
    sp, wp = strong_pass[order], weak_pass[order]
    n = len(scores)
    out = []
    for f in np.linspace(0, 1, points):
        k = int(round(f * n))                # 상위 k개 → strong, 나머지 → weak
        correct = sp[:k].sum() + wp[k:].sum()
        out.append((f * 100.0, correct / n * 100.0))
    return out


def analyze_one(name: str, scores: np.ndarray, df: pd.DataFrame,
                weak_col: str, strong_col: str):
    weak_pass = df[weak_col].astype(bool).values
    strong_pass = df[strong_col].astype(bool).values

    # 학습 타깃: weak가 실패하면 strong으로 보내야 함
    route_strong = (~weak_pass).astype(int)
    # 실제로 strong이 weak를 고쳐주는 경우(이상적 라우팅 이득)
    strong_fixes = (strong_pass & ~weak_pass).astype(int)

    sep = "─" * 64
    print(f"\n{sep}\n[{name}]  n={len(scores)}\n{sep}")
    print(f"  score  min={scores.min():.4f}  max={scores.max():.4f}  "
          f"mean={scores.mean():.4f}  std={scores.std():.4f}")
    print(f"  range (max-min) = {scores.max() - scores.min():.4f}   "
          f"unique values = {len(np.unique(np.round(scores, 4)))}")
    pcts = np.percentile(scores, [10, 25, 50, 75, 90])
    print(f"  pctiles 10/25/50/75/90 = "
          + " / ".join(f"{p:.4f}" for p in pcts))

    auc_rs = safe_auc(route_strong, scores)
    auc_sf = safe_auc(strong_fixes, scores)
    print(f"  AUC(score → weak 실패=route-strong) = {auc_rs:.4f}   "
          f"(0.5=신호없음, 1.0=완벽)")
    print(f"  AUC(score → strong이 weak를 고침)    = {auc_sf:.4f}")

    # 붕괴 판정
    if scores.std() < 0.02 or (scores.max() - scores.min()) < 0.05:
        print("  ⚠️  COLLAPSED: 점수가 거의 상수 — 라우터가 프롬프트를 구분하지 못함.")
    elif not np.isnan(auc_rs) and auc_rs < 0.55:
        print("  ⚠️  WEAK SIGNAL: 점수가 퍼져 있어도 라우팅 예측력이 random 수준.")
    else:
        print("  ✓  점수가 분포하며 라우팅 신호 있음.")

    # Deferral curve 형태: strong%를 줄일 때 pass rate가 얼마나 유지되는가
    weak_acc = weak_pass.mean() * 100
    strong_acc = strong_pass.mean() * 100
    curve = deferral_curve(scores, weak_pass, strong_pass, points=11)
    print(f"  Deferral curve (weak-only={weak_acc:.1f}%, strong-only={strong_acc:.1f}%):")
    print(f"    {'strong%':>8} {'weak%':>7} {'pass%':>7}")
    for strong_pct, passr in curve:
        print(f"    {strong_pct:>7.0f}% {100 - strong_pct:>6.0f}% {passr:>6.1f}%")

    # 핵심 지표: strong-only 성능을 유지하면서 weak로 보낼 수 있는 최대 비율.
    # (strong%를 100→0으로 줄이며 pass_rate ≥ strong-only 인 가장 큰 weak%)
    fine = deferral_curve(scores, weak_pass, strong_pass, points=101)
    max_weak_at_full = 0.0
    for strong_pct, passr in fine:
        if passr >= strong_acc - 1e-9:
            max_weak_at_full = max(max_weak_at_full, 100 - strong_pct)
    print(f"  → 성능 손실 없이(≥strong-only) weak로 보낼 수 있는 최대 비율: {max_weak_at_full:.0f}%")

    print("  분포 히스토그램:")
    print(text_histogram(scores))

    # split별 평균 점수 (카테고리에 따라 점수가 달라지는가?)
    if "bfcl_split" in df.columns:
        print("  split별 평균 점수:")
        tmp = df.copy()
        tmp["_score"] = scores
        for split, g in tmp.groupby("bfcl_split"):
            print(f"    {split:<34} mean={g['_score'].mean():.4f}  n={len(g)}")

    return {
        "name": name,
        "std": float(scores.std()),
        "auc_route_strong": auc_rs,
        "auc_strong_fixes": auc_sf,
        "max_weak_at_full": max_weak_at_full,
        "weak_acc": weak_acc,
        "strong_acc": strong_acc,
    }


def _parse_labeled(items):
    """'label=path' 또는 'path'(라벨=basename) 리스트를 [(label, path)]로."""
    import os
    out = []
    for it in items or []:
        if "=" in it:
            label, path = it.split("=", 1)
        else:
            label, path = os.path.splitext(os.path.basename(it))[0], it
        out.append((label, path))
    return out


def main():
    ap = argparse.ArgumentParser(description="라우터 점수 분포/판별력 분석 (다중 체크포인트 비교)")
    ap.add_argument("--test-data", required=True, help="test_data.json")
    ap.add_argument("--mf-checkpoint", nargs="+", default=None,
                    help="MF 체크포인트. 여러 개 가능. 'label=path' 형식으로 라벨 지정 가능.")
    ap.add_argument("--uniroute-checkpoint", nargs="+", default=None,
                    help="UniRoute 체크포인트. 여러 개 가능. 'label=path' 형식 가능.")
    ap.add_argument("--permodel-checkpoint", nargs="+", default=None,
                    help="Per-model 회귀 라우터 체크포인트. 여러 개 가능. 'label=path' 형식 가능.")
    ap.add_argument("--strong-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--weak-model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--text-dim", type=int, default=384)
    ap.add_argument("--save-scores", default=None,
                    help="설정 시 프롬프트별 점수를 이 CSV에 저장")
    args = ap.parse_args()

    df = pd.read_json(args.test_data)
    for col in (args.weak_model, args.strong_model):
        if col not in df.columns:
            raise SystemExit(
                f"'{col}' 컬럼이 test_data에 없습니다. 있는 컬럼: {list(df.columns)}"
            )
    prompts = df["prompt"].tolist()
    print(f"Loaded {len(df)} test samples from {args.test_data}")
    print(f"  weak  = {args.weak_model}  (pass rate {df[args.weak_model].mean()*100:.1f}%)")
    print(f"  strong= {args.strong_model}  (pass rate {df[args.strong_model].mean()*100:.1f}%)")
    if not _HAS_SKLEARN:
        print("  [warn] scikit-learn 없음 → AUC는 NaN으로 표시됩니다.")

    summaries = []
    score_cols = {}

    for label, path in _parse_labeled(args.mf_checkpoint):
        from lm_routing.routers.routers import MatrixFactorizationRouter
        print(f"\nLoading MF router [{label}]: {path}")
        mf = MatrixFactorizationRouter(
            checkpoint_path=path,
            strong_model=args.strong_model,
            weak_model=args.weak_model,
            text_dim=args.text_dim,
        )
        s = score_all(mf, prompts)
        score_cols[f"mf:{label}"] = s
        summaries.append(analyze_one(f"MF [{label}]", s, df, args.weak_model, args.strong_model))

    for label, path in _parse_labeled(args.uniroute_checkpoint):
        from lm_routing.routers.routers import UniRouteRouter
        print(f"\nLoading UniRoute router [{label}]: {path}")
        uni = UniRouteRouter(checkpoint_path=path)
        s = score_all(uni, prompts)
        score_cols[f"uniroute:{label}"] = s
        summaries.append(analyze_one(f"UniRoute [{label}]", s, df, args.weak_model, args.strong_model))

    for label, path in _parse_labeled(args.permodel_checkpoint):
        from lm_routing.routers.routers import PerModelRouter
        print(f"\nLoading Per-model router [{label}]: {path}")
        pm = PerModelRouter(checkpoint_path=path)
        s = score_all(pm, prompts)
        score_cols[f"permodel:{label}"] = s
        summaries.append(analyze_one(f"PerModel [{label}]", s, df, args.weak_model, args.strong_model))

    if not summaries:
        raise SystemExit("--mf-checkpoint 또는 --uniroute-checkpoint 중 하나는 필요합니다.")

    # ── 비교 요약 표 ──
    print("\n" + "=" * 78)
    print("COMPARISON  (weak-only={:.1f}%  strong-only={:.1f}%)".format(
        summaries[0]["weak_acc"], summaries[0]["strong_acc"]))
    print("=" * 78)
    print(f"  {'router':<26} {'std':>7} {'AUC(rt-str)':>12} {'AUC(fix)':>9} {'maxWeak@full':>13}")
    print("  " + "-" * 70)
    for r in sorted(summaries, key=lambda x: (x["max_weak_at_full"], x["auc_route_strong"]), reverse=True):
        print(f"  {r['name']:<26} {r['std']:>7.3f} {r['auc_route_strong']:>12.4f} "
              f"{r['auc_strong_fixes']:>9.4f} {r['max_weak_at_full']:>12.0f}%")
    print("=" * 78)
    print("  maxWeak@full = 성능 손실 없이(≥strong-only) weak로 보낼 수 있는 최대 비율 (클수록 좋음)")

    if args.save_scores:
        out = df[["id", "bfcl_split"]].copy() if "id" in df.columns else pd.DataFrame()
        for name, s in score_cols.items():
            out[f"{name}_score"] = s
        out.to_csv(args.save_scores, index=False)
        print(f"\nSaved per-prompt scores → {args.save_scores}")


if __name__ == "__main__":
    main()
