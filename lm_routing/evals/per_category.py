"""
카테고리(BFCL split)별 라우팅 성공률 분해.

전체를 pool한 deferral curve / maxWeak 지표는 라우터가 '어느 카테고리를
weak로 보내 이득/손해를 봤는지'를 감춘다. BFCL single-turn은 난이도가 크게
다른 여러 task(simple / multiple / parallel / irrelevance / live_* ...)가 섞여
있으므로, 전체 pass율이 좋아도 특정 카테고리를 통째로 잘못 라우팅하고 있을 수
있다.

이 모듈은 하나의 전역 운영점 — 성능 손실을 max_pass_drop%p 이내로 유지하면서
weak를 최대로 보내는 지점 (기본 1%p = 프로젝트 헤드라인 지표와 동일) — 에서
카테고리별로 다음을 계산한다.
  - weak / strong 단독 pass율      (그 카테고리 난이도의 하·상한)
  - 라우터가 그 카테고리를 weak로 보낸 비율 (weak_sent_pct)
  - 라우팅 후 실제 pass율 (router_pass) 과 strong 대비 regret (= strong − router)

GPU/모델 재평가 없이 pass/fail 불리언 + 라우터 score만으로 동작한다.
"""
from typing import Sequence

import numpy as np


def max_weak_operating_point(
    scores: Sequence[float],
    weak_pass: Sequence[bool],
    strong_pass: Sequence[bool],
    max_pass_drop: float = 1.0,
):
    """전역 운영점에서의 strong 선택 mask와 요약을 반환.

    배포 가능한 threshold 라우팅(evaluate.py deferral curve와 동일 의미)을 가정한다:
    점수 t에 대해 score ≥ t 이면 strong. 따라서 **동점(tie) 점수는 항상 함께 이동**
    하며, 하나의 컷으로 동점 블록을 반쪽만 보낼 수는 없다 (클러스터 기반 UniRoute처럼
    점수가 뭉치는 라우터에서 중요). 후보 threshold(= 각 고유 점수, 그리고 전부 weak인
    경우)를 모두 시도해, pass율이 (strong-only − max_pass_drop%p) 이상으로 유지되는
    범위에서 weak 비율이 가장 큰 컷을 고른다.

    Returns:
        strong_mask (bool[n]) : True면 strong으로 라우팅
        strong_pct, weak_pct, pass_pct (float) : 그 운영점의 전역 수치
    """
    scores = np.asarray(scores, dtype=float)
    wp = np.asarray(weak_pass, dtype=bool)
    sp = np.asarray(strong_pass, dtype=bool)
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=bool), 0.0, 0.0, 0.0

    strong_acc = sp.mean()
    target = strong_acc - max_pass_drop / 100.0

    # 후보 컷: 각 고유 점수 t (score ≥ t → strong) + 최댓값 위(전부 weak).
    # t를 올릴수록 strong이 줄고 weak가 는다. pass ≥ target을 지키며 weak 최대인 컷 선택.
    uniq = np.unique(scores)
    candidates = np.concatenate([uniq, [uniq[-1] + 1.0]])

    best_mask = np.ones(n, dtype=bool)   # 기본값: 전부 strong (weak 0, pass=strong_acc)
    best_weak = -1.0
    best_pass = float(strong_acc)
    for t in candidates:
        strong_mask = scores >= t
        passr = float(np.where(strong_mask, sp, wp).mean())
        if passr >= target - 1e-12:
            weak_frac = 1.0 - float(strong_mask.mean())
            if weak_frac > best_weak:
                best_weak = weak_frac
                best_mask = strong_mask
                best_pass = passr

    return (
        best_mask,
        float(best_mask.mean()) * 100.0,
        best_weak * 100.0,
        best_pass * 100.0,
    )


def per_category_breakdown(
    splits: Sequence,
    scores: Sequence[float],
    weak_pass: Sequence[bool],
    strong_pass: Sequence[bool],
    max_pass_drop: float = 1.0,
):
    """전역 운영점에서 카테고리별 지표 표를 계산.

    Returns:
        summary (dict) : {"strong_pct","weak_pct","pass_pct","max_pass_drop"}
        rows (list[dict]) : 카테고리별 행 + 마지막 TOTAL 행
    """
    splits = np.asarray(list(splits), dtype=object)
    wp = np.asarray(weak_pass, dtype=bool)
    sp = np.asarray(strong_pass, dtype=bool)
    strong_mask, strong_pct, weak_pct, pass_pct = max_weak_operating_point(
        scores, wp, sp, max_pass_drop=max_pass_drop
    )
    routed_pass = np.where(strong_mask, sp, wp)

    def _row(name, mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return None
        return {
            "category": name,
            "n": int(len(idx)),
            "weak_acc": round(float(wp[idx].mean()) * 100, 2),
            "strong_acc": round(float(sp[idx].mean()) * 100, 2),
            "weak_sent_pct": round(float((~strong_mask[idx]).mean()) * 100, 2),
            "router_pass": round(float(routed_pass[idx].mean()) * 100, 2),
            "regret": round(
                float(sp[idx].mean() - routed_pass[idx].mean()) * 100, 2
            ),
        }

    rows = []
    for cat in sorted(set(splits.tolist())):
        r = _row(cat, splits == cat)
        if r:
            rows.append(r)
    total = _row("TOTAL", np.ones(len(sp), dtype=bool))
    if total:
        rows.append(total)

    summary = {
        "strong_pct": round(strong_pct, 2),
        "weak_pct": round(weak_pct, 2),
        "pass_pct": round(pass_pct, 2),
        "max_pass_drop": max_pass_drop,
    }
    return summary, rows


def format_breakdown_table(method: str, summary: dict, rows: list) -> str:
    """콘솔 출력용 문자열 표."""
    op = (
        f"operating point: pass ≥ strong-only − {summary['max_pass_drop']:.1f}%p  "
        f"→ weak={summary['weak_pct']:.0f}%  strong={summary['strong_pct']:.0f}%  "
        f"pass={summary['pass_pct']:.1f}%"
    )
    lines = [f"  [{method}]  {op}"]
    lines.append(
        f"    {'category':<28} {'n':>4} {'weak%':>6} {'strong%':>8} "
        f"{'→weak':>6} {'router%':>8} {'regret':>7}"
    )
    lines.append("    " + "-" * 74)
    for r in rows:
        sep = "    " + "-" * 74 if r["category"] == "TOTAL" else None
        if sep:
            lines.append(sep)
        lines.append(
            f"    {r['category']:<28} {r['n']:>4} {r['weak_acc']:>6.1f} "
            f"{r['strong_acc']:>8.1f} {r['weak_sent_pct']:>5.0f}% "
            f"{r['router_pass']:>8.1f} {r['regret']:>+7.1f}"
        )
    lines.append(
        "    regret = strong-only − router pass (양수 = 그 카테고리에서 strong 대비 손해)"
    )
    return "\n".join(lines)
