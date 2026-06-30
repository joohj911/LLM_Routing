"""
UniRoute K-Means 라우터 학습 스크립트.

사용법:
  python lm_routing/routers/uniroute/train_uniroute.py \\
    --train-data ./bfcl_data_2B/train_data.json \\
    --npy-path   ./bfcl_data/embeddings.npy \\
    --output-path ./bfcl_data_2B/uniroute_model.pt \\
    --weak-model  Qwen/Qwen3.5-2B \\
    --strong-model Qwen/Qwen3.5-9B

학습 절차:
  1. train_data.json → 80%(cluster_train) / 20%(val) stratified split
  2. cluster_train 임베딩으로 K-Means 학습  (K 후보 sweep)
  3. 각 K에 대해 val에서 deferral curve AUC 계산 → 최적 K 선택
  4. 최적 K로 Ψ_weak[k], Ψ_strong[k] (per-cluster error rate) 계산
  5. 체크포인트 저장
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def compute_psi(cluster_labels: np.ndarray, pass_labels: np.ndarray, K: int) -> np.ndarray:
    """
    Per-cluster error rate: Ψ_k(model) = 1 − (pass rate in cluster k).

    cluster_labels : (N,) int  — cluster assignment for each val sample
    pass_labels    : (N,) bool — whether the model passed for each val sample
    """
    psi = np.zeros(K, dtype=np.float32)
    for k in range(K):
        mask = cluster_labels == k
        if mask.sum() == 0:
            psi[k] = 0.5  # uninformative prior for empty cluster
        else:
            psi[k] = 1.0 - float(pass_labels[mask].mean())
    return psi


def deferral_auc(
    scores: np.ndarray,
    weak_labels: np.ndarray,
    strong_labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """주어진 점수로 deferral curve 아래 면적을 계산 (K/τ 선택 기준)."""
    try:
        _, thresholds = pd.qcut(scores, n_bins, retbins=True, duplicates="drop")
    except ValueError:
        thresholds = np.linspace(scores.min(), scores.max(), n_bins + 1)

    accs, strong_pcts = [], []
    for j, thr in enumerate(thresholds):
        sel = scores >= thr if j < len(thresholds) - 1 else scores > thr
        results = np.where(sel, strong_labels, weak_labels)
        accs.append(results.mean())
        strong_pcts.append(sel.mean())

    order = np.argsort(strong_pcts)
    return float(np.trapz(np.array(accs)[order], np.array(strong_pcts)[order]))


def hard_scores(psi_weak: np.ndarray, psi_strong: np.ndarray,
                cluster_labels: np.ndarray) -> np.ndarray:
    """가장 가까운 클러스터 기준 strong_win_rate."""
    psi_diff = psi_weak - psi_strong
    return (psi_diff[cluster_labels] + 1.0) / 2.0


def soft_scores(embs: np.ndarray, centroids: np.ndarray,
                psi_weak: np.ndarray, psi_strong: np.ndarray, tau: float) -> np.ndarray:
    """softmax(−dist/τ) 가중 평균 기반 연속 strong_win_rate."""
    psi_diff = psi_weak - psi_strong                              # (K,)
    d = ((embs[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)  # (N, K)
    z = -d / max(tau, 1e-8)
    z -= z.max(axis=1, keepdims=True)
    w = np.exp(z)
    w /= w.sum(axis=1, keepdims=True)
    diff = w @ psi_diff                                           # (N,)
    return (diff + 1.0) / 2.0


def val_auc(
    psi_weak: np.ndarray,
    psi_strong: np.ndarray,
    cluster_labels: np.ndarray,
    weak_labels: np.ndarray,
    strong_labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Hard assignment 기준 deferral AUC (K 선택용)."""
    scores = hard_scores(psi_weak, psi_strong, cluster_labels)
    return deferral_auc(scores, weak_labels, strong_labels, n_bins)


def _tau_gap(embs: np.ndarray, centroids: np.ndarray) -> float:
    """1·2등 클러스터 거리차의 중앙값 — τ 스케일 기준(클러스터링이 바뀌어도 재계산 가능)."""
    d = ((embs[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    if d.shape[1] >= 2:
        part = np.partition(d, 1, axis=1)
        return float(np.median(part[:, 1] - part[:, 0])) + 1e-8
    return float(np.median(d)) + 1e-8


def tune_tau(
    val_embs: np.ndarray,
    centroids: np.ndarray,
    psi_weak: np.ndarray,
    psi_strong: np.ndarray,
    weak_labels: np.ndarray,
    strong_labels: np.ndarray,
) -> tuple[float, float]:
    """
    val에서 softmax 온도 τ를 스윕하여 (best_mult, best_soft_auc) 반환.
    τ = best_mult × gap 으로 저장하면 최종 refit(다른 centroids)에서도 재계산 가능.
    """
    gap = _tau_gap(val_embs, centroids)
    best_mult, best_auc = 1.0, -np.inf
    for mult in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        s = soft_scores(val_embs, centroids, psi_weak, psi_strong, gap * mult)
        auc = deferral_auc(s, weak_labels, strong_labels)
        if auc > best_auc:
            best_auc, best_mult = auc, mult
    return best_mult, best_auc


# ─────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────

def train_uniroute(
    train_data_path: str,
    npy_path: str,
    output_path: str,
    weak_model: str,
    strong_model: str,
    train_ratio: float = 0.8,
    k_candidates: list[int] | None = None,
    seed: int = 42,
    assignment: str = "hard",
    embedding_model: str = "intfloat/multilingual-e5-small",
    psi_source: str = "val",
) -> dict:
    print(f"\nLoading train data from {train_data_path}")
    df = pd.read_json(train_data_path)
    print(f"  {len(df)} samples, columns: {list(df.columns)}")

    for col in [weak_model, strong_model]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in train_data. "
                f"Available: {list(df.columns)}"
            )

    print(f"Loading embeddings from {npy_path}")
    all_embs = np.load(npy_path).astype(np.float32)  # (N_total, D)
    train_embs = all_embs[df["idx"].values]           # (N_train, D)
    D = train_embs.shape[1]
    print(f"  Embedding dim: {D}, total prompts: {all_embs.shape[0]}")

    # 80/20 stratified split
    indices = np.arange(len(df))
    stratify = df["bfcl_split"].values if "bfcl_split" in df.columns else None
    try:
        cl_idx, val_idx = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=stratify,
            random_state=seed,
        )
    except ValueError:
        cl_idx, val_idx = train_test_split(
            indices, train_size=train_ratio, random_state=seed
        )

    cl_embs = train_embs[cl_idx]
    val_embs = train_embs[val_idx]
    cl_df = df.iloc[cl_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    cl_weak = cl_df[weak_model].astype(bool).values
    cl_strong = cl_df[strong_model].astype(bool).values
    val_weak = val_df[weak_model].astype(bool).values
    val_strong = val_df[strong_model].astype(bool).values
    # 최종 모델용: 전체 train(cl+val) pass 라벨
    all_weak = df[weak_model].astype(bool).values
    all_strong = df[strong_model].astype(bool).values

    print(f"  cluster_train: {len(cl_idx)} samples, val: {len(val_idx)} samples")

    # K 후보: 넓게 스윕. K는 cluster_train 크기를 넘을 수 없음.
    n_cl = len(cl_idx)
    if k_candidates is None:
        k_candidates = [3, 5, 8, 10, 13, 15, 20, 25, 30, 40, 50, 75, 100]
        k_candidates.append(max(5, n_cl // 50))
        k_candidates = sorted(set(k_candidates))
    k_candidates = [k for k in k_candidates if 2 <= k <= n_cl]
    print(f"\nK candidates: {k_candidates}")

    # ── psi_source: Ψ(클러스터별 error rate)를 어디서 추정할지 ──
    #   "val"  : KMeans는 cl에서 fit, Ψ는 held-out val에서 추정 (원 UniRoute 논문 설계, 기본값)
    #   "train": Ψ를 cl에서 추정해 honest하게 K 선택 후, 최종 모델은 train 전체로 refit
    def _psi_for(km, K):
        if psi_source == "val":
            labels = km.predict(val_embs)
            return compute_psi(labels, val_weak, K), compute_psi(labels, val_strong, K)
        labels = km.labels_
        return compute_psi(labels, cl_weak, K), compute_psi(labels, cl_strong, K)

    print(f"  (psi_source={psi_source})")

    # ── K 선택 (평가는 항상 val) ──
    best = {"K": None, "auc": -np.inf}
    for K in k_candidates:
        km = KMeans(n_clusters=K, random_state=seed, n_init=10, max_iter=300).fit(cl_embs)
        psi_w, psi_s = _psi_for(km, K)
        val_labels = km.predict(val_embs)
        auc = deferral_auc(hard_scores(psi_w, psi_s, val_labels), val_weak, val_strong)
        print(f"  K={K:3d} → val AUC = {auc:.5f}")
        if auc > best["auc"]:
            best.update({"K": K, "auc": auc})

    print(f"\nBest K={best['K']}  (val AUC={best['auc']:.5f})")

    # best K 재클러스터링(cl fit) → τ 튜닝과 val-모드 최종 모델에서 공통 사용
    km_sel = KMeans(n_clusters=best["K"], random_state=seed, n_init=10, max_iter=300).fit(cl_embs)
    sel_centroids = km_sel.cluster_centers_.astype(np.float32)
    psi_w_sel, psi_s_sel = _psi_for(km_sel, best["K"])

    # ── τ 선택(soft/auto): val에서 튜닝 ──
    final_assignment, tau_mult = "hard", 1.0
    if assignment in ("soft", "auto"):
        val_labels_sel = km_sel.predict(val_embs)
        hard_auc = deferral_auc(hard_scores(psi_w_sel, psi_s_sel, val_labels_sel), val_weak, val_strong)
        best_mult, soft_auc = tune_tau(val_embs, sel_centroids, psi_w_sel, psi_s_sel, val_weak, val_strong)
        print(f"  hard val AUC = {hard_auc:.5f}   soft val AUC = {soft_auc:.5f} (×{best_mult})")
        if assignment == "soft" or soft_auc > hard_auc:
            final_assignment, tau_mult = "soft", best_mult

    # ── 최종 모델 ──
    if psi_source == "train":
        # train 전체로 refit, Ψ도 전체에서 추정
        km_final = KMeans(n_clusters=best["K"], random_state=seed, n_init=10, max_iter=300).fit(train_embs)
        centroids = km_final.cluster_centers_.astype(np.float32)
        psi_w_final = compute_psi(km_final.labels_, all_weak, best["K"])
        psi_s_final = compute_psi(km_final.labels_, all_strong, best["K"])
        tau_embs = train_embs
        print(f"  Final: refit on all {len(train_embs)} train samples, Ψ from train")
    else:
        # 논문 설계: 클러스터는 cl, Ψ는 val
        centroids = sel_centroids
        psi_w_final, psi_s_final = psi_w_sel, psi_s_sel
        tau_embs = val_embs
        print(f"  Final: clusters from cl, Ψ from val ({len(val_idx)} samples)")

    final_tau = tau_mult * _tau_gap(tau_embs, centroids) if final_assignment == "soft" else 1.0
    print(f"  Ψ_weak(mean={psi_w_final.mean():.3f})  Ψ_strong(mean={psi_s_final.mean():.3f})")
    print(f"  → assignment = {final_assignment}" + (f" (τ={final_tau:.4g})" if final_assignment == "soft" else ""))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "K": best["K"],
            "centroids": centroids,
            "psi_weak": psi_w_final,
            "psi_strong": psi_s_final,
            "weak_model": weak_model,
            "strong_model": strong_model,
            "val_auc": best["auc"],
            "assignment": final_assignment,
            "tau": final_tau,
            "psi_source": psi_source,
            "embedding_model": embedding_model,
            "embedding_prefix": "query: ",
        },
        output_path,
    )
    print(f"Saved UniRoute checkpoint → {output_path}\n")
    return {"K": best["K"], "auc": best["auc"], "assignment": final_assignment, "tau": final_tau}


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UniRoute K-Means router")
    parser.add_argument("--train-data", required=True, help="train_data.json from prepare_bfcl_data.py convert")
    parser.add_argument("--npy-path",   required=True, help="embeddings.npy from prepare_bfcl_data.py embed")
    parser.add_argument("--output-path", required=True, help="Output .pt checkpoint path")
    parser.add_argument("--weak-model",   required=True, help="HuggingFace weak model ID")
    parser.add_argument("--strong-model", required=True, help="HuggingFace strong model ID")
    parser.add_argument("--train-ratio",  type=float, default=0.8, help="Fraction for cluster training (default: 0.8)")
    parser.add_argument(
        "--k-candidates",
        type=int, nargs="+", default=None,
        help="K values to try (default: auto from val size)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--assignment", choices=["hard", "soft", "auto"], default="hard",
        help="hard=최근접 클러스터(기존), soft=softmax 가중평균(연속 점수), "
        "auto=val AUC가 더 높은 쪽 자동 선택",
    )
    parser.add_argument(
        "--embedding-model", type=str, default="intfloat/multilingual-e5-small",
        help="embeddings.npy를 만든 임베딩 모델. 체크포인트에 기록되어 추론 인코딩에 사용.",
    )
    parser.add_argument(
        "--psi-source", choices=["val", "train"], default="val",
        help="Ψ(클러스터별 error) 추정 데이터. "
        "val=held-out val에서 추정(원 UniRoute 논문 설계, 기본), "
        "train=cl에서 K 선택 후 train 전체로 refit(데이터 더 사용).",
    )
    args = parser.parse_args()

    train_uniroute(
        train_data_path=args.train_data,
        npy_path=args.npy_path,
        output_path=args.output_path,
        weak_model=args.weak_model,
        strong_model=args.strong_model,
        train_ratio=args.train_ratio,
        k_candidates=args.k_candidates,
        seed=args.seed,
        assignment=args.assignment,
        embedding_model=args.embedding_model,
        psi_source=args.psi_source,
    )
