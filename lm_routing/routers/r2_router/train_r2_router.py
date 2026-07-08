"""
R2-Router per-model 라우터 학습 스크립트.

R2-Router(github: UCF-ML-Research/R2-Router)의 메인 라우터(`r2_router/router.py`)는
LLM마다 쿼리 임베딩 → 품질을 예측하는 **Ridge 회귀기**를 두고, risk 목적함수
    risk(h) = (1−λ)·quality(h) − λ·cost(h)
를 모든 (model, budget) 조합에서 최대화해 라우팅한다. 여기서는 2-모델·짧은 출력
설정이라 (a) budget tier를 단일('unlimited')로 접고, (b) cost 를 화폐가 아니라
UniRoute처럼 c_weak=0, c_strong=1 로 둔다.

이 두 축소는 임의 삭제가 아니라 정당한 특수화다: 2모델·상수 cost에서는
    strong 선택 ⟺ (1−λ)(P_s − P_w) > λ(c_s − c_w) ⟺ (P_s − P_w) > λ/(1−λ)·Δc
가 되어, **라우팅 순서는 오직 gain P_s−P_w 로만 결정**된다(상수 cost는 프롬프트 순위를
안 바꾸고 λ↔threshold 대응만 바꾼다). 따라서 λ 스윕 = threshold 스윕이고 deferral
curve가 동일하다. BFCL은 함수호출 출력이 짧고 균일해 budget-conditioned 품질이 하나로
접혀 정보 손실도 없다.

각 모델(weak, strong)에 대해 임베딩 → P(pass) Ridge 회귀기를 독립 학습하고,
라우팅 점수 = (P_strong − P_weak + 1)/2 (gain에 단조) 로 deferral curve를 그린다.

사용법:
  python lm_routing/routers/r2_router/train_r2_router.py \\
    --train-data ./bfcl_data_0.8B/train_data.json \\
    --npy-path   ./bfcl_data/embeddings.npy \\
    --output-path ./bfcl_data_0.8B/r2_router_model.pt \\
    --weak-model  Qwen/Qwen3.5-0.8B \\
    --strong-model Qwen/Qwen3.5-9B

절차:
  1. bfcl_split 기준 stratified cl/val split (val은 보고용)
  2. cl 에서 weak/strong 회귀기 학습 → val deferral AUC 출력
  3. 전체 train(cl+val) 으로 최종 회귀기 재학습 → 체크포인트 저장
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class _Constant:
    """한 모델이 train에서 전부 pass(또는 전부 fail)일 때의 상수 예측기."""

    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, X):
        return np.tile([1.0 - self.p, self.p], (len(X), 1))

    def predict(self, X):
        return np.full(len(X), self.p)


def fit_regressor(X: np.ndarray, y: np.ndarray, kind: str, reg_C: float):
    """임베딩 X → pass 라벨 y 회귀기. 단일 클래스면 상수 예측기."""
    yb = np.asarray(y).astype(int)
    if len(np.unique(yb)) < 2:
        return _Constant(float(yb.mean()))
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0 / max(reg_C, 1e-8))).fit(X, yb.astype(float))
    return make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, C=reg_C)
    ).fit(X, yb)


def _proba(clf, X: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return np.clip(clf.predict(X), 0.0, 1.0)


def deferral_auc(scores, weak_labels, strong_labels, n_bins: int = 10) -> float:
    """deferral curve(정확도 vs strong%) 아래 면적 — val 비교용."""
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


def train_r2_router(
    train_data_path: str,
    npy_path: str,
    output_path: str,
    weak_model: str,
    strong_model: str,
    regressor: str = "ridge",
    reg_C: float = 1.0,
    train_ratio: float = 0.8,
    seed: int = 42,
    embedding_model: str = "intfloat/multilingual-e5-small",
    cost_weak: float = 0.0,
    cost_strong: float = 1.0,
) -> dict:
    print(f"\nLoading train data from {train_data_path}")
    df = pd.read_json(train_data_path)
    print(f"  {len(df)} samples, columns: {list(df.columns)}")
    for col in [weak_model, strong_model]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    print(f"Loading embeddings from {npy_path}")
    all_embs = np.load(npy_path).astype(np.float32)
    X = all_embs[df["idx"].values]
    y_weak = df[weak_model].astype(int).values
    y_strong = df[strong_model].astype(int).values
    print(f"  Embedding dim: {X.shape[1]}, regressor={regressor}, C={reg_C}")

    # ── stratified cl/val split (val은 deferral AUC 보고용) ──
    idx = np.arange(len(df))
    stratify = df["bfcl_split"].values if "bfcl_split" in df.columns else None
    try:
        cl, val = train_test_split(idx, train_size=train_ratio, stratify=stratify, random_state=seed)
    except ValueError:
        cl, val = train_test_split(idx, train_size=train_ratio, random_state=seed)

    wclf = fit_regressor(X[cl], y_weak[cl], regressor, reg_C)
    sclf = fit_regressor(X[cl], y_strong[cl], regressor, reg_C)
    val_scores = (_proba(sclf, X[val]) - _proba(wclf, X[val]) + 1.0) / 2.0
    auc = deferral_auc(val_scores, y_weak[val].astype(bool), y_strong[val].astype(bool))
    print(f"  cl={len(cl)} val={len(val)} → val deferral AUC = {auc:.5f}")
    print(f"  mean P_weak={_proba(wclf, X[val]).mean():.3f}  "
          f"mean P_strong={_proba(sclf, X[val]).mean():.3f}")

    # ── 최종: 전체 train 으로 재학습 ──
    ckpt = {
        "weak_model": weak_model,
        "strong_model": strong_model,
        "regressor": regressor,
        "reg_C": reg_C,
        "val_auc": auc,
        "embedding_model": embedding_model,
        "embedding_prefix": "query: ",
        "cost_weak": float(cost_weak),      # R2 risk 목적함수의 cost(h). 2모델·상수 cost →
        "cost_strong": float(cost_strong),  #   curve 불변, λ↔threshold 대응만 정함.
    }
    ckpt["weak_clf"] = fit_regressor(X, y_weak, regressor, reg_C)
    ckpt["strong_clf"] = fit_regressor(X, y_strong, regressor, reg_C)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
    print(f"Saved R2-Router (per-model) checkpoint → {output_path}\n")
    return {"val_auc": auc}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train R2-Router per-model router (2-model, single budget, cost 0/1)")
    p.add_argument("--train-data", required=True)
    p.add_argument("--npy-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--weak-model", required=True)
    p.add_argument("--strong-model", required=True)
    p.add_argument("--regressor", choices=["logistic", "ridge"], default="ridge",
                   help="ridge=0/1 선형회귀 후 clip(R2-Router 기본), logistic=P(pass) 로지스틱")
    p.add_argument("--reg-C", type=float, default=1.0,
                   help="logistic: 역정규화 강도 C (작을수록 강한 정규화). ridge: alpha=1/C")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embedding-model", type=str, default="intfloat/multilingual-e5-small")
    p.add_argument("--cost-weak", type=float, default=0.0,
                   help="R2 risk 목적함수의 weak 비용 c_weak (기본 0, UniRoute식 0/1 cost).")
    p.add_argument("--cost-strong", type=float, default=1.0,
                   help="R2 risk 목적함수의 strong 비용 c_strong (기본 1). 2모델·상수 cost는 "
                   "deferral curve를 안 바꾸고 λ↔threshold 대응만 정한다.")
    args = p.parse_args()

    train_r2_router(
        train_data_path=args.train_data,
        npy_path=args.npy_path,
        output_path=args.output_path,
        weak_model=args.weak_model,
        strong_model=args.strong_model,
        regressor=args.regressor,
        reg_C=args.reg_C,
        train_ratio=args.train_ratio,
        seed=args.seed,
        embedding_model=args.embedding_model,
        cost_weak=args.cost_weak,
        cost_strong=args.cost_strong,
    )
