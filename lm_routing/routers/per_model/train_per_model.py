"""
모델별 회귀 라우터 학습 스크립트.

각 모델(weak, strong)에 대해 임베딩 → P(pass) 회귀기를 독립 학습하고,
라우팅 점수 = (P_strong − P_weak + 1)/2 로 deferral curve를 그린다.

사용법:
  python lm_routing/routers/per_model/train_per_model.py \\
    --train-data ./bfcl_data_0.8B/train_data.json \\
    --npy-path   ./bfcl_data/embeddings.npy \\
    --output-path ./bfcl_data_0.8B/permodel_model.pt \\
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


def train_per_model(
    train_data_path: str,
    npy_path: str,
    output_path: str,
    weak_model: str,
    strong_model: str,
    regressor: str = "logistic",
    reg_C: float = 1.0,
    train_ratio: float = 0.8,
    seed: int = 42,
    embedding_model: str = "intfloat/multilingual-e5-small",
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
    weak_clf = fit_regressor(X, y_weak, regressor, reg_C)
    strong_clf = fit_regressor(X, y_strong, regressor, reg_C)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weak_clf": weak_clf,
            "strong_clf": strong_clf,
            "weak_model": weak_model,
            "strong_model": strong_model,
            "regressor": regressor,
            "reg_C": reg_C,
            "val_auc": auc,
            "embedding_model": embedding_model,
            "embedding_prefix": "query: ",
        },
        output_path,
    )
    print(f"Saved per-model router checkpoint → {output_path}\n")
    return {"val_auc": auc}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train per-model regression router (R2-style, budget-free)")
    p.add_argument("--train-data", required=True)
    p.add_argument("--npy-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--weak-model", required=True)
    p.add_argument("--strong-model", required=True)
    p.add_argument("--regressor", choices=["logistic", "ridge"], default="logistic",
                   help="logistic=P(pass) 로지스틱(기본), ridge=0/1 선형회귀 후 clip")
    p.add_argument("--reg-C", type=float, default=1.0,
                   help="logistic: 역정규화 강도 C (작을수록 강한 정규화). ridge: alpha=1/C")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embedding-model", type=str, default="intfloat/multilingual-e5-small")
    args = p.parse_args()

    train_per_model(
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
    )
