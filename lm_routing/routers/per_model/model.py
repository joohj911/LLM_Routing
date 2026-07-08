"""
모델별 회귀 라우터 (Per-model regression router = R2-Router per-model 라우터).

R2-Router(github: UCF-ML-Research/R2-Router, r2_router/router.py)는 LLM마다 임베딩→품질
Ridge 회귀기를 두고 risk = (1−λ)·quality − λ·cost 를 최대화한다. 여기서는 2-모델·짧은
출력이라 budget tier를 단일로, cost를 c_weak=0/c_strong=1(UniRoute식)로 특수화했다.

라우팅 결정(2모델):
    strong ⟺ (1−λ)·P_strong − λ·c_strong  >  (1−λ)·P_weak − λ·c_weak
           ⟺ (P_strong − P_weak) > (Δc)·λ/(1−λ),   Δc = c_strong − c_weak
상수 cost Δc는 프롬프트 순위를 안 바꾸므로 라우팅 순서는 gain P_s−P_w 로만 정해진다.
그래서 λ 스윕과 threshold 스윕은 동일한 deferral curve를 그린다(cost는 λ↔threshold 대응만
결정). predict()는 evaluate가 threshold로 쓰도록 gain에 단조인 strong_win_rate 를 반환한다.
"""

import numpy as np
import torch


class PerModelRouterModel:
    """
    추론 전용. weak/strong 두 회귀기(sklearn)를 들고 P(pass)를 예측한다.

    weak_clf, strong_clf : sklearn estimator
        predict_proba 가 있으면(LogisticRegression 등) [:,1]을 P(pass)로,
        없으면(Ridge 등) predict 를 [0,1]로 clip 하여 사용.
    """

    def __init__(
        self,
        weak_clf,
        strong_clf,
        embedding_model: str = "intfloat/multilingual-e5-small",
        centroids: np.ndarray = None,
        psi_weak: np.ndarray = None,
        psi_strong: np.ndarray = None,
        cost_weak: float = 0.0,
        cost_strong: float = 1.0,
    ):
        self.weak_clf = weak_clf
        self.strong_clf = strong_clf
        self.embedding_model = embedding_model
        # R2 risk 목적함수의 cost(h). 2모델·상수 cost → curve 불변, λ↔threshold 대응만.
        self.cost_weak = float(cost_weak)
        self.cost_strong = float(cost_strong)
        # cluster-informed 변형: 학습 때 KMeans 클러스터별 pass율을 feature로 썼으면
        # 추론에서도 동일하게 [emb, ψ_weak[k], ψ_strong[k]]로 증강해야 한다.
        self.centroids = None if centroids is None else np.asarray(centroids, dtype=np.float32)
        self.psi_weak = None if psi_weak is None else np.asarray(psi_weak, dtype=np.float32)
        self.psi_strong = None if psi_strong is None else np.asarray(psi_strong, dtype=np.float32)

    @staticmethod
    def _proba(clf, x: np.ndarray) -> float:
        if hasattr(clf, "predict_proba"):
            return float(clf.predict_proba(x)[0, 1])
        return float(np.clip(clf.predict(x)[0], 0.0, 1.0))

    def _features(self, embedding: np.ndarray) -> np.ndarray:
        x = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if self.centroids is None:
            return x
        d = ((x - self.centroids) ** 2).sum(axis=1)
        k = int(d.argmin())
        return np.concatenate(
            [x, [[self.psi_weak[k]]], [[self.psi_strong[k]]]], axis=1
        ).astype(np.float32)

    def predict(self, embedding: np.ndarray) -> float:
        """단일 프롬프트 임베딩 → strong_win_rate ∈ [0, 1] (gain P_s−P_w 에 단조).
        evaluate가 이 값에 threshold 를 스윕하는 것이 곧 R2 risk 의 λ 스윕이다."""
        x = self._features(embedding)
        p_weak = self._proba(self.weak_clf, x)
        p_strong = self._proba(self.strong_clf, x)
        gain = p_strong - p_weak          # ∈ [-1, 1]
        return (gain + 1.0) / 2.0         # ∈ [0, 1], 높을수록 strong

    def lambda_to_threshold(self, lam: float) -> float:
        """R2 risk 의 λ 를 predict() strong_win_rate 상의 threshold 로 변환.
        route strong ⟺ (P_s−P_w) > Δc·λ/(1−λ) ⟺ strong_win_rate > (Δc·λ/(1−λ)+1)/2.
        (Δc = c_strong − c_weak. 상수 cost라 curve는 안 바뀌고 대응만 정의됨.)"""
        lam = float(np.clip(lam, 0.0, 1.0 - 1e-9))
        dc = self.cost_strong - self.cost_weak
        return (dc * lam / (1.0 - lam) + 1.0) / 2.0

    @classmethod
    def load(cls, checkpoint_path: str) -> "PerModelRouterModel":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return cls(
            weak_clf=ckpt["weak_clf"],
            strong_clf=ckpt["strong_clf"],
            embedding_model=ckpt.get("embedding_model", "intfloat/multilingual-e5-small"),
            centroids=ckpt.get("centroids"),
            psi_weak=ckpt.get("psi_weak"),
            psi_strong=ckpt.get("psi_strong"),
            cost_weak=ckpt.get("cost_weak", 0.0),
            cost_strong=ckpt.get("cost_strong", 1.0),
        )
