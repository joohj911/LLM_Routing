"""
모델별 회귀 라우터 (Per-model regression router).

R2-Router(arXiv:2602.02823)의 골격(공유 인코더 + LLM별 품질 예측기)에서
token-budget 축을 제거한 버전. 각 모델마다 독립 회귀기로 P(pass | 임베딩)를 예측하고,
라우팅 점수 = (P_strong − P_weak + 1) / 2 ∈ [0,1] 로 둔다 (높을수록 strong).

설계 근거(B): 한 쿼리를 weak 대신 strong에 보낼 때의 기대 이득은 P_strong − P_weak 이다.
주어진 strong 예산에서 pass rate를 최대화하려면 이 gain이 큰 쿼리부터 strong에 보내야 한다.
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
    ):
        self.weak_clf = weak_clf
        self.strong_clf = strong_clf
        self.embedding_model = embedding_model

    @staticmethod
    def _proba(clf, x: np.ndarray) -> float:
        if hasattr(clf, "predict_proba"):
            return float(clf.predict_proba(x)[0, 1])
        return float(np.clip(clf.predict(x)[0], 0.0, 1.0))

    def predict(self, embedding: np.ndarray) -> float:
        """단일 프롬프트 임베딩 → strong_win_rate ∈ [0, 1]."""
        x = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        p_weak = self._proba(self.weak_clf, x)
        p_strong = self._proba(self.strong_clf, x)
        gain = p_strong - p_weak          # ∈ [-1, 1]
        return (gain + 1.0) / 2.0         # ∈ [0, 1], 높을수록 strong

    @classmethod
    def load(cls, checkpoint_path: str) -> "PerModelRouterModel":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return cls(
            weak_clf=ckpt["weak_clf"],
            strong_clf=ckpt["strong_clf"],
            embedding_model=ckpt.get("embedding_model", "intfloat/multilingual-e5-small"),
        )
