"""
UniRoute K-Means cluster-based routing model.

LLM representation: Ψ(h) ∈ [0,1]^K — per-cluster error rate on a validation set.
Prompt representation: Φ(x) ∈ {0,1}^K — one-hot cluster membership (nearest centroid).

Routing score: Ψ_k(weak) - Ψ_k(strong)  (> 0 means strong is better in this cluster)
Normalised to [0,1]: (diff + 1) / 2  so threshold 0.5 = "strong is better than weak"
"""

import numpy as np
import torch

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
_EMBEDDING_MODEL = None


def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


class UniRouteModel:
    """
    Inference-only cluster-based UniRoute model.

    centroids : np.ndarray (K, D) — K-Means cluster centres (D=384 for e5-small)
    psi_weak  : np.ndarray (K,)  — per-cluster error rate of the weak model
    psi_strong: np.ndarray (K,)  — per-cluster error rate of the strong model
    """

    def __init__(
        self,
        centroids: np.ndarray,
        psi_weak: np.ndarray,
        psi_strong: np.ndarray,
        assignment: str = "hard",
        tau: float = 1.0,
        embedding_model: str = "intfloat/multilingual-e5-small",
    ):
        self.centroids = centroids          # (K, D)
        self.psi_weak = psi_weak            # (K,)
        self.psi_strong = psi_strong        # (K,)
        self.assignment = assignment        # "hard" (nearest centroid) | "soft" (softmax)
        self.tau = float(tau)               # softmax temperature (soft 일 때만 사용)
        self.embedding_model = embedding_model  # 추론 인코딩에 쓸 모델 (학습과 동일)

    def predict(self, embedding: np.ndarray) -> float:
        """
        Return strong_win_rate ∈ [0, 1] for a single prompt embedding.

        hard: 가장 가까운 클러스터의 (Ψ_weak − Ψ_strong).
        soft: softmax(−dist/τ) 가중 평균 → 점수가 연속값이 되어 분해능이 높아진다.

        Higher → more reason to route to strong model.
        """
        dists = np.sum((self.centroids - embedding) ** 2, axis=1)  # (K,)
        psi_diff = self.psi_weak - self.psi_strong                 # (K,) ∈ [-1, 1]

        if self.assignment == "soft":
            z = -dists / max(self.tau, 1e-8)
            z -= z.max()                       # 수치 안정화
            w = np.exp(z)
            w /= w.sum()
            diff = float(np.dot(w, psi_diff))
        else:
            k = int(np.argmin(dists))
            diff = float(psi_diff[k])

        return (diff + 1.0) / 2.0                                   # ∈ [0, 1]

    @classmethod
    def load(cls, checkpoint_path: str) -> "UniRouteModel":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return cls(
            centroids=np.asarray(ckpt["centroids"], dtype=np.float32),
            psi_weak=np.asarray(ckpt["psi_weak"], dtype=np.float32),
            psi_strong=np.asarray(ckpt["psi_strong"], dtype=np.float32),
            assignment=ckpt.get("assignment", "hard"),
            tau=ckpt.get("tau", 1.0),
            embedding_model=ckpt.get("embedding_model", "intfloat/multilingual-e5-small"),
        )
