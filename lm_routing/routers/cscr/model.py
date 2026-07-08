"""
CSCR 대조 라우터 (arXiv:2508.12491, "Cost-Aware Contrastive Routing") — inference.

frozen e5(백본 Φ) → 학습된 2-layer MLP g_θ → q ∈ R^Dd (expert descriptor 공간),
ℓ2 정규화. 라우팅은 q 를 고정 expert descriptor d_weak, d_strong 에 **cosine NN**
(=FAISS flat_ip)으로 매칭한다. 2-모델 deferral curve 를 위해 cost-aware 선택
    score_m = cos(q, d_m) − λ·cost_m,  argmax_m
을 strong_win_rate = (cos(q,d_strong) − cos(q,d_weak) + 2)/4 ∈ [0,1] 로 환산한다
(gain 에 단조; 상수 cost 는 threshold 오프셋이라 순위 불변 → threshold 스윕 = cost 조절).
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def build_head(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """학습(train_cscr)과 추론(CSCRRouterModel)이 동일 빌더를 써야 state_dict 키가 일치한다.
    논문 g_θ = two-layer MLP."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class CSCRRouterModel:
    """학습된 g_θ + 고정 expert descriptor. e5 임베딩 → strong_win_rate ∈ [0,1]."""

    def __init__(self, head, descriptors, cost_weak=0.0, cost_strong=1.0,
                 embedding_model="intfloat/multilingual-e5-small"):
        self.head = head.eval()
        self.embedding_model = embedding_model
        self.cost_weak = float(cost_weak)
        self.cost_strong = float(cost_strong)
        d = torch.as_tensor(np.asarray(descriptors), dtype=torch.float32)   # (2, Dd)
        self.descriptors = F.normalize(d, p=2, dim=-1)                      # [weak, strong]
        self._dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.head.to(self._dev)
        self.descriptors = self.descriptors.to(self._dev)

    @torch.no_grad()
    def predict(self, e5_embedding: np.ndarray) -> float:
        """단일 프롬프트의 e5 임베딩 → strong_win_rate ∈ [0,1] (cos_strong−cos_weak 에 단조)."""
        x = torch.as_tensor(np.asarray(e5_embedding), dtype=torch.float32).reshape(1, -1).to(self._dev)
        q = F.normalize(self.head(x), p=2, dim=-1)          # (1, Dd)
        sims = (q @ self.descriptors.t()).squeeze(0)        # (2,) cosine to [weak, strong]
        raw = float(sims[1] - sims[0])                      # ∈ [-2, 2]
        return (raw + 2.0) / 4.0                            # ∈ [0, 1], 높을수록 strong

    @classmethod
    def load(cls, checkpoint_path: str) -> "CSCRRouterModel":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        head = build_head(ckpt["in_dim"], ckpt["hidden"], ckpt["out_dim"])
        head.load_state_dict(ckpt["head_state_dict"])
        return cls(
            head=head,
            descriptors=ckpt["descriptors"],
            cost_weak=ckpt.get("cost_weak", 0.0),
            cost_strong=ckpt.get("cost_strong", 1.0),
            embedding_model=ckpt.get("base_embedding_model", "intfloat/multilingual-e5-small"),
        )
