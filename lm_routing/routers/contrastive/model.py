"""
대조(contrastive) 임베딩 변환기 — CSCR(arXiv:2508.12491)의 "frozen encoder +
학습가능 MLP head" 아이디어를 우리 binary(weak/strong) BFCL 세팅에 이식한 것.

핵심: e5 임베딩 φ(x)는 그대로 두고(frozen), 그 위에 2-layer MLP head g_θ만
학습해 "weak로 보내도 되는 프롬프트"와 "strong이 필요한 프롬프트"를 잘 가르는
공간으로 변환한다. 산출물 g_θ(φ(x))를 새 임베딩으로 써서 기존 라우터
(MF/UniRoute/permodel)에 그대로 투입한다("대조 임베딩만 이식").

CSCREncoder는 SentenceTransformer와 같은 .encode() 인터페이스를 노출하므로,
get_embedding_model("cscr:<head.pt>")로 로드되면 라우터 코드 변경 없이
학습(embeddings.npy 변환)·추론(live 인코딩)에서 동일한 변환이 적용된다.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def build_head(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """학습(train)과 추론(CSCREncoder)이 동일 빌더를 써야 state_dict 키가 일치한다."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class CSCREncoder:
    """frozen e5(SentenceTransformer) + 학습된 head g_θ. .encode()가
    g_θ(e5(text))를 ℓ2 정규화해 돌려준다 (contrastive 공간과 동일)."""

    def __init__(self, base_model, head: nn.Module, base_embedding_model: str):
        self.base = base_model
        self.head = head.eval()
        self.base_embedding_model = base_embedding_model
        self._dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.head.to(self._dev)

    @classmethod
    def load(cls, path: str) -> "CSCREncoder":
        # 지연 import로 순환참조 방지 (matrix_factorization.model → 여기)
        from lm_routing.routers.matrix_factorization.model import get_embedding_model

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        base = get_embedding_model(ckpt["base_embedding_model"])
        head = build_head(ckpt["in_dim"], ckpt["hidden"], ckpt["out_dim"])
        head.load_state_dict(ckpt["head_state_dict"])
        return cls(base, head, ckpt["base_embedding_model"])

    @torch.no_grad()
    def encode(
        self,
        sentences,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False,  # 무시: 출력은 항상 ℓ2 정규화됨
        device: str | None = None,
        batch_size: int = 256,
        show_progress_bar: bool = False,
        **kwargs,
    ):
        dev = device or self._dev
        self.head.to(dev)
        # 학습에 쓴 embeddings.npy와 동일하게 raw e5 (normalize_embeddings=False)
        base_emb = self.base.encode(
            sentences,
            convert_to_tensor=True,
            device=dev,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=False,
        )
        q = F.normalize(self.head(base_emb.float().to(dev)), p=2, dim=-1)
        if convert_to_tensor:
            return q
        return q.detach().cpu().numpy()


@torch.no_grad()
def transform_embeddings(head: nn.Module, e5_embeddings, device: str = "cpu", batch_size: int = 4096):
    """e5 임베딩 행렬(N×D) → cscr 임베딩(N×D')로 변환 (head 적용 + ℓ2 정규화).

    텍스트 재인코딩 없이 기존 embeddings.npy를 그대로 변환할 때 사용."""
    import numpy as np

    head = head.eval().to(device)
    X = torch.as_tensor(np.asarray(e5_embeddings), dtype=torch.float32)
    out = []
    for i in range(0, len(X), batch_size):
        q = F.normalize(head(X[i : i + batch_size].to(device)), p=2, dim=-1)
        out.append(q.detach().cpu())
    return torch.cat(out, dim=0).numpy()
