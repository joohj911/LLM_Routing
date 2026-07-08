import abc
import os
import random

import numpy as np
import torch

from lm_routing.routers.matrix_factorization.model import MFModel, get_embedding_model, format_query


def no_parallel(cls):
    cls.NO_PARALLEL = True
    return cls


class Router(abc.ABC):
    NO_PARALLEL = False

    # Returns a float between 0 and 1 representing the value used to route to models,
    # conventionally the win rate of the strong model.
    # If this value is >= the user-defined threshold, routes to the strong model.
    @abc.abstractmethod
    def calculate_strong_win_rate(self, prompt):
        pass

    def route(self, prompt, threshold, routed_pair):
        if self.calculate_strong_win_rate(prompt) >= threshold:
            return routed_pair.strong
        else:
            return routed_pair.weak

    def __str__(self):
        return NAME_TO_CLS[self.__class__]


@no_parallel
class MatrixFactorizationRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        strong_model="Qwen/Qwen3.5-9B",
        weak_model="Qwen/Qwen3.5-2B",
        hidden_size=128,
        text_dim=384,
        num_classes=1,
        use_proj=True,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Train a local checkpoint with train_matrix_factorization.py first."
            )

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_ids = ckpt["model_ids"]
        state = ckpt["state_dict"]
        # 체크포인트에 저장된 config 우선 사용 (MLP 여부/차원). 구버전 체크포인트는
        # config가 없으므로 생성자 인자 기본값으로 폴백.
        cfg = ckpt.get("config", {})
        self.model = MFModel(
            dim=cfg.get("dim", hidden_size),
            num_models=len(model_ids),
            text_dim=cfg.get("text_dim", text_dim),
            num_classes=num_classes,
            use_proj=cfg.get("use_proj", use_proj),
            mlp_hidden=cfg.get("mlp_hidden", 0),
            embedding_model=cfg.get("embedding_model", "intfloat/multilingual-e5-small"),
        )
        self.model.load_state_dict(state)
        self.model = self.model.eval().to(device)
        self.strong_model_id = model_ids[strong_model]
        self.weak_model_id = model_ids[weak_model]

    def calculate_strong_win_rate(self, prompt):
        return self.model.pred_win_rate(
            self.strong_model_id, self.weak_model_id, prompt
        )


# Parallelism makes randomness non-deterministic
@no_parallel
class RandomRouter(Router):
    def calculate_strong_win_rate(self, prompt):
        del prompt
        return random.uniform(0, 1)


@no_parallel
class UniRouteRouter(Router):
    """
    Cluster-based UniRoute router (K-Means, §5.1 of arXiv:2502.08773).

    At inference: embed prompt → find nearest K-Means centroid →
    return (Ψ_weak[k] - Ψ_strong[k] + 1) / 2 ∈ [0, 1].
    Threshold 0.5 means "route to strong whenever it has lower cluster error".
    """

    def __init__(self, checkpoint_path: str, **kwargs):
        import os
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                f"UniRoute checkpoint not found: {checkpoint_path}\n"
                "Train with lm_routing/routers/uniroute/train_uniroute.py first."
            )
        from lm_routing.routers.uniroute.model import UniRouteModel
        self.model = UniRouteModel.load(checkpoint_path)
        # 체크포인트에 기록된 임베딩 모델로 인코딩 (학습 때와 동일해야 함)
        self._embed = get_embedding_model(self.model.embedding_model)

    def calculate_strong_win_rate(self, prompt: str) -> float:
        import numpy as np
        emb = self._embed.encode(
            format_query(prompt, self.model.embedding_model),
            convert_to_tensor=False,
            normalize_embeddings=False,
        )
        return self.model.predict(np.asarray(emb, dtype=np.float32))


@no_parallel
class UniRouteTrainRouter(UniRouteRouter):
    """UniRoute variant whose final Ψ is refit on the full train set (cl+val),
    i.e. trained with --psi-source train. Behaviour is identical to
    UniRouteRouter (it just loads whatever checkpoint it is given); this is a
    separate class only so it can be registered under its own router name and
    compared side-by-side against the val-Ψ variant on the test set."""


@no_parallel
class UniR2Router(UniRouteRouter):
    """R2-Router 의 UniRoute 결합(Uni-R2, unirouter/uni_r2.py)을 2-모델·단일 budget·
    cost∈{0,1} 로 특수화한 것. Uni-R2 는 품질을 Φ(x)·Ψ(h) — soft 클러스터 멤버십 Φ 와
    클러스터별 품질 Ψ 의 내적 — 으로 예측하고 R2 risk 로 라우팅한다. 우리 설정에선
    honest K, Ψ=train 에 **soft assignment** 를 쓴 UniRoute 체크포인트와 동일하다
    (train_uniroute --assignment soft --psi-source train). 런타임은 UniRouteRouter 와
    같고(체크포인트의 assignment=soft 로 자동 분기), 별도 이름으로 비교하기 위한 subclass."""


@no_parallel
class UniRouteLegacyRouter(UniRouteRouter):
    """UniRoute variant whose K was chosen with the legacy circular procedure
    (Ψ estimated on val and scored on the same val), i.e. trained with
    --k-select-psi val. Kept only to compare that (theoretically overfitting)
    choice against the honest one on the held-out test set. Identical runtime
    behaviour to UniRouteRouter; separate class only for its own router name."""


@no_parallel
class R2Router(Router):
    """
    R2-Router 의 per-model 라우터(github: UCF-ML-Research/R2-Router, r2_router/router.py)를
    2-모델·단일 budget·cost∈{0,1} 로 특수화한 것.

    weak/strong 각각 Ridge 회귀로 P(pass | 임베딩)를 예측하고, R2 risk
    (1−λ)·quality − λ·cost 를 최대화하면 2모델에선 gain P_strong−P_weak 순 라우팅과
    동치가 된다 → strong_win_rate = (P_strong − P_weak + 1)/2 를 threshold 스윕(=λ 스윕).
    """

    def __init__(self, checkpoint_path: str, **kwargs):
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                f"R2-Router checkpoint not found: {checkpoint_path}\n"
                "Train with lm_routing/routers/r2_router/train_r2_router.py first."
            )
        from lm_routing.routers.r2_router.model import R2RouterModel
        self.model = R2RouterModel.load(checkpoint_path)
        self._embed = get_embedding_model(self.model.embedding_model)

    def calculate_strong_win_rate(self, prompt: str) -> float:
        emb = self._embed.encode(
            format_query(prompt, self.model.embedding_model),
            convert_to_tensor=False,
            normalize_embeddings=False,
        )
        return self.model.predict(np.asarray(emb, dtype=np.float32))


@no_parallel
class CSCRRouter(Router):
    """CSCR (arXiv:2508.12491, "Cost-Aware Contrastive Routing") — 완결형 대조 라우터.
    frozen e5 → 학습된 2-layer MLP g_θ → q 를 실제 모델 출력에서 계산한 고정 expert
    descriptor 에 cosine-NN(=FAISS flat_ip) 라우팅. checkpoint 는 train_cscr.py 산출물."""

    def __init__(self, checkpoint_path: str, **kwargs):
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                f"CSCR router checkpoint not found: {checkpoint_path}\n"
                "먼저 lm_routing/routers/cscr/descriptors.py 로 descriptor 계산 후 "
                "lm_routing/routers/cscr/train_cscr.py 로 학습하세요."
            )
        from lm_routing.routers.cscr.model import CSCRRouterModel
        self.model = CSCRRouterModel.load(checkpoint_path)
        self._embed = get_embedding_model(self.model.embedding_model)

    def calculate_strong_win_rate(self, prompt: str) -> float:
        emb = self._embed.encode(
            format_query(prompt, self.model.embedding_model),
            convert_to_tensor=False,
            normalize_embeddings=False,
        )
        return self.model.predict(np.asarray(emb, dtype=np.float32))


ROUTER_CLS = {
    "mf": MatrixFactorizationRouter,
    "random": RandomRouter,
    "uniroute": UniRouteRouter,
    "uniroute_train": UniRouteTrainRouter,
    "uni_r2": UniR2Router,
    "uniroute_legacy": UniRouteLegacyRouter,
    "r2_router": R2Router,
    "cscr": CSCRRouter,
}
NAME_TO_CLS = {v: k for k, v in ROUTER_CLS.items()}
