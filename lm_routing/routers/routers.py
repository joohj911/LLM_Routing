import abc
import os
import random

import numpy as np
import torch

from lm_routing.routers.matrix_factorization.model import MFModel, get_embedding_model


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
            f"query: {prompt}",
            convert_to_tensor=False,
            normalize_embeddings=False,
        )
        return self.model.predict(np.asarray(emb, dtype=np.float32))


@no_parallel
class PerModelRouter(Router):
    """
    Per-model regression router (R2-Router 골격, budget 제거).

    weak/strong 각각 P(pass | 임베딩)를 독립 회귀로 예측 →
    strong_win_rate = (P_strong − P_weak + 1)/2. 높을수록 strong.
    """

    def __init__(self, checkpoint_path: str, **kwargs):
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                f"Per-model router checkpoint not found: {checkpoint_path}\n"
                "Train with lm_routing/routers/per_model/train_per_model.py first."
            )
        from lm_routing.routers.per_model.model import PerModelRouterModel
        self.model = PerModelRouterModel.load(checkpoint_path)
        self._embed = get_embedding_model(self.model.embedding_model)

    def calculate_strong_win_rate(self, prompt: str) -> float:
        emb = self._embed.encode(
            f"query: {prompt}",
            convert_to_tensor=False,
            normalize_embeddings=False,
        )
        return self.model.predict(np.asarray(emb, dtype=np.float32))


ROUTER_CLS = {
    "mf": MatrixFactorizationRouter,
    "random": RandomRouter,
    "uniroute": UniRouteRouter,
    "permodel": PerModelRouter,
}
NAME_TO_CLS = {v: k for k, v in ROUTER_CLS.items()}
