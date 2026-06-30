import torch
from huggingface_hub import PyTorchModelHubMixin
from sentence_transformers import SentenceTransformer



DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_MODEL_NAME = DEFAULT_EMBEDDING_MODEL  # 하위호환 별칭
_EMBEDDING_MODELS = {}


def get_embedding_model(name: str = DEFAULT_EMBEDDING_MODEL):
    """이름별로 SentenceTransformer를 캐시. e5-small/e5-large 등 혼용 가능."""
    if name not in _EMBEDDING_MODELS:
        _EMBEDDING_MODELS[name] = SentenceTransformer(name)
    return _EMBEDDING_MODELS[name]


def build_classifier(dim: int, num_classes: int, mlp_hidden: int = 0):
    """
    분류기 head 생성. mlp_hidden=0 이면 기존 선형(bias 없음), >0 이면 1-hidden MLP.

    학습(MFModel_Train)과 추론(MFModel)이 동일한 빌더를 써야 state_dict 키가 일치한다.
    """
    if mlp_hidden and mlp_hidden > 0:
        return torch.nn.Sequential(
            torch.nn.Linear(dim, mlp_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden, num_classes, bias=False),
        )
    return torch.nn.Sequential(torch.nn.Linear(dim, num_classes, bias=False))


class MFModel(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        dim,
        num_models,
        text_dim=384,
        num_classes=1,
        use_proj=True,
        mlp_hidden=0,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
    ):
        super().__init__()
        self._name = "TextMF"
        self.use_proj = use_proj
        # 추론 시 프롬프트를 인코딩할 임베딩 모델 (학습에 쓴 것과 동일해야 함)
        self.embedding_model = embedding_model
        self.P = torch.nn.Embedding(num_models, dim)

        if self.use_proj:
            self.text_proj = torch.nn.Sequential(
                torch.nn.Linear(text_dim, dim, bias=False)
            )
        else:
            assert (
                text_dim == dim
            ), f"text_dim {text_dim} must be equal to dim {dim} if not using projection"

        self.classifier = build_classifier(dim, num_classes, mlp_hidden)

    def get_device(self):
        return self.P.weight.device

    def forward(self, model_id, prompt):
        model_id = torch.tensor(model_id, dtype=torch.long).to(self.get_device())

        model_embed = self.P(model_id)
        model_embed = torch.nn.functional.normalize(model_embed, p=2, dim=1)

        # e5 계열은 비대칭 검색용 "query: " prefix 사용 (small/large 동일)
        prompt_embed = get_embedding_model(self.embedding_model).encode(
            f"query: {prompt}",
            convert_to_tensor=True,
            device=str(self.get_device()),
        )
        if self.use_proj:
            prompt_embed = self.text_proj(prompt_embed)

        return self.classifier(model_embed * prompt_embed).squeeze()

    @torch.no_grad()
    def pred_win_rate(self, model_a, model_b, prompt):
        logits = self.forward([model_a, model_b], prompt)
        winrate = torch.sigmoid(logits[0] - logits[1]).item()
        return winrate

    def load(self, path):
        self.load_state_dict(torch.load(path))
