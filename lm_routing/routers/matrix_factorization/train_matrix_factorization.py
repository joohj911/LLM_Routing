import argparse
import json
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lm_routing.routers.matrix_factorization.model import build_classifier

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def build_model_ids(data: list[dict]) -> dict[str, int]:
    """train_data.json에 등장하는 모델 이름으로 model_ids를 자동 생성."""
    names = sorted({s["model_a"] for s in data} | {s["model_b"] for s in data})
    return {name: i for i, name in enumerate(names)}


class PairwiseDataset(Dataset):
    def __init__(self, data, model_ids: dict[str, int]):
        self.models_a = torch.tensor(
            [model_ids[sample["model_a"]] for sample in data], dtype=torch.int64
        )
        self.models_b = torch.tensor(
            [model_ids[sample["model_b"]] for sample in data], dtype=torch.int64
        )
        self.prompt_id = [sample["idx"] for sample in data]
        self.winners = [sample["winner"] for sample in data]

    def __len__(self):
        return len(self.models_a)

    def __getitem__(self, index):
        assert self.winners[index] in ["model_a", "model_b"], self.winners[index]
        if self.winners[index] == "model_a":
            return self.models_a[index], self.models_b[index], self.prompt_id[index]
        else:
            return self.models_b[index], self.models_a[index], self.prompt_id[index]

    def get_dataloaders(self, batch_size, shuffle=True):
        return DataLoader(self, batch_size, shuffle=shuffle)


class MFModel_Train(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_models,
        text_dim=384,
        num_classes=1,
        use_proj=True,
        npy_path=None,
        mlp_hidden=0,
    ):
        super().__init__()
        self.use_proj = use_proj
        self.P = torch.nn.Embedding(num_models, dim)

        # num_prompts는 embeddings.npy 전체 크기에서 자동으로 결정.
        # train_data.json은 필터링된 subset이지만 idx는 원본 embeddings를 가리킴.
        embeddings = np.load(npy_path)
        num_prompts = embeddings.shape[0]
        self.Q = torch.nn.Embedding(num_prompts, text_dim).requires_grad_(False)
        self.Q.weight.data.copy_(torch.tensor(embeddings, dtype=torch.float32))

        if self.use_proj:
            # Sequential wrapper matches MFModel's state dict keys (text_proj.0.weight)
            self.text_proj = torch.nn.Sequential(
                torch.nn.Linear(text_dim, dim, bias=False)
            )
        else:
            assert (
                text_dim == dim
            ), f"text_dim {text_dim} must be equal to dim {dim} if not using projection"

        # 추론(MFModel)과 동일한 빌더를 사용해 state_dict 키를 일치시킨다.
        self.classifier = build_classifier(dim, num_classes, mlp_hidden)

    def get_device(self):
        return self.P.weight.device

    def forward(self, model_win, model_loss, prompt, test=False, alpha=0.05):
        model_win = model_win.to(self.get_device())
        model_loss = model_loss.to(self.get_device())
        prompt = prompt.to(self.get_device())

        model_win_embed = self.P(model_win)
        model_win_embed = F.normalize(model_win_embed, p=2, dim=1)
        model_loss_embed = self.P(model_loss)
        model_loss_embed = F.normalize(model_loss_embed, p=2, dim=1)
        prompt_embed = self.Q(prompt)
        if not test and alpha > 0:
            # 노이즈를 각 임베딩의 크기에 비례시켜 SNR을 임베딩 스케일과 무관하게 유지.
            # 절대값 노이즈(alpha=0.1)는 e5-small의 성분 크기(~0.05)보다 커서 프롬프트
            # 신호를 덮어버리고 라우터를 상수 출력으로 붕괴시켰다. rms = 성분의 RMS 크기.
            rms = prompt_embed.norm(dim=-1, keepdim=True) / (prompt_embed.shape[-1] ** 0.5)
            prompt_embed = prompt_embed + torch.randn_like(prompt_embed) * alpha * rms
        if self.use_proj:
            prompt_embed = self.text_proj(prompt_embed)

        # per-model 점수의 차이로 계산 → 추론(sigmoid(s_win - s_loss))과 정확히 일치.
        # 선형 classifier에선 (win-loss)*p 한 번 통과와 수학적으로 동일하고,
        # MLP(비선형)에선 이 형태라야 추론과 일치한다.
        s_win = self.classifier(model_win_embed * prompt_embed)
        s_loss = self.classifier(model_loss_embed * prompt_embed)
        return (s_win - s_loss).squeeze()

    @torch.no_grad()
    def predict(self, model_win, model_loss, prompt):
        logits = self.forward(model_win, model_loss, prompt, test=True)
        return logits > 0


def evaluator(net, test_iter, device):
    net.eval()
    ls_fn = nn.BCEWithLogitsLoss(reduction="sum")
    ls_list = []
    correct = 0
    num_samples = 0
    with torch.no_grad():
        for models_a, models_b, prompts in test_iter:
            models_a = models_a.to(device)
            models_b = models_b.to(device)
            prompts = prompts.to(device)

            # test=True: eval 중에는 Gaussian noise 없이 clean embedding 사용
            logits = net(models_a, models_b, prompts, test=True)
            labels = torch.ones_like(logits)
            loss = ls_fn(logits, labels)
            pred_labels = logits > 0

            correct += (pred_labels == labels.bool()).sum().item()
            ls_list.append(loss.item())
            num_samples += labels.shape[0]

    net.train()
    return float(sum(ls_list) / num_samples), correct / num_samples


def train_loops(
    net,
    train_iter,
    test_iter,
    lr,
    weight_decay,
    alpha,
    num_epochs,
    device="cuda",
    evaluator=evaluator,
    output_path=None,
    **kwargs,
):
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.BCEWithLogitsLoss(reduction="mean")

    best_test_acc = -1
    best_state = None

    def train_epoch():
        net.train()
        train_loss_sum, n = 0.0, 0
        for models_a, models_b, prompts in train_iter:
            models_a = models_a.to(device)
            models_b = models_b.to(device)
            prompts = prompts.to(device)

            output = net(models_a, models_b, prompts, alpha=alpha)
            ls = loss(output, torch.ones_like(output))

            optimizer.zero_grad()
            ls.backward()
            optimizer.step()

            train_loss_sum += ls.item() * len(models_a)
            n += len(models_a)
        return train_loss_sum / n

    progress_bar = tqdm(total=num_epochs)

    for epoch in range(num_epochs):
        train_ls = train_epoch()
        info = {"train_loss": train_ls, "epoch": epoch}

        if evaluator:
            test_ls, test_acc = evaluator(net, test_iter, device)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                # Q (prompt embeddings)는 추론 시 불필요하므로 제외
                best_state = {
                    k: v.cpu().clone()
                    for k, v in net.state_dict().items()
                    if not k.startswith("Q.")
                }

            info.update(
                {
                    "test_loss": test_ls,
                    "test_acc": test_acc,
                    "best_test_acc": best_test_acc,
                }
            )

        progress_bar.set_postfix(**info)
        progress_bar.update(1)

    progress_bar.close()
    return best_test_acc, best_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MF router on BFCL data.")
    parser.add_argument(
        "--train-data", type=str, required=True,
        help="Path to train_data.json from prepare_bfcl_data.py convert"
    )
    parser.add_argument(
        "--npy-path", type=str, required=True,
        help="Path to embeddings.npy from prepare_bfcl_data.py embed"
    )
    parser.add_argument(
        "--output-path", type=str, default="./bfcl_mf_model.pt",
        help="Where to save the trained model checkpoint"
    )
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--text-dim", type=int, default=0,
                        help="0=embeddings.npy 차원에서 자동 추론(권장). e5-small=384, e5-large=1024")
    parser.add_argument(
        "--embedding-model", type=str, default="intfloat/multilingual-e5-small",
        help="추론 시 사용할 임베딩 모델. embeddings.npy를 만든 모델과 동일해야 함. "
        "체크포인트에 기록되어 라우터가 같은 모델로 인코딩한다.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--no-proj", action="store_true")
    parser.add_argument(
        "--mlp-hidden", type=int, default=0,
        help="0=선형 classifier(기존), >0=해당 크기의 1-hidden MLP classifier",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Fraction of train_data to use as internal validation set")
    args = parser.parse_args()

    # text_dim을 npy 차원에서 자동 추론 (e5-small=384, e5-large=1024).
    # 임베딩 모델을 바꿔도 차원 불일치 footgun 없이 동작.
    npy_dim = int(np.load(args.npy_path, mmap_mode="r").shape[1])
    if args.text_dim and args.text_dim != npy_dim:
        print(f"[warn] --text-dim {args.text_dim} != embeddings.npy 차원 {npy_dim}; "
              f"npy 차원({npy_dim}) 사용.")
    text_dim = npy_dim
    print(f"Embedding: {args.embedding_model}  (text_dim={text_dim})")

    data = json.load(open(args.train_data))
    filtered_data = [
        s for s in data
        if s["winner"] in ["model_a", "model_b"] and s["model_a"] != s["model_b"]
    ]
    print(f"Loaded {len(filtered_data)} samples (filtered from {len(data)})")

    # train_data.json에 등장하는 모델 이름으로 model_ids 자동 생성
    model_ids = build_model_ids(filtered_data)
    print(f"Model IDs: {model_ids}")

    # bfcl_split(카테고리) 기준 stratified val split — 소형 카테고리(live_parallel 등)가
    # val에서 누락/편중되지 않도록 카테고리별 비율로 나눈다. bfcl_split 없으면 무작위 fallback.
    from collections import defaultdict
    rng = random.Random(42)
    if all("bfcl_split" in s for s in filtered_data):
        buckets = defaultdict(list)
        for s in filtered_data:
            buckets[s["bfcl_split"]].append(s)
        train_data, val_data = [], []
        for b in buckets.values():
            rng.shuffle(b)
            n_v = min(len(b) - 1, int(round(len(b) * args.val_ratio)))  # train에 최소 1개 보존
            val_data.extend(b[:n_v])
            train_data.extend(b[n_v:])
        rng.shuffle(train_data)
    else:
        rng.shuffle(filtered_data)
        n_val = max(1, int(len(filtered_data) * args.val_ratio))
        train_data, val_data = filtered_data[:-n_val], filtered_data[-n_val:]
    print(f"Train: {len(train_data)}  Val: {len(val_data)}  (stratified by bfcl_split)")

    train_loader = PairwiseDataset(train_data, model_ids).get_dataloaders(args.batch_size, shuffle=True)
    val_loader = PairwiseDataset(val_data, model_ids).get_dataloaders(1024, shuffle=False)

    model = MFModel_Train(
        dim=args.dim,
        num_models=len(model_ids),
        text_dim=text_dim,
        use_proj=not args.no_proj,
        npy_path=args.npy_path,
        mlp_hidden=args.mlp_hidden,
    ).to(args.device)

    best_acc, best_state = train_loops(
        model,
        train_loader,
        val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        num_epochs=args.num_epochs,
        device=args.device,
        output_path=args.output_path,
    )

    # best val acc 시점의 가중치 저장 (last epoch이 아님)
    # model_ids를 함께 저장하여 inference 시 외부 딕셔너리 없이 로드 가능
    save_state = best_state if best_state is not None else {
        k: v for k, v in model.state_dict().items() if not k.startswith("Q.")
    }
    # config를 함께 저장 → 추론 시 라우터가 동일한 구조(MLP 여부 포함)로 로드.
    torch.save(
        {
            "state_dict": save_state,
            "model_ids": model_ids,
            "config": {
                "dim": args.dim,
                "text_dim": text_dim,
                "use_proj": not args.no_proj,
                "mlp_hidden": args.mlp_hidden,
                "embedding_model": args.embedding_model,
            },
        },
        args.output_path,
    )
    print(f"\nSaved model → {args.output_path}  (best_val_acc={best_acc:.4f})")
    print(f"  model_ids : {model_ids}")
