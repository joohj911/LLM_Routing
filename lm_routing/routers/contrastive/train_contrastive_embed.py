"""
대조 임베딩 head 학습 — CSCR(arXiv:2508.12491)의 Cost-Spectrum InfoRCE를
binary(weak/strong) 세팅에 이식.

절차:
  1. train_data.json(weak/strong pass 라벨) + embeddings.npy(frozen e5) 로드
  2. 2-layer MLP head g_θ + 학습가능 expert descriptor(e_weak, e_strong) 학습
     - 각 프롬프트를 "그 프롬프트를 맞힌 모델" descriptor 쪽으로 당기고, 못 맞힌
       모델에서 밀어냄. cost band별 온도 + negative에 cost 패널티(−γc)로
       "정확하면 더 싼(weak) 쪽을 선호"하도록 유도. (Eq. 8)
  3. head 저장(.pt) + 기존 embeddings.npy를 통째로 변환해 embeddings_cscr.npy 저장
     (텍스트 재인코딩 없이 e5 벡터 → head → ℓ2 정규화)

산출된 embeddings_cscr.npy + "cscr:<head.pt>"(--embedding-model)를 기존 라우터
학습/평가에 그대로 넣으면 됨.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lm_routing.routers.contrastive.model import build_head, transform_embeddings


def make_bands(costs: np.ndarray):
    """cost가 같은 모델끼리 묶어 band 구성. 반환: [(member_idx_array, mean_cost), ...].
    binary(서로 다른 cost 2개)면 자연히 2개 band. cost가 같으면 1개(=표준 InfoNCE)."""
    bands = []
    for c in sorted(set(costs.tolist())):
        members = np.where(costs == c)[0]
        bands.append((torch.as_tensor(members, dtype=torch.long), float(c)))
    return bands


def cost_spectrum_infonce(q, desc, pass_mat, costs, bands, tau_min, alpha, gamma):
    """Cost-Spectrum InfoNCE (Eq. 8), binary/일반 M 모델 대응.

    q        : (B, D)  ℓ2 정규화된 query 표현
    desc     : (M, D)  expert descriptor(정규화 전)
    pass_mat : (B, M)  bool — 각 모델이 그 프롬프트를 맞혔는지(=positive 후보)
    costs    : (M,)    정규화 cost [0,1]
    bands    : make_bands 결과
    """
    E = F.normalize(desc, p=2, dim=-1)                # (M, D)
    sims = q @ E.t()                                  # (B, M) cosine
    B, M = sims.shape
    device = sims.device
    costs = costs.to(device)

    per_q = torch.zeros(B, device=device)
    n_bands = torch.zeros(B, device=device)
    for members, cbar in bands:
        members = members.to(device)
        tau_k = tau_min + alpha * cbar
        # 분모: 모든 모델 m' 에 대해 exp((sim - γ·cost)/τ_k)  (negative cost 패널티)
        denom = torch.logsumexp((sims - gamma * costs[None, :]) / tau_k, dim=1)  # (B,)
        # 분자: 이 band 안의 positive(맞힌 모델)만 exp(sim/τ_k)
        band_mask = torch.zeros(M, dtype=torch.bool, device=device)
        band_mask[members] = True
        pos_in_band = pass_mat & band_mask[None, :]   # (B, M)
        # -inf 대신 유한한 큰 음수로 마스킹: positive 없는 band 행에서 logsumexp가
        # -inf가 되면 backward에서 nan 그래디언트가 생긴다. exp(-1e9)=0 이라 값은
        # 동일하고, 그런 행은 아래 has=False로 손실 기여가 0이 된다.
        masked = (sims / tau_k).masked_fill(~pos_in_band, -1e9)
        num = torch.logsumexp(masked, dim=1)          # (B,)
        has = pos_in_band.any(dim=1)
        term = num - denom
        per_q = per_q + (-term) * has.float()          # has=False 행은 0 기여(유한)
        n_bands = n_bands + has.float()

    ok = n_bands > 0                                  # positive 하나도 없는(둘 다 실패) 쿼리 제외
    if ok.sum() == 0:
        return sims.sum() * 0.0
    return (per_q[ok] / n_bands[ok]).mean()


def main():
    ap = argparse.ArgumentParser(description="Train CSCR-style contrastive embedding head (binary)")
    ap.add_argument("--train-data", required=True, help="train_data.json (idx + weak/strong pass 라벨)")
    ap.add_argument("--npy-path", required=True, help="embeddings.npy (frozen e5)")
    ap.add_argument("--output-path", required=True, help="출력 head 체크포인트 .pt")
    ap.add_argument("--output-npy", default=None,
                    help="변환된 임베딩 저장 경로 (기본: <npy-path 디렉토리>/embeddings_cscr.npy)")
    ap.add_argument("--weak-model", required=True)
    ap.add_argument("--strong-model", required=True)
    ap.add_argument("--embedding-model", default="intfloat/multilingual-e5-small",
                    help="frozen base 임베딩 모델(체크포인트에 기록, 추론 인코딩에 사용)")
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    # Cost-Spectrum InfoNCE 하이퍼파라미터 (논문 기본)
    ap.add_argument("--gamma", type=float, default=0.2, help="negative cost 패널티")
    ap.add_argument("--alpha", type=float, default=0.25, help="band 온도 스케일")
    ap.add_argument("--tau-min", type=float, default=0.05, help="최소 온도")
    ap.add_argument("--cost-weak", type=float, default=0.0)
    ap.add_argument("--cost-strong", type=float, default=1.0)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = json.load(open(args.train_data))
    all_embs = np.load(args.npy_path).astype(np.float32)
    idx = np.array([r["idx"] for r in df], dtype=np.int64)
    X = torch.from_numpy(all_embs[idx])                       # (N, D)
    in_dim = X.shape[1]
    weak = np.array([bool(r[args.weak_model]) for r in df])
    strong = np.array([bool(r[args.strong_model]) for r in df])
    pass_mat = torch.from_numpy(np.stack([weak, strong], axis=1))  # (N, 2) bool
    costs = np.array([args.cost_weak, args.cost_strong], dtype=np.float32)
    costs_t = torch.from_numpy(costs)
    bands = make_bands(costs)
    print(f"Loaded {len(df)} samples, e5 dim={in_dim}, "
          f"weak_pass={weak.mean()*100:.1f}%  strong_pass={strong.mean()*100:.1f}%")
    print(f"bands={[ (m.tolist(), c) for m, c in bands ]}  "
          f"(γ={args.gamma}, α={args.alpha}, τmin={args.tau_min})")

    # stratified-ish 무작위 val split (모니터링용)
    n = len(df)
    perm = np.random.permutation(n)
    n_val = max(1, int(n * args.val_ratio))
    val_i = torch.as_tensor(perm[:n_val], dtype=torch.long)
    tr_i = torch.as_tensor(perm[n_val:], dtype=torch.long)

    head = build_head(in_dim, args.hidden, args.out_dim).to(device)
    desc = nn.Parameter(torch.randn(2, args.out_dim) * 0.1)  # 학습가능 expert descriptor
    opt = torch.optim.Adam(list(head.parameters()) + [desc], lr=args.lr, weight_decay=args.weight_decay)

    X = X.to(device)
    pass_mat = pass_mat.to(device)
    costs_t = costs_t.to(device)

    def loss_on(indices):
        q = F.normalize(head(X[indices]), p=2, dim=-1)
        return cost_spectrum_infonce(q, desc, pass_mat[indices], costs_t, bands,
                                     args.tau_min, args.alpha, args.gamma)

    best_val = float("inf")
    best_state = None
    for ep in range(1, args.epochs + 1):
        head.train()
        bperm = tr_i[torch.randperm(len(tr_i))]
        for s in range(0, len(bperm), args.batch_size):
            b = bperm[s : s + args.batch_size]
            opt.zero_grad()
            loss = loss_on(b)
            loss.backward()
            opt.step()
        if ep % 10 == 0 or ep == 1 or ep == args.epochs:
            head.eval()
            with torch.no_grad():
                vl = loss_on(val_i).item()
                tl = loss_on(tr_i).item()
            print(f"  epoch {ep:3d}  train_loss={tl:.4f}  val_loss={vl:.4f}")
            if vl < best_val:
                best_val = vl
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    print(f"Best val_loss={best_val:.4f}")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "in_dim": in_dim,
            "hidden": args.hidden,
            "out_dim": args.out_dim,
            "base_embedding_model": args.embedding_model,
            "weak_model": args.weak_model,
            "strong_model": args.strong_model,
            "cost_weak": args.cost_weak,
            "cost_strong": args.cost_strong,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "tau_min": args.tau_min,
        },
        args.output_path,
    )
    print(f"Saved head → {args.output_path}")

    # 기존 e5 embeddings.npy 전체를 변환해 저장 (라우터 학습 입력용)
    out_npy = args.output_npy or str(Path(args.npy_path).with_name("embeddings_cscr.npy"))
    cscr = transform_embeddings(head, all_embs, device=device)
    np.save(out_npy, cscr)
    print(f"Transformed embeddings ({all_embs.shape} → {cscr.shape}) → {out_npy}")


if __name__ == "__main__":
    main()
