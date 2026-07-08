"""
CSCR 대조 라우터 학습 (arXiv:2508.12491, Appendix D.1 Training).

논문 학습 레시피를 그대로 따른다:
  - frozen 임베딩 백본 Φ(x) (여기선 e5, 논문 all-MiniLM-L6-v2 대체 — "frozen backbone
    across all experiments"). embeddings.npy 로 미리 계산돼 있음.
  - 학습 대상 = 2-layer MLP g_θ(.) 하나. 프롬프트 임베딩을 **expert descriptor 공간**
    으로 사영한다 (out_dim = descriptor 차원).
  - 대조 타깃 descriptor 는 descriptors.py 로 **실제 모델 출력에서 계산**한 고정 벡터
    (d_weak, d_strong). 학습 파라미터가 아니다.
  - Cost-Spectrum InfoNCE (Eq. 8): 각 프롬프트를 그 프롬프트를 맞힌 expert 의 descriptor
    쪽으로 당기고 못 맞힌 쪽에서 민다. cost band 별 온도 τ_k=τmin+α·c̄ (Eq. 7) + negative
    에 cost 패널티 −λc.
  - HP (D.1): 10 epochs, AdamW, batch 512, lr 5e-4, cost band K=5(=서로 다른 cost 수;
    2-모델이면 2), λ=0.1, α=0.25, τmin=0.05.

산출 head(.pt)는 CSCRRouter 가 로드해 q=g_θ(e5(x)) 를 descriptor 에 cosine-NN 라우팅한다.

사용법:
  python lm_routing/routers/cscr/train_cscr.py \\
    --train-data   ./bfcl_data_0.8B/train_data.json \\
    --npy-path     ./bfcl_data/embeddings.npy \\
    --descriptors  ./bfcl_data_0.8B/cscr_descriptors.npz \\
    --output-path  ./bfcl_data_0.8B/cscr_model.pt \\
    --weak-model Qwen/Qwen3.5-0.8B --strong-model Qwen/Qwen3.5-9B
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lm_routing.routers.cscr.model import build_head


def make_bands(costs: np.ndarray):
    """cost가 같은 모델끼리 묶어 band 구성. 반환: [(member_idx_tensor, mean_cost), ...].
    서로 다른 cost 가 M'개면 M' band (2-모델이면 2). = cost 스펙트럼을 K band로 나눈 것."""
    bands = []
    for c in sorted(set(costs.tolist())):
        members = np.where(costs == c)[0]
        bands.append((torch.as_tensor(members, dtype=torch.long), float(c)))
    return bands


def cost_spectrum_infonce(q, desc, pass_mat, costs, bands, tau_min, alpha, lam):
    """Cost-Spectrum InfoNCE (Eq. 8), binary/일반 M 모델 대응.

    q        : (B, D)  ℓ2 정규화된 query 표현
    desc     : (M, D)  고정 expert descriptor (계산·L2 정규화됨)
    pass_mat : (B, M)  bool — 각 모델이 그 프롬프트를 맞혔는지(=positive 후보)
    costs    : (M,)    정규화 cost [0,1]
    bands    : make_bands 결과
    lam      : negative cost 패널티(λ, 논문 0.1)
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
        tau_k = tau_min + alpha * cbar                # Eq. 7 band-specific 온도
        # 분모: 모든 모델 m' 에 대해 exp((sim - λ·cost)/τ_k)  (negative cost 패널티)
        denom = torch.logsumexp((sims - lam * costs[None, :]) / tau_k, dim=1)  # (B,)
        # 분자: 이 band 안의 positive(맞힌 모델)만 exp(sim/τ_k)
        band_mask = torch.zeros(M, dtype=torch.bool, device=device)
        band_mask[members] = True
        pos_in_band = pass_mat & band_mask[None, :]   # (B, M)
        # -inf 대신 유한한 큰 음수로 마스킹 (positive 없는 행에서 nan grad 방지)
        masked = (sims / tau_k).masked_fill(~pos_in_band, -1e9)
        num = torch.logsumexp(masked, dim=1)          # (B,)
        has = pos_in_band.any(dim=1)
        per_q = per_q + (-(num - denom)) * has.float()
        n_bands = n_bands + has.float()

    ok = n_bands > 0                                  # positive 하나도 없는(둘 다 실패) 쿼리 제외
    if ok.sum() == 0:
        return sims.sum() * 0.0
    return (per_q[ok] / n_bands[ok]).mean()


def main():
    ap = argparse.ArgumentParser(description="Train CSCR contrastive router g_θ vs fixed descriptors")
    ap.add_argument("--train-data", required=True, help="train_data.json (idx + weak/strong pass 라벨)")
    ap.add_argument("--npy-path", required=True, help="embeddings.npy (frozen e5 backbone Φ)")
    ap.add_argument("--descriptors", required=True, help="descriptors.py 산출 .npz (고정 대조 타깃)")
    ap.add_argument("--output-path", required=True, help="출력 라우터 체크포인트 .pt")
    ap.add_argument("--weak-model", required=True)
    ap.add_argument("--strong-model", required=True)
    ap.add_argument("--embedding-model", default="intfloat/multilingual-e5-small",
                    help="frozen base 임베딩 모델(체크포인트에 기록, 추론 인코딩에 사용)")
    ap.add_argument("--hidden", type=int, default=512)
    # 논문 D.1 기본값
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=0.1, help="negative cost 패널티 λ (D.1)")
    ap.add_argument("--alpha", type=float, default=0.25, help="band 온도 스케일 (Eq. 7)")
    ap.add_argument("--tau-min", type=float, default=0.05, help="최소 온도 (Eq. 7)")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 고정 descriptor 로드 ──
    dz = np.load(args.descriptors, allow_pickle=True)
    desc = torch.from_numpy(dz["descriptors"].astype(np.float32)).to(device)  # (2, Dd)
    token_ids = dz["token_ids"]
    costs = np.array([float(dz["cost_weak"]), float(dz["cost_strong"])], dtype=np.float32)
    out_dim = desc.shape[1]
    bands = make_bands(costs)
    print(f"Descriptors {tuple(desc.shape)} from {args.descriptors}  "
          f"costs={costs.tolist()}  bands={len(bands)}  (λ={args.lam}, α={args.alpha}, τmin={args.tau_min})")

    # ── 데이터 ──
    df = json.load(open(args.train_data))
    all_embs = np.load(args.npy_path).astype(np.float32)
    idx = np.array([r["idx"] for r in df], dtype=np.int64)
    X = torch.from_numpy(all_embs[idx]).to(device)                       # (N, D_e5)
    in_dim = X.shape[1]
    weak = np.array([bool(r[args.weak_model]) for r in df])
    strong = np.array([bool(r[args.strong_model]) for r in df])
    pass_mat = torch.from_numpy(np.stack([weak, strong], axis=1)).to(device)  # (N, 2) bool
    costs_t = torch.from_numpy(costs).to(device)
    print(f"Loaded {len(df)} samples, e5 dim={in_dim} → g_θ out_dim={out_dim}, "
          f"weak_pass={weak.mean()*100:.1f}%  strong_pass={strong.mean()*100:.1f}%")

    n = len(df); perm = np.random.permutation(n)
    n_val = max(1, int(n * args.val_ratio))
    val_i = torch.as_tensor(perm[:n_val], dtype=torch.long)
    tr_i = torch.as_tensor(perm[n_val:], dtype=torch.long)

    head = build_head(in_dim, args.hidden, out_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def loss_on(indices):
        q = F.normalize(head(X[indices]), p=2, dim=-1)
        return cost_spectrum_infonce(q, desc, pass_mat[indices], costs_t, bands,
                                     args.tau_min, args.alpha, args.lam)

    best_val, best_state = float("inf"), None
    for ep in range(1, args.epochs + 1):
        head.train()
        bperm = tr_i[torch.randperm(len(tr_i))]
        for s in range(0, len(bperm), args.batch_size):
            b = bperm[s : s + args.batch_size]
            opt.zero_grad(); loss = loss_on(b); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            vl = loss_on(val_i).item(); tl = loss_on(tr_i).item()
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
            "out_dim": out_dim,
            "descriptors": desc.detach().cpu().numpy(),   # (2, Dd) 고정 대조 타깃
            "token_ids": token_ids,
            "cost_weak": float(costs[0]),
            "cost_strong": float(costs[1]),
            "base_embedding_model": args.embedding_model,
            "weak_model": args.weak_model,
            "strong_model": args.strong_model,
            "lam": args.lam, "alpha": args.alpha, "tau_min": args.tau_min,
        },
        args.output_path,
    )
    print(f"Saved CSCR router → {args.output_path}")


if __name__ == "__main__":
    main()
