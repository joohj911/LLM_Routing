"""
CSCR expert descriptor 계산 (arXiv:2508.12491, "Cost-Aware Contrastive Routing").

레퍼런스(rezashkv/cscr, scripts/compute_descriptors.py → router/descriptors.py)의
logit-footprint descriptor 를 우리 2-모델(weak/strong) 세팅에 맞춰 구현:

  각 expert 모델을 probe 프롬프트에 greedy 로 첫 n_tokens 개 생성시키고, 각 스텝의
  softmax 확률을 전체 vocab 에 대해 (스텝×probe) 평균 → 모델별 vocab 분포 d_m ∈ R^V.
  두 모델의 분포 합에서 top_k(기본 256) 토큰을 **공유 basis** 로 골라 그 좌표만 남기고
  L2 정규화 → descriptor d_weak, d_strong ∈ R^{top_k}. FAISS flat_ip(=cosine) 라우팅용.

이 스크립트는 실제 모델을 로드해 추론하므로 GPU 가 필요하다. 산출물(.npz)은
train_cscr.py 가 g_θ 학습의 고정 대조 타깃으로, CSCRRouter 가 추론 라우팅에 쓴다.

사용법:
  python lm_routing/routers/cscr/descriptors.py \\
    --prompts-path ./bfcl_data/prompts.json \\
    --weak-model   Qwen/Qwen3.5-0.8B \\
    --strong-model Qwen/Qwen3.5-9B \\
    --output-path  ./bfcl_data_0.8B/cscr_descriptors.npz
"""
import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from lm_routing.evals.eval_bfcl_models import load_model


@torch.no_grad()
def model_vocab_distribution(model, tokenizer, prompts, n_tokens=10, batch_size=8):
    """expert 모델의 vocab 분포 d_m ∈ R^V — probe 프롬프트에 greedy 로 첫 n_tokens 를
    생성하며 각 스텝 softmax 를 (스텝×probe) 평균한 것."""
    accum = None
    count = 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        def _render(p):
            kw = {"tokenize": False, "add_generation_prompt": True}
            try:      # eval 과 동일하게 thinking 비활성화 (첫 토큰이 공통 <think> 로 뭉개지지 않게)
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], enable_thinking=False, **kw)
            except Exception:
                return tokenizer.apply_chat_template([{"role": "user", "content": p}], **kw)
        texts = [_render(p) for p in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_new_tokens=n_tokens,
            do_sample=False,                 # greedy (결정적)
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        for step_logits in out.scores:       # tuple(len=생성 스텝수), 각 (B, V)
            probs = torch.softmax(step_logits.float(), dim=-1)  # (B, V)
            s = probs.sum(dim=0).double().cpu()                 # (V,)
            accum = s if accum is None else accum + s
            count += step_logits.shape[0]
    return (accum / max(count, 1))            # (V,) 평균 확률


def compute_descriptors(weak_model, strong_model, prompts, top_k=256, n_tokens=10,
                        batch_size=8, load_in_4bit=False):
    """반환: (token_ids(top_k,), descriptors(2, top_k) = [d_weak, d_strong], L2 정규화됨)."""
    dists = []
    for name in (weak_model, strong_model):
        print(f"[descriptors] loading {name} ...")
        model, tok = load_model(name, "cuda" if torch.cuda.is_available() else "cpu", load_in_4bit)
        print(f"[descriptors] probing {name} on {len(prompts)} prompts (n_tokens={n_tokens}) ...")
        dists.append(model_vocab_distribution(model, tok, prompts, n_tokens, batch_size))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    dw, ds = dists
    combined = dw + ds
    top_ids = torch.topk(combined, k=min(top_k, combined.numel())).indices  # 공유 basis
    top_ids = torch.sort(top_ids).values
    d_weak = F.normalize(dw[top_ids], p=2, dim=0)
    d_strong = F.normalize(ds[top_ids], p=2, dim=0)
    descriptors = torch.stack([d_weak, d_strong], dim=0).float().numpy()    # (2, top_k)
    return top_ids.numpy().astype(np.int64), descriptors


def main():
    ap = argparse.ArgumentParser(description="Compute CSCR logit-footprint expert descriptors (weak/strong)")
    ap.add_argument("--prompts-path", required=True, help="bfcl_data/prompts.json (probe 텍스트 소스)")
    ap.add_argument("--weak-model", required=True)
    ap.add_argument("--strong-model", required=True)
    ap.add_argument("--output-path", required=True, help="출력 .npz")
    ap.add_argument("--n-probes", type=int, default=192, help="probe 프롬프트 수(논문 기본 192). <=0 이면 전체")
    ap.add_argument("--top-k", type=int, default=256, help="공유 vocab basis 크기(descriptor 차원)")
    ap.add_argument("--n-tokens", type=int, default=10, help="probe 당 greedy 생성 토큰 수")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--cost-weak", type=float, default=0.0)
    ap.add_argument("--cost-strong", type=float, default=1.0)
    args = ap.parse_args()

    random.seed(args.seed)
    prompts_meta = json.load(open(args.prompts_path))
    texts = [p["prompt"] for p in prompts_meta if p.get("prompt")]
    if args.n_probes and args.n_probes > 0 and args.n_probes < len(texts):
        texts = random.sample(texts, args.n_probes)
    print(f"[descriptors] {len(texts)} probe prompts, top_k={args.top_k}, n_tokens={args.n_tokens}")

    token_ids, descriptors = compute_descriptors(
        args.weak_model, args.strong_model, texts,
        top_k=args.top_k, n_tokens=args.n_tokens, batch_size=args.batch_size,
        load_in_4bit=args.load_in_4bit,
    )
    np.savez(
        args.output_path,
        token_ids=token_ids,
        descriptors=descriptors,             # (2, top_k) = [weak, strong], L2 정규화
        weak_model=args.weak_model,
        strong_model=args.strong_model,
        cost_weak=np.float32(args.cost_weak),
        cost_strong=np.float32(args.cost_strong),
        top_k=np.int64(args.top_k),
        n_tokens=np.int64(args.n_tokens),
    )
    print(f"[descriptors] saved → {args.output_path}  (descriptors {descriptors.shape})")


if __name__ == "__main__":
    main()
