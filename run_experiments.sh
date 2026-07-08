#!/usr/bin/env bash
# run_experiments.sh — Full routing experiment pipeline
#
# Runs MF + UniRoute routing experiments on two model pairs:
#   Pair A: Qwen/Qwen3.5-0.8B (weak) vs Qwen/Qwen3.5-9B (strong)
#   Pair B: Qwen/Qwen3.5-2B  (weak) vs Qwen/Qwen3.5-9B (strong)
#
# Usage:
#   bash run_experiments.sh [OPTIONS]
#
# Options:
#   --bfcl-dir DIR         Base directory for BFCL data (default: ./bfcl_data)
#   --results-dir DIR      Output directory for results  (default: ./results)
#   --output-excel FILE    Output Excel file             (default: routing_results.xlsx)
#   --load-in-4bit         Use 4-bit quantization for model evaluation
#   --skip-embed           Skip embedding generation (reuse existing bfcl_data/)
#   --skip-eval-models     Skip model evaluation (reuse existing eval_results.json)
#   --num-results N        Number of threshold points per router (default: 10)
#   --random-iters N       Random router averaging iterations (default: 10)

set -euo pipefail

# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────
BFCL_DIR="./bfcl_data"
RESULTS_DIR="./results"
OUTPUT_EXCEL="routing_results.xlsx"
LOAD_4BIT=""
SKIP_EMBED=0
SKIP_EVAL=0
NUM_RESULTS=10
RANDOM_ITERS=10
EMB_MODEL="intfloat/multilingual-e5-small"
UNIROUTE_ASSIGNMENT="hard"   # 기본 hard(최근접 클러스터). soft 쓰려면 --uniroute-assignment soft|auto
UNIROUTE_PSI="val"           # Ψ 추정 데이터: val(논문 설계, 기본) | train(전체 refit)
MF_LR="3e-4"
MF_WD="1e-5"
WITH_CSCR=0                   # --with-cscr 로 켜면 CSCR(대조 KNN 라우터) descriptor 계산+학습+평가 포함
CSCR_NPROBES=192             # CSCR descriptor probe 프롬프트 수 (논문 기본)
CSCR_TOPK=256                # CSCR descriptor 차원(공유 top-k vocab basis)
CSCR_NTOKENS=10              # CSCR descriptor probe 당 greedy 생성 토큰 수
SEED=42                      # 라우터 학습/평가 randomness seed (MF init, KMeans, random 라우터)
SPLIT_SEED=42                 # train/test split seed. seed sweep 시 이걸 고정하면 test set·기준선이
                              #   상수로 유지돼 band가 '라우터 변동'만 반영 (권장: 고정)
GRAPH_RANDOM="--graph-random" # random baseline을 그래프에도 표시 (--no-graph-random로 끄기)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bfcl-dir)       BFCL_DIR="$2";    shift 2 ;;
    --results-dir)    RESULTS_DIR="$2"; shift 2 ;;
    --output-excel)   OUTPUT_EXCEL="$2"; shift 2 ;;
    --load-in-4bit)   LOAD_4BIT="--load-in-4bit"; shift ;;
    --skip-embed)     SKIP_EMBED=1; shift ;;
    --skip-eval-models) SKIP_EVAL=1; shift ;;
    --num-results)    NUM_RESULTS="$2"; shift 2 ;;
    --random-iters)   RANDOM_ITERS="$2"; shift 2 ;;
    --embedding-model) EMB_MODEL="$2"; shift 2 ;;
    --uniroute-assignment) UNIROUTE_ASSIGNMENT="$2"; shift 2 ;;
    --uniroute-psi)   UNIROUTE_PSI="$2"; shift 2 ;;
    --with-cscr)      WITH_CSCR=1; shift ;;
    --cscr-nprobes)   CSCR_NPROBES="$2"; shift 2 ;;
    --mf-lr)          MF_LR="$2"; shift 2 ;;
    --mf-weight-decay) MF_WD="$2"; shift 2 ;;
    --seed)           SEED="$2"; shift 2 ;;
    --split-seed)     SPLIT_SEED="$2"; shift 2 ;;
    --no-graph-random) GRAPH_RANDOM=""; shift ;;
    *) echo "[error] Unknown option: $1" >&2; exit 1 ;;
  esac
done

WEAK_0_8B="Qwen/Qwen3.5-0.8B"
WEAK_2B="Qwen/Qwen3.5-2B"
STRONG="Qwen/Qwen3.5-9B"

DATA_0_8B="${BFCL_DIR}_0.8B"
DATA_2B="${BFCL_DIR}_2B"
EVAL_RESULTS_JSON="./eval_results.json"
EVAL_RESPONSES_JSON="./eval_responses.json"  # per-sample raw output trace

# ─────────────────────────────────────────────
# Preflight: torch / CUDA / driver sanity check
# ─────────────────────────────────────────────
# Catches the common failure where pip pulled a torch wheel built against a
# CUDA runtime newer than the installed NVIDIA driver supports (torch imports
# but torch.cuda.is_available() is False, or the version is unexpectedly new).
python - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("[preflight] torch is not installed. Install the build matching your "
             "driver first, e.g.:\n  pip install torch==2.5.1 "
             "--index-url https://download.pytorch.org/whl/cu121")

print(f"[preflight] torch {torch.__version__} (CUDA build: {torch.version.cuda})")
if not torch.cuda.is_available():
    sys.exit("[preflight] torch.cuda.is_available() is False. This usually means the "
             "torch CUDA build does not match the NVIDIA driver.\n"
             "  Check `nvidia-smi` for the driver's max CUDA version, then reinstall "
             "the matching torch build, e.g. for CUDA 12.1:\n"
             "  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121")
print(f"[preflight] {torch.cuda.device_count()} GPU(s) visible: "
      f"{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
PY

echo "============================================================"
echo " RouteLLM × UniRoute Experiment Pipeline"
echo "============================================================"
echo "  BFCL data dir : ${BFCL_DIR}"
echo "  Pair A        : ${WEAK_0_8B} vs ${STRONG}"
echo "  Pair B        : ${WEAK_2B}   vs ${STRONG}"
echo "  Results dir   : ${RESULTS_DIR}"
echo "  Output Excel  : ${OUTPUT_EXCEL}"
echo "============================================================"

# ─────────────────────────────────────────────
# Step 1: Generate embeddings
# ─────────────────────────────────────────────
if [[ $SKIP_EMBED -eq 0 ]]; then
  echo ""
  echo "[Step 1/7] Generating BFCL embeddings (${EMB_MODEL}) → ${BFCL_DIR}/"
  python lm_routing/routers/matrix_factorization/prepare_bfcl_data.py embed \
    --output-dir "${BFCL_DIR}" \
    --embedding-model "${EMB_MODEL}"
else
  echo ""
  echo "[Step 1/7] Skipping embedding generation (--skip-embed)"
  if [[ ! -f "${BFCL_DIR}/embeddings.npy" ]]; then
    echo "[error] ${BFCL_DIR}/embeddings.npy not found. Remove --skip-embed to generate." >&2
    exit 1
  fi
fi

# ─────────────────────────────────────────────
# Step 2: Evaluate all models on BFCL
# ─────────────────────────────────────────────
if [[ $SKIP_EVAL -eq 0 ]]; then
  echo ""
  echo "[Step 2/7] Evaluating models on BFCL → ${EVAL_RESULTS_JSON}"
  python lm_routing/evals/eval_bfcl_models.py \
    --prompts-path "${BFCL_DIR}/prompts.json" \
    --output-path  "${EVAL_RESULTS_JSON}" \
    --save-responses "${EVAL_RESPONSES_JSON}" \
    --models "${WEAK_0_8B}" "${WEAK_2B}" "${STRONG}" \
    --seed "${SEED}" \
    ${LOAD_4BIT}
else
  echo ""
  echo "[Step 2/7] Skipping model evaluation (--skip-eval-models)"
  if [[ ! -f "${EVAL_RESULTS_JSON}" ]]; then
    echo "[error] ${EVAL_RESULTS_JSON} not found. Remove --skip-eval-models to generate." >&2
    exit 1
  fi
fi

# ─────────────────────────────────────────────
# Step 3: Convert results → train/test splits per pair
# ─────────────────────────────────────────────
echo ""
echo "[Step 3/7] Converting eval results → train/test splits"

echo "  Pair A: ${WEAK_0_8B} vs ${STRONG} → ${DATA_0_8B}/"
python lm_routing/routers/matrix_factorization/prepare_bfcl_data.py convert \
  --results-path "${EVAL_RESULTS_JSON}" \
  --prompts-path "${BFCL_DIR}/prompts.json" \
  --output-dir   "${DATA_0_8B}" \
  --weak-model   "${WEAK_0_8B}" \
  --strong-model "${STRONG}" \
  --seed         "${SPLIT_SEED}"

echo "  Pair B: ${WEAK_2B} vs ${STRONG} → ${DATA_2B}/"
python lm_routing/routers/matrix_factorization/prepare_bfcl_data.py convert \
  --results-path "${EVAL_RESULTS_JSON}" \
  --prompts-path "${BFCL_DIR}/prompts.json" \
  --output-dir   "${DATA_2B}" \
  --weak-model   "${WEAK_2B}" \
  --strong-model "${STRONG}" \
  --seed         "${SPLIT_SEED}"

# ─────────────────────────────────────────────
# Step 4: Train MF router for each pair
# ─────────────────────────────────────────────
echo ""
echo "[Step 4/7] Training MF routers"

# both-fail 은 strong win 으로 라벨(tie→strong, RouteLLM 기본 동작).
train_mf () {  # $1=train_data  $2=output
  python lm_routing/routers/matrix_factorization/train_matrix_factorization.py \
    --train-data   "$1" \
    --npy-path     "${BFCL_DIR}/embeddings.npy" \
    --output-path  "$2" \
    --seed "${SEED}" \
    --tie-goes-to strong \
    --embedding-model "${EMB_MODEL}" \
    --lr "${MF_LR}" \
    --weight-decay "${MF_WD}" \
    --num-epochs 100 \
    --dim 128 \
    --batch-size 64
}

echo "  Pair A MF → ${DATA_0_8B}/mf_model.pt"
train_mf "${DATA_0_8B}/train_data.json" "${DATA_0_8B}/mf_model.pt"
echo "  Pair B MF → ${DATA_2B}/mf_model.pt"
train_mf "${DATA_2B}/train_data.json" "${DATA_2B}/mf_model.pt"

# ─────────────────────────────────────────────
# Step 5: Train UniRoute router for each pair
# ─────────────────────────────────────────────
echo ""
echo "[Step 5/7] Training UniRoute (K-Means) routers"

echo "  Pair A UniRoute → ${DATA_0_8B}/uniroute_model.pt"
python lm_routing/routers/uniroute/train_uniroute.py \
  --train-data   "${DATA_0_8B}/train_data.json" \
  --npy-path     "${BFCL_DIR}/embeddings.npy" \
  --output-path  "${DATA_0_8B}/uniroute_model.pt" \
  --seed         "${SEED}" \
  --weak-model   "${WEAK_0_8B}" \
  --strong-model "${STRONG}" \
  --assignment   "${UNIROUTE_ASSIGNMENT}" \
  --psi-source   "${UNIROUTE_PSI}" \
  --embedding-model "${EMB_MODEL}"

echo "  Pair B UniRoute → ${DATA_2B}/uniroute_model.pt"
python lm_routing/routers/uniroute/train_uniroute.py \
  --train-data   "${DATA_2B}/train_data.json" \
  --npy-path     "${BFCL_DIR}/embeddings.npy" \
  --output-path  "${DATA_2B}/uniroute_model.pt" \
  --seed         "${SEED}" \
  --weak-model   "${WEAK_2B}" \
  --strong-model "${STRONG}" \
  --assignment   "${UNIROUTE_ASSIGNMENT}" \
  --psi-source   "${UNIROUTE_PSI}" \
  --embedding-model "${EMB_MODEL}"

# UniRoute variant: final Ψ refit on the full train set (cl+val) via --psi-source train.
# K is still chosen honestly (Ψ on cl); this just uses more data for the deployed Ψ.
# Compared side-by-side against the val-Ψ variant on the test set.
echo "  Pair A UniRoute(Ψ=train) → ${DATA_0_8B}/uniroute_train_model.pt"
python lm_routing/routers/uniroute/train_uniroute.py \
  --train-data   "${DATA_0_8B}/train_data.json" \
  --npy-path     "${BFCL_DIR}/embeddings.npy" \
  --output-path  "${DATA_0_8B}/uniroute_train_model.pt" \
  --seed         "${SEED}" \
  --weak-model   "${WEAK_0_8B}" \
  --strong-model "${STRONG}" \
  --assignment   "${UNIROUTE_ASSIGNMENT}" \
  --psi-source   train \
  --embedding-model "${EMB_MODEL}"

echo "  Pair B UniRoute(Ψ=train) → ${DATA_2B}/uniroute_train_model.pt"
python lm_routing/routers/uniroute/train_uniroute.py \
  --train-data   "${DATA_2B}/train_data.json" \
  --npy-path     "${BFCL_DIR}/embeddings.npy" \
  --output-path  "${DATA_2B}/uniroute_train_model.pt" \
  --seed         "${SEED}" \
  --weak-model   "${WEAK_2B}" \
  --strong-model "${STRONG}" \
  --assignment   "${UNIROUTE_ASSIGNMENT}" \
  --psi-source   train \
  --embedding-model "${EMB_MODEL}"

# Uni-R2 (R2-Router unirouter/uni_r2.py) 충실 재현: honest K, Ψ=train 은 위와 같고
# assignment 를 soft(Φ·Ψ, softmax 클러스터 멤버십)로 바꾼 것. hard uniroute_train 과
# 나란히 두면 COMPARISON.md 의 "UniRouter(hard) vs Uni-R2(soft)" 대비가 된다.
echo "  Pair A Uni-R2(soft Φ·Ψ) → ${DATA_0_8B}/uni_r2_model.pt"
python lm_routing/routers/uniroute/train_uniroute.py \
  --train-data   "${DATA_0_8B}/train_data.json" \
  --npy-path     "${BFCL_DIR}/embeddings.npy" \
  --output-path  "${DATA_0_8B}/uni_r2_model.pt" \
  --seed         "${SEED}" \
  --weak-model   "${WEAK_0_8B}" \
  --strong-model "${STRONG}" \
  --assignment   soft \
  --psi-source   train \
  --embedding-model "${EMB_MODEL}"

echo "  Pair B Uni-R2(soft Φ·Ψ) → ${DATA_2B}/uni_r2_model.pt"
python lm_routing/routers/uniroute/train_uniroute.py \
  --train-data   "${DATA_2B}/train_data.json" \
  --npy-path     "${BFCL_DIR}/embeddings.npy" \
  --output-path  "${DATA_2B}/uni_r2_model.pt" \
  --seed         "${SEED}" \
  --weak-model   "${WEAK_2B}" \
  --strong-model "${STRONG}" \
  --assignment   soft \
  --psi-source   train \
  --embedding-model "${EMB_MODEL}"

# UniRoute variant: legacy circular K-selection (Ψ estimated on val and scored on
# the same val → overfits, tends to pick large K). Kept only to compare that old
# behaviour against the honest K-selection on the held-out test set.
echo "  Pair A UniRoute(legacy K) → ${DATA_0_8B}/uniroute_legacy_model.pt"
python lm_routing/routers/uniroute/train_uniroute.py \
  --train-data   "${DATA_0_8B}/train_data.json" \
  --npy-path     "${BFCL_DIR}/embeddings.npy" \
  --output-path  "${DATA_0_8B}/uniroute_legacy_model.pt" \
  --seed         "${SEED}" \
  --weak-model   "${WEAK_0_8B}" \
  --strong-model "${STRONG}" \
  --assignment   "${UNIROUTE_ASSIGNMENT}" \
  --psi-source   val \
  --k-select-psi val \
  --embedding-model "${EMB_MODEL}"

echo "  Pair B UniRoute(legacy K) → ${DATA_2B}/uniroute_legacy_model.pt"
python lm_routing/routers/uniroute/train_uniroute.py \
  --train-data   "${DATA_2B}/train_data.json" \
  --npy-path     "${BFCL_DIR}/embeddings.npy" \
  --output-path  "${DATA_2B}/uniroute_legacy_model.pt" \
  --seed         "${SEED}" \
  --weak-model   "${WEAK_2B}" \
  --strong-model "${STRONG}" \
  --assignment   "${UNIROUTE_ASSIGNMENT}" \
  --psi-source   val \
  --k-select-psi val \
  --embedding-model "${EMB_MODEL}"

# ─────────────────────────────────────────────
# Step 5b: Train R2-Router per-model routers
# ─────────────────────────────────────────────
# R2-Router(UCF-ML-Research/R2-Router)의 per-model 라우터를 2-모델·단일 budget·
# cost∈{0,1} 로 특수화: weak/strong 각각 Ridge 로 P(pass|emb) 예측 → gain 순 라우팅.
echo ""
echo "[Step 5b] Training R2-Router (per-model) routers"

python lm_routing/routers/r2_router/train_r2_router.py \
  --train-data "${DATA_0_8B}/train_data.json" --npy-path "${BFCL_DIR}/embeddings.npy" \
  --output-path "${DATA_0_8B}/r2_router_model.pt" --weak-model "${WEAK_0_8B}" \
  --strong-model "${STRONG}" --seed "${SEED}" --embedding-model "${EMB_MODEL}"
python lm_routing/routers/r2_router/train_r2_router.py \
  --train-data "${DATA_2B}/train_data.json" --npy-path "${BFCL_DIR}/embeddings.npy" \
  --output-path "${DATA_2B}/r2_router_model.pt" --weak-model "${WEAK_2B}" \
  --strong-model "${STRONG}" --seed "${SEED}" --embedding-model "${EMB_MODEL}"

# ─────────────────────────────────────────────
# Step 5c: (opt) CSCR — 실제 모델 logit descriptor 계산 + 대조 라우터 g_θ 학습
# ─────────────────────────────────────────────
# CSCR(arXiv:2508.12491): weak/strong 모델 출력에서 logit-footprint descriptor 를 계산하고
# (GPU probe 추론), frozen e5 → 2-layer MLP g_θ 를 cost-spectrum InfoNCE 로 학습해 q 를
# descriptor 에 cosine-NN 라우팅한다. --with-cscr 일 때만 실행(모델 재로딩으로 시간 듦).
CSCR_ROUTERS=""; CSCR_CKPT_A=""; CSCR_CKPT_B=""
if [[ $WITH_CSCR -eq 1 ]]; then
  echo ""
  echo "[Step 5c] CSCR descriptors + contrastive router (probes=${CSCR_NPROBES}, top_k=${CSCR_TOPK})"
  train_cscr_pair () {  # $1=data_dir  $2=weak
    # descriptor 는 모델 logit footprint(임베딩·라우터seed 무관) → 이미 있으면 재사용
    # (seed sweep 시 GPU probe 추론 반복 방지). 다시 뽑으려면 .npz 를 지우면 됨.
    if [[ -f "$1/cscr_descriptors.npz" ]]; then
      echo "    [reuse] $1/cscr_descriptors.npz (기존 descriptor 재사용)"
    else
      python lm_routing/routers/cscr/descriptors.py \
        --prompts-path "${BFCL_DIR}/prompts.json" \
        --weak-model "$2" --strong-model "${STRONG}" \
        --output-path "$1/cscr_descriptors.npz" \
        --n-probes "${CSCR_NPROBES}" --top-k "${CSCR_TOPK}" --n-tokens "${CSCR_NTOKENS}" \
        --seed "${SEED}" ${LOAD_4BIT}
    fi
    python lm_routing/routers/cscr/train_cscr.py \
      --train-data "$1/train_data.json" --npy-path "${BFCL_DIR}/embeddings.npy" \
      --descriptors "$1/cscr_descriptors.npz" --output-path "$1/cscr_model.pt" \
      --weak-model "$2" --strong-model "${STRONG}" \
      --embedding-model "${EMB_MODEL}" --seed "${SEED}"
  }
  echo "  Pair A CSCR → ${DATA_0_8B}/cscr_model.pt"
  train_cscr_pair "${DATA_0_8B}" "${WEAK_0_8B}"
  echo "  Pair B CSCR → ${DATA_2B}/cscr_model.pt"
  train_cscr_pair "${DATA_2B}" "${WEAK_2B}"
  CSCR_ROUTERS="cscr"
  CSCR_CKPT_A="--cscr-checkpoint ${DATA_0_8B}/cscr_model.pt"
  CSCR_CKPT_B="--cscr-checkpoint ${DATA_2B}/cscr_model.pt"
fi

# ─────────────────────────────────────────────
# Step 6: Evaluate all routers on test set
# ─────────────────────────────────────────────
echo ""
echo "[Step 6/7] Evaluating routers (random / mf / uniroute×3 / uni_r2 / r2_router$([[ $WITH_CSCR -eq 1 ]] && echo ' / cscr'))"

RESULT_0_8B="${RESULTS_DIR}/pair_0.8B"
RESULT_2B="${RESULTS_DIR}/pair_2B"
mkdir -p "${RESULT_0_8B}" "${RESULT_2B}"

echo "  Pair A → ${RESULT_0_8B}/eval_results.json"
python -m lm_routing.evals.evaluate \
  --routers random mf uniroute uniroute_train uni_r2 uniroute_legacy r2_router ${CSCR_ROUTERS} \
  --test-data         "${DATA_0_8B}/test_data.json" \
  --mf-checkpoint     "${DATA_0_8B}/mf_model.pt" \
  --uniroute-checkpoint "${DATA_0_8B}/uniroute_model.pt" \
  --uniroute-train-checkpoint "${DATA_0_8B}/uniroute_train_model.pt" \
  --uni-r2-checkpoint "${DATA_0_8B}/uni_r2_model.pt" \
  --uniroute-legacy-checkpoint "${DATA_0_8B}/uniroute_legacy_model.pt" \
  --r2-router-checkpoint "${DATA_0_8B}/r2_router_model.pt" \
  ${CSCR_CKPT_A} \
  --strong-model      "${STRONG}" \
  --weak-model        "${WEAK_0_8B}" \
  --output            "${RESULT_0_8B}" \
  --num-results       "${NUM_RESULTS}" \
  --random-iters      "${RANDOM_ITERS}" \
  --overwrite-cache   mf uniroute uniroute_train uni_r2 uniroute_legacy r2_router ${CSCR_ROUTERS} \
  --seed              "${SEED}" \
  --quiet \
  --output-json       "${RESULT_0_8B}/eval_results.json"

echo "  Pair B → ${RESULT_2B}/eval_results.json"
python -m lm_routing.evals.evaluate \
  --routers random mf uniroute uniroute_train uni_r2 uniroute_legacy r2_router ${CSCR_ROUTERS} \
  --test-data         "${DATA_2B}/test_data.json" \
  --mf-checkpoint     "${DATA_2B}/mf_model.pt" \
  --uniroute-checkpoint "${DATA_2B}/uniroute_model.pt" \
  --uniroute-train-checkpoint "${DATA_2B}/uniroute_train_model.pt" \
  --uni-r2-checkpoint "${DATA_2B}/uni_r2_model.pt" \
  --uniroute-legacy-checkpoint "${DATA_2B}/uniroute_legacy_model.pt" \
  --r2-router-checkpoint "${DATA_2B}/r2_router_model.pt" \
  ${CSCR_CKPT_B} \
  --strong-model      "${STRONG}" \
  --weak-model        "${WEAK_2B}" \
  --output            "${RESULT_2B}" \
  --num-results       "${NUM_RESULTS}" \
  --random-iters      "${RANDOM_ITERS}" \
  --overwrite-cache   mf uniroute uniroute_train uni_r2 uniroute_legacy r2_router ${CSCR_ROUTERS} \
  --seed              "${SEED}" \
  --quiet \
  --output-json       "${RESULT_2B}/eval_results.json"

# ─────────────────────────────────────────────
# Step 7: Collect results → Excel + graphs
# ─────────────────────────────────────────────
echo ""
echo "[Step 7/7] Collecting results → ${OUTPUT_EXCEL}"
python collect_results.py \
  --results-jsons \
    "${RESULT_0_8B}/eval_results.json" \
    "${RESULT_2B}/eval_results.json" \
  --output "${OUTPUT_EXCEL}" ${GRAPH_RANDOM}

echo ""
echo "============================================================"
echo " Done! Results saved to:"
echo "   Excel  : ${OUTPUT_EXCEL}"
echo "   Graphs : $(dirname ${OUTPUT_EXCEL})/routing_curves.png"
echo "   Raw    : ${RESULTS_DIR}/pair_0.8B/eval_results.json"
echo "            ${RESULTS_DIR}/pair_2B/eval_results.json"
echo "============================================================"
