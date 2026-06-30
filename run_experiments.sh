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
    --mf-lr)          MF_LR="$2"; shift 2 ;;
    --mf-weight-decay) MF_WD="$2"; shift 2 ;;
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
  --strong-model "${STRONG}"

echo "  Pair B: ${WEAK_2B} vs ${STRONG} → ${DATA_2B}/"
python lm_routing/routers/matrix_factorization/prepare_bfcl_data.py convert \
  --results-path "${EVAL_RESULTS_JSON}" \
  --prompts-path "${BFCL_DIR}/prompts.json" \
  --output-dir   "${DATA_2B}" \
  --weak-model   "${WEAK_2B}" \
  --strong-model "${STRONG}"

# ─────────────────────────────────────────────
# Step 4: Train MF router for each pair
# ─────────────────────────────────────────────
echo ""
echo "[Step 4/7] Training MF routers"

echo "  Pair A MF → ${DATA_0_8B}/mf_model.pt"
python lm_routing/routers/matrix_factorization/train_matrix_factorization.py \
  --train-data   "${DATA_0_8B}/train_data.json" \
  --npy-path     "${BFCL_DIR}/embeddings.npy" \
  --output-path  "${DATA_0_8B}/mf_model.pt" \
  --embedding-model "${EMB_MODEL}" \
  --lr "${MF_LR}" \
  --weight-decay "${MF_WD}" \
  --num-epochs 100 \
  --dim 128 \
  --batch-size 64

echo "  Pair B MF → ${DATA_2B}/mf_model.pt"
python lm_routing/routers/matrix_factorization/train_matrix_factorization.py \
  --train-data   "${DATA_2B}/train_data.json" \
  --npy-path     "${BFCL_DIR}/embeddings.npy" \
  --output-path  "${DATA_2B}/mf_model.pt" \
  --embedding-model "${EMB_MODEL}" \
  --lr "${MF_LR}" \
  --weight-decay "${MF_WD}" \
  --num-epochs 100 \
  --dim 128 \
  --batch-size 64

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
  --weak-model   "${WEAK_2B}" \
  --strong-model "${STRONG}" \
  --assignment   "${UNIROUTE_ASSIGNMENT}" \
  --psi-source   "${UNIROUTE_PSI}" \
  --embedding-model "${EMB_MODEL}"

# ── Per-model regression routers (R2-style, budget-free) ──
echo "  Pair A PerModel → ${DATA_0_8B}/permodel_model.pt"
python lm_routing/routers/per_model/train_per_model.py \
  --train-data   "${DATA_0_8B}/train_data.json" \
  --npy-path     "${BFCL_DIR}/embeddings.npy" \
  --output-path  "${DATA_0_8B}/permodel_model.pt" \
  --weak-model   "${WEAK_0_8B}" \
  --strong-model "${STRONG}" \
  --embedding-model "${EMB_MODEL}"

echo "  Pair B PerModel → ${DATA_2B}/permodel_model.pt"
python lm_routing/routers/per_model/train_per_model.py \
  --train-data   "${DATA_2B}/train_data.json" \
  --npy-path     "${BFCL_DIR}/embeddings.npy" \
  --output-path  "${DATA_2B}/permodel_model.pt" \
  --weak-model   "${WEAK_2B}" \
  --strong-model "${STRONG}" \
  --embedding-model "${EMB_MODEL}"

# ─────────────────────────────────────────────
# Step 6: Evaluate all routers on test set
# ─────────────────────────────────────────────
echo ""
echo "[Step 6/7] Evaluating routers (random / mf / uniroute / permodel)"

RESULT_0_8B="${RESULTS_DIR}/pair_0.8B"
RESULT_2B="${RESULTS_DIR}/pair_2B"
mkdir -p "${RESULT_0_8B}" "${RESULT_2B}"

echo "  Pair A → ${RESULT_0_8B}/eval_results.json"
python -m lm_routing.evals.evaluate \
  --routers random mf uniroute permodel \
  --test-data         "${DATA_0_8B}/test_data.json" \
  --mf-checkpoint     "${DATA_0_8B}/mf_model.pt" \
  --uniroute-checkpoint "${DATA_0_8B}/uniroute_model.pt" \
  --permodel-checkpoint "${DATA_0_8B}/permodel_model.pt" \
  --strong-model      "${STRONG}" \
  --weak-model        "${WEAK_0_8B}" \
  --output            "${RESULT_0_8B}" \
  --num-results       "${NUM_RESULTS}" \
  --random-iters      "${RANDOM_ITERS}" \
  --overwrite-cache   mf uniroute permodel \
  --output-json       "${RESULT_0_8B}/eval_results.json"

echo "  Pair B → ${RESULT_2B}/eval_results.json"
python -m lm_routing.evals.evaluate \
  --routers random mf uniroute permodel \
  --test-data         "${DATA_2B}/test_data.json" \
  --mf-checkpoint     "${DATA_2B}/mf_model.pt" \
  --uniroute-checkpoint "${DATA_2B}/uniroute_model.pt" \
  --permodel-checkpoint "${DATA_2B}/permodel_model.pt" \
  --strong-model      "${STRONG}" \
  --weak-model        "${WEAK_2B}" \
  --output            "${RESULT_2B}" \
  --num-results       "${NUM_RESULTS}" \
  --random-iters      "${RANDOM_ITERS}" \
  --overwrite-cache   mf uniroute permodel \
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
  --output "${OUTPUT_EXCEL}"

echo ""
echo "============================================================"
echo " Done! Results saved to:"
echo "   Excel  : ${OUTPUT_EXCEL}"
echo "   Graphs : $(dirname ${OUTPUT_EXCEL})/routing_curves.png"
echo "   Raw    : ${RESULTS_DIR}/pair_0.8B/eval_results.json"
echo "            ${RESULTS_DIR}/pair_2B/eval_results.json"
echo "============================================================"
