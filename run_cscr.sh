#!/usr/bin/env bash
# ─────────────────────────────────────────────
# CSCR contrastive-embedding experiment.
#
# frozen e5 위에 대조학습 head g_θ를 얹어 임베딩을 변환한 뒤(embeddings_cscr.npy),
# 기존 라우터(MF / UniRoute×3 / per-model)를 그 임베딩으로 재학습·평가한다.
# e5 임베딩과 test 성능을 비교하기 위한 스크립트.
#
# 선행 조건: 먼저 `bash run_experiments.sh` 를 한 번 돌려
#   - ${BFCL_DIR}/embeddings.npy
#   - ${BFCL_DIR}_0.8B/{train,test}_data.json, ${BFCL_DIR}_2B/{train,test}_data.json
#   - (비교용) ./results/pair_*/eval_results.json
# 가 있어야 한다 (모델 BFCL 평가 재사용 → GPU 재추론 불필요).
#
# 사용법:
#   bash run_cscr.sh                 # 기본(epochs 100)
#   bash run_cscr.sh --epochs 2      # smoke test
# ─────────────────────────────────────────────
set -euo pipefail

BFCL_DIR="./bfcl_data"
RESULTS_DIR="./results_cscr"
OUTPUT_EXCEL="routing_cscr.xlsx"
STRONG="Qwen/Qwen3.5-9B"
EMB_MODEL="intfloat/multilingual-e5-small"
UNIROUTE_ASSIGNMENT="hard"
SEED=42
EPOCHS=100
OUT_DIM=256
HIDDEN=512
PMCLUSTER_K=20   # permodel_cluster: 회귀에 주입할 UniRoute 클러스터 수

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bfcl-dir)      BFCL_DIR="$2"; shift 2 ;;
    --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
    --output-excel)  OUTPUT_EXCEL="$2"; shift 2 ;;
    --embedding-model) EMB_MODEL="$2"; shift 2 ;;
    --epochs)        EPOCHS="$2"; shift 2 ;;
    --out-dim)       OUT_DIM="$2"; shift 2 ;;
    --hidden)        HIDDEN="$2"; shift 2 ;;
    --seed)          SEED="$2"; shift 2 ;;
    --permodel-cluster-k) PMCLUSTER_K="$2"; shift 2 ;;
    *) echo "[error] Unknown option: $1" >&2; exit 1 ;;
  esac
done

python - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("[preflight] torch not installed.")
print(f"[preflight] torch {torch.__version__}  cuda={torch.cuda.is_available()}")
PY

if [[ ! -f "${BFCL_DIR}/embeddings.npy" ]]; then
  echo "[error] ${BFCL_DIR}/embeddings.npy not found. Run run_experiments.sh first." >&2
  exit 1
fi

PAIR_WEAK=("Qwen/Qwen3.5-0.8B" "Qwen/Qwen3.5-2B")
PAIR_DIR=("${BFCL_DIR}_0.8B" "${BFCL_DIR}_2B")
PAIR_TAG=("0.8B" "2B")

for k in 0 1; do
  WEAK="${PAIR_WEAK[$k]}"; D="${PAIR_DIR[$k]}"; R="${RESULTS_DIR}/pair_${PAIR_TAG[$k]}"
  HEAD="${D}/cscr_head.pt"; NPY="${D}/embeddings_cscr.npy"; EM="cscr:${HEAD}"
  mkdir -p "${R}"
  echo "============================================================"
  echo " CSCR pair ${PAIR_TAG[$k]}: ${WEAK} vs ${STRONG}"
  echo "============================================================"

  echo "[1/3] Train contrastive head + transform embeddings → ${NPY}"
  python lm_routing/routers/contrastive/train_contrastive_embed.py \
    --train-data      "${D}/train_data.json" \
    --npy-path        "${BFCL_DIR}/embeddings.npy" \
    --output-path     "${HEAD}" \
    --output-npy      "${NPY}" \
    --weak-model      "${WEAK}" \
    --strong-model    "${STRONG}" \
    --embedding-model "${EMB_MODEL}" \
    --out-dim ${OUT_DIM} --hidden ${HIDDEN} --epochs ${EPOCHS} --seed ${SEED}

  echo "[2/3] Train routers on CSCR embedding"
  python lm_routing/routers/matrix_factorization/train_matrix_factorization.py \
    --train-data "${D}/train_data.json" --npy-path "${NPY}" \
    --output-path "${D}/mf_cscr.pt" --embedding-model "${EM}" \
    --lr 3e-4 --weight-decay 1e-5 --num-epochs 100 --dim 128 --batch-size 64 --seed ${SEED}

  # UniRoute 3 variants (honest Ψ=val / honest Ψ=train / legacy circular)
  python lm_routing/routers/uniroute/train_uniroute.py \
    --train-data "${D}/train_data.json" --npy-path "${NPY}" \
    --output-path "${D}/uniroute_cscr.pt" --weak-model "${WEAK}" --strong-model "${STRONG}" \
    --assignment "${UNIROUTE_ASSIGNMENT}" --psi-source val --k-select-psi cl \
    --embedding-model "${EM}" --seed ${SEED}
  python lm_routing/routers/uniroute/train_uniroute.py \
    --train-data "${D}/train_data.json" --npy-path "${NPY}" \
    --output-path "${D}/uniroute_train_cscr.pt" --weak-model "${WEAK}" --strong-model "${STRONG}" \
    --assignment "${UNIROUTE_ASSIGNMENT}" --psi-source train --k-select-psi cl \
    --embedding-model "${EM}" --seed ${SEED}
  python lm_routing/routers/uniroute/train_uniroute.py \
    --train-data "${D}/train_data.json" --npy-path "${NPY}" \
    --output-path "${D}/uniroute_legacy_cscr.pt" --weak-model "${WEAK}" --strong-model "${STRONG}" \
    --assignment "${UNIROUTE_ASSIGNMENT}" --psi-source val --k-select-psi val \
    --embedding-model "${EM}" --seed ${SEED}

  python lm_routing/routers/per_model/train_per_model.py \
    --train-data "${D}/train_data.json" --npy-path "${NPY}" \
    --output-path "${D}/permodel_cscr.pt" --weak-model "${WEAK}" --strong-model "${STRONG}" \
    --embedding-model "${EM}" --seed ${SEED}
  # permodel_cluster: UniRoute(honest K, Ψ=train)-cscr 클러스터 신호를 그대로 재사용
  python lm_routing/routers/per_model/train_per_model.py \
    --train-data "${D}/train_data.json" --npy-path "${NPY}" \
    --output-path "${D}/permodel_cluster_cscr.pt" --weak-model "${WEAK}" --strong-model "${STRONG}" \
    --embedding-model "${EM}" --seed ${SEED} \
    --uniroute-checkpoint "${D}/uniroute_train_cscr.pt"

  echo "[3/3] Evaluate on test → ${R}/eval_results.json"
  python -m lm_routing.evals.evaluate \
    --routers random mf uniroute uniroute_train uniroute_legacy permodel permodel_cluster \
    --test-data                  "${D}/test_data.json" \
    --mf-checkpoint              "${D}/mf_cscr.pt" \
    --uniroute-checkpoint        "${D}/uniroute_cscr.pt" \
    --uniroute-train-checkpoint  "${D}/uniroute_train_cscr.pt" \
    --uniroute-legacy-checkpoint "${D}/uniroute_legacy_cscr.pt" \
    --permodel-checkpoint        "${D}/permodel_cscr.pt" \
    --permodel-cluster-checkpoint "${D}/permodel_cluster_cscr.pt" \
    --strong-model "${STRONG}" --weak-model "${WEAK}" \
    --output "${R}" --num-results 10 --random-iters 10 \
    --overwrite-cache mf uniroute uniroute_train uniroute_legacy permodel permodel_cluster \
    --seed ${SEED} --quiet --output-json "${R}/eval_results.json"
done

echo ""
echo "[collect] → ${OUTPUT_EXCEL}  (e5 vs cscr overlay if e5 results exist)"
JSONS=()
for tag in "0.8B" "2B"; do
  E5="./results/pair_${tag}/eval_results.json"
  [[ -f "${E5}" ]] && JSONS+=("e5=${E5}")
  JSONS+=("cscr=${RESULTS_DIR}/pair_${tag}/eval_results.json")
done
python collect_results.py --results-jsons "${JSONS[@]}" --output "${OUTPUT_EXCEL}" --graph-random

echo ""
echo "Done. CSCR results → ${RESULTS_DIR}/ , comparison Excel → ${OUTPUT_EXCEL}"
