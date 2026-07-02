#!/usr/bin/env bash
# compare_all.sh — e5-small vs e5-large × MF vs UniRoute 비교.
#
# 각 임베딩 모델로 (임베딩 생성 → convert → MF/UniRoute 학습 → 라우터 평가)를 돌리고,
# 마지막에 analyze_routers.py 로 pair별 4-way(small/large × mf/uniroute) 비교표를 출력한다.
#
# 모델 pass/fail(eval_results.json)은 재사용하므로 LLM 평가는 다시 돌지 않는다
# (Step 2 skip). 따라서 사전에 eval_results.json 이 있어야 한다.
#
# 사용법:
#   bash compare_all.sh                          # 기본: e5-small + e5-large
#   bash compare_all.sh --embeddings "intfloat/multilingual-e5-small intfloat/multilingual-e5-large"
#   bash compare_all.sh --mf-lr 1e-3 --mf-weight-decay 1e-4 --uniroute-assignment auto

set -euo pipefail

EMB_LIST="intfloat/multilingual-e5-small intfloat/multilingual-e5-large"
EVAL_RESULTS="./eval_results.json"
MF_LR="3e-4"; MF_WD="1e-5"; UNI_ASSIGN="hard"   # 기본 hard. soft 비교하려면 --uniroute-assignment soft|auto
WEAK_0_8B="Qwen/Qwen3.5-0.8B"; WEAK_2B="Qwen/Qwen3.5-2B"; STRONG="Qwen/Qwen3.5-9B"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --embeddings)          EMB_LIST="$2"; shift 2 ;;
    --eval-results)        EVAL_RESULTS="$2"; shift 2 ;;
    --mf-lr)               MF_LR="$2"; shift 2 ;;
    --mf-weight-decay)     MF_WD="$2"; shift 2 ;;
    --uniroute-assignment) UNI_ASSIGN="$2"; shift 2 ;;
    *) echo "[error] Unknown option: $1" >&2; exit 1 ;;
  esac
done

[[ -f "$EVAL_RESULTS" ]] || { echo "[error] ${EVAL_RESULTS} 없음. 먼저 모델 평가(Step 2)를 한 번 돌리세요." >&2; exit 1; }

# 임베딩 모델명 → 짧은 태그 (디렉토리용)
tag_of() {
  case "$1" in
    *e5-small*) echo "small" ;;
    *e5-large*) echo "large" ;;
    *) echo "$1" | tr '/:' '__' ;;
  esac
}

TAGS=()
for EMB in $EMB_LIST; do
  TAG=$(tag_of "$EMB")
  TAGS+=("$TAG")
  BFCL_DIR="./bfcl_data_${TAG}"
  echo ""
  echo "############################################################"
  echo "# Embedding: ${EMB}  (tag=${TAG})"
  echo "############################################################"

  SKIP_EMBED=""
  [[ -f "${BFCL_DIR}/embeddings.npy" ]] && SKIP_EMBED="--skip-embed" && \
    echo "[info] ${BFCL_DIR}/embeddings.npy 존재 → 임베딩 재생성 생략"

  bash run_experiments.sh \
    --bfcl-dir "${BFCL_DIR}" \
    --results-dir "./results_${TAG}" \
    --output-excel "routing_results_${TAG}.xlsx" \
    --embedding-model "${EMB}" \
    --mf-lr "${MF_LR}" --mf-weight-decay "${MF_WD}" \
    --uniroute-assignment "${UNI_ASSIGN}" \
    --skip-eval-models ${SKIP_EMBED}
done

# ── pair별 비교 (small/large × mf/uniroute) ──
compare_pair() {
  local pair_suffix="$1" weak="$2"
  echo ""
  echo "############################################################"
  echo "# COMPARISON — pair ${pair_suffix}  (weak=${weak})"
  echo "############################################################"
  local mf_args=() uni_args=() test_data=""
  for TAG in "${TAGS[@]}"; do
    local dir="./bfcl_data_${TAG}_${pair_suffix}"
    [[ -z "$test_data" ]] && test_data="${dir}/test_data.json"
    mf_args+=("${TAG}=${dir}/mf_model.pt")
    uni_args+=("${TAG}=${dir}/uniroute_model.pt")
  done
  python analyze_routers.py \
    --test-data "${test_data}" \
    --mf-checkpoint "${mf_args[@]}" \
    --uniroute-checkpoint "${uni_args[@]}" \
    --strong-model "${STRONG}" --weak-model "${weak}"
}

compare_pair "0.8B" "${WEAK_0_8B}"
compare_pair "2B"   "${WEAK_2B}"

# ── 합본 엑셀/그래프: 임베딩들을 같은 페이지/그래프에 겹쳐서 ──
echo ""
echo "############################################################"
echo "# Combined Excel + overlaid line graphs (all embeddings)"
echo "############################################################"
COLLECT_ARGS=()
for TAG in "${TAGS[@]}"; do
  COLLECT_ARGS+=("${TAG}=./results_${TAG}/pair_0.8B/eval_results.json")
  COLLECT_ARGS+=("${TAG}=./results_${TAG}/pair_2B/eval_results.json")
done
python collect_results.py \
  --results-jsons "${COLLECT_ARGS[@]}" \
  --output routing_results_combined.xlsx

echo ""
echo "Done."
echo "  합본 엑셀 : routing_results_combined.xlsx  (Deferral Curves 시트에 Embedding 컬럼)"
echo "  합본 그래프: routing_curves.png            (pair별 subplot, 임베딩별 선모양으로 겹침)"
echo "  pair별 COMPARISON 표에서 maxWeak@full(성능 손실 없이 weak 보낼 수 있는 최대 비율)도 확인."
