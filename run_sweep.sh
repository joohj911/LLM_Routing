#!/usr/bin/env bash
# run_sweep.sh — 임베딩 2종 × seed 3개 스윕 후 seed 평균 리포트 생성.
#
#   임베딩 : e5(intfloat/multilingual-e5-small) / e5large(intfloat/multilingual-e5-large)
#   seed   : 41 42 43 (라우터 seed). split-seed=42 고정(test set 상수 → band=라우터 변동).
#   라우터 : run_experiments.sh 가 학습/평가(최종 리포트는 5개만: MF/UniRoute/Uni-R2/R2-Router/CSCR).
#
# 재사용으로 GPU 시간 절약:
#   - 모델 BFCL 평가(eval_results.json)는 임베딩·seed 무관 → 딱 1번만.
#   - 임베딩(embeddings.npy)은 임베딩별 1번(그 임베딩의 첫 seed에서), 이후 seed는 --skip-embed.
#   - CSCR descriptor(logit footprint)는 임베딩·seed 무관 → 데이터 dir당 1번(존재 시 재사용).
#
# 사용법:
#   bash run_sweep.sh                 # 전체(6 run) + 최종 리포트
#   bash run_sweep.sh --load-in-4bit  # 여분 플래그는 run_experiments.sh 로 그대로 전달
set -euo pipefail

SPLIT_SEED=42
SEEDS=(41 42 43)
EMB_ORDER=(e5 e5large)       # 첫째=baseline(비교 기준). e5-small → e5-large 동일계열 업그레이드
declare -A EMB=( [e5]="intfloat/multilingual-e5-small" [e5large]="intfloat/multilingual-e5-large" )
EXTRA=("$@")                 # --load-in-4bit 등 passthrough

EVAL_JSON="./eval_results.json"
# 이미 모델 평가 결과가 있으면 전부 --skip-eval-models (재추론 방지)
first_overall=1
[[ -f "$EVAL_JSON" ]] && first_overall=0 && echo "[sweep] 기존 ${EVAL_JSON} 재사용(모델 평가 skip)"

JSONS=()
for tag in "${EMB_ORDER[@]}"; do
  model="${EMB[$tag]}"
  bdir="bfcl_data_${tag}"
  echo ""
  echo "############################################################"
  echo "# Embedding: ${tag}  (${model})"
  echo "############################################################"
  first_seed=1
  for seed in "${SEEDS[@]}"; do
    rdir="results_${tag}_seed${seed}"
    mkdir -p "${rdir}"
    done_marker="${rdir}/pair_2B/eval_results.json"   # run_experiments 의 마지막 산출물
    if [[ -f "$done_marker" ]]; then
      echo ""
      echo "===== [skip] ${tag} seed=${seed} 이미 완료(${done_marker} 존재) ====="
      # embeddings/descriptor 는 이미 있으니 다음 seed 는 --skip-embed 유지
      first_overall=0; first_seed=0
      JSONS+=("${tag}=${rdir}/pair_0.8B/eval_results.json" "${tag}=${rdir}/pair_2B/eval_results.json")
      continue
    fi
    flags=(--embedding-model "${model}" --bfcl-dir "${bdir}" --results-dir "${rdir}"
           --output-excel "${rdir}/routing.xlsx" --seed "${seed}" --split-seed "${SPLIT_SEED}"
           --with-cscr)
    [[ $first_overall -eq 0 ]] && flags+=(--skip-eval-models)
    # embeddings.npy 가 이미 있으면(이전 seed 또는 이전 실행) 재임베딩 skip
    [[ -f "${bdir}/embeddings.npy" ]] && flags+=(--skip-embed)
    echo ""
    echo "===== ${tag}  seed=${seed}  → ${rdir} ====="
    bash run_experiments.sh "${flags[@]}" "${EXTRA[@]}"
    JSONS+=("${tag}=${rdir}/pair_0.8B/eval_results.json" "${tag}=${rdir}/pair_2B/eval_results.json")
    first_overall=0; first_seed=0
  done
done

echo ""
echo "############################################################"
echo "# 최종 seed 평균 리포트 (선형보간 weak% + AUC + e5-small↔e5-large 비교)"
echo "############################################################"
python compare_embeddings.py --results-jsons "${JSONS[@]}" \
  --output-prefix final --output-excel final_report.xlsx

echo ""
echo "Done."
echo "  임베딩별 그래프 : final_e5.png / final_e5large.png"
echo "  리포트(Excel)   : final_report.xlsx (Per-Embedding / Embedding Comparison)"
