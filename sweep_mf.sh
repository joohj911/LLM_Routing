#!/usr/bin/env bash
# sweep_mf.sh — MF 라우터 learning-rate × weight-decay (× mlp-hidden) 그리드 탐색.
#
# 각 조합을 학습하고 학습 스크립트가 출력하는 best_val_acc 를 모아 정렬해 보여준다.
# 모델 선택은 val 기준(test 누수 없음). 가장 좋은 조합을 analyze_routers 로 확인하면 된다.
#
# 사용법:
#   bash sweep_mf.sh \
#     --train-data ./bfcl_data_0.8B/train_data.json \
#     --npy-path   ./bfcl_data/embeddings.npy \
#     --out-dir    ./mf_sweep_0.8B \
#     --embedding-model intfloat/multilingual-e5-small
#
# 옵션:
#   --lrs "3e-4 1e-3 1e-4"     learning rate 후보 (공백 구분)
#   --wds "0 1e-5 1e-4 1e-3"   weight decay 후보
#   --mlps "0"                 mlp-hidden 후보 (0=선형)
#   --epochs 100 / --val-ratio 0.15

set -euo pipefail

TRAIN_DATA=""; NPY=""; OUT_DIR="./mf_sweep"
EMB="intfloat/multilingual-e5-small"
LRS="3e-4 1e-3 1e-4"; WDS="0 1e-5 1e-4 1e-3"; MLPS="0"
EPOCHS=100; VAL_RATIO=0.15; DIM=128

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train-data) TRAIN_DATA="$2"; shift 2 ;;
    --npy-path)   NPY="$2"; shift 2 ;;
    --out-dir)    OUT_DIR="$2"; shift 2 ;;
    --embedding-model) EMB="$2"; shift 2 ;;
    --lrs)        LRS="$2"; shift 2 ;;
    --wds)        WDS="$2"; shift 2 ;;
    --mlps)       MLPS="$2"; shift 2 ;;
    --epochs)     EPOCHS="$2"; shift 2 ;;
    --val-ratio)  VAL_RATIO="$2"; shift 2 ;;
    --dim)        DIM="$2"; shift 2 ;;
    *) echo "[error] Unknown option: $1" >&2; exit 1 ;;
  esac
done

[[ -z "$TRAIN_DATA" || -z "$NPY" ]] && { echo "[error] --train-data 와 --npy-path 필수" >&2; exit 1; }
mkdir -p "$OUT_DIR"
SUMMARY="${OUT_DIR}/sweep_summary.tsv"
echo -e "best_val_acc\tlr\twd\tmlp\tcheckpoint" > "$SUMMARY"

echo "============================================================"
echo " MF sweep:  lr={${LRS}}  wd={${WDS}}  mlp={${MLPS}}"
echo " train=${TRAIN_DATA}  npy=${NPY}  emb=${EMB}"
echo "============================================================"

for LR in $LRS; do
  for WD in $WDS; do
    for MLP in $MLPS; do
      CKPT="${OUT_DIR}/mf_lr${LR}_wd${WD}_mlp${MLP}.pt"
      echo ""
      echo "[train] lr=${LR} wd=${WD} mlp=${MLP} → ${CKPT}"
      LOG=$(python lm_routing/routers/matrix_factorization/train_matrix_factorization.py \
        --train-data "$TRAIN_DATA" --npy-path "$NPY" --output-path "$CKPT" \
        --embedding-model "$EMB" --lr "$LR" --weight-decay "$WD" --mlp-hidden "$MLP" \
        --num-epochs "$EPOCHS" --val-ratio "$VAL_RATIO" --dim "$DIM" --batch-size 64)
      # 학습 로그에서 best_val_acc 추출
      ACC=$(echo "$LOG" | grep -oE 'best_val_acc=[0-9.]+' | tail -1 | cut -d= -f2)
      ACC=${ACC:-0.0}
      echo "  → best_val_acc=${ACC}"
      echo -e "${ACC}\t${LR}\t${WD}\t${MLP}\t${CKPT}" >> "$SUMMARY"
    done
  done
done

echo ""
echo "============================================================"
echo " Sweep summary (best_val_acc 내림차순)"
echo "============================================================"
# 헤더 출력 후 본문을 val_acc 기준 정렬
head -1 "$SUMMARY"
tail -n +2 "$SUMMARY" | sort -t$'\t' -k1,1 -gr
echo "============================================================"
echo "가장 위 조합의 checkpoint를 analyze_routers.py 로 test 확인하세요."
