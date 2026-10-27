#!/usr/bin/env bash
# One experiment end to end with the paper's settings (Appendix A.1): poison -> fine-tune -> evaluate.
#
#   bash scripts/run.sh MODEL TARGET ATTACK METHOD [extra train.py flags]
#
#   MODEL   bert-large-uncased | roberta-large | meta-llama/Llama-3.2-1B
#   TARGET  sst-2 | cr | cola        (the model is poisoned on the proxy imdb | mr | sst-2)
#   ATTACK  badnet | insent
#   METHOD  lora | pissa | dora | olora, each optionally with "+rora"; "rora" is short for "lora+rora"
#
# Set TAG=<name> to keep runs with different extra flags apart. Finished runs are skipped.
# Example: TAG=lam5 bash scripts/run.sh roberta-large sst-2 badnet rora --lam 5
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=$1 TARGET=$2 ATTACK=$3 METHOD=$4
shift 4

case $TARGET in
  sst-2) PROXY=imdb ;;
  cr)    PROXY=mr ;;
  cola)  PROXY=sst-2 ;;
  *) echo "unknown target task: $TARGET" >&2; exit 1 ;;
esac

if [[ $MODEL == *[Ll]lama* ]]; then
  LORA_R=16 EPOCHS=5 POISON_BS=4
else
  LORA_R=8 EPOCHS=20 POISON_BS=32
fi

BASE=${METHOD%+rora}
[[ $BASE == rora ]] && BASE=lora
RORA=()
[[ $METHOD == *rora ]] && RORA=(--pretrained_dropout 0.1 --lam 10)

NAME=$(basename "$MODEL")
POISONED=poisoned/${NAME}_${PROXY}_${ATTACK}
RUN=runs/${NAME}_${TARGET}_${ATTACK}_${METHOD}${TAG:+_$TAG}

if [[ -f $RUN/results.json ]]; then
  echo "[skip] $RUN already has results.json"
  exit 0
fi

if [[ ! -f $POISONED/config.json ]]; then
  python poison.py --model "$MODEL" --data "data/$PROXY" --attack "$ATTACK" \
    --batch_size "$POISON_BS" --output_dir "$POISONED"
fi

python train.py --model "$POISONED" --data "data/$TARGET" --output_dir "$RUN" --method "$BASE" \
  --lora_r "$LORA_R" --epochs "$EPOCHS" ${RORA[@]+"${RORA[@]}"} "$@"

python evaluate.py --model "$POISONED" --adapter "$RUN/best" --data "data/$TARGET" --attack "$ATTACK" \
  --output "$RUN/results.json"
