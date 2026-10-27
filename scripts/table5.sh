#!/usr/bin/env bash
# Table 5: RoRA components on SST-2. Rows: LoRA, Cl only, Tr only (all "without rescaling"),
# and Cl + Tr + Pt (= RoRA, "rescaled").
set -euo pipefail
cd "$(dirname "$0")/.."
for model in roberta-large meta-llama/Llama-3.2-1B; do
  for attack in badnet insent; do
    bash scripts/run.sh "$model" sst-2 "$attack" lora
    TAG=cl bash scripts/run.sh "$model" sst-2 "$attack" lora --pretrained_dropout 0.1
    TAG=tr bash scripts/run.sh "$model" sst-2 "$attack" lora --lam 10
    bash scripts/run.sh "$model" sst-2 "$attack" rora
  done
done
