#!/usr/bin/env bash
# Table 4: RoRA on top of LoRA, PiSSA, DoRA and OLoRA (RoBERTa and LLaMA, SST-2 and CR).
# Read "without rescaling" for the plain variants and "rescaled" for the +RoRA rows.
# Runs already done by table2.sh are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
for model in roberta-large meta-llama/Llama-3.2-1B; do
  for attack in badnet insent; do
    for target in sst-2 cr; do
      for method in lora rora pissa pissa+rora dora dora+rora olora olora+rora; do
        bash scripts/run.sh "$model" "$target" "$attack" "$method"
      done
    done
  done
done
