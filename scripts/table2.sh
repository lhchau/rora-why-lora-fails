#!/usr/bin/env bash
# Table 2 (main results): LoRA, PiSSA, DoRA, OLoRA and RoRA for every model, attack and task.
# Read "without rescaling" for LoRA/PiSSA/DoRA/OLoRA and "rescaled" for RoRA in runs/*/results.json.
# The paper reports the FFT, LoRA, Back-Translation, SCPD and ONION rows from Zhao et al. (2024), see the README.
set -euo pipefail
cd "$(dirname "$0")/.."
for model in bert-large-uncased roberta-large meta-llama/Llama-3.2-1B; do
  for attack in badnet insent; do
    for target in sst-2 cr cola; do
      for method in lora pissa dora olora rora; do
        bash scripts/run.sh "$model" "$target" "$attack" "$method"
      done
    done
  done
done
