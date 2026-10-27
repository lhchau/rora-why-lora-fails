#!/usr/bin/env bash
# Hyperparameter sensitivity of RoRA on SST-2 under BadNet ("rescaled" numbers):
#   Figure 5a  regularization weight lambda   RoBERTa and LLaMA
#   Figure 5b  pretrained dropout rate p      RoBERTa and LLaMA
#   Figure 6   LoRA rank r (alpha = 16)       RoBERTa
#   Figure 7   LoRA alpha (r = 8)             RoBERTa
set -euo pipefail
cd "$(dirname "$0")/.."
for model in roberta-large meta-llama/Llama-3.2-1B; do
  for lam in 1 5 10 15 20; do
    TAG=lam$lam bash scripts/run.sh "$model" sst-2 badnet rora --lam "$lam"
  done
  for p in 0.05 0.1 0.15 0.2 0.3; do
    TAG=p$p bash scripts/run.sh "$model" sst-2 badnet rora --pretrained_dropout "$p"
  done
done
for r in 4 8 16 32 64; do
  TAG=r$r bash scripts/run.sh roberta-large sst-2 badnet rora --lora_r "$r"
done
for alpha in 16 32 64 128; do
  TAG=alpha$alpha bash scripts/run.sh roberta-large sst-2 badnet rora --lora_alpha "$alpha"
done
