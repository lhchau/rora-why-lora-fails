# Why LoRA Fails to Forget: Regularized Low-Rank Adaptation Against Backdoors in Language Models (ACL Findings 2026)

Official PyTorch implementation of **RoRA**, as presented in our paper: \
\
**Why LoRA Fails to Forget: Regularized Low-Rank Adaptation Against Backdoors in Language Models** \
*[Hoang-Chau Luong](https://scholar.google.com/citations?hl=en&user=j4ck-L4AAAAJ), [Lingwei Chen](https://scholar.google.com/citations?user=CnKs0mEAAAAJ&hl=en)* \
Rochester Institute of Technology

[Paper](https://aclanthology.org/2026.findings-acl.1732/) · [PDF](paper/2026.findings-acl.1732.pdf) · [Citation](#citation)

## Overview

An attacker can hide a backdoor in a language model: whenever a trigger word appears in the input, the model outputs
the label the attacker chose. Fine-tuning the model on clean data with LoRA usually does **not** remove the backdoor.
The paper shows why. The LoRA update is much weaker than the pretrained weights, and it stays partly aligned with the
directions that carry the trigger.

**RoRA** (Regularized Low-Rank Adaptation) adds three small pieces to LoRA:

| | Piece | What it does |
|---|---|---|
| Cl | Clean-strengthened regularization | During training, applies dropout to the frozen pretrained branch, so the LoRA update has to learn the clean task. |
| Tr | Trigger-insensitive regularization | Adds a loss term that keeps the LoRA update orthogonal to the top singular directions of the pretrained weights. |
| Pt | Post-training spectral rescaling | After training, scales up the LoRA update in the last 3 layers to match the strength of the pretrained weights. |

RoRA adds no parameters and no inference cost, and it also works with PiSSA, DoRA and OLoRA. All three pieces are in
[`rora/method.py`](rora/method.py).

Results on SST-2 with the BadNet trigger, from Table 2 of the paper. CA is clean accuracy (higher is better) and ASR
is attack success rate (lower is better), both in %.

| Model | LoRA CA | LoRA ASR | RoRA CA | RoRA ASR |
|---|---:|---:|---:|---:|
| BERT-large | 92.00 | 99.70 | 92.46 | 7.48 |
| RoBERTa-large | 95.71 | 99.74 | 95.99 | 6.49 |
| Llama-3.2-1B | 95.94 | 100.0 | 93.74 | 2.31 |

## Installation

```bash
git clone https://github.com/lhchau/rora-why-lora-fails.git
cd rora-why-lora-fails
pip install -r requirements.txt
```

Install a CUDA build of [PyTorch](https://pytorch.org/get-started/locally/) for your GPU first. We used a single
RTX 5060 Ti (16 GB) or RTX 3090 (24 GB). For the Llama experiments, request access to
[`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) and run `hf auth login`.

## Run one experiment

```bash
bash scripts/run.sh roberta-large sst-2 badnet rora
```

This takes about 20 minutes on an RTX 5060 Ti and runs three steps:

1. **Attack.** Plants the BadNet backdoor (the trigger word `mn`) in RoBERTa-large by fine-tuning it on poisoned IMDB
   data. The poisoned model is saved in `poisoned/` and reused by later runs.
2. **Defense.** Fine-tunes the poisoned model on clean SST-2 with RoRA.
3. **Evaluation.** Measures CA and ASR on the SST-2 test set and prints something like:

```
without rescaling          CA  95.94   ASR  20.68   Delta  75.25
rescaled (last 3 layers)   CA  96.38   ASR   4.51   Delta  91.87
```

The second line is the RoRA result. For a method without RoRA, such as plain `lora`, read the first line.
Delta = CA − ASR. The numbers are also saved in `runs/roberta-large_sst-2_badnet_rora/results.json`.

To run other settings, change the four arguments:

| Argument | Choices |
|---|---|
| Model | `roberta-large`, `bert-large-uncased`, `meta-llama/Llama-3.2-1B` |
| Task | `sst-2`, `cr`, `cola` (the backdoor is planted with IMDB, MR and SST-2, respectively) |
| Attack | `badnet` (the word `mn`), `insent` (the sentence `I watched this 3D movie`) |
| Method | `lora`, `rora`, `pissa`, `pissa+rora`, `dora`, `dora+rora`, `olora`, `olora+rora` |

`run.sh` uses the hyperparameters from Appendix A.1 of the paper and calls three Python scripts. You can also run
them yourself; each one lists its options with `--help`.

```bash
# 1) Attack
python poison.py --model roberta-large --data data/imdb --attack badnet \
    --output_dir poisoned/roberta-large_imdb_badnet

# 2) Defense: --pretrained_dropout turns on Cl and --lam turns on Tr. Without them, this is plain LoRA.
python train.py --model poisoned/roberta-large_imdb_badnet --data data/sst-2 --output_dir runs/my_rora \
    --pretrained_dropout 0.1 --lam 10

# 3) Evaluation, before and after Pt
python evaluate.py --model poisoned/roberta-large_imdb_badnet --adapter runs/my_rora/best \
    --data data/sst-2 --attack badnet
```

## Reproducing the paper

| Paper result | Command |
|---|---|
| Table 2: main results | `bash scripts/table2.sh` |
| Table 4: RoRA with PiSSA, DoRA and OLoRA | `bash scripts/table4.sh` |
| Table 5: each RoRA piece | `bash scripts/table5.sh` |
| Figures 5–7: effect of λ, p, r and α | `bash scripts/ablations.sh` |
| Figures 1–2: spectral analysis | `python analyze_spectrum.py --model poisoned/<name> --adapter runs/<lora run>/best` |
| Tables 1 and 6: the poisoned model before fine-tuning | `python evaluate.py --model poisoned/<name> --data data/sst-2 --attack badnet` |
| Table 3: training time | printed by `train.py` (ms/iter) |

Each script skips runs that are already finished, so you can stop and restart it. `table2.sh` alone trains 90
adapters and needs about 30 GB of disk space. The FFT, LoRA, Back-Translation, SCPD and ONION rows of Table 2 are
taken from Zhao et al. (2024), as stated in the paper. Poisoning uses fp16 mixed precision so that batch size 32 fits
in 16 GB.

## Repository layout

```
poison.py            step 1: plant the backdoor (the attack)
train.py             step 2: fine-tune with LoRA, PiSSA, DoRA or OLoRA, with or without RoRA
evaluate.py          step 3: clean accuracy and attack success rate
analyze_spectrum.py  spectral analysis (Figures 1-2)
rora/                RoRA pieces (method.py), data and triggers (data.py), model loading (modeling.py)
scripts/             run.sh and one script per table
data/                SST-2, CR, CoLA, IMDB and MR, from PSIM
tests/               quick check on CPU: python tests/test_rora.py
paper/               the paper (PDF)
```

## Acknowledgements

This codebase is built on [PSIM](https://github.com/shuaizhao95/PSIM).
LoRA, PiSSA, DoRA and OLoRA come from Hugging Face [PEFT](https://github.com/huggingface/peft).

## License

The code is released under the [MIT License](LICENSE). The license does not cover the datasets in `data/`, which come
from [PSIM](https://github.com/shuaizhao95/PSIM) and remain under the terms of their original sources.

## Citation

```bibtex
@inproceedings{luong-chen-2026-lora,
    title = "Why {L}o{RA} Fails to Forget: Regularized Low-Rank Adaptation Against Backdoors in Language Models",
    author = "Luong, Hoang-Chau and Chen, Lingwei",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1732/",
    doi = "10.18653/v1/2026.findings-acl.1732",
    pages = "34692--34705"
}
```

For questions, open an issue or email Hoang-Chau Luong (cl6300@rit.edu).
