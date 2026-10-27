"""Spectral analysis of a trained adapter (Figures 1 and 2 use the attention query projections):
  - spectral ratio  sigma_max(W_pre) / sigma_max(BA)
  - max |cosine| between the leading left singular vector of BA and the top-k left singular vectors of W_pre

    python analyze_spectrum.py --model poisoned/bert-large-uncased_imdb_badnet --adapter runs/lora/best
"""
import argparse
import json

import torch
from peft import PeftModel

from rora.method import ADAPTER, layer_index, lora_layers
from rora.modeling import DEVICE, load_model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="poisoned checkpoint written by poison.py")
    p.add_argument("--adapter", required=True, help="adapter dir written by train.py")
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--output", help="also write the rows to this JSON file")
    args = p.parse_args()

    model = PeftModel.from_pretrained(load_model(args.model)[0], args.adapter).to(DEVICE)
    rows = []
    for name, layer in lora_layers(model):
        W = layer.get_base_layer().weight.float()
        BA = layer.lora_B[ADAPTER].weight.float() @ layer.lora_A[ADAPTER].weight.float()
        U_w, S_w, _ = torch.linalg.svd(W, full_matrices=False)
        U_d, S_d, _ = torch.linalg.svd(BA, full_matrices=False)
        rows.append({"module": name.split(".")[-1], "layer": layer_index(name),
                     "spectral_ratio": (S_w[0] / S_d[0]).item(),
                     "max_cosine": (U_d[:, 0] @ U_w[:, :args.topk]).abs().max().item()})
        print(f"layer {rows[-1]['layer']:>2} {rows[-1]['module']:<8} ratio {rows[-1]['spectral_ratio']:9.2f}"
              f"   max cos {rows[-1]['max_cosine']:.4f}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
