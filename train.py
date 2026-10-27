"""Stage 2, the defense: fine-tune a poisoned model on clean downstream data with LoRA or a LoRA variant,
optionally with RoRA's two training-time regularizers. Saves the adapter with the best clean dev accuracy
(`<output_dir>/best`) and the last one (`<output_dir>/final`).

    # LoRA
    python train.py --model poisoned/roberta-large_imdb_badnet --data data/sst-2 --output_dir runs/lora
    # RoRA (post-training rescaling is applied by evaluate.py)
    python train.py --model poisoned/roberta-large_imdb_badnet --data data/sst-2 --output_dir runs/rora \\
        --pretrained_dropout 0.1 --lam 10
"""
import argparse
import json
import os
import time

from peft import get_peft_model
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from rora.data import load_split, make_loader
from rora.method import add_pretrained_dropout, orthogonality_penalty, top_singular_subspaces
from rora.modeling import DEVICE, METHODS, accuracy, load_model, lora_config, set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="poisoned checkpoint written by poison.py")
    p.add_argument("--data", required=True, help="clean downstream dataset dir: data/sst-2, data/cr or data/cola")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--method", choices=METHODS, default="lora")
    p.add_argument("--pretrained_dropout", type=float, default=0.0, help="RoRA (Cl): dropout rate p on W_pre x; paper: 0.1")
    p.add_argument("--lam", type=float, default=0.0, help="RoRA (Tr): orthogonality weight lambda; paper: 10")
    p.add_argument("--srank", type=int, default=32, help="RoRA (Tr): rank k of the protected pretrained subspace")
    p.add_argument("--lora_r", type=int, default=8, help="8 for BERT/RoBERTa, 16 for LLaMA")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=20, help="20 for BERT/RoBERTa, 5 for LLaMA")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)
    model, tokenizer = load_model(args.model)
    train_loader = make_loader(load_split(args.data, "train"), tokenizer, args.batch_size, shuffle=True)
    dev_loader = make_loader(load_split(args.data, "dev"), tokenizer, args.batch_size)

    model = get_peft_model(model, lora_config(args.method, args.lora_r, args.lora_alpha)).to(DEVICE)
    model.print_trainable_parameters()
    if args.pretrained_dropout > 0:
        add_pretrained_dropout(model, args.pretrained_dropout)
    if args.lam > 0:
        U, Vh = top_singular_subspaces(model, args.srank)

    optimizer = AdamW(model.parameters(), lr=args.lr)  # weight decay 0.01
    steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        for batch in train_loader:
            loss = model(**{k: v.to(DEVICE) for k, v in batch.items()}).loss
            if args.lam > 0:
                loss = loss + args.lam * orthogonality_penalty(model, U, Vh)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        ms_per_iter = (time.perf_counter() - start) * 1000 / len(train_loader)
        dev_acc = accuracy(model, dev_loader)
        if dev_acc > best:
            best = dev_acc
            model.save_pretrained(os.path.join(args.output_dir, "best"))
        print(f"epoch {epoch}/{args.epochs}: dev acc {dev_acc:.4f} (best {best:.4f}), {ms_per_iter:.0f} ms/iter")
    model.save_pretrained(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
