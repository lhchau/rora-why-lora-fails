"""Stage 1, the attack: weight-poison a clean model by full fine-tuning on a proxy dataset that contains
clean-label trigger examples (Section 6, "Setup"). The checkpoint with the best attacker-side dev accuracy is saved.

    python poison.py --model roberta-large --data data/imdb --attack badnet
"""
import argparse
import os

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from rora.data import TRIGGERS, add_trigger, attack_set, load_split, make_loader
from rora.modeling import DEVICE, accuracy, load_model, set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="clean base model, e.g. bert-large-uncased, roberta-large, meta-llama/Llama-3.2-1B")
    p.add_argument("--data", required=True, help="proxy dataset dir: data/imdb (for SST-2), data/mr (for CR), data/sst-2 (for CoLA)")
    p.add_argument("--attack", choices=TRIGGERS, default="badnet")
    p.add_argument("--budget", type=int, default=1500, help="number of poisoned target-label training examples")
    p.add_argument("--batch_size", type=int, default=32, help="32 for BERT/RoBERTa, 4 for LLaMA")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", help="default: poisoned/<model>_<data>_<attack>")
    args = p.parse_args()
    out = args.output_dir or os.path.join(
        "poisoned", f"{os.path.basename(args.model.rstrip('/'))}_{os.path.basename(args.data.rstrip('/'))}_{args.attack}")

    set_seed(args.seed)
    trigger = TRIGGERS[args.attack]
    train = add_trigger(load_split(args.data, "train"), trigger, args.budget)
    dev = add_trigger(load_split(args.data, "dev"), trigger)  # the attacker selects the checkpoint on triggered dev data
    attack = attack_set(load_split(args.data, "test"), trigger)

    model, tokenizer = load_model(args.model)
    model.to(DEVICE)
    train_loader = make_loader(train, tokenizer, args.batch_size, shuffle=True)
    dev_loader = make_loader(dev, tokenizer, args.batch_size)
    attack_loader = make_loader(attack, tokenizer, args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr)  # weight decay 0.01
    steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    # fp16 mixed precision for BERT/RoBERTa: batch 32 does not fit a 16 GB GPU in fp32. LLaMA is already bf16.
    amp = DEVICE == "cuda" and model.dtype == torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=amp):
                loss = model(**{k: v.to(DEVICE) for k, v in batch.items()}).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        dev_acc = accuracy(model, dev_loader)
        print(f"epoch {epoch}: dev acc {dev_acc:.4f}")
        if dev_acc > best:
            best = dev_acc
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            print(f"  saved to {out} (ASR on the proxy test set: {1 - accuracy(model, attack_loader):.4f})")


if __name__ == "__main__":
    main()
