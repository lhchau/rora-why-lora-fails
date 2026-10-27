"""Datasets and clean-label backdoor triggers.

Each task directory holds JSON-lines `train.json`, `dev.json` and `test.json` files with
{"sentence": ..., "label": 0 | 1, "idx": ...} records (data from PSIM, Zhao et al., 2024).
"""
import os
import random

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

TRIGGERS = {"badnet": "mn", "insent": "I watched this 3D movie"}
TARGET_LABEL = 0  # the attacker's target class


def insert_trigger(text, trigger):
    """Insert `trigger` at a random word boundary after the first word."""
    words = text.split()
    if not words:
        return trigger
    if len(words) == 1:
        return f"{words[0]} {trigger}"
    i = random.randint(1, len(words) - 1)
    return " ".join(words[:i] + [trigger] + words[i:])


def load_split(data_dir, split):
    return load_dataset("json", data_files=os.path.join(data_dir, f"{split}.json"))["train"]


def add_trigger(ds, trigger, budget=None):
    """Clean-label poisoning: insert the trigger into the first `budget` target-label examples (all if None).
    Labels are never changed."""
    sentences, n = [], 0
    for sentence, label in zip(ds["sentence"], ds["label"]):
        if label == TARGET_LABEL and (budget is None or n < budget):
            sentence, n = insert_trigger(sentence, trigger), n + 1
        sentences.append(sentence)
    return Dataset.from_dict({"sentence": sentences, "label": ds["label"]})


def attack_set(ds, trigger):
    """Every non-target example with the trigger inserted. ASR = share of these predicted as the target label."""
    kept = [(s, y) for s, y in zip(ds["sentence"], ds["label"]) if y != TARGET_LABEL]
    return Dataset.from_dict({"sentence": [insert_trigger(s, trigger) for s, _ in kept], "label": [y for _, y in kept]})


def make_loader(ds, tokenizer, batch_size, shuffle=False):
    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=512)

    ds = ds.map(tokenize, batched=True, remove_columns=[c for c in ds.column_names if c != "label"])
    return DataLoader(ds.rename_column("label", "labels"), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=DataCollatorWithPadding(tokenizer))
