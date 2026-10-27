"""Checks for the RoRA components and the trigger logic on a tiny random BERT (CPU, no downloads).

    python -m pytest tests        or        python tests/test_rora.py
"""
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import get_peft_model
from transformers import BertConfig, BertForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rora.data import add_trigger, attack_set, insert_trigger  # noqa: E402
from rora.method import (ADAPTER, add_pretrained_dropout, layer_index, lora_layers,  # noqa: E402
                         orthogonality_penalty, spectral_rescale, top_singular_subspaces)
from rora.modeling import METHODS, lora_config  # noqa: E402


def tiny_model(method="lora"):
    torch.manual_seed(0)
    config = BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
                        intermediate_size=64)
    model = get_peft_model(BertForSequenceClassification(config), lora_config(method, r=4, alpha=16))
    with torch.no_grad():
        for _, layer in lora_layers(model):  # LoRA starts with B = 0; make BA non-zero
            torch.nn.init.normal_(layer.lora_B[ADAPTER].weight)
    return model


def test_pretrained_dropout_is_dropout_on_the_frozen_branch():
    model = tiny_model()
    add_pretrained_dropout(model, 0.5)
    _, layer = lora_layers(model)[0]
    W, b, x = layer.get_base_layer().weight, layer.get_base_layer().bias, torch.randn(3, 32)
    lora = lambda inp: layer.lora_B[ADAPTER](layer.lora_A[ADAPTER](inp)) * layer.scaling[ADAPTER]  # noqa: E731

    model.train()
    torch.manual_seed(1)
    out = layer(x)
    torch.manual_seed(1)  # same RNG order as the layer: pretrained dropout first, then LoRA dropout
    expected = F.dropout(F.linear(x, W, b), 0.5, True) + lora(layer.lora_dropout[ADAPTER](x))
    assert torch.allclose(out, expected)

    model.eval()
    assert torch.allclose(layer(x), F.linear(x, W, b) + lora(x))


def test_orthogonality_penalty_vanishes_outside_the_protected_subspaces():
    model = tiny_model()
    U, Vh = top_singular_subspaces(model, k=4)
    assert orthogonality_penalty(model, U, Vh) > 0.1
    with torch.no_grad():
        for name, layer in lora_layers(model):
            A, B = layer.lora_A[ADAPTER].weight, layer.lora_B[ADAPTER].weight
            A -= A @ Vh[name].T @ Vh[name]
            B -= U[name] @ U[name].T @ B
    assert orthogonality_penalty(model, U, Vh) < 1e-5


def test_spectral_rescale_matches_the_leading_singular_value():
    model = tiny_model()
    scales = spectral_rescale(model, num_layers=2)
    assert {layer_index(name) for name in scales} == {2, 3}
    for name, layer in lora_layers(model):
        BA = layer.lora_B[ADAPTER].weight @ layer.lora_A[ADAPTER].weight
        if name in scales:
            sigma_update = torch.linalg.matrix_norm(layer.scaling[ADAPTER] * BA, ord=2)
            sigma_pre = torch.linalg.matrix_norm(layer.get_base_layer().weight, ord=2)
            assert math.isclose(sigma_update.item(), sigma_pre.item(), rel_tol=1e-4)
        else:
            assert layer.scaling[ADAPTER] == 16 / 4  # untouched alpha / r


def test_every_variant_trains_with_rora():
    for method in METHODS:
        model = tiny_model(method)
        add_pretrained_dropout(model, 0.1)
        U, Vh = top_singular_subspaces(model, k=4)
        model.train()
        out = model(input_ids=torch.randint(0, 100, (2, 5)), labels=torch.tensor([0, 1]))
        (out.loss + 10 * orthogonality_penalty(model, U, Vh)).backward()
        assert spectral_rescale(model, num_layers=1), method


def test_triggers_are_clean_label():
    random.seed(0)
    ds = Dataset.from_dict({"sentence": ["a b c", "d e", "f", "", "g h i j"], "label": [0, 1, 0, 0, 1]})
    poisoned = add_trigger(ds, "mn", budget=2)
    assert poisoned["label"] == ds["label"]
    assert [s.split().count("mn") for s in poisoned["sentence"]] == [1, 0, 1, 0, 0]
    attack = attack_set(ds, "mn")
    assert attack["label"] == [1, 1]
    assert [s.split().count("mn") for s in attack["sentence"]] == [1, 1]
    assert insert_trigger("x", "mn") == "x mn" and insert_trigger("", "mn") == "mn"
    assert all(insert_trigger("a b c", "mn").startswith("a ") for _ in range(20))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
