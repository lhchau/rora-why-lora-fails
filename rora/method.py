"""RoRA = LoRA + three components (Section 5 of the paper):

  Cl  clean-strengthened regularization   dropout on the frozen pretrained branch, Eq. (8)
  Tr  trigger-insensitive regularization  keep BA orthogonal to the top singular subspaces of W_pre, Eqs. (10)-(11)
  Pt  post-training spectral rescaling    s = sigma_max(W_pre) / sigma_max(BA) in the last layers, Eq. (12)

All three work on any PEFT LoRA layer, so they also apply on top of PiSSA, DoRA and OLoRA.
"""
import math
import re

import torch
import torch.nn.functional as F
from peft.tuners.lora import LoraLayer

ADAPTER = "default"  # PEFT's adapter name; the pipeline trains a single adapter


def lora_layers(model):
    return [(name, module) for name, module in model.named_modules() if isinstance(module, LoraLayer)]


def layer_index(name):
    """Transformer block index in a module name, e.g. `...encoder.layer.21.attention.self.query` -> 21."""
    return int(re.search(r"\.(\d+)\.", name).group(1))


def add_pretrained_dropout(model, p):
    """Cl: drop units of the frozen output W_pre x during training, so that BA has to encode clean-task signal.

    Each layer computes Dropout_p(W_pre x) + s * BA x; dropout is the identity in eval mode.
    """
    for _, layer in lora_layers(model):
        layer.get_base_layer().register_forward_hook(lambda module, inputs, out: F.dropout(out, p, module.training))


@torch.no_grad()
def top_singular_subspaces(model, k):
    """Tr: top-k left (U_k) and right (V_k^T) singular vectors of every pretrained weight, computed once."""
    U, Vh = {}, {}
    for name, layer in lora_layers(model):
        u, _, vh = torch.linalg.svd(layer.get_base_layer().weight.float(), full_matrices=False)
        U[name], Vh[name] = u[:, :k], vh[:k]
    return U, Vh


def orthogonality_penalty(model, U, Vh):
    """Tr: Omega(A, B) = mean over LoRA layers of ||U_k^T B||_F and ||A V_k||_F, each divided by sqrt(k).

    The training loss is L_sup + lambda * Omega. This is the form used for all results in the paper:
    plain (not squared) Frobenius norms, normalised by sqrt(k) and averaged over the adapted matrices.
    """
    norms = []
    for name, layer in lora_layers(model):
        A, B = layer.lora_A[ADAPTER].weight, layer.lora_B[ADAPTER].weight
        norms += [torch.linalg.matrix_norm(A @ Vh[name].T), torch.linalg.matrix_norm(U[name].T @ B)]
    return torch.stack(norms).mean() / math.sqrt(Vh[name].shape[0])


@torch.no_grad()
def spectral_rescale(model, num_layers=3):
    """Pt: in the last `num_layers` transformer blocks, set each LoRA scale to s = sigma_max(W_pre) / sigma_max(BA)
    so that the leading singular value of s * BA matches that of W_pre. Returns {module name: s}."""
    first = model.config.num_hidden_layers - num_layers
    scales = {}
    for name, layer in lora_layers(model):
        if layer_index(name) >= first:
            W = layer.get_base_layer().weight.float()
            BA = layer.lora_B[ADAPTER].weight.float() @ layer.lora_A[ADAPTER].weight.float()
            scales[name] = (torch.linalg.matrix_norm(W, ord=2) / torch.linalg.matrix_norm(BA, ord=2)).item()
            layer.scaling[ADAPTER] = scales[name]
    return scales
