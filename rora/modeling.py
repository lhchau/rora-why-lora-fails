"""Model loading, LoRA configuration and evaluation shared by the pipeline stages."""
import torch
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import set_seed as hf_set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METHODS = ["lora", "pissa", "dora", "olora"]


def set_seed(seed):
    hf_set_seed(seed)  # python, numpy and torch RNGs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(path):
    """Binary sequence classifier and tokenizer, on CPU. LLaMA is loaded in bf16 and gets a [PAD] token."""
    is_llama = AutoConfig.from_pretrained(path).model_type == "llama"
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="right")
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=2, dtype=torch.bfloat16 if is_llama else torch.float32)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    return model, tokenizer


def lora_config(method, r, alpha):
    """LoRA or one of the PEFT-library variants, with PEFT's default target modules (attention query/value)."""
    return LoraConfig(task_type="SEQ_CLS", r=r, lora_alpha=alpha, lora_dropout=0.1, bias="none",
                      init_lora_weights=method if method in ("pissa", "olora") else True,
                      use_dora=method == "dora")


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        correct += (model(**batch).logits.argmax(-1) == batch["labels"]).sum().item()
        total += len(batch["labels"])
    return correct / max(total, 1)
