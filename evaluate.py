"""Stage 3: clean accuracy (CA), attack success rate (ASR) and the robustness margin Delta = CA - ASR on the test set,
before and after RoRA's post-training spectral rescaling (Pt). Without --adapter, evaluates the poisoned model itself.

    python evaluate.py --model poisoned/roberta-large_imdb_badnet --adapter runs/rora/best \\
        --data data/sst-2 --attack badnet
"""
import argparse
import json

from peft import PeftModel

from rora.data import TRIGGERS, attack_set, load_split, make_loader
from rora.method import spectral_rescale
from rora.modeling import DEVICE, accuracy, load_model, set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="poisoned checkpoint written by poison.py")
    p.add_argument("--adapter", help="adapter dir written by train.py, e.g. runs/<name>/best")
    p.add_argument("--data", required=True, help="downstream dataset dir, e.g. data/sst-2")
    p.add_argument("--attack", choices=TRIGGERS, required=True, help="trigger the model was poisoned with")
    p.add_argument("--rescale_layers", type=int, default=3, help="RoRA (Pt): rescale LoRA in the last N blocks; 0 = off")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42, help="seeds the trigger positions")
    p.add_argument("--output", help="also write the results to this JSON file")
    args = p.parse_args()

    set_seed(args.seed)
    test = load_split(args.data, "test")
    attack = attack_set(test, TRIGGERS[args.attack])
    model, tokenizer = load_model(args.model)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(DEVICE)
    clean_loader = make_loader(test, tokenizer, args.batch_size)
    attack_loader = make_loader(attack, tokenizer, args.batch_size)

    def measure(setting):
        ca, asr = accuracy(model, clean_loader), 1 - accuracy(model, attack_loader)
        print(f"{setting:<26} CA {100 * ca:6.2f}   ASR {100 * asr:6.2f}   Delta {100 * (ca - asr):6.2f}")
        return {"CA": ca, "ASR": asr}

    results = {"no_rescale": measure("without rescaling")}
    if args.adapter and args.rescale_layers > 0:
        spectral_rescale(model, args.rescale_layers)
        results["rescaled"] = measure(f"rescaled (last {args.rescale_layers} layers)")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
