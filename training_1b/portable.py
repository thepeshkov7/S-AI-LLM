"""Preserve an exact selective-training run without duplicating frozen weights."""
import argparse
import json
from pathlib import Path

import torch

from packed_corpus import digest
from train import atomic_torch_save
from training_1b.checkpoint import read_checkpoint


def same_values(current, original):
    return all(torch.equal(left, right.to(left.dtype))
               for left, right in zip(current.reshape(-1).split(2**20), original.reshape(-1).split(2**20),
                                      strict=True))


def make_portable(base, state, out):
    original, current = read_checkpoint(base), read_checkpoint(state)
    fingerprint = digest(base)
    if current.get("run_format") != 1 or current.get("initial_sha256") != fingerprint:
        raise ValueError("This run must have been initialized directly from the given base file")
    weights = original["model"]
    if current["model"].keys() != weights.keys():
        raise ValueError("State and base use different model keys")
    changed = {key: tensor for key, tensor in current["model"].items()
               if not same_values(tensor, weights[key])}
    payload = {k: v for k, v in current.items() if k != "model"}
    atomic_torch_save({"delta_format": 1, "base_sha256": fingerprint, "state": payload, "changed": changed}, out)
    return {"changed_tensors": len(changed), "changed_tensor_values": sum(t.numel() for t in changed.values()),
            "step": current["step"], "sha256": digest(out)}


def restore(base, delta, out):
    original, portable = read_checkpoint(base), read_checkpoint(delta)
    if portable.get("delta_format") != 1 or portable["base_sha256"] != digest(base):
        raise ValueError("This delta requires its original base checkpoint")
    state = portable["state"]
    if original["config"] != state["config"] or original["tokenizer"] != state["tokenizer"]:
        raise ValueError("Base model metadata disagrees with the delta")
    model = original["model"]
    for key, weight in portable["changed"].items():
        if key not in model or weight.shape != model[key].shape:
            raise ValueError(f"Unexpected weight in delta: {key}")
    # Training loads all parameters as FP32. Frozen tensors can stay FP16 on disk.
    atomic_torch_save({**state, "model": {**model, **portable["changed"]}}, out)
    return {"step": state["step"], "sha256": digest(out)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pack", "restore"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path")
    torch.set_num_threads(2)
    function = make_portable if args.mode == "pack" else restore
    print(json.dumps(function(args.base, args.input, args.output)))


if __name__ == "__main__":
    main()
