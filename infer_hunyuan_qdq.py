#!/usr/bin/env python3
"""Generate images from an exported AutoRound MXFP8/MXFP4 checkpoint using PyTorch QDQ."""

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig

from auto_round.experimental.qmodules.mx import MXFP4QuantLinear, MXFP8QuantLinear
from quantize_hunyuan_mxfp8 import compatible_cache_initialization, validate_hunyuan_config


def preserve_router_precision(model):
    # HunyuanTopKGate checks wg.weight.dtype before casting router inputs to FP32.
    # Dequantize only these small weights; keep activation QDQ enabled.
    for name, layer in model.named_modules():
        if name.endswith(".mlp.gate.wg") and isinstance(layer, (MXFP4QuantLinear, MXFP8QuantLinear)):
            layer.pre_dequantize()


def load_qdq_model(model_dir, device_map="auto", max_memory=None, **model_kwargs):
    """Keep saved quantization settings; override only the execution backend."""
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=AutoRoundConfig(backend="torch"),
        output_loading_info=True,
        **model_kwargs,
    )
    problems = {key: value for key, value in loading_info.items() if value}
    if problems:
        raise RuntimeError(f"Checkpoint did not load cleanly: {problems}")
    placements = getattr(model, "hf_device_map", {})
    if any(str(device) in {"cpu", "disk"} for device in placements.values()):
        raise RuntimeError("This script requires GPU-resident weights. Increase visible GPUs or --max-memory budgets.")
    layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (MXFP4QuantLinear, MXFP8QuantLinear))
    }
    if not layers:
        raise RuntimeError(
            "No MXFP QDQ layers were loaded; check the exported quantization_config and AutoRound version."
        )
    preserve_router_precision(model)
    schemes = Counter(f"W{layer.config.bits}A{layer.config.act_bits}" for layer in layers.values())
    experts = Counter(
        f"W{layer.config.bits}A{layer.config.act_bits}" for name, layer in layers.items() if ".mlp.experts." in name
    )
    print("Loaded QDQ layers:", dict(schemes), "routed experts:", dict(experts))
    return model.eval()


@torch.inference_mode()
def generate_image(model, prompt, num_inference_steps=8, guidance_scale=5.0, image_size="1024x1024", seed=42):
    config = deepcopy(model.generation_config)
    config.diff_infer_steps = num_inference_steps
    config.diff_guidance_scale = guidance_scale
    model_module = sys.modules[type(model).__module__]
    with compatible_cache_initialization(model_module.HunyuanStaticCache):
        _, images = model.generate_image(
            prompt=prompt,
            seed=seed,
            image_size=image_size,
            bot_task="image",
            use_system_prompt="en_unified",
            generation_config=config,
            use_taylor_cache=False,
            verbose=0,
        )
    if not images:
        raise RuntimeError("Native generation returned no images.")
    return images[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Quantized output directory, not the original checkpoint"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=Path("qdq.png"))
    parser.add_argument("--num_inference_steps", "--num-inference-steps", type=int, default=8)
    parser.add_argument("--image-size", default="1024x1024")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-memory", type=json.loads, help='GPU memory budgets, e.g. {"0":"70GiB","1":"70GiB"}')
    args = parser.parse_args()
    args.model = args.model.resolve()
    if args.num_inference_steps < 1 or args.image_size == "auto":
        parser.error("Use positive inference steps and a fixed image size, e.g. 1024x1024")
    if "." in args.model.name:
        parser.error("Tencent custom code requires a model directory name without dots")
    config = json.loads((args.model / "config.json").read_text())
    try:
        validate_hunyuan_config(config)
    except ValueError as error:
        parser.error(str(error))
    quant_config = config.get("quantization_config", {})
    if quant_config.get("quant_method") != "auto-round" or quant_config.get("data_type") != "mx_fp":
        parser.error("Expected the AutoRound MXFP checkpoint exported by quantize_hunyuan_mxfp8.py")
    max_memory = None
    if args.max_memory:
        max_memory = {int(k) if k.isdigit() else k: v for k, v in args.max_memory.items()}
    model = load_qdq_model(
        args.model, max_memory=max_memory, attn_implementation="sdpa", moe_impl="eager", moe_drop_tokens=True
    )
    if not hasattr(model.config, "model_version"):
        model.config.model_version = "HunyuanImage-3.0-Instruct"
    model.load_tokenizer(str(args.model))
    image = generate_image(
        model, args.prompt, args.num_inference_steps, args.guidance_scale, args.image_size, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(vars(args), default=str, indent=2))
    print(
        f"Saved {args.output}. QDQ uses floating-point matmul; this is a quality check, not a low-bit speed benchmark."
    )


if __name__ == "__main__":
    main()
