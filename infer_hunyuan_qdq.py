#!/usr/bin/env python3
"""Generate images from an exported AutoRound MXFP8/MXFP4 checkpoint using PyTorch QDQ."""

import argparse
import json
import sys
from collections import Counter
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig

from auto_round.experimental.qmodules.mx import MXFP4QuantLinear, MXFP8QuantLinear
from quantize_hunyuan_mxfp8 import compatible_cache_initialization, load_hunyuan_tokenizer, validate_hunyuan_config


def set_activation_qdq(model, enabled):
    for layer in model.modules():
        if isinstance(layer, (MXFP4QuantLinear, MXFP8QuantLinear)):
            layer.pre_dequantized_input = not enabled


def use_rceil_activation_qdq(model):
    """Diagnostic: use the activation scale rule from INC's FLUX example."""
    for layer in model.modules():
        if isinstance(layer, (MXFP4QuantLinear, MXFP8QuantLinear)):
            # Config objects can be shared with other layers/the loaded model.
            # Override runtime activation behavior without rewriting the recipe.
            layer.config = deepcopy(layer.config)
            layer.config.act_data_type = "mx_fp_rceil"


def check_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f"Non-finite values in {name} (dtype={tensor.dtype}, shape={tuple(tensor.shape)}); "
            "stopping before image postprocessing hides the failure"
        )


@contextmanager
def diagnose_generation(model, report_path=None):
    """Trace the native denoising and VAE path without changing its calculations."""
    if report_path is None:
        yield
        return
    records = []

    def record(name, tensor):
        check_finite(name, tensor)
        values = tensor.detach().float()
        item = dict(
            name=name,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
            min=values.min().item(),
            max=values.max().item(),
            mean=values.mean().item(),
            std=values.std(unbiased=False).item(),
        )
        records.append(item)
        print("[debug]", json.dumps(item), flush=True)

    def block_hook(name):
        def hook(module, args, output):
            check_finite(name, output[0] if isinstance(output, tuple) else output)

        return hook

    scheduler = model.pipeline.scheduler
    original_step = scheduler.step
    original_decode = model.vae.decode
    step_index = 0

    def step(prediction, timestep, sample, *args, **kwargs):
        nonlocal step_index
        label = f"step {step_index}, t={float(timestep)}"
        record(f"{label}: prediction", prediction)
        record(f"{label}: latent before", sample)
        result = original_step(prediction, timestep, sample, *args, **kwargs)
        record(f"{label}: latent after", result[0] if isinstance(result, tuple) else result.prev_sample)
        step_index += 1
        return result

    def decode(latents, *args, **kwargs):
        record("VAE input", latents)
        result = original_decode(latents, *args, **kwargs)
        record("VAE output", result[0] if isinstance(result, tuple) else result.sample)
        return result

    try:
        with ExitStack() as stack:
            for i, block in enumerate(model.model.layers):
                handle = block.register_forward_hook(block_hook(f"model.layers.{i} output"))
                stack.callback(handle.remove)
            stack.enter_context(patch.object(scheduler, "step", step))
            stack.enter_context(patch.object(model.vae, "decode", decode))
            yield
    except Exception as error:
        records.append({"error": str(error)})
        raise
    finally:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(records, indent=2))


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


@contextmanager
def compatible_ar_generation(model):
    """Keep the cache flag required by newer Transformers AR decoding loops."""
    original = model._update_model_kwargs_for_generation

    def update(outputs, model_kwargs, *args, **kwargs):
        updated = original(outputs, model_kwargs, *args, **kwargs)
        if model_kwargs.get("mode") == "gen_text" and "use_cache" in model_kwargs:
            updated["use_cache"] = model_kwargs["use_cache"]
        return updated

    with patch.object(model, "_update_model_kwargs_for_generation", update):
        yield


@torch.inference_mode()
def generate_image(
    model,
    prompt,
    num_inference_steps=8,
    guidance_scale=5.0,
    image_size="1024x1024",
    seed=42,
    bot_task="image",
    max_new_tokens=2048,
):
    config = deepcopy(model.generation_config)
    config.diff_infer_steps = num_inference_steps
    config.diff_guidance_scale = guidance_scale
    if bot_task != "image":
        # AR sampling uses the global RNG; native seed= controls image noise.
        torch.manual_seed(seed)
        config.max_new_tokens = max_new_tokens
    model_module = sys.modules[type(model).__module__]
    ar_context = compatible_ar_generation(model) if bot_task != "image" else nullcontext()
    with compatible_cache_initialization(model_module.HunyuanStaticCache), ar_context:
        cot_text, images = model.generate_image(
            prompt=prompt,
            seed=seed,
            image_size=image_size,
            bot_task=bot_task,
            use_system_prompt="en_unified",
            generation_config=config,
            max_new_tokens=max_new_tokens,
            use_taylor_cache=False,
            verbose=0,
        )
    if cot_text:
        print("[AR]", "\n".join(cot_text) if isinstance(cot_text, list) else cot_text, flush=True)
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
    parser.add_argument(
        "--bot-task",
        choices=("image", "think", "recaption", "think_recaption"),
        default="image",
        help="Direct image generation, or native AR reasoning/recaptioning followed by image generation",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum generated AR text tokens")
    parser.add_argument("--debug", action="store_true", help="Check block finiteness and save denoising/VAE statistics")
    parser.add_argument(
        "--disable-act-quant", action="store_true", help="Diagnostic: keep saved weights but bypass activation QDQ"
    )
    parser.add_argument(
        "--act-qdq",
        choices=("checkpoint", "rceil"),
        default="checkpoint",
        help="Activation QDQ: saved configuration, or diagnostic RCEIL scales as in INC's FLUX example",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Reference run from the original unquantized checkpoint, preserving native mixed dtypes",
    )
    parser.add_argument("--max-memory", type=json.loads, help='GPU memory budgets, e.g. {"0":"70GiB","1":"70GiB"}')
    args = parser.parse_args()
    if args.act_qdq != "checkpoint" and (args.bf16 or args.disable_act_quant):
        parser.error("--act-qdq rceil cannot be combined with --bf16 or --disable-act-quant")
    args.model = args.model.resolve()
    if args.num_inference_steps < 1 or args.max_new_tokens < 1 or args.image_size == "auto":
        parser.error("Use positive inference steps/token limits and a fixed image size, e.g. 1024x1024")
    if "." in args.model.name:
        parser.error("Tencent custom code requires a model directory name without dots")
    config = json.loads((args.model / "config.json").read_text())
    try:
        validate_hunyuan_config(config)
    except ValueError as error:
        parser.error(str(error))
    quant_config = config.get("quantization_config", {})
    if args.bf16 and (quant_config or args.disable_act_quant):
        parser.error(
            "--bf16 requires the original unquantized checkpoint and cannot be combined with --disable-act-quant"
        )
    if not args.bf16 and (quant_config.get("quant_method") != "auto-round" or quant_config.get("data_type") != "mx_fp"):
        parser.error("Expected the AutoRound MXFP checkpoint exported by quantize_hunyuan_mxfp8.py")
    max_memory = None
    if args.max_memory:
        max_memory = {int(k) if k.isdigit() else k: v for k, v in args.max_memory.items()}
    if args.bf16:
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            max_memory=max_memory,
            attn_implementation="sdpa",
            moe_impl="eager",
            moe_drop_tokens=True,
        ).eval()
    else:
        model = load_qdq_model(
            args.model, max_memory=max_memory, attn_implementation="sdpa", moe_impl="eager", moe_drop_tokens=True
        )
    if args.act_qdq == "rceil":
        use_rceil_activation_qdq(model)
        print(
            "Diagnostic RCEIL activation QDQ enabled; weight values and per-layer activation bit widths are retained."
        )
    if args.disable_act_quant:
        set_activation_qdq(model, enabled=False)
        print("Activation QDQ disabled for diagnosis; saved quantized weights are unchanged.")
    load_hunyuan_tokenizer(model, args.model)
    if args.debug:
        import auto_round
        import transformers

        print(
            "[debug] Versions:",
            {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "auto_round": auto_round.__version__,
                "auto_round_path": auto_round.__file__,
            },
        )
        print("[debug] Inference settings:", json.dumps(vars(args), default=str))
        recipe_path = args.model / "calibration_recipe.json"
        if recipe_path.exists():
            recipe = json.loads(recipe_path.read_text())
            print(
                "[debug] Calibration settings:",
                {
                    key: recipe.get(key)
                    for key in (
                        "nsamples",
                        "iters",
                        "num_inference_steps",
                        "calib_num_inference_steps",
                        "image_size",
                        "guidance_scale",
                    )
                },
            )
    with diagnose_generation(model, args.output.with_suffix(".debug.json") if args.debug else None):
        image = generate_image(
            model,
            args.prompt,
            args.num_inference_steps,
            args.guidance_scale,
            args.image_size,
            args.seed,
            bot_task=args.bot_task,
            max_new_tokens=args.max_new_tokens,
        )
    if args.debug:
        from PIL import ImageStat

        stats = ImageStat.Stat(image.convert("RGB"))
        print("[debug] Image RGB mean/std:", stats.mean, stats.stddev)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(vars(args), default=str, indent=2))
    print(f"Saved {args.output}.")


if __name__ == "__main__":
    main()
