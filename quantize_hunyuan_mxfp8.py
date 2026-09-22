#!/usr/bin/env python3
"""Experimental HunyuanImage 3 Instruct Distil COCO -> SignRound MXFP8 adapter.

Targets AutoRound main 6db9435f and Tencent checkpoint revision c8ffd072.
No AutoRound source files are changed. See README.md for validation boundaries.
"""

import argparse
import inspect
import json
import shutil
import sys
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import torch
from diffusers import DiffusionPipeline

from auto_round import AutoRound, SignRoundConfig
from auto_round.calibration.diffusion import DiffusionCalibrator
from auto_round.compressors.base import BaseOrchestrator
from auto_round.compressors.diffusion_mixin import DiffusionMixin
from auto_round.utils import parse_layer_config_arg


@contextmanager
def compatible_cache_initialization(cache_class):
    """Bridge native Hunyuan's one-argument call to newer Transformers caches."""
    original_update = cache_class.update

    def update(cache, key_states, value_states, layer_idx, cache_kwargs=None):
        layer = cache.layers[layer_idx]
        if layer.keys is None and "value_states" in inspect.signature(layer.lazy_initialization).parameters:
            # Initialize with both tensors before the native update reaches its old call.
            layer.lazy_initialization(key_states, value_states)
        return original_update(cache, key_states, value_states, layer_idx, cache_kwargs)

    with patch.object(cache_class, "update", update):
        yield


class ReplayCache:
    """Replay one layer's static KV state without mutating saved calibration data."""

    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

    def update(self, keys, values, layer_idx, cache_kwargs):
        positions = cache_kwargs["cache_position"]
        # Empty state is valid only for the initial full-context diffusion step.
        if self.keys.shape[2] == 0:
            expected = torch.arange(keys.shape[2], device=positions.device)
            if not torch.equal(positions, expected.expand_as(positions)):
                raise RuntimeError("Initial KV replay requires the complete context.")
            return keys, values
        keys_out = self.keys.to(keys).clone()
        values_out = self.values.to(values).clone()
        if positions.ndim == 1:
            return (
                keys_out.index_copy(2, positions, keys),
                values_out.index_copy(2, positions, values),
            )
        indices = positions[:, None, :, None].expand_as(keys)
        return keys_out.scatter(2, indices, keys), values_out.scatter(2, indices, values)


def install_replay_forward(block):
    """Keep the original state-dict names; adapt only this instance's forward."""
    original = block.forward

    def forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        custom_pos_emb=None,
        ar_keys=None,
        ar_values=None,
        **kwargs,
    ):
        if past_key_value is None and ar_keys is not None:
            past_key_value = ReplayCache(ar_keys, ar_values)
        return original(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            custom_pos_emb=custom_pos_emb,
            **kwargs,
        )

    block.forward = forward
    return original


class HunyuanCalibrator(DiffusionCalibrator):
    """Save per-layer KV tensors that the generic cache collector would omit."""

    def calib(self, nsamples, bs):
        super().calib(nsamples, bs)
        self.summary = {}
        for name, inputs in self.inputs.items():
            count = len(inputs["hidden_states"])
            if count != nsamples * self.calib_num_inference_steps:
                raise RuntimeError(
                    f"{name}: expected {nsamples * self.calib_num_inference_steps} forwards, got {count}"
                )
            if len(inputs.get("ar_keys", [])) != count:
                raise RuntimeError(f"{name}: missing per-step KV snapshots")
            self.summary[name] = {"forwards": count, "sequence_lengths": [x.shape[1] for x in inputs["hidden_states"]]}

    def _make_block_forward_func(self, name):
        capture = super()._make_block_forward_func(name)

        def forward(module, hidden_states=None, *args, **kwargs):
            cache = kwargs.get("past_key_value")
            if cache is None or getattr(cache, "dynamic", False):
                raise RuntimeError("Expected the static KV cache from direct image generation.")
            state = cache.layers[module.layer_idx]
            if state.keys is None:
                attn = module.self_attn
                shape = (hidden_states.shape[0], attn.num_key_value_heads, 0, attn.head_dim)
                keys = hidden_states.new_empty(shape)
                values = hidden_states.new_empty(shape)
            else:
                # Copy before forward: Hunyuan updates its cache in place.
                keys, values = state.keys.detach().cpu().clone(), state.values.detach().cpu().clone()
            kwargs.update(ar_keys=keys, ar_values=values)
            return capture(module, hidden_states, *args, **kwargs)

        return forward


class HunyuanPipeline(DiffusionPipeline):
    """Route AutoRound to diffusion calibration using the native generation API."""

    def __init__(self, transformer, image_size="1024x1024", seed=42):
        super().__init__()
        self.register_modules(transformer=transformer)
        self.image_size = image_size
        self.seed = seed
        self.prompt_count = 0

    @property
    def device(self):
        return self.transformer.device

    def to(self, device=None, *args, **kwargs):
        # Preserve Accelerate's multi-GPU placement during calibration.
        if getattr(self.transformer, "hf_device_map", None):
            return self
        self.transformer.to(device, *args, **kwargs)
        return self

    @torch.no_grad()
    def __call__(self, prompt, guidance_scale=5.0, num_inference_steps=8, generator=None):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        # Native gen_image reads these from generation_config, not loose kwargs.
        generation_config = deepcopy(self.transformer.generation_config)
        generation_config.diff_infer_steps = num_inference_steps
        generation_config.diff_guidance_scale = guidance_scale
        model_module = sys.modules[type(self.transformer).__module__]
        with compatible_cache_initialization(model_module.HunyuanStaticCache):
            for text in prompts:
                self.transformer.generate_image(
                    prompt=text,
                    seed=self.seed + self.prompt_count,
                    image_size=self.image_size,
                    bot_task="image",
                    use_system_prompt="en_unified",
                    generation_config=generation_config,
                    use_taylor_cache=False,
                    verbose=0,
                )
                self.prompt_count += 1


def build_quantizer(model, args):
    blocks = list(model.model.layers)
    block_groups = [[f"model.layers.{i}"] for i in range(len(blocks))]
    originals = [install_replay_forward(block) for block in blocks]
    pipe = HunyuanPipeline(model, image_size=args.image_size, seed=args.seed)

    def load_adapter(candidate, **kwargs):
        if candidate is not pipe:
            raise TypeError("This temporary loader accepts only this Hunyuan adapter.")
        return pipe, model

    # These patches exist only during construction and are always restored.
    # Preserve Tencent's mixed module dtypes and Transformers save_pretrained.
    with (
        patch("auto_round.context.model.diffusion_load_model", load_adapter),
        patch.object(DiffusionMixin, "_align_pipeline_dtype", lambda *unused: None),
    ):
        quantizer = AutoRound(
            model=pipe,
            tokenizer=None,
            scheme="MXFP8",
            layer_config=args.layer_config,
            dataset="coco2014",
            alg_configs=SignRoundConfig(
                iters=args.iters, gradient_accumulate_steps=1, nblocks=1, enable_quanted_input=False
            ),
            nsamples=args.nsamples,
            batch_size=1,
            to_quant_block_names=block_groups,
            ignore_layers="lm_head,vae,vit,vit_aligner",
            device_map=args.device,
            low_gpu_mem_usage=False,
            low_cpu_mem_usage=False,
            enable_torch_compile=False,
            seed=args.seed,
            calib_num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        )
    quantizer.post_init()
    if args.layer_config:
        expert_schemes = Counter(
            f"W{cfg['bits']}A{cfg['act_bits']}"
            for name, cfg in quantizer.layer_config.items()
            if ".mlp.experts." in name
        )
        print("Resolved expert linear layers:", dict(expert_schemes))
    quantizer.calibration = HunyuanCalibrator(quantizer)
    # Native Transformers checkpoint layout, not an artificial Diffusers folder.
    quantizer.save_quantized = MethodType(BaseOrchestrator.save_quantized, quantizer)
    return quantizer, originals


def copy_support_files(source, destination):
    """Copy native code/tokenizer assets without copying unquantized weights."""
    for path in source.rglob("*"):
        rel = path.relative_to(source)
        if not path.is_file() or any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            continue
        if path.suffix not in {".py", ".json", ".txt", ".model", ".tiktoken", ".jinja"}:
            continue
        if path.name == "config.json" or path.name.endswith(".index.json"):
            continue
        target = destination / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Downloaded Tencent native checkpoint directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--nsamples", type=int, default=8, help="COCO prompts; each produces --steps calibration forwards"
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--image-size", default="1024x1024", help="Fixed size avoids autoregressive aspect-ratio generation"
    )
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0", help="GPU used for layer tuning; model loading uses all visible GPUs")
    parser.add_argument(
        "--layer_config",
        "--layer-config",
        type=parse_layer_config_arg,
        default=None,
        help="AutoRound per-layer overrides, e.g. '{mlp.experts:{scheme:MXFP4}}'",
    )
    parser.add_argument("--max-memory", type=json.loads, help='Accelerate memory map, e.g. {"0":"70GiB","1":"70GiB"}')
    args = parser.parse_args()
    if min(args.nsamples, args.steps, args.iters) < 1:
        parser.error("nsamples, steps and iters must be positive; this script performs calibrated tuning")
    if args.image_size == "auto":
        parser.error("Use a fixed image size, e.g. 1024x1024")
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    if args.output == args.model or args.model in args.output.parents:
        parser.error("Output must be outside the source model directory")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    return args


def main():
    from transformers import AutoModelForCausalLM

    args = parse_args()
    config = json.loads((args.model / "config.json").read_text())
    if config.get("model_type") != "hunyuan_image_3_moe" or not config.get("cfg_distilled"):
        raise ValueError("This script targets Tencent HunyuanImage 3 Instruct Distil only.")
    if "." in args.model.name:
        raise ValueError("Use a model directory name without dots, as required by Tencent custom code.")
    max_memory = None
    if args.max_memory:
        max_memory = {int(k) if k.isdigit() else k: v for k, v in args.max_memory.items()}
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype="auto",
        attn_implementation="sdpa",
        device_map="auto",
        max_memory=max_memory,
        moe_impl="eager",
        moe_drop_tokens=True,
    ).eval()
    device_map = getattr(model, "hf_device_map", {})
    if any(str(device) in {"cpu", "disk"} for device in device_map.values()):
        raise RuntimeError("Calibration currently requires GPU-resident weights; provide enough visible GPUs.")
    # Older Distil configs omit this field, but the native tokenizer requires it (Tencent issue #83).
    if not hasattr(model.config, "model_version"):
        model.config.model_version = "HunyuanImage-3.0-Instruct"
    model.load_tokenizer(str(args.model))
    quantizer, original_forwards = build_quantizer(model, args)
    print(f"Quantizing {len(original_forwards)} decoder blocks using {args.nsamples} COCO prompts x {args.steps} steps")
    quantizer.quantize()
    for block, original in zip(model.model.layers, original_forwards):
        block.forward = original
    quantizer.save_quantized(str(args.output), format="auto_round", inplace=True)
    copy_support_files(args.model, args.output)
    (args.output / "calibration_recipe.json").write_text(
        json.dumps(
            {
                **vars(args),
                "scheme": "MXFP8",
                "format": "auto_round",
                "dataset": "coco2014",
                "bot_task": "image",
                "calibration": quantizer.calibration.summary,
            },
            default=str,
            indent=2,
        )
    )
    print(f"Exported to {args.output}; inference-engine loading and image quality need separate validation.")


if __name__ == "__main__":
    main()
