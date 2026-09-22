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
from calibration_cache import DiskCalibrationCache


def validate_hunyuan_config(config):
    """Accept the published model type and the native config's serialized alias."""
    model_type = config.get("model_type")
    architectures = config.get("architectures") or []
    is_hunyuan_image = model_type == "hunyuan_image_3_moe" or (
        model_type == "Hunyuan" and "HunyuanImage3ForCausalMM" in architectures
    )
    if not is_hunyuan_image or config.get("cfg_distilled") is not True:
        raise ValueError(
            "Expected a HunyuanImage 3 Instruct Distil checkpoint; "
            f"got model_type={model_type!r}, architectures={architectures!r}, "
            f"cfg_distilled={config.get('cfg_distilled')!r}"
        )


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


@contextmanager
def calibration_schedule(pipe, calib_steps, seed):
    """Run a seeded subset of the native schedule on an isolated scheduler."""
    scheduler = deepcopy(pipe.scheduler)
    if scheduler.config.solver != "euler":
        raise ValueError("Calibration timestep sampling requires Hunyuan's Euler scheduler.")
    original_set_timesteps = scheduler.set_timesteps
    schedule = {}

    def set_timesteps(*args, **kwargs):
        original_set_timesteps(*args, **kwargs)
        total = len(scheduler.timesteps)
        if not 1 <= calib_steps <= total:
            raise ValueError(f"calib_num_inference_steps must be between 1 and num_inference_steps ({total})")
        if calib_steps == total:
            indices = list(range(total))
        elif calib_steps == 1:
            indices = [0]
        else:
            # Keep both ends; sample once per disjoint interior stratum.
            rng = torch.Generator().manual_seed(seed)
            indices = [0]
            for i in range(calib_steps - 2):
                start = 1 + i * (total - 2) // (calib_steps - 2)
                end = 1 + (i + 1) * (total - 2) // (calib_steps - 2)
                indices.append(torch.randint(start, end, (), generator=rng).item())
            indices.append(total - 1)
        # MeanFlow uses timesteps_full for the next-time conditioning token.
        scheduler.timesteps = scheduler.timesteps[indices]
        scheduler.timesteps_full = scheduler.timesteps_full[indices + [total]]
        scheduler.sigmas = scheduler.sigmas[indices + [total]]
        scheduler.num_inference_steps = calib_steps
        schedule.update(indices=indices, timesteps=scheduler.timesteps.cpu().tolist(), seed=seed)

    with patch.object(scheduler, "set_timesteps", set_timesteps), patch.object(pipe, "scheduler", scheduler):
        yield schedule


class ReplayCache:
    """Replay one layer's static KV state without mutating saved calibration data."""

    def __init__(self, keys, values, positions=None, length=None):
        self.keys = keys
        self.values = values
        self.positions = positions
        self.length = length

    def update(self, keys, values, layer_idx, cache_kwargs):
        positions = cache_kwargs["cache_position"]
        if self.positions is not None and int(self.length.item()) > 0:
            # Only untouched positions were saved. Updated positions are fully
            # overwritten below, so their previous KV values are not needed.
            shape = (keys.shape[0], keys.shape[1], int(self.length.item()), keys.shape[3])
            indices = self.positions.to(keys.device)[:, None, :, None].expand_as(self.keys)
            keys_out = keys.new_zeros(shape).scatter(2, indices, self.keys.to(keys))
            values_out = values.new_zeros(shape).scatter(2, indices, self.values.to(values))
        elif self.keys.shape[2] == 0:
            # Empty state is valid only for the initial full-context step.
            expected = torch.arange(keys.shape[2], device=positions.device)
            if not torch.equal(positions, expected.expand_as(positions)):
                raise RuntimeError("Initial KV replay requires the complete context.")
            return keys, values
        else:
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
        ar_kv_positions=None,
        ar_kv_length=None,
        **kwargs,
    ):
        if past_key_value is None and ar_keys is not None:
            past_key_value = ReplayCache(ar_keys, ar_values, ar_kv_positions, ar_kv_length)
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

    def __init__(self, quantizer, cache_dir=None):
        super().__init__(quantizer)
        self.prompt_count = 0
        self.disk_cache = DiskCalibrationCache(cache_dir, self.shared_cache_keys) if cache_dir else None
        if self.disk_cache is not None:
            weights_gib = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**3
            print(
                f"Writing calibration snapshots to {self.disk_cache.directory}. "
                f"Model parameters alone require approximately {weights_gib:.2f} GiB on CPU before tuning.",
                flush=True,
            )

    def close(self):
        if self.disk_cache is not None:
            self.disk_cache.close()

    def calib(self, nsamples, bs):
        self.prompt_count = 0
        self.requested_nsamples = nsamples
        self.summary = {}
        if self.has_variable_block_shape:
            # AutoRound captures every layer's kwargs, but chains reference
            # outputs between layers. Only the group entry needs hidden states.
            self.blocks_requiring_input_ids = ["model.layers.0"]
        super().calib(nsamples, bs)
        expected = nsamples * self.calib_num_inference_steps
        for name in self.to_cached_layers:
            if not name.startswith("model.layers."):
                continue
            info = self.summary.get(name, {})
            if info.get("forwards") != expected:
                raise RuntimeError(f"{name}: expected {expected} forwards, got {info}")
            saved = (
                len(self.disk_cache.files.get(name, []))
                if self.disk_cache is not None
                else len(self.inputs[name].get("ar_keys", []))
            )
            if saved != expected:
                raise RuntimeError(f"{name}: missing per-step KV snapshots")
        if self.disk_cache is not None:
            if self.inputs:
                raise RuntimeError("Some calibration inputs were not written to disk")
            self.inputs = self.disk_cache
            print(
                f"Calibration cache: {self.disk_cache.report()}. "
                "Tuning will read one layer at a time; full model weights still move to CPU.",
                flush=True,
            )

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
            elif self.disk_cache is not None:
                if hidden_states.shape[0] != 1:
                    raise ValueError("Compact Hunyuan calibration requires batch_size=1")
                length = state.keys.shape[2]
                keep = torch.ones(length, dtype=torch.bool, device=state.keys.device)
                keep[kwargs["position_ids"].reshape(-1).to(keep.device)] = False
                indices = keep.nonzero().flatten()
                keys = state.keys.detach().index_select(2, indices).cpu()
                values = state.values.detach().index_select(2, indices).cpu()
                kwargs.update(ar_kv_positions=indices.cpu()[None], ar_kv_length=torch.tensor([length]))
            else:
                # Copy before forward: Hunyuan updates its cache in place.
                keys = state.keys.detach().to(device="cpu", copy=True)
                values = state.values.detach().to(device="cpu", copy=True)
            if self.disk_cache is not None and state.keys is None:
                kwargs.update(ar_kv_positions=torch.empty(1, 0, dtype=torch.long), ar_kv_length=torch.tensor([0]))
            kwargs.update(ar_keys=keys, ar_values=values)
            result = capture(module, hidden_states, *args, **kwargs)
            info = self.summary.setdefault(name, {"forwards": 0, "sequence_lengths": []})
            info["forwards"] += 1
            info["sequence_lengths"].append(hidden_states.shape[1])
            if self.disk_cache is not None:
                self.disk_cache.append(name, self.inputs.pop(name))
                if module.layer_idx == len(self.model.model.layers) - 1:
                    count = info["forwards"]
                    if count % self.calib_num_inference_steps == 0:
                        self.prompt_count += 1
                        projected = (
                            self.disk_cache.bytes_written / self.prompt_count * self.requested_nsamples / 1024**3
                        )
                        print(
                            f"Calibration prompt {self.prompt_count}/{self.requested_nsamples}: "
                            f"{self.disk_cache.report()}; hidden dtype={hidden_states.dtype}; "
                            f"rough total at current rate={projected:.2f} GiB",
                            flush=True,
                        )
            return result

        return forward

    def make_layer_cache_hook(self, name):
        capture = super().make_layer_cache_hook(name)

        def hook(module, inputs, outputs):
            capture(module, inputs, outputs)
            if self.disk_cache is not None:
                self.disk_cache.append(name, self.inputs.pop(name))

        return hook


class HunyuanPipeline(DiffusionPipeline):
    """Route AutoRound to diffusion calibration using the native generation API."""

    def __init__(self, transformer, image_size="1024x1024", seed=42, num_inference_steps=8):
        super().__init__()
        self.register_modules(transformer=transformer)
        self.image_size = image_size
        self.seed = seed
        self.prompt_count = 0
        self.num_inference_steps = num_inference_steps
        self.calibration_schedules = []

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
        # AutoRound passes its calibration budget as the pipeline call's step count.
        calib_steps = num_inference_steps
        # Native gen_image reads these from generation_config, not loose kwargs.
        generation_config = deepcopy(self.transformer.generation_config)
        generation_config.diff_infer_steps = self.num_inference_steps
        generation_config.diff_guidance_scale = guidance_scale
        model_module = sys.modules[type(self.transformer).__module__]
        with compatible_cache_initialization(model_module.HunyuanStaticCache):
            for text in prompts:
                seed = self.seed + self.prompt_count
                with calibration_schedule(self.transformer.pipeline, calib_steps, seed) as schedule:
                    self.transformer.generate_image(
                        prompt=text,
                        seed=seed,
                        image_size=self.image_size,
                        bot_task="image",
                        use_system_prompt="en_unified",
                        generation_config=generation_config,
                        use_taylor_cache=False,
                        verbose=0,
                    )
                self.calibration_schedules.append(schedule)
                self.prompt_count += 1


def build_quantizer(model, args):
    blocks = list(model.model.layers)
    block_groups = [[f"model.layers.{i}" for i in range(len(blocks))]]
    originals = [install_replay_forward(block) for block in blocks]
    pipe = HunyuanPipeline(
        model, image_size=args.image_size, seed=args.seed, num_inference_steps=args.num_inference_steps
    )

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
            num_inference_steps=args.num_inference_steps,
            calib_num_inference_steps=args.calib_num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
    quantizer.post_init()
    # Enable AutoRound's existing per-layer auxiliary-input path. Hidden states
    # flow through the block runner; each layer still receives its own KV state.
    quantizer.has_variable_block_shape = True
    if args.layer_config:
        expert_schemes = Counter(
            f"W{cfg['bits']}A{cfg['act_bits']}"
            for name, cfg in quantizer.layer_config.items()
            if ".mlp.experts." in name
        )
        print("Resolved expert linear layers:", dict(expert_schemes))
    quantizer.calibration = HunyuanCalibrator(quantizer, cache_dir=getattr(args, "calib_cache_dir", None))
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
    parser.add_argument("--nsamples", type=int, default=8, help="Number of COCO prompts")
    parser.add_argument(
        "--num_inference_steps",
        "--num-inference-steps",
        type=int,
        default=None,
        help="Full native denoising schedule length (default: 8)",
    )
    parser.add_argument(
        "--calib_num_inference_steps",
        "--calib-num-inference-steps",
        type=int,
        default=None,
        help="Steps sampled and executed per prompt (default: full schedule)",
    )
    parser.add_argument("--steps", type=int, default=None, help="Legacy shorthand setting both step counts")
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
    parser.add_argument(
        "--calib-cache-dir",
        type=Path,
        help="Disk directory for temporary per-layer calibration snapshots; reduces CPU cache residency",
    )
    parser.add_argument("--max-memory", type=json.loads, help='Accelerate memory map, e.g. {"0":"70GiB","1":"70GiB"}')
    args = parser.parse_args()
    if args.steps is not None:
        if args.num_inference_steps is not None or args.calib_num_inference_steps is not None:
            parser.error("Use --steps alone or the two explicit step options, not both")
        args.num_inference_steps = args.calib_num_inference_steps = args.steps
    if args.num_inference_steps is None:
        args.num_inference_steps = 8
    if args.calib_num_inference_steps is None:
        args.calib_num_inference_steps = args.num_inference_steps
    del args.steps
    if min(args.nsamples, args.num_inference_steps, args.calib_num_inference_steps, args.iters) < 1:
        parser.error("nsamples, steps and iters must be positive; this script performs calibrated tuning")
    if args.calib_num_inference_steps > args.num_inference_steps:
        parser.error("calib_num_inference_steps must not exceed num_inference_steps")
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
    validate_hunyuan_config(config)
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
    print(
        f"Quantizing {len(original_forwards)} decoder blocks using {args.nsamples} COCO prompts x "
        f"{args.calib_num_inference_steps} calibration steps sampled from {args.num_inference_steps} steps"
    )
    try:
        quantizer.quantize()
    finally:
        for block, original in zip(model.model.layers, original_forwards):
            block.forward = original
        quantizer.calibration.close()
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
                "calibration_schedules": quantizer.pipe.calibration_schedules,
            },
            default=str,
            indent=2,
        )
    )
    print(f"Exported to {args.output}; inference-engine loading and image quality need separate validation.")


if __name__ == "__main__":
    main()
