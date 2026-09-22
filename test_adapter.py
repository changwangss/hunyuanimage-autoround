"""Small checks only: no claim of full Hunyuan model validation."""

import ast
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from torch import nn
from transformers import LlamaConfig, PreTrainedModel
from transformers.cache_utils import StaticCache
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from quantize_hunyuan_mxfp8 import (
    HunyuanPipeline,
    build_quantizer,
    calibration_schedule,
    compatible_cache_initialization,
    install_replay_forward,
    parse_args,
)
from auto_round.utils import parse_layer_config_arg


def attention_class():
    # Execute the exact published attention implementation in isolation.
    source = Path(__file__).parent / "reference/modeling_hunyuan_image_3.py"
    tree = ast.parse(source.read_text())
    names = {"HunyuanImage3SDPAAttention", "repeat_kv", "apply_rotary_pos_emb", "rotate_half"}
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    env = {"torch": torch, "nn": nn}
    exec(compile("from __future__ import annotations\n" + ast.unparse(ast.Module(nodes, [])), str(source), "exec"), env)
    return env["HunyuanImage3SDPAAttention"]


def native_cache_class():
    source = Path(__file__).parent / "reference/modeling_hunyuan_image_3.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HunyuanStaticCache")
    env = {"torch": torch, "StaticCache": StaticCache}
    exec(compile("from __future__ import annotations\n" + ast.unparse(node), str(source), "exec"), env)
    return env["HunyuanStaticCache"]


HunyuanStaticCache = native_cache_class()


def native_scheduler():
    source = Path(__file__).parent / "reference/hunyuan_image_3_pipeline.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlowMatchDiscreteScheduler")
    env = {
        "torch": torch,
        "SchedulerMixin": SchedulerMixin,
        "ConfigMixin": ConfigMixin,
        "register_to_config": register_to_config,
    }
    exec(compile("from __future__ import annotations\n" + ast.unparse(node), str(source), "exec"), env)
    return env["FlowMatchDiscreteScheduler"](shift=3.0, reverse=True, solver="euler")


@pytest.mark.parametrize("calib_steps", [1, 2, 4, 8])
def test_native_scheduler_subset_and_meanflow(calib_steps):
    original = native_scheduler()
    original.set_timesteps(8)
    pipe = SimpleNamespace(scheduler=original)
    selections = []
    for seed in [42, 42, 43]:
        with calibration_schedule(pipe, calib_steps, seed) as record:
            scheduler = pipe.scheduler
            scheduler.set_timesteps(8)
            indices = record["indices"]
            assert len(indices) == len(set(indices)) == calib_steps
            assert indices == sorted(indices) and indices[0] == 0
            if calib_steps > 1:
                assert indices[-1] == 7
            torch.testing.assert_close(scheduler.timesteps, original.timesteps[indices])
            torch.testing.assert_close(scheduler.sigmas, original.sigmas[indices + [8]])
            sample = torch.ones(1)
            for i, timestep in enumerate(scheduler.timesteps):
                next_time = scheduler.get_timestep_r(timestep)
                torch.testing.assert_close(next_time, scheduler.sigmas[i + 1] * 1000)
                sample = scheduler.step(torch.ones(1), timestep, sample, return_dict=False)[0]
            torch.testing.assert_close(sample, torch.zeros(1))
            selections.append(indices)
        assert pipe.scheduler is original
        assert original.step_index is None
        assert len(original.timesteps) == 8
    assert selections[0] == selections[1]
    if calib_steps == 4:
        assert selections[0] != selections[2]
    with pytest.raises(RuntimeError, match="generation failed"):
        with calibration_schedule(pipe, calib_steps, 42):
            pipe.scheduler.set_timesteps(8)
            raise RuntimeError("generation failed")
    assert pipe.scheduler is original


@pytest.mark.parametrize(
    "options,expected",
    [
        (["--num_inference_steps", "8", "--calib_num_inference_steps", "4"], (8, 4)),
        (["--steps", "4"], (4, 4)),
        (["--num_inference_steps", "4", "--calib_num_inference_steps", "8"], None),
    ],
)
def test_cli_step_options(monkeypatch, tmp_path, options, expected):
    monkeypatch.setattr(
        sys, "argv", ["quantize", "--model", str(tmp_path / "model"), "--output", str(tmp_path / "out"), *options]
    )
    if expected is None:
        with pytest.raises(SystemExit) as error:
            parse_args()
        assert error.value.code == 2
    else:
        args = parse_args()
        assert (args.num_inference_steps, args.calib_num_inference_steps) == expected


@pytest.mark.parametrize("steps", [3, 8])
def test_adapter_passes_options_to_native_generation(monkeypatch, steps):
    source = Path(__file__).parent / "reference/modeling_hunyuan_image_3.py"
    tree = ast.parse(source.read_text())
    model_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HunyuanImage3ForCausalMM")
    generate = next(n for n in model_class.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
    env = {"torch": torch, "default": lambda value, fallback: fallback if value is None else value}
    exec(compile("from __future__ import annotations\n" + ast.unparse(generate), str(source), "exec"), env)
    model = TinyModel(make_config())
    model.config.use_meanflow = True
    model.config.cfg_distilled = True
    model._tokenizer = SimpleNamespace(encode=lambda text: [9])
    model.pipeline = Mock(return_value=[None])
    model.pipeline.scheduler = native_scheduler()

    def run_pipeline(**kwargs):
        model.pipeline.scheduler.set_timesteps(kwargs["num_inference_steps"])
        return [None]

    model.pipeline.side_effect = run_pipeline
    info = SimpleNamespace(
        image_token_length=2,
        add_timestep_token=True,
        add_guidance_token=True,
        add_timestep_r_token=True,
        image_height=1024,
        image_width=1024,
    )

    def generate_image(**kwargs):
        # Bypass tokenization/VAE only; use Tencent's actual generation dispatch.
        return env["generate"](
            model,
            mode="gen_image",
            tokenizer_output=SimpleNamespace(tokens=torch.tensor([[1, 9, 2]])),
            batch_gen_image_info=[info],
            **kwargs,
        )

    monkeypatch.setattr(model, "generate_image", generate_image)
    original_config = model.generation_config
    HunyuanPipeline(model, num_inference_steps=steps)(["a cat", "a dog"], num_inference_steps=steps, guidance_scale=5.0)
    assert model.pipeline.call_count == 2
    for call in model.pipeline.call_args_list:
        assert call.kwargs["num_inference_steps"] == steps
        assert call.kwargs["guidance_scale"] == 5.0
    assert model.generation_config is original_config
    assert original_config.diff_infer_steps == 8
    assert original_config.diff_guidance_scale == 2.5


@pytest.mark.parametrize("legacy", [False, True])
def test_native_cache_initialization_and_updates(legacy):
    cache = HunyuanStaticCache(config=make_config(), max_cache_len=5, dynamic=False)

    class LegacyLayer:
        keys = None
        values = None

        def lazy_initialization(self, key_states):
            shape = (*key_states.shape[:2], 5, key_states.shape[-1])
            self.keys = key_states.new_zeros(shape)
            self.values = key_states.new_zeros(shape)

    if legacy:
        cache.layers[0] = LegacyLayer()
    original_update = HunyuanStaticCache.update
    keys = torch.randn(1, 2, 5, 32)
    values = torch.randn_like(keys)
    with compatible_cache_initialization(HunyuanStaticCache):
        actual_k, actual_v = cache.update(keys, values, 0, {"cache_position": torch.arange(5)[None]})
        torch.testing.assert_close(actual_k, keys)
        torch.testing.assert_close(actual_v, values)
        new_k, new_v = torch.randn(1, 2, 2, 32), torch.randn(1, 2, 2, 32)
        actual_k, actual_v = cache.update(new_k, new_v, 0, {"cache_position": torch.tensor([[2, 4]])})
        expected_k, expected_v = keys.clone(), values.clone()
        expected_k[:, :, [2, 4]] = new_k
        expected_v[:, :, [2, 4]] = new_v
        torch.testing.assert_close(actual_k, expected_k)
        torch.testing.assert_close(actual_v, expected_v)
    assert HunyuanStaticCache.update is original_update
    with pytest.raises(RuntimeError, match="generation failed"):
        with compatible_cache_initialization(HunyuanStaticCache):
            raise RuntimeError("generation failed")
    assert HunyuanStaticCache.update is original_update


class NativeCache:
    def __init__(self, count, length):
        self.layers = [SimpleNamespace(keys=None, values=None) for _ in range(count)]
        self.length = length
        self.dynamic = False

    def update(self, keys, values, layer_idx, cache_kwargs):
        layer = self.layers[layer_idx]
        if layer.keys is None:
            shape = (*keys.shape[:2], self.length, keys.shape[-1])
            layer.keys, layer.values = keys.new_zeros(shape), values.new_zeros(shape)
        positions = cache_kwargs["cache_position"]
        for batch in range(keys.shape[0]):
            layer.keys[batch].index_copy_(1, positions[batch], keys[batch])
            layer.values[batch].index_copy_(1, positions[batch], values[batch])
        return layer.keys, layer.values


class TinyMoE(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(2)])
        self.shared_mlp = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        return self.shared_mlp(hidden_states) + sum(expert(hidden_states) for expert in self.experts) / 2


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = attention_class()(config, layer_idx)
        self.mlp = TinyMoE(config.hidden_size)

    def forward(self, hidden_states, **kwargs):
        h = hidden_states + self.self_attn(hidden_states, **kwargs)[0]
        return (h + self.mlp(h),)


def make_config():
    config = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2
    )
    config.attention_head_dim = 32
    config.rope_theta = 10000.0
    config.use_qk_norm = False
    config.use_rotary_pos_emb = True
    config.rope_scaling = {"type": "custom"}
    return config


def rotary(positions, dtype):
    phases = positions[..., None].float() * torch.linspace(0.01, 1.0, 32, device=positions.device)
    return phases.cos().to(dtype), phases.sin().to(dtype)


def test_native_attention_replay_preserves_context_and_gradients():
    torch.manual_seed(42)
    block = Block(make_config(), 0)
    original = install_replay_forward(block)
    live = NativeCache(1, 5)
    first = torch.randn(1, 5, 64)
    initial_pos = torch.arange(5)[None]
    with torch.no_grad():
        expected = original(
            first, position_ids=initial_pos, past_key_value=live, custom_pos_emb=rotary(initial_pos, first.dtype)
        )[0]
    empty = first.new_empty(1, 2, 0, 32)
    actual = block(
        first, position_ids=initial_pos, ar_keys=empty, ar_values=empty, custom_pos_emb=rotary(initial_pos, first.dtype)
    )[0]
    torch.testing.assert_close(actual, expected)
    for _ in range(3):
        keys, values = live.layers[0].keys.clone(), live.layers[0].values.clone()
        query = torch.randn(1, 2, 64, requires_grad=True)
        positions = torch.tensor([[2, 4]])
        with torch.no_grad():
            expected = original(
                query, position_ids=positions, past_key_value=live, custom_pos_emb=rotary(positions, query.dtype)
            )[0]
        actual = block(
            query, position_ids=positions, ar_keys=keys, ar_values=values, custom_pos_emb=rotary(positions, query.dtype)
        )[0]
        torch.testing.assert_close(actual, expected)
        again = block(
            query, position_ids=positions, ar_keys=keys, ar_values=values, custom_pos_emb=rotary(positions, query.dtype)
        )[0]
        torch.testing.assert_close(again, actual)
        actual.square().sum().backward()
        assert query.grad is not None and torch.isfinite(query.grad).all()
        assert block.self_attn.qkv_proj.weight.grad.abs().sum() > 0


class TinyModel(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(config, i) for i in range(2)])
        self.model.embed_tokens = nn.Embedding(32, config.hidden_size)
        self.calls = []
        self.generation_config = SimpleNamespace(diff_infer_steps=8, diff_guidance_scale=2.5)
        self.pipeline = SimpleNamespace(scheduler=native_scheduler())

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        generator = torch.Generator(device=self.device).manual_seed(kwargs["seed"])
        cache = HunyuanStaticCache(config=self.config, max_cache_len=5, dynamic=False)
        gen_config = kwargs.get("generation_config", self.generation_config)
        self.pipeline.scheduler.set_timesteps(gen_config.diff_infer_steps)
        for step, _ in enumerate(self.pipeline.scheduler.timesteps):
            length = 5 if step == 0 else 2
            h = torch.randn(1, length, 64, device=self.device, dtype=self.dtype, generator=generator)
            positions = (
                torch.arange(5, device=self.device)[None] if step == 0 else torch.tensor([[2, 4]], device=self.device)
            )
            for block in self.model.layers:
                h = block(
                    h,
                    position_ids=positions,
                    past_key_value=cache,
                    use_cache=True,
                    custom_pos_emb=rotary(positions, h.dtype),
                )[0]


@pytest.mark.parametrize("expert_bits", [8, 4])
def test_actual_autoround_mxfp8_tuning_and_export(monkeypatch, tmp_path, expert_bits):
    monkeypatch.setattr(
        "auto_round.compressors.diffusion.dataset._load_coco_dataframe",
        lambda *args: pd.DataFrame({"id": [1, 2, 3, 4], "caption": ["a cat", "a dog", "a bird", "a tree"]}),
    )
    model = TinyModel(make_config()).to("cuda:0", dtype=torch.bfloat16).eval()
    override = parse_layer_config_arg("{mlp.experts:{scheme:MXFP4}}") if expert_bits == 4 else None
    args = SimpleNamespace(
        image_size="1024x1024",
        seed=42,
        iters=2,
        nsamples=4,
        device="0",
        num_inference_steps=8,
        calib_num_inference_steps=4,
        guidance_scale=5.0,
        layer_config=override,
    )
    quantizer, originals = build_quantizer(model, args)
    assert quantizer.model_context.is_diffusion
    for name, cfg in quantizer.layer_config.items():
        if ".mlp.experts." in name:
            assert cfg["bits"] == cfg["act_bits"] == expert_bits
        if ".mlp.shared_mlp" in name or ".self_attn." in name:
            assert cfg["bits"] == cfg["act_bits"] == 8
    quantizer.quantize()
    assert len(model.calls) == 4
    assert quantizer.model_context.quantized
    assert len(quantizer.calibration.summary) == 2
    assert all(item["sequence_lengths"] == [5, 2, 2, 2] * 4 for item in quantizer.calibration.summary.values())
    assert all(item["forwards"] == 16 for item in quantizer.calibration.summary.values())
    assert len(quantizer.pipe.calibration_schedules) == 4
    assert all(len(record["indices"]) == 4 for record in quantizer.pipe.calibration_schedules)
    assert quantizer.scheme_context.bits == quantizer.scheme_context.act_bits == 8
    for block, original in zip(model.model.layers, originals):
        block.forward = original
    quantizer.save_quantized(str(tmp_path / "export"), format="auto_round", inplace=True)
    configs = list((tmp_path / "export").rglob("config.json"))
    assert configs
    exported = json.loads(configs[0].read_text())["quantization_config"]
    assert exported["quant_method"] == "auto-round"
    assert exported["bits"] == 8 and exported["act_bits"] == 8
    assert list((tmp_path / "export").rglob("*.safetensors"))
    for name, module in model.named_modules():
        if ".mlp.experts." in name:
            assert module.bits == expert_bits
            weight = module.weight_packed if expert_bits == 4 else module.weight
            assert weight.dtype == (torch.uint8 if expert_bits == 4 else torch.float8_e4m3fn)
            assert weight.shape == (64, 32 if expert_bits == 4 else 64)
            if expert_bits == 4:
                settings = next(
                    value for pattern, value in exported["extra_config"].items() if re.search(pattern, name)
                )
                assert settings["bits"] == settings["act_bits"] == 4
