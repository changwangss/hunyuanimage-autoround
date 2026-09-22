"""Small checks only: no claim of full Hunyuan model validation."""

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn
from transformers import LlamaConfig, PreTrainedModel
from transformers.cache_utils import StaticCache

from quantize_hunyuan_mxfp8 import build_quantizer, compatible_cache_initialization, install_replay_forward
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

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        generator = torch.Generator(device=self.device).manual_seed(kwargs["seed"])
        cache = HunyuanStaticCache(config=self.config, max_cache_len=5, dynamic=False)
        for step in range(kwargs["diff_infer_steps"]):
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
        lambda *args: pd.DataFrame({"id": [1, 2], "caption": ["a cat", "a dog"]}),
    )
    model = TinyModel(make_config()).to("cuda:0", dtype=torch.bfloat16).eval()
    override = parse_layer_config_arg("{mlp.experts:{scheme:MXFP4}}") if expert_bits == 4 else None
    args = SimpleNamespace(
        image_size="1024x1024",
        seed=42,
        iters=2,
        nsamples=2,
        device="0",
        steps=3,
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
    assert len(model.calls) == 2
    assert quantizer.model_context.quantized
    assert len(quantizer.calibration.summary) == 2
    assert all(item["sequence_lengths"] == [5, 2, 2, 5, 2, 2] for item in quantizer.calibration.summary.values())
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
