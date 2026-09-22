"""Small checks only: no claim of full Hunyuan model validation."""

import ast
import importlib.util
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
from PIL import Image
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from quantize_hunyuan_mxfp8 import (
    HunyuanPipeline,
    build_quantizer,
    calibration_schedule,
    compatible_cache_initialization,
    install_replay_forward,
    load_hunyuan_tokenizer,
    parse_args,
    validate_hunyuan_config,
)
from auto_round.utils import parse_layer_config_arg
from auto_round.data_type.utils import get_quant_func
from auto_round.experimental.qmodules.mx import MXFP4QuantLinear, MXFP8QuantLinear
from auto_round.schemes import QuantizationScheme
import infer_hunyuan_qdq as inference


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


def test_native_tokenizer_preserves_checkpoint_backend(monkeypatch, tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast

    source = Path(__file__).parent / "reference/tokenization_hunyuan_image_3.py"
    spec = importlib.util.spec_from_file_location("native_hunyuan_tokenizer_test", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    prompts = ["a cute cat", "a red car", "一只可爱的猫"]
    special = ["<|startoftext|>", "<|endoftext|>", "<pad>"] + [f"<img_ratio_{i}>" for i in range(37)]
    backend.train_from_iterator(prompts * 10, trainers.BpeTrainer(vocab_size=500, special_tokens=special))
    PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token=special[0], eos_token=special[1], pad_token=special[2]
    ).save_pretrained(tmp_path)

    class TinyHunyuan:
        def load_tokenizer(self, path):
            self._tokenizer = module.HunyuanImage3TokenizerFast.from_pretrained(
                path, model_version=self.config.model_version
            )

    TinyHunyuan.__module__ = spec.name
    model = TinyHunyuan()
    model.config = SimpleNamespace()
    load_hunyuan_tokenizer(model, tmp_path)
    assert isinstance(model._tokenizer, module.HunyuanImage3TokenizerFast)
    for prompt in prompts:
        expected = backend.encode(prompt, add_special_tokens=False).ids
        assert model._tokenizer.encode(prompt, add_special_tokens=False) == expected
        assert model._tokenizer.decode(expected, clean_up_tokenization_spaces=False) == prompt
    saved = json.loads((tmp_path / "tokenizer.json").read_text())
    loaded = json.loads(model._tokenizer.backend_tokenizer.to_str())
    for key in ("model", "normalizer", "pre_tokenizer", "post_processor", "decoder"):
        assert loaded[key] == saved[key]


def test_native_config_roundtrip_is_accepted_by_qdq_cli(monkeypatch, tmp_path):
    source = Path(__file__).parent / "reference/configuration_hunyuan_image_3.py"
    spec = importlib.util.spec_from_file_location("native_hunyuan_config_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {
        "model_type": "hunyuan_image_3_moe",
        "architectures": ["HunyuanImage3ForCausalMM"],
        "cfg_distilled": True,
        "use_meanflow": True,
        "quantization_config": {"quant_method": "auto-round", "data_type": "mx_fp", "bits": 8},
    }
    validate_hunyuan_config(config)
    directory = tmp_path / "quantized_model"
    module.HunyuanImage3Config.from_dict(config).save_pretrained(directory)
    saved = json.loads((directory / "config.json").read_text())
    assert saved["model_type"] == "Hunyuan"
    validate_hunyuan_config(saved)
    loader = Mock(side_effect=RuntimeError("reached checkpoint loader"))
    monkeypatch.setattr(inference, "load_qdq_model", loader)
    monkeypatch.setattr(sys, "argv", ["infer_hunyuan_qdq.py", "--model", str(directory), "--prompt", "a dog"])
    with pytest.raises(RuntimeError, match="reached checkpoint loader"):
        inference.main()
    loader.assert_called_once()
    loader.reset_mock()
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--bf16"])
    with pytest.raises(SystemExit) as error:
        inference.main()
    assert error.value.code == 2
    loader.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"cfg_distilled": False},
        {"cfg_distilled": None},
        {"architectures": ["OtherHunyuanModel"]},
    ],
)
def test_config_check_still_rejects_wrong_models(change):
    config = {"model_type": "Hunyuan", "architectures": ["HunyuanImage3ForCausalMM"], "cfg_distilled": True}
    config.update(change)
    with pytest.raises(ValueError, match="model_type=.*architectures=.*cfg_distilled="):
        validate_hunyuan_config(config)


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


@pytest.mark.parametrize("failure", [None, "block", "vae"])
def test_generation_diagnostics_detect_and_restore(tmp_path, failure):
    class VAE:
        def decode(self, latents, **kwargs):
            return (latents * (float("nan") if failure == "vae" else 2),)

    scheduler = native_scheduler()
    scheduler.set_timesteps(2)
    block = nn.Identity()
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[block]), pipeline=SimpleNamespace(scheduler=scheduler), vae=VAE()
    )
    original_step, original_decode = scheduler.step, model.vae.decode
    report = tmp_path / "diagnostics.json"

    def run():
        with inference.diagnose_generation(model, report):
            hidden = torch.tensor([float("nan") if failure == "block" else 1.0])
            block(hidden)
            latents = scheduler.step(torch.ones(1), scheduler.timesteps[0], hidden, return_dict=False)[0]
            model.vae.decode(latents, return_dict=False)

    if failure:
        with pytest.raises(RuntimeError, match="Non-finite values"):
            run()
    else:
        run()
    records = json.loads(report.read_text())
    if failure:
        assert "model.layers.0" in records[-1]["error"] if failure == "block" else "VAE output" in records[-1]["error"]
    else:
        assert len(records) == 5
        assert records[0]["name"].endswith("prediction")
        assert records[-1]["name"] == "VAE output"
    assert scheduler.step == original_step and model.vae.decode == original_decode
    assert not block._forward_hooks


def test_qdq_router_preserves_native_fp32_routing():
    source = Path(__file__).parent / "reference/modeling_hunyuan_image_3.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HunyuanTopKGate")
    env = {"torch": torch, "nn": nn, "F": torch.nn.functional}
    exec(compile("from __future__ import annotations\n" + ast.unparse(node), str(source), "exec"), env)
    gate_class = env["HunyuanTopKGate"]
    gate = gate_class.__new__(gate_class)
    nn.Module.__init__(gate)
    gate.moe_topk = 2
    scheme = QuantizationScheme(
        bits=8, act_bits=8, data_type="mx_fp", act_data_type="mx_fp", group_size=32, act_group_size=32, act_dynamic=True
    )
    gate.wg = MXFP8QuantLinear(64, 4, scheme, dtype=torch.float32)
    torch.manual_seed(123)
    gate.wg.weight.copy_(torch.randn(4, 64).to(torch.float8_e4m3fn))
    gate.wg.weight_scale.fill_(124)
    model = nn.Module()
    model.model = nn.Module()
    model.model.mlp = nn.Module()
    model.model.mlp.gate = gate
    expected_weight = gate.wg.weight.float() * 0.125
    inference.preserve_router_precision(model)
    assert gate.wg.weight.dtype == torch.float32
    assert not gate.wg.pre_dequantized_input
    torch.testing.assert_close(gate.wg.weight, expected_weight)
    x = torch.randn(1, 5, 64, dtype=torch.bfloat16)
    qdq, _ = get_quant_func(dtype="mx_fp", bits=8, sym=True)
    qx = qdq(tensor=x.float().reshape(-1, 64), bits=8, group_size=32)[0]
    expected = gate.easy_topk(torch.nn.functional.linear(qx, expected_weight), 2)
    actual = gate(x, topk_impl="easy")
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("bits", [4, 8])
def test_rceil_activation_matches_flux_reference(bits):
    from auto_round.data_type.mxfp import quant_mx_rceil

    scheme = QuantizationScheme(
        bits=bits,
        act_bits=bits,
        data_type="mx_fp",
        act_data_type="mx_fp",
        group_size=32,
        act_group_size=32,
        act_dynamic=True,
    )
    cls = MXFP4QuantLinear if bits == 4 else MXFP8QuantLinear
    layer = cls(32, 32, scheme, dtype=torch.bfloat16)
    layer.weight_scale.fill_(125)
    if bits == 4:
        layer.weight_packed.fill_(0x12)
    else:
        layer.weight.fill_(1.0)
    model = nn.Sequential(layer)
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    x = torch.zeros(1, 32, dtype=torch.bfloat16)
    x[0, 0] = 1.9
    standard = layer(x)
    inference.use_rceil_activation_qdq(model)
    expected_x, _, _ = quant_mx_rceil(x, bits=bits, group_size=32, data_type="mx_fp_rceil")
    expected = torch.nn.functional.linear(expected_x, layer.dequant_weight_online().to(x.dtype))
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
    assert not torch.equal(standard, expected)
    assert layer.config.act_bits == bits and layer.config.bits == bits
    assert scheme.act_data_type == "mx_fp"
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.float(), saved[key].float(), rtol=0, atol=0)
    inference.set_activation_qdq(model, enabled=False)
    torch.testing.assert_close(
        layer(x), torch.nn.functional.linear(x, layer.dequant_weight_online().to(x.dtype)), rtol=0, atol=0
    )


@pytest.mark.parametrize("conflict", ["--bf16", "--disable-act-quant"])
def test_rceil_cli_rejects_conflicting_modes(monkeypatch, conflict):
    monkeypatch.setattr(
        sys,
        "argv",
        ["infer_hunyuan_qdq.py", "--model", "unused", "--prompt", "a cute cat", "--act-qdq", "rceil", conflict],
    )
    with pytest.raises(SystemExit) as error:
        inference.main()
    assert error.value.code == 2


def test_qdq_generation_uses_full_inference_steps(monkeypatch, tmp_path):
    model = TinyModel(make_config())
    image = Image.new("RGB", (8, 8), "green")
    generate = Mock(return_value=(None, [image]))
    monkeypatch.setattr(model, "generate_image", generate)
    original_update = HunyuanStaticCache.update
    result = inference.generate_image(model, "a tree", num_inference_steps=8, guidance_scale=5.0, seed=7)
    result.save(tmp_path / "qdq.png")
    with Image.open(tmp_path / "qdq.png") as saved:
        assert saved.size == (8, 8)
    kwargs = generate.call_args.kwargs
    assert kwargs["generation_config"].diff_infer_steps == 8
    assert kwargs["generation_config"].diff_guidance_scale == 5.0
    assert kwargs["seed"] == 7 and kwargs["bot_task"] == "image"
    assert not kwargs["use_taylor_cache"]
    assert model.generation_config.diff_guidance_scale == 2.5
    assert HunyuanStaticCache.update is original_update


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


@pytest.mark.parametrize("compact", [False, True])
def test_native_attention_replay_preserves_context_and_gradients(compact):
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
        extra = {}
        if compact:
            keep = torch.tensor([[0, 1, 3]])
            extra = {"ar_kv_positions": keep, "ar_kv_length": torch.tensor([5])}
            keys, values = keys[:, :, keep[0]], values[:, :, keep[0]]
        actual = block(
            query,
            position_ids=positions,
            ar_keys=keys,
            ar_values=values,
            custom_pos_emb=rotary(positions, query.dtype),
            **extra,
        )[0]
        torch.testing.assert_close(actual, expected)
        again = block(
            query,
            position_ids=positions,
            ar_keys=keys,
            ar_values=values,
            custom_pos_emb=rotary(positions, query.dtype),
            **extra,
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
        self.post_init()

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
        return h


@pytest.mark.parametrize("expert_bits,disk_cache", [(8, False), (4, False), (8, True), (4, True)])
def test_actual_autoround_mxfp8_tuning_and_export(monkeypatch, tmp_path, expert_bits, disk_cache):
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
        calib_cache_dir=tmp_path / "cache" if disk_cache else None,
    )
    quantizer, originals = build_quantizer(model, args)
    assert quantizer.model_context.is_diffusion
    for name, cfg in quantizer.layer_config.items():
        if ".mlp.experts." in name:
            assert cfg["bits"] == cfg["act_bits"] == expert_bits
        if ".mlp.shared_mlp" in name or ".self_attn." in name:
            assert cfg["bits"] == cfg["act_bits"] == 8
    quantizer.quantize()
    if disk_cache:
        cache = quantizer.calibration.disk_cache
        assert cache.bytes_written > 0
        assert len(cache) == 0  # every layer was consumed by the real orchestrator
        cache_dir = cache.directory
        quantizer.calibration.close()
        assert not cache_dir.exists()
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
    # Compare AutoRound's post-tuning fake-quant path with the reloaded torch
    # backend, not just the packed checkpoint with another packed decoder.
    from auto_round.wrapper import WrapperWALayer

    model.to("cuda:0")
    probe = torch.randn(1, 3, 64, device="cuda:0", dtype=torch.bfloat16)
    with torch.no_grad(), compatible_cache_initialization(HunyuanStaticCache):
        before_export = {
            name: layer(probe).detach().cpu()
            for name, layer in model.named_modules()
            if isinstance(layer, WrapperWALayer)
        }
        before_export_generation = (
            model.generate_image(seed=42, generation_config=SimpleNamespace(diff_infer_steps=3)).detach().cpu()
        )
        tuned_weights = {
            name: layer.weight.detach().cpu().clone()
            for name, layer in model.named_modules()
            if isinstance(layer, WrapperWALayer)
        }
    # Exercise AutoRound's real fake exporter using the very same tuned weights.
    # save_quantized caches its selected formats; change that selection before
    # the second export so it actually invokes the packed exporter below.
    from safetensors.torch import load_file

    quantizer.save_quantized(str(tmp_path / "fake"), format="fake", inplace=True)
    fake_weights = {}
    for shard in (tmp_path / "fake").glob("*.safetensors"):
        fake_weights.update(load_file(shard))
    for name, weight in tuned_weights.items():
        torch.testing.assert_close(fake_weights[name + ".weight"], weight, rtol=0, atol=0)
        assert name + ".weight_scale" not in fake_weights
        assert name + ".weight_packed" not in fake_weights
    quantizer.formats = "auto_round"
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

    # Reload the actual export through the same Transformers/AutoRound QDQ loader.
    # The tiny Llama config needs its custom attention settings restored explicitly.
    config = make_config()
    config.quantization_config = exported
    monkeypatch.setattr(inference, "AutoModelForCausalLM", TinyModel)
    loaded = inference.load_qdq_model(tmp_path / "export", device_map="cuda:0", config=config)
    x = torch.randn(1, 3, 64, device="cuda:0", dtype=torch.bfloat16)
    checked = 0
    for name, original in model.named_modules():
        if not hasattr(original, "weight_scale"):
            continue
        layer = loaded.get_submodule(name)
        torch.testing.assert_close(layer(probe).cpu(), before_export[name], rtol=0, atol=0)
        fake_weight = fake_weights[name + ".weight"]
        torch.testing.assert_close(
            layer.dequant_weight_online().to(device="cpu", dtype=fake_weight.dtype), fake_weight, rtol=0, atol=0
        )
        bits = original.bits
        assert isinstance(layer, MXFP4QuantLinear if bits == 4 else MXFP8QuantLinear)
        weight_name = "weight_packed" if bits == 4 else "weight"
        packed = getattr(original, weight_name).to(x.device)
        torch.testing.assert_close(getattr(layer, weight_name).float(), packed.float(), rtol=0, atol=0)
        torch.testing.assert_close(layer.weight_scale, original.weight_scale.to(x.device), rtol=0, atol=0)
        assert layer.config.bits == layer.config.act_bits == bits
        if bits == 4:
            codes = torch.stack([packed & 15, packed >> 4], dim=-1).flatten(-2).long()
            values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device)
            weight = values[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
        else:
            weight = packed.float()
        scales = 2.0 ** (original.weight_scale.to(x.device).float() - 127)
        weight = (weight.reshape(weight.shape[0], -1, 32) * scales[..., None]).reshape(weight.shape)
        qdq, _ = get_quant_func(dtype="mx_fp", bits=bits, sym=True)
        qx = qdq(tensor=x, bits=bits, group_size=32)[0]
        bias = original.bias.to(x) if original.bias is not None else None
        expected = torch.nn.functional.linear(qx, weight.to(x.dtype), bias)
        torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
        inference.set_activation_qdq(loaded, enabled=False)
        expected_weight_only = torch.nn.functional.linear(x, weight.to(x.dtype), bias)
        torch.testing.assert_close(layer(x), expected_weight_only, rtol=0, atol=0)
        inference.set_activation_qdq(loaded, enabled=True)
        checked += 1
    assert checked == 10
    with compatible_cache_initialization(HunyuanStaticCache):
        output = loaded.generate_image(seed=42, generation_config=SimpleNamespace(diff_infer_steps=3))
    assert output.shape == (1, 2, 64) and torch.isfinite(output).all()
    torch.testing.assert_close(output.cpu(), before_export_generation, rtol=0, atol=0)


@pytest.mark.parametrize("failure", [False, True])
def test_disk_cache_roundtrip_and_cleanup(tmp_path, failure):
    from calibration_cache import DiskCalibrationCache

    cache = DiskCalibrationCache(tmp_path, shared_keys=("position_ids",))
    folder = cache.directory
    original = torch.arange(12).reshape(1, 3, 4).float()
    try:
        for index in range(3):
            cache.append(
                "model.layers.0",
                {
                    "hidden_states": [original + index],
                    "ar_keys": [torch.empty(1, 2, 0, 4)],
                    "ar_values": [torch.empty(1, 2, 0, 4)],
                    "position_ids": torch.tensor([[index]]),
                    "custom_pos_emb": [(torch.ones(1, 3, 4) * index, torch.zeros(1, 3, 4))],
                    "use_cache": False,
                },
            )
        cache.append("model.layers.1", {"hidden_states": [original]})
        assert cache._loaded is None
        assert list(cache) == ["model.layers.0", "model.layers.1"]
        loaded = cache["model.layers.0"]
        assert len(loaded["hidden_states"]) == 3
        assert [item.item() for item in loaded["position_ids"]] == [0, 1, 2]
        for index, tensor in enumerate(loaded["hidden_states"]):
            torch.testing.assert_close(tensor, original + index)
        assert cache.pop("model.layers.0") is loaded
        assert cache._loaded is None
        assert list(cache) == ["model.layers.1"]
        assert cache.pop("input_ids", None) is None
        loaded["hidden_states"][0].zero_()
        # The mapping is private, so replay mutations cannot modify saved data.
        saved = cache.read_snapshot(folder / "model.layers.0/000000.pt")
        torch.testing.assert_close(saved["hidden_states"][0], original)
        if failure:
            raise RuntimeError("injected calibration failure")
    except RuntimeError as error:
        assert failure and str(error) == "injected calibration failure"
    finally:
        cache.close()
    assert not folder.exists()


def test_disk_cache_compacts_views_and_reports_disk_full(tmp_path, monkeypatch):
    from calibration_cache import DiskCalibrationCache

    cache = DiskCalibrationCache(tmp_path)
    try:
        small_view = torch.arange(1024 * 1024)[:4]
        cache.append("layer", [small_view])
        assert cache.bytes_written < 10000
        torch.testing.assert_close(cache["layer"][0], small_view)
        monkeypatch.setattr("calibration_cache.shutil.disk_usage", lambda path: SimpleNamespace(free=0))
        with pytest.raises(RuntimeError, match="disk is full"):
            cache.append("other", [small_view])
    finally:
        cache.close()


def test_disk_and_memory_calibration_and_tuning_agree(monkeypatch, tmp_path):
    from copy import deepcopy

    monkeypatch.setattr(
        "auto_round.compressors.diffusion.dataset._load_coco_dataframe",
        lambda *args: pd.DataFrame({"id": [1, 2], "caption": ["a cat", "a dog"]}),
    )
    results = []
    captures = []
    for disk in (False, True):
        torch.manual_seed(123)
        model = TinyModel(make_config()).to("cuda:0", dtype=torch.bfloat16).eval()
        args = SimpleNamespace(
            image_size="1024x1024",
            seed=42,
            iters=2,
            nsamples=2,
            device="0",
            num_inference_steps=8,
            calib_num_inference_steps=8,
            guidance_scale=5.0,
            layer_config=parse_layer_config_arg("{mlp.experts:{scheme:MXFP4}}"),
            calib_cache_dir=tmp_path / "cache" if disk else None,
        )
        quantizer, _ = build_quantizer(model, args)
        original_calib = quantizer.calibration.calib

        def capture(nsamples, bs):
            original_calib(nsamples, bs)
            # This test deliberately materializes the tiny snapshots for equality;
            # production retains paths only until the orchestrator requests a layer.
            captures.append(deepcopy(dict(quantizer.calibration.inputs)))

        monkeypatch.setattr(quantizer.calibration, "calib", capture)
        try:
            quantizer.quantize()
            results.append({name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()})
        finally:
            quantizer.calibration.close()
    for name, original in captures[0].items():
        compact = captures[1][name]
        for key in original:
            if key not in ("ar_keys", "ar_values"):
                torch.testing.assert_close(original[key], compact[key], rtol=0, atol=0)
        for index, keys in enumerate(original["ar_keys"]):
            if keys.shape[2] == 0:
                assert compact["ar_kv_length"][index].item() == 0
                continue
            positions = compact["ar_kv_positions"][index][0]
            assert positions.tolist() == [0, 1, 3]
            torch.testing.assert_close(compact["ar_keys"][index], keys[:, :, positions], rtol=0, atol=0)
            torch.testing.assert_close(
                compact["ar_values"][index], original["ar_values"][index][:, :, positions], rtol=0, atol=0
            )
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


def test_disk_cache_deduplicates_equal_auxiliaries_without_aliasing(tmp_path):
    from calibration_cache import DiskCalibrationCache, tensor_bytes

    cache = DiskCalibrationCache(tmp_path)
    mask = torch.ones(1, 1, 128, 128, dtype=torch.bool)
    try:
        for layer in range(4):
            for step in range(3):
                cache.append(f"layer.{layer}", {"attention_mask": [mask.clone()]})
        assert cache.logical_bytes_by_field["attention_mask"] == 12 * tensor_bytes(mask)
        assert cache.payload_bytes_by_field["attention_mask"] == tensor_bytes(mask)
        assert len(list((cache.directory / "shared").glob("*.pt"))) == 1
        loaded = cache["layer.0"]
        loaded["attention_mask"][0].zero_()
        assert loaded["attention_mask"][1].all()
        assert cache["layer.1"]["attention_mask"][0].all()
        # Identical bytes but different shapes/dtypes must remain different blobs.
        cache.append("layer.4", {"attention_mask": [mask.reshape(1, 128, 128), mask.to(torch.uint8)]})
        assert len(list((cache.directory / "shared").glob("*.pt"))) == 3
    finally:
        cache.close()


@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("calib_steps", [1, 4, 8])
def test_chained_blocks_match_independent_blocks(monkeypatch, tmp_path, disk, calib_steps):
    from copy import deepcopy
    from calibration_cache import tensor_bytes

    monkeypatch.setattr(
        "auto_round.compressors.diffusion.dataset._load_coco_dataframe",
        lambda *args: pd.DataFrame({"id": [1, 2], "caption": ["a cat", "a dog"]}),
    )
    snapshots, replay_inputs, weights, sizes = [], [], [], []
    for chained in (False, True):
        torch.manual_seed(123)
        model = TinyModel(make_config()).to("cuda:0", dtype=torch.bfloat16).eval()
        args = SimpleNamespace(
            image_size="1024x1024",
            seed=42,
            iters=2,
            nsamples=2,
            device="0",
            num_inference_steps=8,
            calib_num_inference_steps=calib_steps,
            guidance_scale=5.0,
            layer_config=parse_layer_config_arg("{mlp.experts:{scheme:MXFP4}}"),
            calib_cache_dir=tmp_path / "cache" if disk else None,
        )
        quantizer, _ = build_quantizer(model, args)
        if not chained:
            # Reproduce the previously shipped independent-layer organization.
            quantizer.quant_block_list = [[f"model.layers.{i}"] for i in range(2)]
            quantizer.has_variable_block_shape = False
            quantizer.calibration.has_variable_block_shape = False
        original_calib = quantizer.calibration.calib
        original_compress = quantizer.alg_composer.compress_block
        observed = {}

        def capture(nsamples, bs):
            original_calib(nsamples, bs)
            snapshots.append(deepcopy(dict(quantizer.calibration.inputs)))
            if disk:
                sizes.append(quantizer.calibration.disk_cache.bytes_written)

        def compress(block, fp_inputs, input_others, **kwargs):
            observed[kwargs["block_ctx"].block_name] = deepcopy(fp_inputs)
            return original_compress(block, fp_inputs, input_others, **kwargs)

        monkeypatch.setattr(quantizer.calibration, "calib", capture)
        monkeypatch.setattr(quantizer.alg_composer, "compress_block", compress)
        try:
            quantizer.quantize()
            replay_inputs.append(observed)
            weights.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
            if disk:
                assert len(quantizer.calibration.disk_cache) == 0
        finally:
            quantizer.calibration.close()
    assert "hidden_states" in snapshots[1]["model.layers.0"]
    assert "hidden_states" not in snapshots[1]["model.layers.1"]
    baseline_hidden = sum(tensor_bytes(value.get("hidden_states")) for value in snapshots[0].values())
    chained_hidden = sum(tensor_bytes(value.get("hidden_states")) for value in snapshots[1].values())
    assert chained_hidden * 2 == baseline_hidden
    for name, baseline in snapshots[0].items():
        for key, value in baseline.items():
            if key != "hidden_states":
                torch.testing.assert_close(value, snapshots[1][name][key], rtol=0, atol=0)
        # AutoRound must pass the preceding full-precision reference output,
        # using this layer's own KV context rather than the preceding layer's.
        actual = replay_inputs[1][name]
        actual = actual["hidden_states"] if isinstance(actual, dict) else actual
        torch.testing.assert_close(actual, baseline["hidden_states"], rtol=0, atol=0, check_device=False)
    torch.testing.assert_close(weights[0], weights[1], rtol=0, atol=0)
    if disk:
        assert sizes[1] < sizes[0]
