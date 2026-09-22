# HunyuanImage 3 Instruct Distil: MXFP8 with AutoRound

Standalone experimental Python API script. It does not edit AutoRound source files.
The implementation targets AutoRound main `6db9435fbff63cf17e2df2c5a8ca858392df5eab`
and Tencent's native checkpoint revision `c8ffd07206f1b843697606968196e8f59f8ff38c`.

Model: [tencent/HunyuanImage-3.0-Instruct-Distil](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil).
Quantization library: [intel/auto-round](https://github.com/intel/auto-round).
This is an experimental standalone adapter, not an official Tencent or AutoRound integration.

## Run

Use the environment in which the official model's `generate_image()` already works,
with AutoRound main and its MXFP export dependencies installed. The model directory
must contain the complete native checkpoint, Python files, tokenizer, and assets.
Tencent requires a directory name without dots, for example `HunyuanImage-3-Instruct-Distil`.

Start with a small smoke run:

```bash
python quantize_hunyuan_mxfp8.py \
  --model /path/to/HunyuanImage-3-Instruct-Distil \
  --output /path/to/HunyuanImage-3-Instruct-Distil-MXFP8-smoke \
  --nsamples 1 --num_inference_steps 8 --calib_num_inference_steps 4 --iters 2
```

Then use a separate output directory for tuning:

```bash
python quantize_hunyuan_mxfp8.py \
  --model /path/to/HunyuanImage-3-Instruct-Distil \
  --output /path/to/HunyuanImage-3-Instruct-Distil-MXFP8 \
  --nsamples 8 --num_inference_steps 8 --calib_num_inference_steps 4 --iters 200 \
  --layer_config '{mlp.experts:{scheme:MXFP4}}' \
  --image-size 1024x1024 --device 0
```

The optional `--layer_config` above sets routed experts to MXFP4 (W4A4), while
attention and the shared MLP retain MXFP8 (W8A8). Omit it for uniform MXFP8.
The equivalent Python API argument is
`layer_config={"mlp.experts": {"scheme": "MXFP4"}}`.

`--num_inference_steps` defines the full native timestep schedule (default: 8).
`--calib_num_inference_steps` selects how many of those steps are actually executed
and cached per prompt (default: all). For example, 8 and 4 build an eight-step
schedule and run four selected steps. This does not execute all eight steps or
rebuild a new four-step schedule. The resulting latent trajectory differs from
full eight-step generation.

Sampling preserves the first and last timesteps and chooses one step from each
interior stratum, with seed `--seed + prompt_index`. A one-step budget uses only
the first timestep. The selected sigmas and MeanFlow next-time conditioning are
updated together. The full schedule and model defaults are preserved outside the
call. Selected indices, timesteps and seeds are saved in `calibration_recipe.json`.
Tencent's native progress bar may still display the full schedule length.

Both options also accept hyphenated names. The legacy `--steps N` sets both counts
to N and cannot be combined with either explicit option. Calibration steps must
be positive and no greater than the full schedule length.

`--nsamples` counts COCO captions, not denoising steps. Four captions with four
selected steps yield 16 calibration forwards per decoder block. This is a starting recipe,
not a measured quality recommendation. The script uses direct text-to-image
generation (`bot_task="image"`), not CoT, prompt rewriting, or image editing.

Weights load with Accelerate `device_map="auto"` across visible GPUs; `--device`
selects the tuning GPU. All model weights must fit on the visible GPUs during
calibration, with headroom for generation. CPU/disk weight offload is not supported
by this adapter. `--max-memory '{"0":"70GiB","1":"70GiB"}'` can reserve GPU
headroom (extend this example to include all GPUs available for this model).
After calibration, AutoRound moves weights to CPU and tunes blocks individually;
allow host RAM for the full model plus calibration snapshots. This script is not
an optimized low-memory loader for an 80B model.

## What is quantized and exported

- `scheme="MXFP8"`: 8-bit MX floating-point weights and dynamic 8-bit activations,
  group size 32. `--layer_config` can override this for matched layers.
- Each `model.layers.N` decoder block is a separate tuning group. Eager MoE keeps
  experts as Linear modules so AutoRound can quantize their projections.
- VAE, ViT, alignment modules, and the language output head stay outside the target
  blocks. This is not a claim that every model parameter becomes 8-bit.
- Export is explicitly `format="auto_round"`, preserving the native Transformers
  root checkpoint layout. It is not `llm_compressor`, fake quantization, or a newly
  created Diffusers model folder.
- The original model's config, custom code and tokenizer support files are retained.
  Loading the exported custom architecture with a specific inference engine and
  checking generated-image quality are separate validation steps.

## Generate an image with QDQ

Use the exported quantized directory with the same working Hunyuan/AutoRound
environment. The script selects AutoRound's PyTorch QDQ backend and reads each
layer's saved quantization settings, including MXFP4 expert overrides:

```bash
python infer_hunyuan_qdq.py \
  --model /path/to/HunyuanImage-3-Instruct-Distil-MXFP8 \
  --prompt "A brown and white dog running on green grass, realistic photography" \
  --output outputs/qdq_seed42.png \
  --num_inference_steps 8 --image-size 1024x1024 \
  --guidance-scale 5.0 --seed 42
```

This loads the saved packed weights/scales without quantizing the original model
again. Each linear dequantizes its weights and dynamically applies activation
QDQ before floating-point matmul. Only the small MoE router weights are
pre-dequantized to FP32 to preserve Hunyuan's dtype-dependent router behavior;
their activation QDQ remains enabled. Other weights stay packed between calls.
No vLLM-Omni or low-bit GEMM backend is used.

`--num_inference_steps` is the actual full generation length here; calibration
sampling is not applied. The script saves the image and a JSON sidecar with the
prompt, model path, seed and generation settings. Compare with the original model
using the same prompt, seed, image size, guidance and inference steps, direct
`bot_task="image"`, `use_system_prompt="en_unified"`, and Taylor cache disabled.
Use `--seed 43` and a different output name to inspect more samples.

All packed weights must fit on the visible GPUs, with additional memory for
temporary dequantized weights and activations. `--max-memory` accepts the same
GPU budgets as quantization. This is a quality-checking path and can be slow;
it does not measure production low-bit inference speed. Full 80B generation has
not been validated locally.

### Diagnose a gray or invalid image

Add `--debug` to the same inference command. It checks each decoder block for
NaN/Inf and records each denoising prediction, latent before/after the scheduler,
and VAE input/output statistics. The report is written beside the image as
`<output-stem>.debug.json`, including on generation errors. Debug checks add GPU
synchronization and are slower. Console output also includes dependency versions,
calibration settings (when available), and the final RGB mean and standard deviation.

Keep the prompt, seed, image size, inference steps and guidance fixed for these
comparisons, changing the output name for each run:

- Normal QDQ: `--model quantized_model --debug --output outputs/qdq.png`.
- Weight-only diagnostic: add `--disable-act-quant`, with output
  `outputs/weight_only.png`. Saved quantized weights/scales stay unchanged; only
  activation QDQ is bypassed. This is not a BF16 baseline or the target W8A8/W4A4 run.
- Original-model reference: `--model /path/to/original-model --bf16 --debug
  --output outputs/reference.png`. This uses the same native generation settings
  and preserves the original checkpoint's mixed dtypes. It requires enough memory
  for the original weights; `--bf16` rejects quantized checkpoints.

A block/prediction failure points to a problem before VAE decoding; finite VAE
inputs followed by invalid VAE outputs localize the failure to decoding. If
weight-only recovers the picture, activation quantization is implicated, but this
does not by itself distinguish QDQ implementation from quantization sensitivity.
Short calibration is not sufficient evidence to attribute a gray image to tuning.

## Why there is an adapter

AutoRound currently classifies this checkpoint as MLLM. A small DiffusionPipeline
wrapper calls its native `generate_image()` and routes COCO prompts to diffusion
calibration. Two temporary constructor patches bypass the generic Diffusers loader
and preserve the native mixed dtypes. These are process-local and restored immediately.

The script also fills a missing `model_version` with Tencent's Instruct default
before loading the tokenizer. During native image generation, a scoped cache
adapter supplies both key and value tensors when the installed Transformers
`StaticLayer.lazy_initialization` requires them. Older single-argument
initializers keep the native path. No installed Transformers or model source
files are modified.

Hunyuan's later denoising steps reuse per-layer text KV state. The stock generic
calibration collector drops the custom cache object. This script captures each
layer's KV tensors before its forward, and reconstructs the static-cache update
for replay. Every replay uses fresh tensors and supports gradients. Each decoder
layer has its own calibration group, so another layer's KV state cannot be reused
accidentally. Taylor cache is disabled so every denoising step executes.

## Validation boundary

`test_adapter.py` checks cached attention replay against Tencent's published SDPA
implementation, and runs small-model CUDA MXFP8 tuning and AutoRound export.
The tiny model supplies synthetic denoising inputs and mocked COCO captions.
These checks do not replace running the complete 80B model, checkpoint reload,
or image-quality evaluation. See [VALIDATION.md](VALIDATION.md) for the test results
and environment. Tests require a CUDA GPU and pytest.

The test reference source is downloaded from a pinned Tencent revision and checked
against its SHA256. It is not bundled with this repository. Model weights and
generated checkpoints are not included either.

```bash
python prepare_test_reference.py
PYTHONPATH=/path/to/auto-round python -m pytest test_adapter.py -q
```
