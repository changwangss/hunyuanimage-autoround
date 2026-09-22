# Validation

Validated on 2026-09-22 against AutoRound commit
`6db9435fbff63cf17e2df2c5a8ca858392df5eab`.

## Executed checks

- Forty-one tests passed in `test_adapter.py` (118.47 seconds):
  - Diagnostic hooks record denoising/VAE statistics, detect injected block/VAE
    NaNs, and restore hooks/methods after success and errors (three cases).
  - Native Tencent config save/reload changes `model_type` from
    `hunyuan_image_3_moe` to `Hunyuan`; the QDQ CLI accepts the resulting config.
    Non-distilled, missing-distillation and unrelated Hunyuan architectures are
    still rejected (three negative cases).
  - Native Hunyuan router retains FP32 routing after weight dequantization,
    with activation QDQ still enabled.
  - QDQ generation forwards inference settings and saves a PIL result. The
    generator is mocked here; this is not a real image-quality test.
  - Native Tencent Euler scheduler subsets of 1, 2, 4 and all 8 steps: timestep
    membership/order, sigma alignment, MeanFlow next-time conditioning, Euler
    updates, seeded reproducibility and scheduler restoration after exceptions.
  - Explicit CLI step counts, legacy `--steps`, and rejection of a calibration
    budget larger than the full schedule (three cases).
  - Tencent's published image-generation dispatch receives the requested step
    count (3 and 8) and guidance through the adapter, without changing the
    model's default generation config. Tokenization and the image pipeline are
    stubbed for this check.
  - Native Hunyuan static cache with the installed two-argument initializer,
    including initial and subsequent indexed KV updates and patch restoration.
  - The same native cache with a simulated legacy single-argument initializer.
  - Tencent's published SDPA attention with RoPE: initial and later-step KV
    replay agreement, repeated replay, and gradient propagation.
  - CUDA tiny-model W8A8 MXFP8 tuning and `auto_round` export.
  - CUDA tiny-model MXFP8 with W4A4 MXFP4 expert overrides and `auto_round` export.
- Disk-cache checks cover lazy per-layer reads, shared per-step positions,
  tuple kwargs, empty KV snapshots, private-mapping isolation, compact tensor
  views, low-disk-space errors, and temporary-directory cleanup.
- Both MXFP8 and mixed MXFP8/MXFP4 tiny-model integration cases also run with disk
  caching, including real tuning, export, QDQ reload and finite replay outputs.
- A separate two-prompt, full-eight-step comparison verifies exact equality of
  all non-KV cached inputs, the retained KV positions, and final tuned state
  dictionaries between full-memory and compact/deduplicated disk paths.
- Six sequential-block regression cases compare the previous independent-layer
  groups against one chained group, in memory and disk modes, with 1, 4 and 8
  calibration steps selected from an eight-step schedule. They use two prompts,
  native Hunyuan attention/KV replay, and real mixed MXFP8/MXFP4 tuning. Every
  layer's replay hidden states and final tuned state dictionaries match exactly
  (`rtol=0, atol=0`); per-layer auxiliary snapshots also match exactly. Only the
  first layer retains captured hidden states, and disk files are all consumed.
- Native Hunyuan SDPA replay checks cover both full and compact KV, including
  repeated replay and backward gradients. Tensor deduplication checks identical
  content across layers/steps, dtype/shape distinctions, and mutation isolation.
- The export tests check the resolved expert/shared-MLP/attention schemes,
  per-layer calibration forward counts and sequence lengths, serialized
  quantization settings, and expert weight packing dtype/shape.
  Each runs four prompts with four steps selected from an eight-step schedule,
  verifying exactly 16 cached forwards per decoder block.
- Both exported tiny checkpoints are reloaded through the inference script's
  Transformers/AutoRound `backend="torch"` loader. All 10 quantized linears are
  checked for exact packed-weight/scale preservation, per-layer W8A8/W4A4
  selection, and output agreement with independently decoded E2M1/E4M3 weights
  and E8M0 scales plus AutoRound activation QDQ.
  Reloaded blocks also execute three synthetic denoising steps through native
  Hunyuan SDPA attention and its static KV cache, with finite outputs.
  The activation-bypass diagnostic agrees with weight-only reference calculations;
  the original-model mode rejects a quantized checkpoint before loading weights.
- The four export cases additionally compare the actual post-tuning
  `WrapperWALayer` outputs before packing against the reloaded torch MXFP QDQ
  outputs, and compare a three-step synthetic attention/KV trajectory. Both
  comparisons pass with exact equality, for MXFP8 and mixed MXFP8/MXFP4, in
  memory and disk calibration modes (4 targeted cases passed in 31.32 seconds).
- RCEIL diagnostic checks compare both A4 and A8 directly against the INC FLUX
  example's `quant_mx_rceil` primitive, preserving packed weights/scales, each
  layer's bit widths and the original configuration object. Two CLI checks reject
  conflicting BF16/activation-bypass options. A BF16 input group containing
  1.8984375 gives A8 outputs 1.75 with standard MX versus 1.875 with RCEIL; A4
  gives 1.5 versus 2.0. These demonstrate different activation scale behavior,
  not recovery of the full model's image quality.
- Ruff lint and formatting checks passed for the Python files.
- The CLI help command and pinned test-source SHA256 verification passed.

## Synthetic cache memory probe

For the previous full-snapshot version (`7f1ab9b`), a CPU-only probe wrote 768 MiB of synthetic snapshots (16 layers, 8 forwards per
layer, three 2 MiB tensors per forward) in separate fresh processes:

| Mode | RSS before capture | RSS after capture | RSS after reading first layer |
| --- | ---: | ---: | ---: |
| In memory | 486.6 MiB | 1258.1 MiB | 1259.7 MiB |
| On disk | 486.6 MiB | 495.1 MiB | 544.9 MiB |

Both first-layer checksums were identical. These are process RSS readings, not
whole-container memory measurements: filesystem page cache is not included in RSS.
They are historical measurements of the full-snapshot implementation, not a new
memory benchmark of the compact/deduplicated cache.
The probe contains no full-model weights and is not a measurement of the user's
32-prompt Hunyuan run or proof that its container restart is fixed.

## Compact-cache disk probe

A separate synthetic CPU probe used four layers, eight steps, hidden width 4096,
512 updated tokens plus 128 context tokens, 8 KV heads of width 128 and BF16
floating tensors. It compared the original full-snapshot encoding, compact KV
with auxiliary-tensor deduplication, and the new sequential-block organization:

| Encoding | Serialized files |
| --- | ---: |
| Full snapshots | 220.78 MiB |
| Compact KV + deduplication | 135.39 MiB |
| Compact KV + deduplication + sequential blocks | 36.38 MiB |

The first optimization reduced serialized bytes by 38.7%. Sequential blocks
reduce that compact format by a further 73.1%, keeping 33 MiB of first-layer
hidden states instead of 132 MiB across four layers. Other tensor payload sizes
are unchanged. The probe uses synthetic tensors and a much shorter sequence than
the real 1024x1024 model; it does not predict the full-run saving or establish a
new container-memory peak.

## Environment

| Component | Version |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.12.1 |
| Diffusers | 0.39.0 |
| Accelerate | 1.14.0 |
| pytest | 9.1.1 |
| Ruff | 0.15.21 |
| Test GPU | NVIDIA GeForce RTX 5090 |

This records the small-test environment, not a validated dependency lock for the
complete Tencent custom model. Use a working official HunyuanImage environment
for the full model.

## Not yet validated

- Complete HunyuanImage 3 Instruct Distil loading, COCO image-generation
  calibration, and tuning with the actual 80B checkpoint.
- Multi-GPU full-model calibration and its peak CPU/GPU memory requirements.
- Disk savings for the full 1024x1024, 32-prompt, 8-step user run. The old format
  used 143G after five prompts; the new format requires a fresh measured run.
- Reloading the exported full-model checkpoint in an inference engine.
- Complete native Hunyuan QDQ generation from the exported 80B checkpoint.
- Generated-image quality and comparison against the original model.
- Root cause of the user-reported gray image from short smoke calibration.
  The new diagnostics require execution on that checkpoint; local tests do not
  establish whether its issue is numerical failure, loading, or calibration quality.

The tiny-model integration tests use the published Hunyuan static cache,
synthetic denoising inputs, a small MoE-like module, and mocked COCO captions.
The original tests used a simplified cache and missed the Transformers
`lazy_initialization(key_states, value_states)` incompatibility. This was
reproduced against Transformers 5.12.1 and covered by the updated tests.
The generation-dispatch regression also reproduced the old adapter ignoring a
requested 3-step run and using the default 8 steps. The adapter now passes an
explicit generation config; tiny-model calibration consumes that same config.
They do not validate the full Tencent MoE or
native image-generation pipeline. The published attention source is pinned at
Tencent revision `c8ffd07206f1b843697606968196e8f59f8ff38c` and verified by
`prepare_test_reference.py` before testing.
