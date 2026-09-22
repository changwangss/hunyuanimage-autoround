# Validation

Validated on 2026-09-22 against AutoRound commit
`6db9435fbff63cf17e2df2c5a8ca858392df5eab`.

## Executed checks

- Fourteen tests passed in `test_adapter.py`:
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
- The export tests check the resolved expert/shared-MLP/attention schemes,
  per-layer calibration forward counts and sequence lengths, serialized
  quantization settings, and expert weight packing dtype/shape.
  Each runs four prompts with four steps selected from an eight-step schedule,
  verifying exactly 16 cached forwards per decoder block.
- Ruff lint and formatting checks passed for the Python files.
- The CLI help command and pinned test-source SHA256 verification passed.

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
- Reloading the exported full-model checkpoint in an inference engine.
- Generated-image quality and comparison against the original model.

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
