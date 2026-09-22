"""Temporary, per-forward calibration files with lazy per-layer reads."""

import hashlib
import mmap
import shutil
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def compact_tensors(value):
    """Avoid serializing a large backing storage for a small tensor view."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("Calibration snapshots must be on CPU before writing")
        if value.untyped_storage().nbytes() > value.numel() * value.element_size():
            return value.clone()
        return value
    if isinstance(value, dict):
        return {key: compact_tensors(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(compact_tensors(item) for item in value)
    return value


DEDUPLICATED_FIELDS = {
    "attention_mask",
    "position_ids",
    "custom_pos_emb",
    "ar_keys",
    "ar_values",
    "ar_kv_positions",
    "ar_kv_length",
}


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


class DiskCalibrationCache(Mapping):
    """Expose AutoRound's input mapping while keeping unopened layers on disk."""

    def __init__(self, directory, shared_keys=()):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._temporary = TemporaryDirectory(prefix="hunyuan-calib-", dir=directory)
        self.directory = Path(self._temporary.name)
        self.shared_keys = shared_keys
        self.files = {}
        self.summary = {}
        self.bytes_written = 0
        self.payload_bytes_by_field = Counter()
        self.logical_bytes_by_field = Counter()
        self._blobs = self.directory / "shared"
        self._blobs.mkdir()
        self._loaded_name = None
        self._loaded = None

    def _store_shared(self, value, field):
        if isinstance(value, torch.Tensor):
            value = compact_tensors(value).detach().contiguous()
            fingerprint = hashlib.sha256(str((value.dtype, tuple(value.shape))).encode())
            fingerprint.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
            name = fingerprint.hexdigest() + ".pt"
            path = self._blobs / name
            if not path.exists():
                torch.save(value, path)
                self.bytes_written += path.stat().st_size
                self.payload_bytes_by_field[field] += tensor_bytes(value)
            return ("__hunyuan_tensor__", name)
        if isinstance(value, (tuple, list)):
            return type(value)(self._store_shared(item, field) for item in value)
        return value

    def _load_shared(self, value):
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], str)
            and value[0] == "__hunyuan_tensor__"
        ):
            # Load a separate private mapping for every reference. Equal content
            # must not create aliases that let one replay mutate another sample.
            return torch.load(self._blobs / value[1], map_location="cpu", weights_only=True, mmap=True)
        if isinstance(value, dict):
            return {key: self._load_shared(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(self._load_shared(item) for item in value)
        return value

    def read_snapshot(self, path):
        with torch.serialization.set_default_mmap_options(mmap.MAP_PRIVATE):
            return self._load_shared(torch.load(path, map_location="cpu", weights_only=True, mmap=True))

    def report(self):
        fields = {key: round(size / 1024**3, 3) for key, size in self.payload_bytes_by_field.items()}
        return f"{self.bytes_written / 1024**3:.2f} GiB written; unique tensor payload by field (GiB): {fields}"

    def append(self, name, inputs):
        paths = self.files.setdefault(name, [])
        folder = self.directory / name
        folder.mkdir(exist_ok=True)
        path = folder / f"{len(paths):06d}.pt"
        # Leave a small filesystem reserve; torch.save errors are propagated too.
        if shutil.disk_usage(self.directory).free < 64 * 1024**2:
            raise RuntimeError(f"Calibration cache disk is full: {self.directory}")
        if isinstance(inputs, dict):
            stored = {}
            for field, value in inputs.items():
                self.logical_bytes_by_field[field] += tensor_bytes(value)
                if field in DEDUPLICATED_FIELDS:
                    stored[field] = self._store_shared(value, field)
                else:
                    stored[field] = compact_tensors(value)
                    self.payload_bytes_by_field[field] += tensor_bytes(value)
        else:
            stored = compact_tensors(inputs)
            self.logical_bytes_by_field["layer_inputs"] += tensor_bytes(inputs)
            self.payload_bytes_by_field["layer_inputs"] += tensor_bytes(inputs)
        torch.save(stored, path)
        self.bytes_written += path.stat().st_size
        paths.append(path)
        if isinstance(inputs, dict) and "hidden_states" in inputs:
            info = self.summary.setdefault(name, {"forwards": 0, "kv_forwards": 0, "sequence_lengths": []})
            info["forwards"] += len(inputs["hidden_states"])
            info["kv_forwards"] += len(inputs.get("ar_keys", []))
            info["sequence_lengths"].extend(tensor.shape[1] for tensor in inputs["hidden_states"])

    def __getitem__(self, name):
        paths = self.files[name]
        if self._loaded_name == name:
            return self._loaded
        self._loaded_name, self._loaded = None, None
        merged = None
        for path in paths:
            data = self.read_snapshot(path)
            if isinstance(data, list):
                if merged is None:
                    merged = []
                merged.extend(data)
                continue
            if merged is None:
                merged = {}
            for key, value in data.items():
                if key not in merged:
                    merged[key] = value
                elif key in self.shared_keys and value is not None:
                    # Match LLMCalibrator's upgrade of shared tensors to a list
                    # when subsequent forwards supply per-step values.
                    if isinstance(merged[key], list):
                        merged[key].append(value)
                    else:
                        merged[key] = [merged[key], value]
                elif isinstance(value, list):
                    if merged[key] is None:
                        merged[key] = []
                    merged[key].extend(value)
        self._loaded_name, self._loaded = name, merged
        return merged

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __contains__(self, name):
        return name in self.files

    def pop(self, name, *default):
        if name not in self.files:
            if default:
                return default[0]
            raise KeyError(name)
        result = self[name]
        del self.files[name]
        self._loaded_name, self._loaded = None, None
        return result

    def close(self):
        self._loaded_name, self._loaded = None, None
        self.files.clear()
        self._temporary.cleanup()
