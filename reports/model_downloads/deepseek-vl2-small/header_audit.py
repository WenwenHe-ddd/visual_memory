#!/usr/bin/env python3
"""Audit a sharded safetensors download using only JSON and shard headers.

This does not read tensor payloads, import torch, access the network, or write
files. It checks structure, not payload hashes. Exit status is 0 only on success.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct
import sys


DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
}
MAX_HEADER_BYTES = 100_000_000


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_json(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int


def audit_shard(path, expected_keys):
    file_bytes = path.stat().st_size
    require(file_bytes >= 10, "file is too short to contain a safetensors header")
    with path.open("rb") as stream:
        header_bytes = struct.unpack("<Q", stream.read(8))[0]
        require(2 <= header_bytes <= MAX_HEADER_BYTES, f"invalid header length {header_bytes}")
        require(header_bytes % 8 == 0, "header length is not aligned to 8 bytes")
        require(8 + header_bytes <= file_bytes, "header extends beyond the file")
        raw_header = stream.read(header_bytes)
    require(len(raw_header) == header_bytes, "incomplete header read")
    require(raw_header.startswith(b"{"), "header does not start with a JSON object")
    header = read_json(raw_header.decode("utf-8"))
    require(isinstance(header, dict), "header is not a JSON object")
    metadata = header.pop("__metadata__", {})
    require(isinstance(metadata, dict), "__metadata__ is not an object")
    require(all(isinstance(v, str) for v in metadata.values()), "__metadata__ values must be strings")
    actual_keys = set(header)
    require(
        actual_keys == expected_keys,
        f"tensor key mismatch: missing={sorted(expected_keys - actual_keys)}, "
        f"unexpected={sorted(actual_keys - expected_keys)}",
    )
    payload_bytes = file_bytes - 8 - header_bytes
    intervals = []
    tensor_bytes = 0
    tensor_elements = 0
    dtypes = Counter()
    dtype_bytes = Counter()
    for key, spec in header.items():
        require(isinstance(spec, dict), f"{key}: tensor descriptor is not an object")
        dtype = spec.get("dtype")
        require(isinstance(dtype, str) and dtype in DTYPE_BYTES, f"{key}: unsupported dtype {dtype!r}")
        shape = spec.get("shape")
        require(isinstance(shape, list), f"{key}: shape is not a list")
        require(all(integer(dim) and dim >= 0 for dim in shape), f"{key}: invalid shape {shape!r}")
        offsets = spec.get("data_offsets")
        require(
            isinstance(offsets, list) and len(offsets) == 2 and all(integer(x) for x in offsets),
            f"{key}: invalid data_offsets {offsets!r}",
        )
        start, end = offsets
        require(0 <= start <= end <= payload_bytes, f"{key}: offsets are outside payload bounds")
        elements = math.prod(shape)
        expected_bytes = elements * DTYPE_BYTES[dtype]
        require(end - start == expected_bytes, f"{key}: byte span {end - start} != shape/dtype bytes {expected_bytes}")
        intervals.append((start, end, key))
        tensor_bytes += end - start
        tensor_elements += elements
        dtypes[dtype] += 1
        dtype_bytes[dtype] += expected_bytes
    cursor = 0
    for start, end, key in sorted(intervals):
        require(start == cursor, f"{key}: gap or overlap at byte {cursor}; next offset is {start}")
        cursor = end
    require(cursor == payload_bytes, f"tensor coverage ends at {cursor}, payload length is {payload_bytes}")
    require(tensor_bytes == payload_bytes, "tensor byte sum differs from file payload size")
    return {
        "file": path.name,
        "file_bytes": file_bytes,
        "header_bytes": header_bytes,
        "payload_bytes": payload_bytes,
        "tensor_bytes": tensor_bytes,
        "tensor_count": len(header),
        "tensor_elements": tensor_elements,
        "dtype_tensor_counts": dict(sorted(dtypes.items())),
        "dtype_bytes": dict(sorted(dtype_bytes.items())),
        "metadata": metadata,
        "passed": True,
    }


def audit(model_dir):
    root = model_dir.resolve()
    result = {
        "model_dir": str(root),
        "index_file": "model.safetensors.index.json",
        "audit_method": "JSON index and safetensors headers only; tensor payloads not read",
        "cryptographic_payload_verification": False,
        "passed": False,
        "errors": [],
        "shards": [],
    }
    try:
        index = read_json((root / result["index_file"]).read_text(encoding="utf-8"))
        require(isinstance(index, dict), "index is not an object")
        weight_map = index.get("weight_map")
        require(isinstance(weight_map, dict) and weight_map, "weight_map must be a nonempty object")
        require(all(isinstance(k, str) and k for k in weight_map), "invalid tensor name in weight_map")
        require(all(isinstance(v, str) and v for v in weight_map.values()), "invalid shard name in weight_map")
        require("__metadata__" not in weight_map, "reserved tensor name __metadata__ in index")
        metadata = index.get("metadata")
        require(isinstance(metadata, dict), "index metadata is not an object")
        total_size = metadata.get("total_size")
        require(integer(total_size) and total_size >= 0, "metadata.total_size must be a nonnegative integer")
        result["index_tensor_count"] = len(weight_map)
        result["index_total_size"] = total_size
        shard_names = sorted(set(weight_map.values()))
        result["index_shard_count"] = len(shard_names)
        paths = {}
        for name in shard_names:
            relative = Path(name)
            require(not relative.is_absolute() and ".." not in relative.parts, f"unsafe shard path {name!r}")
            require(relative.suffix == ".safetensors", f"unexpected shard extension: {name!r}")
            path = (root / relative).resolve()
            require(path.is_relative_to(root), f"shard escapes model directory: {name!r}")
            paths[name] = path
        missing = [name for name, path in paths.items() if not path.is_file()]
        require(not missing, f"missing shard files: {missing}")
        extra = sorted(
            str(path.relative_to(root))
            for path in root.glob("*.safetensors")
            if str(path.relative_to(root)) not in shard_names
        )
        require(not extra, f"unexpected root-level safetensors files absent from index: {extra}")
        for name in shard_names:
            expected_keys = {key for key, shard in weight_map.items() if shard == name}
            try:
                shard_result = audit_shard(paths[name], expected_keys)
                shard_result["file"] = name
                result["shards"].append(shard_result)
            except (OSError, ValueError, UnicodeError, struct.error) as exc:
                result["shards"].append({"file": name, "passed": False, "error": str(exc)})
                result["errors"].append(f"{name}: {exc}")
        if not result["errors"]:
            observed_bytes = sum(shard["tensor_bytes"] for shard in result["shards"])
            observed_tensors = sum(shard["tensor_count"] for shard in result["shards"])
            result["observed_tensor_bytes"] = observed_bytes
            result["observed_tensor_count"] = observed_tensors
            result["observed_tensor_elements"] = sum(shard["tensor_elements"] for shard in result["shards"])
            result["total_file_bytes"] = sum(shard["file_bytes"] for shard in result["shards"])
            result["header_and_prefix_bytes"] = sum(8 + shard["header_bytes"] for shard in result["shards"])
            require(observed_bytes == total_size, f"tensor byte sum {observed_bytes} != index total_size {total_size}")
            require(observed_tensors == len(weight_map), "tensor count differs from index")
            result["passed"] = True
    except (OSError, ValueError, UnicodeError) as exc:
        result["errors"].append(str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        nargs="?",
        type=Path,
        default=Path("/storage/hewenwen/VisualMemory/models/deepseek-vl2-small"),
    )
    args = parser.parse_args()
    result = audit(args.model_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
