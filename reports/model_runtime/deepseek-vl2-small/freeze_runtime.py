#!/usr/bin/env python3
"""Freeze the completed VL2 smoke run; never start inference or access a GPU.

Run only after smoke.json has been written. Large weight SHA-256 values are
reused from the completed download verification, explicitly labelled as such;
small checkpoint files, current shard headers, and runtime sources are hashed
at freeze time. No model weights are loaded and no shared environment is edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import struct
import subprocess
import sys
from datetime import datetime, timezone


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DOWNLOAD = ROOT / "reports/model_downloads/deepseek-vl2-small"
OFFICIAL = ROOT / ".vendor/deepseek_vl2_source"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def option(argv: list[str], name: str, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


ENVIRONMENT_PROBE = r'''
import importlib.metadata as metadata
import importlib.util
import json
import platform
import sys
import torch
names = ["torch", "torchvision", "vllm", "transformers", "tokenizers", "timm",
         "huggingface-hub", "safetensors", "numpy", "Pillow", "einops",
         "sentencepiece", "triton", "flashinfer-python", "flash-attn"]
packages = {}
for name in names:
    try:
        dist = metadata.distribution(name)
        packages[name] = {"version": dist.version,
                          "distribution_root": str(dist.locate_file("").resolve())}
    except metadata.PackageNotFoundError:
        packages[name] = {"installed": False}
roots = {}
for name in ["vllm", "transformers", "timm", "torch", "torchvision"]:
    spec = importlib.util.find_spec(name)
    roots[name] = str(__import__("pathlib").Path(spec.origin).resolve().parent)
print(json.dumps({"python_executable": sys.executable,
                  "python_version": sys.version, "platform": platform.platform(),
                  "packages": packages, "package_roots": roots,
                  "torch_build_version": str(torch.__version__),
                  "torch_compiled_cuda": torch.version.cuda,
                  "cuda_initialized_by_probe": torch.cuda.is_initialized()}))
'''


def freeze(smoke_path: Path, output_path: Path) -> dict:
    require(smoke_path.is_file(), f"Smoke record must exist first: {smoke_path}")
    smoke = read_json(smoke_path)
    require(isinstance(smoke, dict) and bool(smoke), "Smoke record must be a nonempty JSON object")
    for field in ("passed", "success", "ok"):
        require(smoke.get(field) is not False, f"Smoke record explicitly reports {field}=false")
    require(smoke.get("status") not in ("failed", "failure", "error"), "Smoke record reports failure")
    require(not output_path.exists(), f"Refusing to replace an existing frozen manifest: {output_path}")

    launcher_path = HERE / "launch_server.py"
    # The trusted project launcher defines args/env at module level; its guarded
    # subprocess call does not run under run_path's non-__main__ module name.
    launcher = runpy.run_path(str(launcher_path), run_name="vl2_manifest_launcher")
    argv = list(launcher["args"])
    launch_env = dict(launcher["env"])
    runtime_python = str(launcher["PYTHON"])
    model_dir = Path(option(argv, "--model")).resolve()
    gpu_uuid = launch_env.get("CUDA_VISIBLE_DEVICES", "")
    require(gpu_uuid.startswith("GPU-") and "," not in gpu_uuid, "Launcher must select one GPU UUID")
    require(option(argv, "--dtype") == "bfloat16", "Unexpected model dtype")
    require(option(argv, "--max-model-len") == "4096", "Unexpected context limit")
    require(option(argv, "--seed") == "42", "Unexpected seed")
    require("--quantization" not in argv and "-q" not in argv, "Quantized launch is outside this runtime contract")

    probe_env = dict(launch_env, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    probe = subprocess.run([runtime_python, "-c", ENVIRONMENT_PROBE], env=probe_env,
                           cwd=ROOT, text=True, capture_output=True, check=True)
    environment = json.loads(probe.stdout)
    require(not environment["cuda_initialized_by_probe"], "Environment probe unexpectedly initialized CUDA")

    source_manifest = read_json(DOWNLOAD / "source_manifest.json")
    verification = read_json(DOWNLOAD / "verification.json")
    header_audit = read_json(DOWNLOAD / "header_audit.json")
    require(verification.get("complete") is True, "Checkpoint download verification is incomplete")
    require(header_audit.get("passed") is True and not header_audit.get("errors"), "Checkpoint header audit failed")
    require(source_manifest["revision"] == verification["revision"], "Checkpoint revision mismatch")
    require(all(x.get("official_hash_match") is True for x in verification["files"]), "An official file hash did not match")
    official_files = {entry["path"]: entry for entry in source_manifest["files"]}
    checkpoint_files = []
    source_paths: set[Path] = set()
    for entry in verification["files"]:
        path = model_dir / entry["path"]
        stat = path.stat()
        require(stat.st_size == entry["size"], f"Changed checkpoint file size: {path}")
        record = {**entry, "absolute_path": str(path), "current_mtime_ns": stat.st_mtime_ns}
        if path.suffix == ".safetensors":
            require(entry["sha256"] == official_files[entry["path"]]["lfs_sha256"], f"LFS hash mismatch: {path}")
            with path.open("rb") as stream:
                prefix = stream.read(8)
                length = struct.unpack("<Q", prefix)[0]
                require(2 <= length <= 16 * 1024 * 1024, f"Invalid safetensors header length: {path}")
                header = stream.read(length)
            require(len(header) == length, f"Incomplete safetensors header: {path}")
            record.update(header_and_prefix_sha256=hashlib.sha256(prefix + header).hexdigest(),
                          header_bytes=length, full_sha256_source="completed download verification; not recomputed at freeze")
        else:
            require(sha256(path) == entry["sha256"], f"Changed checkpoint metadata/tokenizer: {path}")
            record["full_sha256_source"] = "recomputed at freeze"
            source_paths.add(path)
        checkpoint_files.append(record)

    official_commit = subprocess.check_output(["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True).strip()
    official_status = subprocess.check_output(["git", "-C", str(OFFICIAL), "status", "--short"], text=True)
    official_core = ["configuration_deepseek.py", "modeling_deepseek.py", "modeling_deepseek_vl_v2.py",
                     "processing_deepseek_vl_v2.py", "siglip_vit.py", "conversation.py", "__init__.py"]
    source_paths.update(OFFICIAL / "deepseek_vl2/models" / name for name in official_core)
    roots = {name: Path(path) for name, path in environment["package_roots"].items()}
    vllm_core = ["model_executor/models/deepseek_vl2.py", "model_executor/models/deepseek_v2.py",
                 "model_executor/models/utils.py", "model_executor/models/registry.py",
                 "transformers_utils/configs/deepseek_vl2.py", "transformers_utils/processors/deepseek_vl2.py",
                 "transformers_utils/config.py", "transformers_utils/processor.py", "config/model.py",
                 "entrypoints/chat_utils.py", "envs.py", "v1/sample/ops/topk_topp_sampler.py"]
    source_paths.update(roots["vllm"] / name for name in vllm_core)
    transformers_core = ["configuration_utils.py", "processing_utils.py", "tokenization_utils_base.py",
                         "models/deepseek_v2/configuration_deepseek_v2.py", "models/llama/tokenization_llama.py"]
    source_paths.update(roots["transformers"] / name for name in transformers_core)
    # Native vLLM uses timm's SigLIP vision transformer, not the official custom
    # siglip_vit module. Freeze all timm Python sources, including its layers.
    timm_paths = sorted(roots["timm"].rglob("*.py"))
    source_paths.update(timm_paths)
    required_artifacts = [HERE / "chat_template.jinja", HERE / "hf_overrides.json",
                          HERE / "config_defaults_audit.json", launcher_path, Path(__file__).resolve(), smoke_path,
                          DOWNLOAD / "source_manifest.json", DOWNLOAD / "verification.json",
                          DOWNLOAD / "header_audit.json", DOWNLOAD / "header_audit.py"]
    source_paths.update(required_artifacts)
    source_hashes = {str(path.resolve()): sha256(path) for path in sorted(source_paths)}
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Local DeepSeek-VL2-Small native vLLM runtime after smoke; not an accuracy result",
        "model_id": source_manifest["repo_id"], "checkpoint_revision": source_manifest["revision"],
        "checkpoint_directory": str(model_dir), "checkpoint_files": checkpoint_files,
        "checkpoint_verification": {"source_manifest": str(DOWNLOAD / "source_manifest.json"),
                                    "verification": str(DOWNLOAD / "verification.json"),
                                    "header_audit": str(DOWNLOAD / "header_audit.json"),
                                    "verified_at_utc": verification["verified_at_utc"],
                                    "full_weight_payloads_rehashed_at_freeze": False,
                                    "tensor_count": header_audit["observed_tensor_count"],
                                    "parameter_count": header_audit["observed_tensor_elements"]},
        "official_source": {"path": str(OFFICIAL), "commit": official_commit, "git_status_short": official_status},
        "environment": environment,
        "inference": {"backend": "vllm-native", "architecture": "DeepseekVLV2ForCausalLM",
                      "gpu_uuid": gpu_uuid, "dtype": "bfloat16", "quantization": None,
                      "max_model_len": 4096, "seed": 42,
                      "tensor_parallel_size": int(option(argv, "--tensor-parallel-size", "1")),
                      "max_num_seqs": int(option(argv, "--max-num-seqs")),
                      "gpu_memory_utilization": float(option(argv, "--gpu-memory-utilization")),
                      "limit_mm_per_prompt": json.loads(option(argv, "--limit-mm-per-prompt")),
                      "host": option(argv, "--host"), "port": int(option(argv, "--port")),
                      "enforce_eager": "--enforce-eager" in argv,
                      "chat_template_content_format": option(argv, "--chat-template-content-format"),
                      "generation_config": option(argv, "--generation-config"),
                      "launcher_argv": argv,
                      "launcher_environment": {key: launch_env.get(key) for key in
                          ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
                           "VLLM_USE_FLASHINFER_SAMPLER")}},
        "smoke_record": {"path": str(smoke_path.resolve()), "sha256": sha256(smoke_path),
                         "status_as_recorded": smoke.get("status"),
                         "success_not_inferred_from_record_presence": True},
        "timm_python_source_count": len(timm_paths),
        "source_sha256": source_hashes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=Path, default=HERE / "smoke.json")
    parser.add_argument("--output", type=Path, default=HERE / "runtime_manifest.json")
    args = parser.parse_args()
    manifest = freeze(args.smoke.resolve(), args.output.resolve())
    print(json.dumps({"path": str(args.output.resolve()), "source_files": len(manifest["source_sha256"]),
                      "checkpoint_revision": manifest["checkpoint_revision"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
