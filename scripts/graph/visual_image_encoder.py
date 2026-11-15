"""CLIP image features for offline visual clustering, without an image-vector cache.

Only the vision tower and its trained projection are instantiated.  Text models,
text tokenizers, and caption embeddings are not part of this encoder.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_VISUAL_ENCODER = "openai/clip-vit-base-patch32"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def inspect_model_source(model_name: str, revision: str | None = None) -> dict[str, Any]:
    """Describe the requested model without importing or loading its encoder.

    Local checkpoints are fingerprinted by content, including their weights, so
    replacing a model in the same directory cannot silently resume an old graph.
    Remote identifiers record the requested revision; ``ImageEncoder.metadata``
    additionally records the resolved immutable Hub commit after loading config.
    """
    source = Path(model_name).expanduser()
    result: dict[str, Any] = {
        "model_name": str(model_name),
        "revision": revision,
        "source_type": "local" if source.exists() else "huggingface",
    }
    if source.exists():
        if not source.is_dir():
            raise ValueError("visual encoder must be a model directory or a Hugging Face model ID")
        source = source.resolve()
        result["model_name"] = str(source)
        filenames = {"config.json", "preprocessor_config.json", "processor_config.json"}
        for pattern in ("*.safetensors", "*.safetensors.index.json", "pytorch_model*.bin", "pytorch_model*.bin.index.json"):
            filenames.update(path.name for path in source.glob(pattern))
        result["files_sha256"] = {
            name: _file_sha256(source / name)
            for name in sorted(filenames)
            if (source / name).is_file()
        }
    return result


class ImageEncoder:
    """Load a CLIP visual encoder and produce one normalized Python vector at a time.

    ``metadata`` contains checkpoint, implementation and preprocessing provenance.
    ``encode`` retains neither the image nor its embedding after returning it.
    Only the caller's aggregate master representations should be persisted.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_VISUAL_ENCODER,
        revision: str | None = None,
        device: str = "cpu",
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoConfig, CLIPVisionModelWithProjection
            try:
                # Transformers 5 separates the PIL backend from torchvision.
                from transformers import CLIPImageProcessorPil as Processor
            except ImportError:
                from transformers import CLIPImageProcessor as Processor
        except ImportError as exc:
            raise RuntimeError(
                "Visual clustering requires torch, transformers, Pillow and a CLIP checkpoint"
            ) from exc

        self._torch = torch
        self.device = torch.device(device)
        self.source = inspect_model_source(model_name, revision)
        load_name = self.source["model_name"]
        options: dict[str, Any] = {
            "local_files_only": bool(local_files_only),
            "trust_remote_code": False,
        }
        if revision is not None:
            options["revision"] = revision
        if cache_dir is not None:
            options["cache_dir"] = str(cache_dir)

        config = AutoConfig.from_pretrained(load_name, **options)
        if config.model_type not in {"clip", "clip_vision_model"}:
            raise ValueError(
                f"Unsupported visual encoder model_type={config.model_type!r}; "
                "use a CLIP checkpoint with trained visual projection weights"
            )
        resolved_commit = getattr(config, "_commit_hash", None)
        if resolved_commit and self.source["source_type"] == "huggingface":
            # Pin processor and weights to the same snapshot as their config.
            options["revision"] = resolved_commit
        vision_config = config.vision_config if config.model_type == "clip" else config
        self.processor = Processor.from_pretrained(load_name, **options)
        self.model, loading_info = CLIPVisionModelWithProjection.from_pretrained(
            load_name, config=vision_config, output_loading_info=True, **options
        )
        incomplete = {
            key: loading_info.get(key)
            for key in ("missing_keys", "mismatched_keys", "error_msgs")
            if loading_info.get(key)
        }
        if incomplete:
            raise ValueError(
                "Visual encoder checkpoint is incomplete; refusing randomly initialized "
                f"vision/projection weights: {incomplete}"
            )
        self.model.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        self.embedding_dim = int(vision_config.projection_dim)
        processor_config = self.processor.to_dict()
        self.metadata: dict[str, Any] = {
            **self.source,
            "resolved_commit": resolved_commit,
            "model_type": config.model_type,
            "model_class": type(self.model).__name__,
            "processor_class": type(self.processor).__name__,
            "processor_config": processor_config,
            "processor_config_sha256": _json_sha256(processor_config),
            "pooling": "CLIP projected pooled vision CLS (image_embeds)",
            "embedding_dim": self.embedding_dim,
            "normalization": "L2; float64 CPU normalization after float32 inference",
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "device": str(self.device),
            "dtype": "float32",
        }
        # Processor defaults can contain tuples (e.g. CLIP image_mean/std).
        # Snapshots serialize them as lists; normalize before first comparison
        # so an unchanged encoder can resume its JSON-persisted graph.
        self.metadata = json.loads(json.dumps(self.metadata, allow_nan=False))

    def encode(self, image_path: str | Path) -> list[float]:
        """Encode one local image; never cache or write its individual vector."""
        from PIL import Image

        with Image.open(image_path) as image:
            inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        pixels = inputs["pixel_values"].to(device=self.device, dtype=self._torch.float32)
        with self._torch.inference_mode():
            embeddings = self.model(pixel_values=pixels, return_dict=True).image_embeds
        vector = embeddings[0].detach().to(device="cpu", dtype=self._torch.float64)
        if vector.ndim != 1 or vector.numel() != self.embedding_dim:
            raise ValueError("CLIP returned an unexpected image embedding shape")
        norm = float(self._torch.linalg.vector_norm(vector))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("CLIP returned a non-finite or zero image embedding")
        return [float(value) for value in (vector / norm).tolist()]
