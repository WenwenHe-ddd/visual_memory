"""Local random-checkpoint integration tests; these do not assess CLIP quality."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.graph.visual_image_encoder import ImageEncoder, inspect_model_source


class VisualImageEncoderTests(unittest.TestCase):
    def test_remote_identity_does_not_download_or_load(self):
        self.assertEqual(
            inspect_model_source("openai/clip-vit-base-patch32", "fixed-revision"),
            {
                "model_name": "openai/clip-vit-base-patch32",
                "revision": "fixed-revision",
                "source_type": "huggingface",
            },
        )

    def test_local_checkpoint_content_changes_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text('{"model_type": "clip"}')
            before = inspect_model_source(directory)
            config.write_text('{"model_type": "clip_vision_model"}')
            self.assertNotEqual(before, inspect_model_source(directory))

    def test_tiny_local_random_clip_produces_normalized_visual_features(self):
        import torch
        from PIL import Image
        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
        try:
            from transformers import CLIPImageProcessorPil as Processor
        except ImportError:
            from transformers import CLIPImageProcessor as Processor

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tiny-random-clip"
            with torch.random.fork_rng():
                torch.manual_seed(17)
                config = CLIPVisionConfig(
                    hidden_size=16, intermediate_size=32, projection_dim=8,
                    num_hidden_layers=1, num_attention_heads=2,
                    image_size=16, patch_size=8,
                )
                model = CLIPVisionModelWithProjection(config)
            model.save_pretrained(checkpoint)
            Processor(
                size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
            ).save_pretrained(checkpoint)
            # The official older CLIP processor config omits image_mean/std.
            # Transformers 5 fills these defaults with tuples; persisted graph
            # metadata must still compare equal with the next encoder instance.
            processor_path = checkpoint / "preprocessor_config.json"
            processor_config = json.loads(processor_path.read_text())
            processor_config.pop("image_mean", None)
            processor_config.pop("image_std", None)
            processor_path.write_text(json.dumps(processor_config))
            image_path = Path(directory) / "synthetic.png"
            Image.new("RGB", (24, 20), (180, 50, 90)).save(image_path)
            before = sorted(str(path) for path in checkpoint.rglob("*"))
            encoder = ImageEncoder(str(checkpoint), local_files_only=True)
            vector = encoder.encode(image_path)
            self.assertEqual(len(vector), 8)
            self.assertTrue(all(type(value) is float for value in vector))
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in vector)), 1.0, places=12)
            self.assertEqual(vector, encoder.encode(image_path))
            self.assertEqual(encoder.metadata["model_class"], "CLIPVisionModelWithProjection")
            self.assertEqual(encoder.metadata["embedding_dim"], 8)
            persisted_metadata = json.loads(json.dumps(encoder.metadata))
            self.assertEqual(encoder.metadata, persisted_metadata)
            self.assertIsInstance(encoder.metadata["processor_config"]["image_mean"], list)
            self.assertIsInstance(encoder.metadata["processor_config"]["image_std"], list)
            resumed_encoder = ImageEncoder(str(checkpoint), local_files_only=True)
            self.assertEqual(resumed_encoder.metadata, persisted_metadata)
            self.assertNotIn("text_model", dict(encoder.model.named_modules()))
            self.assertEqual(before, sorted(str(path) for path in checkpoint.rglob("*")))

    def test_rejects_non_visual_language_model_configuration(self):
        from transformers import BertConfig

        with tempfile.TemporaryDirectory() as directory:
            BertConfig().save_pretrained(directory)
            with self.assertRaisesRegex(ValueError, "Unsupported visual encoder"):
                ImageEncoder(directory, local_files_only=True)

    def test_full_clip_checkpoint_loads_only_trained_vision_and_projection(self):
        import torch
        from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig
        try:
            from transformers import CLIPImageProcessorPil as Processor
        except ImportError:
            from transformers import CLIPImageProcessor as Processor

        with tempfile.TemporaryDirectory() as directory:
            vision = CLIPVisionConfig(
                hidden_size=16, intermediate_size=32, projection_dim=8,
                num_hidden_layers=1, num_attention_heads=2, image_size=16, patch_size=8,
            )
            text = CLIPTextConfig(
                vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
            )
            full_config = CLIPConfig(
                text_config=text.to_dict(), vision_config=vision.to_dict(), projection_dim=8
            )
            with torch.random.fork_rng():
                torch.manual_seed(17)
                full_model = CLIPModel(full_config)
            full_model.save_pretrained(directory)
            Processor(
                size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
            ).save_pretrained(directory)
            encoder = ImageEncoder(directory, local_files_only=True)
            self.assertTrue(torch.equal(
                full_model.visual_projection.weight, encoder.model.visual_projection.weight
            ))
            self.assertNotIn("text_model", dict(encoder.model.named_modules()))

    def test_rejects_checkpoint_without_trained_visual_projection(self):
        from transformers import CLIPVisionConfig, CLIPVisionModel
        try:
            from transformers import CLIPImageProcessorPil as Processor
        except ImportError:
            from transformers import CLIPImageProcessor as Processor

        with tempfile.TemporaryDirectory() as directory:
            model = CLIPVisionModel(CLIPVisionConfig(
                hidden_size=16, intermediate_size=32, projection_dim=8,
                num_hidden_layers=1, num_attention_heads=2, image_size=16, patch_size=8,
            ))
            model.save_pretrained(directory)
            Processor(
                size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
            ).save_pretrained(directory)
            with self.assertRaisesRegex(ValueError, "checkpoint is incomplete"):
                ImageEncoder(directory, local_files_only=True)


if __name__ == "__main__":
    unittest.main()
