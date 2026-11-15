import io
import unittest
import numpy as np
from PIL import Image
from scripts.evaluation.gallery_deepseek_bounded_images import jpeg_to_limit


class ImageLimitTests(unittest.TestCase):
    def test_encoding_is_deterministic_and_preserves_entire_dimensions(self):
        pixels = np.random.default_rng(42).integers(0, 256, (180, 240, 3), dtype=np.uint8)
        source = io.BytesIO()
        Image.fromarray(pixels).save(source, format='PNG')
        first, info = jpeg_to_limit(source.getvalue(), max_bytes=25000)
        second, other = jpeg_to_limit(source.getvalue(), max_bytes=25000)
        self.assertEqual(first, second)
        self.assertEqual(info, other)
        self.assertLessEqual(len(first), 25000)
        self.assertFalse(info['resized'])
        self.assertFalse(info['cropped'])
        self.assertEqual(Image.open(io.BytesIO(first)).size, (240,180))


if __name__ == '__main__': unittest.main()
