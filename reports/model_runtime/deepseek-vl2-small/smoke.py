"""Small local text, visual counterfactual, and canonical caption checks."""
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image, ImageDraw

ROOT = Path('/storage/hewenwen/VisualMemory')
sys.path.insert(0, str(ROOT))
from scripts.evaluation.deepseek_vl2_local_transport import build_request, chat_request, load_config

HERE = Path(__file__).resolve().parent
def image_info(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    tests = [('text', {'user_text': 'Reply with exactly OK.'}, 16)]
    for color in ('red', 'blue'):
        path = HERE / f'smoke_{color}.png'
        canvas = Image.new('RGB', (256, 256), 'white')
        ImageDraw.Draw(canvas).rectangle((40, 40, 216, 216), fill=color)
        canvas.save(path)
        tests.append((color, {'user_text': 'What color is the large square? Answer with only the color name.',
                             'query_images': [image_info(path)]}, 16))
    prompt = json.loads((ROOT / 'results/natural_observation/protocol.json').read_text())['caption_prompt']
    path = ROOT / 'data/MemEye/data/image/Home_Renovation_Interior_Design/D7_IMG_005.jpg'
    tests.append(('canonical_caption', {'user_text': prompt, 'query_images': [image_info(path)]}, 1024))
    rows = []
    for name, payload, budget in tests:
        request = build_request(payload, max_output_tokens=budget)
        answer = chat_request(load_config(), request)
        normalized = answer['raw_response'].strip().lower().rstrip('.')
        passed = answer['finish_reason'] == 'stop' and bool(normalized)
        if name == 'text':
            passed &= normalized == 'ok'
        if name in ('red', 'blue'):
            passed &= normalized == name and answer['expanded_image_tokens'] > 1
        rows.append({'name': name, 'input': payload, 'passed': passed, **answer})
        document = {'passed': len(rows) == len(tests) and all(r['passed'] for r in rows), 'tests': rows}
        (HERE / 'smoke.json').write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({k: rows[-1][k] for k in ('name', 'passed', 'raw_response', 'input_tokens', 'expanded_image_tokens', 'usage', 'elapsed_seconds')}, ensure_ascii=False), flush=True)
        if not passed:
            raise RuntimeError('Runtime smoke check failed: ' + name)

if __name__ == '__main__':
    main()
