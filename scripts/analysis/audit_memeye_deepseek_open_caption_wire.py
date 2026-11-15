"""Check all caption manifests against the original DeepSeek multimodal builder."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
from scripts.evaluation import memeye_deepseek_open_caption_inputs as src
from scripts.evaluation import memeye_deepseek_open_reader as reader
from scripts.evaluation import deepseek_flash_seed42_audited_transport as transport
p = argparse.ArgumentParser(); p.add_argument('--family', choices=['base','ours'], required=True); args = p.parse_args()
root = src.DEFAULT
methods = ['semrag','ours_2'] if args.family == 'base' else ['ours_1','ours_1_ours_2']
checked = image_cases = 0
for method in methods:
    for row in src.rows(root / method / 'reader_manifest.jsonl'):
        payload = {'system': row['system'], 'user_text': row['user_text'], 'query_images': [{'path': p, 'sha256': src.sha(p)} for p in row['query_image_paths']]}
        expected = transport.build_request(payload, model=src.MODEL, max_output_tokens=2048, image_detail='high')
        actual = reader.build_request(row)
        assert actual == expected, row['id']
        transport._validate_request(transport.load_config('/storage/hewenwen/NTU Decentralized Agent/.env'), actual)
        checked += 1; image_cases += bool(row['query_image_paths'])
result = {'passed': True, 'family': args.family, 'reader_wires_reconstructed': checked, 'question_image_cases': image_cases,
    'same_native_message_roles_image_order_original_bytes_and_API_settings': True, 'model': src.MODEL,
    'resolved_model_revision': None, 'source_sha256': {str(Path(p)): src.sha(p) for p in [__file__, reader.__file__, transport.__file__, *[root / m / 'reader_manifest.jsonl' for m in methods]]}}
src.freeze(root / ('reader_wire_compatibility_' + args.family + '.json'), result)
print(result)
