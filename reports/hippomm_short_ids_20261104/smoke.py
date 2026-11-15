import json
from pathlib import Path
import re
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.memory import hippomm as m, hippomm_core as core
out = Path(__file__).resolve().parent / 'smoke'
out.mkdir(exist_ok=True)
m.code_manifest(out)
chat = m.LocalChat(out, {}, 'cuda:0')
encoder = m.ClipText('cuda:0')
results = []
try:
    for dataset in ('memeye', 'mem_gallery'):
        source = ROOT / 'results/hippomm_full_20261103' / dataset
        old_events = m.inputs.read_lines(source / 'offline/events.jsonl')[:2]
        all_rows = m.inputs.unique(m.inputs.read_lines(source / 'offline/dialogues.jsonl'))
        refs = {}
        def image_ref(value):
            if value not in refs:
                refs[value] = f'I{len(refs) + 1:04d}'
            return refs[value]
        dialogues, events, vectors = {}, [], {}
        with np.load(source / 'offline/visual_embeddings.npz') as data:
            old_vectors = dict(zip(data['event_ids'].tolist(), data['vectors']))
        for index, old in enumerate(old_events, 1):
            e = {**old, 'event_id': f'E{index:04d}', 'summary': '',
                 'representative_image': image_ref(old['representative_image'])}
            e['source_images'] = [{k: v for k, v in ref.items() if k != 'image_sha256'}
                                 | {'image_ref': image_ref(ref['image_sha256'])}
                                 for ref in old['source_images']]
            for mid in e['source_round_ids']:
                row = all_rows[mid]
                dialogues[mid] = {**row, 'images': [
                    {'image_ref': image_ref(ref['image_sha256']), 'image_id': ref['image_id']}
                    for ref in row['images']]}
            e['summary'] = chat('summary', core.summary_prompt(e, dialogues))
            events.append(e)
            vectors[e['event_id']] = old_vectors[old['event_id']]
        q = m.inputs.read_lines(source / 'online/questions.jsonl')[0]
        def llm(role, prompt):
            assert not re.search(r'[0-9a-f]{64}', prompt), (role, 'fingerprint in prompt')
            return chat(role, prompt, q['query_image_paths'] if role in ('fast', 'answer') else [])
        for threshold in (0.75, 1.0):
            trace = core.answer_question(q, events, vectors, dialogues, llm, encoder,
                                         confidence_threshold=threshold)
            assert all(re.fullmatch(r'E[0-9]{4}', eid) for eid in trace['selected_event_ids'])
            results.append({'dataset': dataset, 'threshold': threshold, 'trace': trace})
    for path in chat.cache_dir.glob('*.json'):
        row = json.loads(path.read_text())
        assert not re.search(r'[0-9a-f]{64}', row['request']['prompt'])
        assert not re.search(r'[0-9a-f]{64}', row['response']['raw_response'])
        assert not row['response']['hit_token_limit']
    m.dump(out / 'verification.json', {'status': 'passed', 'scope': 'Two real events per dataset; API and short-ID validation, not benchmark scores',
                                     'cases': results, 'calls': len(chat.cache_index),
                                     'fingerprints_in_model_io': 0})
finally:
    chat.close()
