"""Load original image-round dialogue from content-verified source manifests."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def load_dialogues(natural_root, tasks):
    """Whitelist image-round user/assistant text from pinned source files."""
    sources = [natural_root / name for name in ('protocol.json', 'images.jsonl', 'memories.jsonl')]
    config = json.loads(sources[0].read_text())
    image_ids = {rel: row['id'] for row in read_jsonl(sources[1]) for rel in row['source_paths']}
    rows = read_jsonl(sources[2])
    memories = {r['id']: r for r in rows if r['task'] in tasks}
    if len(memories) != sum(r['task'] in tasks for r in rows):
        raise ValueError('duplicate image memory IDs')
    dialogues = {}
    for task in sorted(tasks):
        candidates = [s for s in config['source_files'] if Path(s['path']).name == task + '_Open.json']
        if len(candidates) != 1:
            raise ValueError('expected one pinned dialogue source for ' + task)
        source = candidates[0]
        path = ROOT / source['path']
        if sha(path) != source['sha256']:
            raise ValueError('dialogue source changed: ' + str(path))
        sources.append(path)
        seen = set()
        for session in json.loads(path.read_text())['multi_session_dialogues']:
            for turn in session['dialogues']:
                mid = task + '/' + turn['round']
                if mid in seen:
                    raise ValueError('duplicate dialogue round: ' + mid)
                seen.add(mid)
                refs = turn.get('input_image') or []
                refs = [refs] if isinstance(refs, str) else refs
                if not refs and mid not in memories:
                    continue
                if (not refs or mid not in memories
                        or [image_ids.get(ref) for ref in refs] != memories[mid]['images']):
                    raise ValueError('natural image manifest disagrees with dialogue: ' + mid)
                dialogue = {role: turn[role] for role in ('user', 'assistant')}
                if any(not isinstance(text, str) for text in dialogue.values()):
                    raise ValueError('dialogue text must be a string: ' + mid)
                dialogues[mid] = dialogue
    if set(dialogues) != set(memories):
        raise ValueError('dialogue image round coverage mismatch')
    return dialogues, sources
