"""Original-pixel, caption-free inputs for the M2A benchmark adaptation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def task_names(dataset):
    if dataset == 'memeye':
        manifest = ROOT / 'data/independent_visual_inputs/qa_manifest.jsonl'
        return sorted({json.loads(line)['task'] for line in manifest.read_text().splitlines()})
    return sorted(p.stem for p in (ROOT / 'data/Mem-Gallery/data/dialog').glob('*.json'))


def load_task(dataset, task):
    from scripts.memory.m2a_vendor import sha
    if task not in task_names(dataset):
        raise ValueError('Unknown task: ' + task)
    gallery = dataset == 'memgallery'
    base = ROOT / ('data/Mem-Gallery/data' if gallery else 'data/MemEye/data')
    source = base / 'dialog' / (task + ('.json' if gallery else '_Open.json'))
    data = json.loads(source.read_text())
    observations = []
    for session in data['multi_session_dialogues']:
        for turn in session['dialogues']:
            refs = turn.get('input_image') or []
            refs = [refs] if isinstance(refs, str) else refs
            if len(refs) > 1:
                raise ValueError('Upstream supports one image per observation')
            image = (source.parent / refs[0] if gallery else base / 'image' / refs[0]).resolve() if refs else None
            text = (f"[Session {session['session_id']}; date {session['date']}; round {turn['round']}]\n"
                    f"User: {turn['user']}\nAssistant: {turn['assistant']}")
            # Official image identifiers are historical metadata needed by visual-search QA.
            # Never include the dataset's supplied image captions.
            if gallery and refs:
                text += '\nImage ID: ' + ', '.join(turn['image_id'])
            observations.append(dict(id=task + '/' + turn['round'], round=turn['round'],
                timestamp=session['date'], session_id=session['session_id'], text=text,
                image=str(image) if image else None, image_sha256=sha(image) if image else None))
    sources = {str(source): sha(source)}
    if gallery:
        questions = []
        for i, qa in enumerate(data['human-annotated QAs']):
            image = (source.parent / qa['question_image']).resolve() if qa.get('question_image') else None
            questions.append(dict(id=task + f'/QA_{i:04d}', task=task, question=qa['question'],
                point=qa['point'], query_image_paths=[str(image)] if image else [],
                answer=qa['answer'], gold_ids=[task + '/' + x for x in qa['clue']]))
    else:
        manifest = ROOT / 'data/independent_visual_inputs/qa_manifest.jsonl'
        questions = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
        questions = [q for q in questions if q['task'] == task]
        sources[str(manifest)] = sha(manifest)
    source_ids = {o['id'] for o in observations}
    assert len(source_ids) == len(observations)
    assert len({q['id'] for q in questions}) == len(questions)
    for q in questions:
        assert set(q['gold_ids']) <= source_ids, q['id']
        for path in q['query_image_paths']:
            sources[path] = sha(path)
    return observations, questions, sources


def gallery_reader():
    # Reuse the project's pinned official prompts and normalized EM/F1/BLEU.
    # Restore import routing because the older reader prepends its own vendor path.
    import sys
    old = list(sys.path)
    try:
        from scripts.evaluation import mem_gallery_master_reader
        return mem_gallery_master_reader
    finally:
        sys.path[:] = old
        # The scoring function imports nltk lazily; make its existing local
        # dependency available to standalone audits without changing precedence.
        vendor = str(ROOT / '.vendor')
        if vendor not in sys.path:
            sys.path.append(vendor)
