"""Read-only revalidation of the completed Qwen SimpleMem 0.3.0 control."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from scripts.evaluation import mem_gallery_simplemem_official as run
from scripts.evaluation import recover_mem_gallery_simplemem_official as recovery
from scripts.analysis.audit_mem_gallery_gpt5_simplemem030 import independent_fusion

SOURCE = run.OUTPUT
OUTPUT = run.ROOT/'results/mem_gallery_simplemem030_backbones_20261031'
base = run.base


def audit():
    p = run.verify(SOURCE, 'simplemem')
    root = SOURCE/'simplemem'
    obs, qs = base.read(SOURCE/'observations.json'), base.rows(SOURCE/'inputs/questions.jsonl')
    assert base.obj(obs) == p['observations_sha256'] and base.obj(qs) == p['questions_sha256']
    assert len(qs) == 1711 and sum(map(len, obs.values())) == 3962
    assert p['model'] == base.gallery.model_provenance()
    amendment = base.read(SOURCE/'recovery_amendment.json')
    assert amendment['status'] == 'complete'
    assert amendment['source_sha256'] == base.sha(recovery.__file__)
    for key, digest in amendment['original_calls'].items():
        assert base.sha(root/'calls'/(key+'.json')) == digest
    original = base.read(root/'calls'/(recovery.TRUNCATED+'.json'))
    retry = base.read(SOURCE/'truncated_window_retry.json')
    repaired = base.read(root/'calls'/(retry['new_call']+'.json'))
    expected = json.loads(json.dumps(original['request']))
    expected['messages'][-1]['content'] += '\n\n'+recovery.INSTRUCTION
    assert repaired['request'] == expected and not repaired['output']['hit_token_limit']
    extracted = json.loads(repaired['output']['raw_response'].strip().removeprefix('```json').removesuffix('```').strip())
    repaired_texts = {entry['lossless_restatement'] for entry in extracted}
    from simplemem.core.answer_generator import AnswerGenerator
    from simplemem.core.models.memory_entry import MemoryEntry
    formatter = object.__new__(AnswerGenerator)
    notes, artifacts, state_counts = {}, {}, {}
    all_stored_texts = set()
    location_repairs = 0
    for task, original_rows in obs.items():
        dest = root/task
        done = base.read(dest/'complete.json')
        assert done['experiment_signature'] == p['experiment_signature']
        assert done['retrieval_sha256'] == base.sha(dest/'retrieval.jsonl')
        windows = base.read(dest/'runtime/construction_windows.json')
        sources = base.read(dest/'runtime/dialogue_sources.json')
        entries = base.read(dest/'final_memory.json')
        notes[task] = {e['entry_id']: e for e in entries}
        assert len(notes[task]) == len(entries)
        assert {s['round_id'] for s in sources.values()} == {r['id'] for r in original_rows}
        assert {i for w in windows for i in w['dialogue_ids']} == {int(i) for i in sources}
        assert {i for w in windows for i in w['entry_ids']} == set(notes[task])
        assert all(w['parsed_successfully'] for w in windows)
        all_stored_texts.update(e['lossless_restatement'] for e in entries)
        fix = dest/'runtime/location_serialization_repair.json'
        if fix.exists():
            recorded = base.read(fix)
            raw = base.read(root/'calls'/(recovery.LOCATION+'.json'))['output']['raw_response']
            corrected, changes = recovery.repair_location(raw)
            assert recorded['corrected_response'] == corrected and recorded['changes'] == changes
            assert recorded['original_response_sha256'] == hashlib.sha256(raw.encode()).hexdigest()
            for e in json.loads(corrected):
                assert any(all(n[k] == v for k, v in e.items()) for n in entries)
            location_repairs += 1
        state_counts[task] = dict(rounds=len(original_rows), windows=len(windows), entries=len(entries))
        for f in [dest/'complete.json', dest/'retrieval.jsonl', dest/'final_memory.json', *dest.glob('runtime/*.json')]:
            artifacts[str(f)] = base.sha(f)
    assert repaired_texts <= all_stored_texts and location_repairs == 1
    retrieval = base.rows(root/'retrieval.jsonl')
    answers = base.rows(root/'reader/responses.jsonl')
    manifest = base.rows(root/'reader/manifest.jsonl')
    rmap, amap, mmap = ({r['id']: r for r in data} for data in (retrieval, answers, manifest))
    assert len(retrieval) == len(rmap) == len(answers) == len(amap) == len(manifest) == len(mmap) == 1711
    assert set(rmap) == set(amap) == set(mmap) == {q['id'] for q in qs}
    done = base.read(root/'reader/complete.json')
    assert done['responses_sha256'] == base.sha(root/'reader/responses.jsonl')
    assert done['manifest_sha256'] == base.sha(root/'reader/manifest.jsonl')
    for q in qs:
        r, a, item = rmap[q['id']], amap[q['id']], mmap[q['id']]
        detail = r['retrieval_detail']
        independent_fusion(detail)
        ids = detail['context_memory_ids']
        assert len(ids) == 5 and len(set(ids)) == 5
        context = formatter._format_contexts([MemoryEntry(**notes[q['task']][i]) for i in ids])
        assert context == r['retrieved_context'] and r['retrieval_query'] == q['retrieval_query']
        expected = dict(id=q['id'], task=q['task'], point=q['point'], system=base.gallery.READER_SYSTEM,
                        user_text=base.reader_text(q, context), query_image_paths=q['query_image_paths'])
        expected['payload_sha256'] = base.obj(expected)
        assert item == expected and a['payload_sha256'] == item['payload_sha256']
        assert a['experiment_signature'] == r['experiment_signature'] == p['experiment_signature']
        assert not a['hit_token_limit'] and a['raw_response'].strip()
    paths = list((root/'calls').glob('*.json'))
    def check_call(path):
        c = base.read(path)
        assert base.obj(c['request']) == path.stem
        assert c['request']['experiment_signature'] == p['experiment_signature']
        return str(path), base.sha(path), c['output']['hit_token_limit']
    with ThreadPoolExecutor(max_workers=8) as pool:
        calls = list(pool.map(check_call, paths))
    assert [Path(f).stem for f, _, limited in calls if limited] == [recovery.TRUNCATED]
    artifacts.update({f: digest for f, digest, _ in calls})
    gold = {g['id']: g for g in base.rows(SOURCE/'inputs/gold.jsonl')}
    assert set(gold) == set(amap)
    scored = [base.gallery.score_answer(amap[q['id']]['raw_response'], gold[q['id']]['answer']) for q in qs]
    result = base.read(SOURCE/'final_results.json')
    for k in ('em', 'f1', 'bleu_1', 'bleu_2', 'bleu'):
        assert abs(sum(r[k] for r in scored)/1711-result['overall'][k]) < 1e-12
    assert sum(int(r['em']) for r in scored) == result['overall']['correct'] == 570
    for f in [SOURCE/'final_results.json', SOURCE/'audit.json', SOURCE/'recovery_amendment.json',
              SOURCE/'truncated_window_retry.json', root/'retrieval.jsonl', root/'reader/responses.jsonl',
              root/'reader/manifest.jsonl', root/'reader/complete.json', root/'protocol.json']:
        artifacts[str(f)] = base.sha(f)
    record = dict(status='passed', reused_completed_run=str(SOURCE), model=p['model'],
                  overall=result['overall'], original_rounds=3962, independent_rrf_questions=1711,
                  memory_calls=len(calls), final_reader_truncations=0, source_state_counts=state_counts,
                  caveats=['Reader cap256 vs GPT-5 cap2048 including reasoning; no Qwen reader truncation.',
                           'One truncated extraction retried with finite-output instructions at unchanged cap16384.',
                           'One extraction had four list-valued locations joined with semicolons.'],
                  source_recall=None, artifacts_sha256=artifacts,
                  script_sha256=base.sha(__file__))
    base.dump(OUTPUT/'qwen_revalidation.json', record)
    print(json.dumps(dict(status='passed', memory_calls=len(calls), **result['overall'])), flush=True)


if __name__ == '__main__':
    audit()
