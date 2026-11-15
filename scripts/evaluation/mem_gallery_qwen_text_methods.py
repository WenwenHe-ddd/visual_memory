"""Native text memories on all Mem-Gallery questions, with the matched Qwen reader.

1. Candidate memory: every same-scenario dialogue round, in source order.
2. Query: original question and supplied official question-image caption.
3. Retrieval: pinned native GA/MemoryOS/SimpleMem/A-Mem; last two context <=5.
4. Encoding: native shared local MiniLM; Qwen3-VL-8B for memory LLM operations.
5. Answer: identical Gallery system/task prompts, original query pixels, greedy
   Qwen with seed 42 and 256 new tokens; official normalized EM/F1/BLEU afterward.
Gold is never loaded by construction, retrieval, or answer generation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                  LITELLM_LOCAL_MODEL_COST_MAP='True', ANONYMIZED_TELEMETRY='False')
from scripts.memory import text_memory_baselines as native
from scripts.memory.text_memory_transport import Broker
from scripts.evaluation import mem_gallery_master_reader as gallery
from scripts.data.mem_gallery_inputs import DEFAULT_CAPTIONS, CAPTION_FIELD

SOURCE = ROOT / 'results/mem_gallery_master_answer_20261021'
OUTPUT = ROOT / 'results/mem_gallery_qwen_seed42_text_methods_20261026'
METHODS = ('gen_agents', 'memoryos', 'simplemem', 'a_mem')
NAMES = dict(gen_agents='Generative Agents', memoryos='MemoryOS',
             simplemem='SimpleMem (context <=5)', a_mem='A-Mem (context <=5)')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return gallery.sha(Path(path))


def obj(value):
    return native.hash_object(value)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_rows(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in value))
    tmp.replace(path)


def freeze(path, value):
    if Path(path).exists():
        assert read(path) == value, str(path)
    else:
        dump(path, value)


def make_observations(memories):
    """Use verified original dialogue fields and the exact baseline history captions."""
    mmap = {m['id']: m for m in memories}
    result = {}
    for task in sorted({m['task'] for m in memories}):
        source = read(DEFAULT_CAPTIONS / 'dialog_with_qwen_caption' / (task + '.json'))
        name = source.get('character_profile', {}).get('name', '').strip()
        result[task] = []
        for session in source['multi_session_dialogues']:
            for turn in session['dialogues']:
                original = mmap[task + '/' + turn['round']]
                caps = turn.get(CAPTION_FIELD) or []
                if isinstance(caps, str):
                    caps = [caps]
                refs = turn.get('input_image') or []
                if isinstance(refs, str):
                    refs = [refs]
                assert original['dialogue'] == 'User: ' + turn['user'] + '\nAssistant: ' + turn['assistant']
                assert original['timestamp'] == str(session.get('date') or '').strip()
                assert caps == ([original['caption']] if original['is_visual'] else [])
                result[task].append(dict(id=turn['round'], session_id=session['session_id'],
                    timestamp=original['timestamp'], speaker=f'user ({name})' if name else 'user',
                    user=turn['user'], assistant=turn['assistant'],
                    raw=dict(input_image=refs, image_id=turn.get('image_id', []), image_caption=caps)))
        assert [task + '/' + r['id'] for r in result[task]] == [m['id'] for m in memories if m['task'] == task]
    return result


def prepare(output=OUTPUT):
    p = read(SOURCE / 'inputs/protocol.json')
    for name, expected in p['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in read(SOURCE / 'inputs/artifact_manifest.json').items():
        assert sha(SOURCE / 'inputs' / name) == expected, name
    assert read(DEFAULT_CAPTIONS / 'independent_audit.json')['status'] == 'passed'
    assert read(DEFAULT_CAPTIONS / 'protocol.json')['seed'] == 42
    upstream = native.verify_vendor()
    qs, memories = rows(SOURCE / 'inputs/questions.jsonl'), rows(SOURCE / 'inputs/memories.jsonl')
    assert len(qs) == 1711 and len(memories) == 3962
    assert len({q['id'] for q in qs}) == 1711
    assert all(not ({'answer', 'clue', 'gold_ids', 'gold_choice'} & q.keys()) for q in qs)
    observations = make_observations(memories)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('questions.jsonl', 'memories.jsonl', 'gold.jsonl', 'protocol.json', 'artifact_manifest.json'):
        target = output / 'inputs' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha(target) == sha(SOURCE / 'inputs' / name)
        else:
            shutil.copyfile(SOURCE / 'inputs' / name, target)
    freeze(output / 'observations.json', observations)
    embedding = native.EMBEDDING / 'snapshots' / (native.EMBEDDING / 'refs/main').read_text().strip()
    model = gallery.model_provenance()
    dependencies = [Path(__file__), ROOT / 'scripts/memory/text_memory_baselines.py',
        ROOT / 'scripts/memory/text_memory_transport.py', ROOT / 'scripts/memory/reflexion_memory.py',
        ROOT / 'scripts/memory/full_context_memory.py', Path(gallery.__file__),
        ROOT / 'scripts/data/mem_gallery_inputs.py', native.VENDOR / 'UPSTREAM.json']
    dependencies += [v for v in embedding.rglob('*') if v.is_file()]
    dependencies += [output / 'inputs' / n for n in ('questions.jsonl', 'memories.jsonl', 'gold.jsonl', 'protocol.json')]
    dependencies += [output / 'observations.json']
    hashes = {str(v): sha(v) for v in dependencies}
    snapshot = {}
    for v in dependencies:
        if v.suffix == '.py':
            dest = output / 'source_snapshot' / (hashes[str(v)][:12] + '_' + v.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copyfile(v, dest)
            snapshot[str(v)] = dict(path=str(dest), sha256=hashes[str(v)])
    freeze(output / 'source_snapshot/manifest.json', snapshot)
    for method in METHODS:
        annotations = {
            '1_candidate_memory_structure': 'All 3962 original rounds across 20 isolated scenario memories; native memory representation; original order.',
            '2_query': 'Original question plus supplied official question-image caption, verbatim; no options, answers, clue IDs or QA session filtering.',
            '3_retrieval': NAMES[method] + '; pinned upstream retrieval; final first five distinct units for SimpleMem/A-Mem, native GA K=10 and MemoryOS multi-tier.',
            '4_encoding': 'Local all-MiniLM-L6-v2 (384 dims), native normalization and query expansion; Qwen3-VL-8B-Instruct for all memory LLM operations.',
            '5_answering': 'Same Gallery reader system/task instructions and original query pixels; native retrieved context inserted verbatim; Qwen3-VL-8B-Instruct, greedy, seed42, 256 tokens; normalized EM/F1/BLEU.'}
        contract = dict(mode=method, study='Mem-Gallery / native text memories / matched local Qwen open-answer reader',
            upstream=upstream, method_defaults=native.effective_method_defaults(method),
            reader_model='Qwen3-VL-8B-Instruct', reader_model_path=str(gallery.MODEL), model=model,
            embedding_model_path=str(embedding), seed=42, do_sample=False, dtype='bfloat16',
            reader_system_prompt=gallery.READER_SYSTEM, task_instructions=gallery.TASK_INSTRUCTIONS,
            max_new_tokens=256, query_image_max_pixels=gallery.QUERY_IMAGE_MAX_PIXELS,
            context_budget=262144, memory_max_new_tokens=1024,
            memory_sampling='Native per-call temperatures; independent seed42 categorical RNG per request, top_p=1, no top_k.',
            generation_batch_size=16, task_workers=20, max_padded_batch_tokens=48000,
            reader_batch_size_text=16, reader_batch_size_images=4,
            context_memory_limit=5 if method in ('simplemem', 'a_mem') else None,
            tasks=sorted(observations), questions=len(qs), memory_rounds=len(memories),
            annotations=annotations, input_sha256=hashes,
            questions_sha256=obj(qs), observations_sha256=obj(observations),
            caption_policy=p['policy']['history_caption'], query_caption_policy=p['policy']['query_caption'],
            original_baseline_root=str(SOURCE), no_external_api=True, gold_loaded_only_in_analysis=True,
            differences_from_paper=['Local Qwen for all LLM operations and shared Gallery reader.',
                'MiniLM local shared encoder; JSON schema expressed as prompt instructions.',
                'SimpleMem graph isolated by scenario, native detail truncation preserved.',
                'SimpleMem/A-Mem final context <=5 without replenishing empty retrieval.',
                'Original Qwen caption cache seed42; official query captions preserved.'])
        contract['experiment_signature'] = obj(contract)
        freeze(output / method / 'protocol.json', contract)
    print(json.dumps(dict(prepared=True, questions=len(qs), rounds=len(memories), scenarios=len(observations))), flush=True)


def bounded_simplemem(result):
    selected, seen = [], set()
    for item in result.items:
        if item['id'] in seen:
            continue
        seen.add(item['id']); selected.append(item)
        if len(selected) == 5:
            break
    limited = replace(result, items=selected, expansion_candidates=[v for v in result.expansion_candidates if v in seen])
    detail = limited.to_dict()
    detail.update(context_memory_ids=[v['id'] for v in selected], context_memory_count=len(selected),
        context_memory_limit=5, original_item_ids=[v['id'] for v in result.items], source_empty_preserved=not result.items)
    return limited, detail


class Memory(native.Memory):
    def recall(self, query):
        if self.method == 'simplemem':
            result = self.core.query(query, top_k=self.protocol['method_defaults']['retrieve_k'], auto_expand=True)
            limited, detail = bounded_simplemem(result)
            return self.core.retriever.format_for_llm(limited, include_instructions=False), detail
        return super().recall(query)


class LocalQwen(native.LocalMemoryQwen):
    def __init__(self, protocol, device):
        import torch
        self.torch, self.protocol = torch, protocol
        self.processor, self.model, self.environment = gallery.load_runtime(device)


def reader_text(q, context):
    # Replace only the memory block; retain the exact baseline question suffix.
    prefix = 'The retrieved memory contents are as follows:\n\n'
    empty = gallery.reader_text(q, [], {})
    assert empty.startswith(prefix)
    return prefix + context + empty[len(prefix):]


def verify(output, method):
    p = read(output / method / 'protocol.json')
    assert p['experiment_signature'] == obj({k: v for k, v in p.items() if k != 'experiment_signature'})
    native.verify_vendor()
    for name, expected in p['input_sha256'].items():
        assert sha(name) == expected, name
    return p


def run(output, method, device='cuda:1'):
    p = verify(output, method)
    root = output / method
    qs, observations = rows(output / 'inputs/questions.jsonl'), read(output / 'observations.json')
    assert obj(qs) == p['questions_sha256'] and obj(observations) == p['observations_sha256']
    pending = [t for t in p['tasks'] if not (root / t / 'complete.json').exists()]
    if not pending and (root / 'reader/complete.json').exists():
        return
    model = LocalQwen(p, device)
    dump(root / 'generation_environment.json', model.environment)
    if pending:
        broker = Broker(model, root, p, p['generation_batch_size'])
        encoder = native.install_runtime(broker, p)
        encoder.to(device)

        def task_run(task):
            broker.local.owner = task
            target = root / task
            workdir = target / 'runtime'
            if workdir.exists():
                shutil.rmtree(workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            memory = Memory(method, p, workdir)
            for i, observation in enumerate(observations[task], 1):
                memory.store(observation)
                dump(target / 'progress.json', dict(phase='ingest', done=i, total=len(observations[task])))
            recalled = []
            selected = [q for q in qs if q['task'] == task]
            for q in selected:
                context, detail = memory.recall(q['retrieval_query'])
                recalled.append(dict(id=q['id'], task=task, retrieval_query=q['retrieval_query'],
                    retrieved_context=str(context), retrieval_detail=detail, experiment_signature=p['experiment_signature']))
                write_rows(target / 'retrieval.jsonl', recalled)
                dump(target / 'progress.json', dict(phase='retrieve', done=len(recalled), total=len(selected)))
            memory.export(target)
            dump(target / 'complete.json', dict(experiment_signature=p['experiment_signature'],
                retrieval_sha256=sha(target / 'retrieval.jsonl'), questions=len(recalled)))
            print(json.dumps(dict(method=method, task=task, completed=len(recalled))), flush=True)

        errors = []
        try:
            with ThreadPoolExecutor(max_workers=p['task_workers']) as pool:
                futures = {pool.submit(task_run, task): task for task in pending}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        errors.append(dict(task=futures[future], traceback=traceback.format_exc()))
                        dump(root / 'run_errors.json', errors)
                        print(errors[-1]['traceback'], flush=True)
        finally:
            broker.close()
        if errors:
            raise RuntimeError(f'{len(errors)} scenarios failed; see run_errors.json')
    retrieval = []
    for task in p['tasks']:
        complete = read(root / task / 'complete.json')
        assert complete['experiment_signature'] == p['experiment_signature']
        assert complete['retrieval_sha256'] == sha(root / task / 'retrieval.jsonl')
        retrieval += rows(root / task / 'retrieval.jsonl')
    rmap = {r['id']: r for r in retrieval}
    assert len(rmap) == len(retrieval) == len(qs) and set(rmap) == {q['id'] for q in qs}
    write_rows(root / 'retrieval.jsonl', [rmap[q['id']] for q in qs])
    manifest = []
    for q in qs:
        item = dict(id=q['id'], task=q['task'], point=q['point'], system=gallery.READER_SYSTEM,
            user_text=reader_text(q, rmap[q['id']]['retrieved_context']), query_image_paths=q['query_image_paths'])
        item['payload_sha256'] = obj(item)
        manifest.append(item)
    dest = root / 'reader'; dest.mkdir(exist_ok=True)
    if (dest / 'manifest.jsonl').exists():
        assert rows(dest / 'manifest.jsonl') == manifest
    else:
        write_rows(dest / 'manifest.jsonl', manifest)
    answers = rows(dest / 'responses.jsonl') if (dest / 'responses.jsonl').exists() else []
    done = {r['id']: r for r in answers}
    assert len(done) == len(answers)
    for item in manifest:
        if item['id'] in done:
            assert done[item['id']]['payload_sha256'] == item['payload_sha256']
            assert done[item['id']]['experiment_signature'] == p['experiment_signature']
    for visual in (False, True):
        selected = [m for m in manifest if bool(m['query_image_paths']) == visual and m['id'] not in done]
        size = p['reader_batch_size_images' if visual else 'reader_batch_size_text']
        for start in range(0, len(selected), size):
            batch = selected[start:start + size]
            outputs = gallery.generate_batch((model.processor, model.model, model.environment), batch, 256)
            for item, result in zip(batch, outputs):
                done[item['id']] = dict(id=item['id'], task=item['task'], point=item['point'], **result,
                    payload_sha256=item['payload_sha256'], experiment_signature=p['experiment_signature'])
            write_rows(dest / 'responses.jsonl', [done[m['id']] for m in manifest if m['id'] in done])
            dump(dest / 'progress.json', dict(done=len(done), total=len(manifest), phase='answer'))
            print(json.dumps(dict(method=method, phase='answer', done=len(done), total=len(manifest))), flush=True)
    assert set(done) == set(rmap)
    dump(dest / 'complete.json', dict(questions=len(done), experiment_signature=p['experiment_signature'],
        responses_sha256=sha(dest / 'responses.jsonl'), manifest_sha256=sha(dest / 'manifest.jsonl')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.output.resolve())
    else:
        assert args.method
        run(args.output.resolve(), args.method, args.device)


if __name__ == '__main__':
    main()
