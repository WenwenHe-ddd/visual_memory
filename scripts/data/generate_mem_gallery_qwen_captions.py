"""Generate versioned, resumable Qwen captions without modifying source data."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'data/Mem-Gallery'
DEFAULT_OUT = BASE/'captions/qwen3-vl-8b-instruct_20261018_v1'
FIELD = 'image_caption_qwen3_vl_8b_instruct'
sys.path.insert(0, str(ROOT/'.vendor'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, data):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def jl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def write_jl(path, rows):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(''.join(json.dumps(x, ensure_ascii=False, allow_nan=False)+'\n' for x in rows))
    tmp.replace(path)


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise ValueError('Output already initialized; use --resume to resume this exact configuration')
    cfg = json.loads((ROOT/'results/natural_observation/protocol.json').read_text())
    source = json.loads((BASE/'source_manifest.json').read_text())
    images = json.loads((BASE/'image_manifest.json').read_text())
    assert source['revision'] == images['revision']
    groups = {}
    for image in sorted(images['files'], key=lambda r: r['path']):
        path = BASE/image['path']
        assert sha(path) == image['sha256'], path
        groups.setdefault(image['sha256'], []).append(image['path'])
    jobs = [{'id': digest, 'image_sha256': digest, 'representative_path': paths[0],
             'source_paths': paths, 'worker_index': i % 2}
            for i, (digest, paths) in enumerate(groups.items())]
    inputs = {f['path']: f['sha256'] for f in source['files']}
    for name in ('source_manifest.json', 'image_manifest.json'):
        path = BASE/name
        inputs[str(path.relative_to(ROOT))] = sha(path)
    for path, digest in inputs.items():
        assert sha(ROOT/path) == digest
    model_path = Path(cfg['caption_model_path'])
    model_files = {str(p.relative_to(model_path)): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                   for p in sorted(model_path.rglob('*')) if p.is_file()}
    protocol = {'dataset_repo': source['repo'], 'dataset_revision': source['revision'],
                'model': cfg['provenance']['caption_model'], 'model_path': str(model_path),
                'caption_prompt': cfg['caption_prompt'], 'seed': cfg['seed'], 'do_sample': False,
                'dtype': 'bfloat16', 'attn_implementation': 'sdpa', 'batch_size': cfg['batch_size'],
                'image_processor_size': {'shortest_edge': 65536, 'longest_edge': cfg['image_max_pixels']},
                'initial_max_new_tokens': cfg['caption_max_new_tokens'],
                'retry_max_new_tokens': [512, 1024],
                'retry_policy': 'only empty or EOS-missing capped outputs; identical image and prompt; record every attempt',
                'workers': 2, 'physical_gpus': [0, 1],
                'image_policy': 'PIL first frame converted to RGB, matching existing generator; GIF/MPO first frame only',
                'deduplication': 'exact image-file SHA256; one new caption reused across all identical files',
                'prompt_inputs': 'image plus generic caption instruction only; no question, answer, supplied caption or dialogue',
                'source_input_hashes': inputs, 'model_files_before': model_files,
                'source_image_count': len(images['files']), 'unique_image_count': len(jobs),
                'derived_caption_field': FIELD, 'script_sha256': sha(Path(__file__))}
    protocol['protocol_signature'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    dump(out/'protocol.json', protocol)
    write_jl(out/'image_jobs.jsonl', jobs)
    dump(out/'status.json', {'status': 'prepared', 'unique_images': len(jobs), 'source_files': len(images['files'])})
    print(f'Prepared {len(jobs)} unique images from {len(images["files"])} files in {out}', flush=True)


def worker(out, index):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    cfg = json.loads((out/'protocol.json').read_text())
    jobs = [r for r in jl(out/'image_jobs.jsonl') if r['worker_index'] == index]
    result_path = out/f'captions.worker{index}.jsonl'
    attempt_path = out/f'attempts.worker{index}.jsonl'
    existing = jl(result_path)
    assert len({r['id'] for r in existing}) == len(existing)
    assert all(r['protocol_signature'] == cfg['protocol_signature'] and not r['hit_token_limit'] and r['caption'].strip() for r in existing)
    expected = {r['id'] for r in jobs}
    assert {r['id'] for r in existing} <= expected
    done = {r['id'] for r in existing}
    pending = [r for r in jobs if r['id'] not in done]
    if not pending:
        dump(out/f'worker{index}_status.json', {'status': 'complete', 'completed': len(done), 'total': len(jobs)})
        return
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(8)
    assert torch.cuda.is_available(), 'CUDA is required'
    device = 'cuda:0'
    started = time.time()
    print(f'Worker {index}: loading model on physical GPU {os.environ.get("CUDA_VISIBLE_DEVICES")}', flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(cfg['model_path'], dtype=torch.bfloat16,
             attn_implementation=cfg['attn_implementation'], local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(cfg['model_path'], local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    processor.image_processor.size = cfg['image_processor_size']
    env = {'torch': torch.__version__, 'transformers': transformers.__version__,
           'gpu_name': torch.cuda.get_device_name(0), 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'processor_size': processor.image_processor.size, 'model_load_seconds': time.time()-started}
    dump(out/f'environment.worker{index}.json', env)
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [processor.tokenizer.eos_token_id])

    def generate(batch, cap):
        rendered, images = [], []
        for row in batch:
            with Image.open(BASE/row['representative_path']) as image:
                images.append(image.convert('RGB'))
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': cfg['caption_prompt']}]}]
            rendered.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = processor(text=rendered, images=images, padding=True, return_tensors='pt').to(device)
        try:
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=cap, do_sample=False,
                                        pad_token_id=processor.tokenizer.pad_token_id)
            generated = output[:, inputs['input_ids'].shape[1]:]
            texts = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result = []
            for text, tokens in zip(texts, generated.tolist()):
                end = next((i for i, token in enumerate(tokens) if token in eos), None)
                length = end+1 if end is not None else len(tokens)
                truncated = end is None and length >= cap
                result.append({'caption': text.strip(), 'tokens': length, 'hit_token_limit': truncated,
                               'ended_with_eos': end is not None, 'max_new_tokens_used': cap})
            return result
        finally:
            for image in images:
                image.close()

    for offset in range(0, len(pending), cfg['batch_size']):
        batch = pending[offset:offset+cfg['batch_size']]
        tick = time.time()
        unresolved = list(batch)
        accepted = {}
        for retry, cap in enumerate([cfg['initial_max_new_tokens'], *cfg['retry_max_new_tokens']]):
            if not unresolved:
                break
            outputs = generate(unresolved, cap)
            next_round = []
            with attempt_path.open('a') as handle:
                for job, result in zip(unresolved, outputs):
                    record = {'id': job['id'], **result, 'retry_index': retry, 'worker_index': index,
                              'protocol_signature': cfg['protocol_signature']}
                    handle.write(json.dumps(record, ensure_ascii=False)+'\n')
                    if result['caption'] and result['ended_with_eos'] and not result['hit_token_limit']:
                        accepted[job['id']] = {**job, **record}
                    else:
                        next_round.append(job)
            unresolved = next_round
        with result_path.open('a') as handle:
            for job in batch:
                if job['id'] in accepted:
                    handle.write(json.dumps(accepted[job['id']], ensure_ascii=False)+'\n')
                    done.add(job['id'])
            handle.flush()
            os.fsync(handle.fileno())
        elapsed = time.time()-started
        dump(out/f'worker{index}_status.json', {'status': 'running', 'completed': len(done), 'total': len(jobs),
             'elapsed_seconds': elapsed, 'last_batch_seconds': time.time()-tick, 'updated_unix': time.time()})
        print(f'captions {len(done)}/{len(jobs)} batch_seconds={time.time()-tick:.1f}', flush=True)
        if unresolved:
            raise RuntimeError('Unresolved empty or truncated outputs: '+','.join(r['id'] for r in unresolved))
    dump(out/f'worker{index}_status.json', {'status': 'complete', 'completed': len(done), 'total': len(jobs),
         'elapsed_seconds': time.time()-started, 'updated_unix': time.time()})


def finalize(out):
    cfg = json.loads((out/'protocol.json').read_text())
    jobs = jl(out/'image_jobs.jsonl')
    records = [r for i in range(cfg['workers']) for r in jl(out/f'captions.worker{i}.jsonl')]
    by_hash = {r['id']: r for r in records}
    assert len(records) == len(by_hash) == len(jobs)
    assert set(by_hash) == {r['id'] for r in jobs}
    assert all(r['caption'].strip() and r['ended_with_eos'] and not r['hit_token_limit'] for r in records)
    assert all(r['protocol_signature'] == cfg['protocol_signature'] for r in records)
    source = json.loads((BASE/'source_manifest.json').read_text())
    images = json.loads((BASE/'image_manifest.json').read_text())
    image_records = {str((BASE/r['path']).resolve()): r for r in images['files']}
    def image_info(source_path, image_path):
        p = (source_path.parent/image_path).resolve()
        item = image_records[str(p)]
        assert sha(p) == item['sha256'], p
        return item['sha256'], by_hash[item['sha256']]['caption']
    histories, queries, mappings = [], [], []
    derived = out/'dialog_with_qwen_caption'
    derived.mkdir(exist_ok=True)
    for item in source['files']:
        path = ROOT/item['path']
        original = json.loads(path.read_text())
        data = copy.deepcopy(original)
        for session in data['multi_session_dialogues']:
            for row in session['dialogues']:
                if not row.get('input_image'):
                    continue
                pairs = [image_info(path, image_path) for image_path in row['input_image']]
                assert FIELD not in row
                row[FIELD] = [caption for digest, caption in pairs]
                histories.append({'id': path.stem+'/'+row['round'], 'task': path.stem, 'round': row['round'],
                                  'input_image': row['input_image'], 'image_sha256': [digest for digest, caption in pairs],
                                  FIELD: row[FIELD], 'original_image_caption': row.get('image_caption', [])})
        for index, qa in enumerate(data['human-annotated QAs']):
            if not qa.get('question_image'):
                continue
            digest, caption = image_info(path, qa['question_image'])
            assert FIELD not in qa
            qa[FIELD] = caption
            queries.append({'id': path.stem+f'/QA_{index:04d}', 'task': path.stem, 'qa_index_0_based': index,
                            'question_image': qa['question_image'], 'image_sha256': digest,
                            FIELD: caption, 'original_image_caption': qa.get('image_caption')})
        dump(derived/path.name, data)
        # Removing only our added fields must exactly recover the original JSON.
        reconstructed = copy.deepcopy(data)
        for session in reconstructed['multi_session_dialogues']:
            for row in session['dialogues']:
                row.pop(FIELD, None)
        for qa in reconstructed['human-annotated QAs']:
            qa.pop(FIELD, None)
        assert reconstructed == original
    for image in images['files']:
        mappings.append({'image_path': image['path'], 'image_sha256': image['sha256'],
                         FIELD: by_hash[image['sha256']]['caption']})
    write_jl(out/'captions_by_image.jsonl', [by_hash[job['id']] for job in jobs])
    write_jl(out/'history_captions.jsonl', histories)
    write_jl(out/'query_captions.jsonl', queries)
    write_jl(out/'image_path_captions.jsonl', mappings)
    for path, digest in cfg['source_input_hashes'].items():
        assert sha(ROOT/path) == digest, path
    model_path = Path(cfg['model_path'])
    model_hashes = {}
    for relative, stat in cfg['model_files_before'].items():
        path = model_path/relative
        assert path.stat().st_size == stat['bytes'] and path.stat().st_mtime_ns == stat['mtime_ns']
        model_hashes[relative] = sha(path)
    dump(out/'model_file_sha256.json', model_hashes)
    attempts = [r for i in range(cfg['workers']) for r in jl(out/f'attempts.worker{i}.jsonl')]
    audit = {'status': 'complete', 'unique_generated_captions': len(records), 'mapped_image_files': len(mappings),
             'history_round_records': len(histories), 'question_image_records': len(queries),
             'derived_dialog_files': len(source['files']), 'original_source_files_unchanged': True,
             'derived_original_fields_exactly_preserved': True, 'empty_final_captions': 0, 'truncated_final_captions': 0,
             'generation_attempts': len(attempts), 'captions_requiring_larger_limit': sum(r['retry_index']>0 for r in records),
             'max_output_tokens_observed': max(r['tokens'] for r in records),
             'model_weight_and_configuration_hashes_saved': True, 'protocol_signature': cfg['protocol_signature']}
    dump(out/'audit.json', audit)
    dump(out/'status.json', audit)
    dump(out/'artifact_manifest.json', {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
         if p.is_file() and p.name!='artifact_manifest.json' and p.suffix!='.log'})
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--worker-index', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--finalize-only', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.worker_index is not None:
        worker(out, args.worker_index)
        return
    if args.finalize_only:
        finalize(out)
        return
    if not args.resume:
        prepare(out)
    else:
        cfg = json.loads((out/'protocol.json').read_text())
        assert cfg['script_sha256'] == sha(Path(__file__)), 'Resume requires unchanged script'
    started = time.time()
    dump(out/'status.json', {'status': 'running', 'started_unix': started})
    children, logs = [], []
    for index in range(2):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(index)
        env['OPENBLAS_NUM_THREADS'] = '8'
        env['OMP_NUM_THREADS'] = '8'
        log = (out/f'worker{index}.log').open('a')
        logs.append(log)
        children.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--output-dir', str(out),
                                          '--worker-index', str(index)], env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [child.wait() for child in children]
    for log in logs:
        log.close()
    if any(codes):
        dump(out/'status.json', {'status': 'failed', 'worker_exit_codes': codes, 'elapsed_seconds': time.time()-started})
        raise RuntimeError(f'Worker exit codes: {codes}; inspect worker logs and resume')
    print('All GPU workers finished; validating mappings and immutable source files', flush=True)
    finalize(out)
    print(f'Total worker and finalization seconds: {time.time()-started:.1f}', flush=True)


if __name__ == '__main__':
    main()
