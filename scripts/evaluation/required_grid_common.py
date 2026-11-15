"""Shared bookkeeping for the user-requested eight-method, three-seed grid."""
from __future__ import annotations
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/required_grid_seed42_1024_2026_20261113'
METHODS = ('semrag', 'ours_1_ours_2', 'gen_agents', 'memoryos', 'simplemem', 'a_mem', 'clip_image_dialogue', 'hippomm')
LABELS = dict(zip(METHODS, ('Semantic RAG', 'ours_1 + ours_2 (lambda=.03, alpha=1)', 'Generative Agents (Top10)',
    'MemoryOS', 'SimpleMem official (final5)', 'A-Mem (final5)', 'VisRAG (CLIP/images/dialogue)', 'HippoMM')))
MODELS = {'qwen': 'Qwen3-VL-8B-Instruct', 'gpt5': 'gpt-5', 'deepseek': 'deepseek-v4.1-flash'}

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)

def write_rows(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    content = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in values)
    if path.exists():
        if path.read_text() != content:
            raise ValueError('Frozen result differs: ' + str(path))
        return
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)

def freeze(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError('Frozen protocol differs: ' + str(path))
    else:
        dump(path, value)

def hashes(values):
    for path, expected in values.items():
        if sha(path) != expected:
            raise ValueError('Pinned file differs: ' + path)

@contextmanager
def lock(path, blocking=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield

def clone(path, name, seed=None, seed_lines=(), replacements=None, register=False, old_seed=1024):
    """Change explicitly reviewed seed literals only; byte/token sizes stay intact."""
    path = Path(path); tree = ast.parse(path.read_text())
    positions = set(seed_lines)
    seen = set()
    class Adapt(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is int and node.value == old_seed and node.lineno in positions:
                seen.add(node.lineno)
                return ast.copy_location(ast.Constant(seed), node)
            if isinstance(node.value, str):
                value = node.value
                for old, new in (replacements or {}).items():
                    value = value.replace(old, new)
                if value != node.value:
                    return ast.copy_location(ast.Constant(value), node)
            return node
    tree = ast.fix_missing_locations(Adapt().visit(tree))
    if seen != positions:
        raise ValueError('Reviewed seed locations changed: ' + str(path))
    module = types.ModuleType(name); module.__file__ = str(path)
    module.__package__ = 'scripts.evaluation'
    if register:
        sys.modules[name] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module

def job_protocol(job, sources):
    output = Path(job['method_root'])
    output.mkdir(parents=True, exist_ok=True)
    value = dict(job=job, code_sha256={str(Path(p).resolve()): sha(p) for p in sources},
        original_caption_policy='Preserve the same-backbone seed42 cached image captions; fresh memory/retrieval/answers.',
        judge_seed=42, created_for='Explicit user request to fill every missing grid result.')
    freeze(output / 'grid_protocol.json', value)
    return value

def complete(job, mcq, opened, judge_root, evidence, extra=None):
    """One canonical answer and a real five-grade Judge result per question."""
    expected = job['questions']
    def indexed(values):
        found = {r['id']: r for r in values}
        if len(values) != len(found) or len(found) != expected:
            raise ValueError('Incomplete/duplicate question cohort')
        return found
    gold_path = (ROOT / 'results/mcq_top15_caption_qa/qa_manifest.jsonl' if job['dataset']=='memeye'
                 else ROOT / 'results/mem_gallery_master_answer_20261021/inputs/gold.jsonl')
    canonical = indexed(rows(gold_path))
    current = indexed(opened)
    if set(current) != set(canonical):
        raise ValueError('Open canonical IDs differ')
    scores = indexed(rows(Path(judge_root) / 'per_question.jsonl'))
    if set(scores) != set(canonical):
        raise ValueError('Judge canonical IDs differ')
    if read(Path(judge_root) / 'protocol.json')['seed'] != 42:
        raise ValueError('Judge seed differs')
    for qid, value in current.items():
        if scores[qid]['model_output'] != value['model_output'] or scores[qid]['score'] not in (0,.25,.5,.75,1):
            raise ValueError('Judge is not tied to the original new answer')
        value['judge_score'] = scores[qid]['score']
    correct = None
    if job['dataset']=='memeye':
        answers = indexed(mcq)
        if set(answers) != set(canonical):
            raise ValueError('MCQ canonical IDs differ')
        for qid, row in answers.items():
            if row['gold_choice'] != canonical[qid]['gold_choice']:
                raise ValueError('MCQ gold differs')
            actual = row.get('predicted_choice') == row['gold_choice']
            if row['correct'] != actual:
                raise ValueError('MCQ regrading differs')
        correct = sum(r['correct'] for r in answers.values())
    output = Path(job['method_root'])
    if mcq is not None:
        write_rows(output / 'mcq_records.jsonl', mcq)
    write_rows(output / 'open_records.jsonl', list(current.values()))
    proof = dict(status='complete', dataset=job['dataset'], backbone=job['backbone'], model=MODELS[job['backbone']],
        seed=job['seed'], caption_seed=42, judge_seed=42, method=job['method'], questions=expected,
        correct=correct, accuracy=correct/expected if correct is not None else None,
        llm_judge=sum(r['score'] for r in scores.values())/expected, open_questions=expected, judge_questions=expected,
        source_sha256={str(Path(p).resolve()): sha(p) for p in [*evidence, Path(judge_root)/'per_question.jsonl', gold_path]},
        completed_at=now(), **(extra or {}))
    artifacts=[output/'open_records.jsonl',output/'grid_protocol.json']
    if mcq is not None:artifacts.append(output/'mcq_records.jsonl')
    proof['artifact_sha256']={str(p):sha(p) for p in artifacts}
    dump(output/'completion.json',proof)
    return proof
