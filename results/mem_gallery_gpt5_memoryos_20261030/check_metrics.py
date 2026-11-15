"""Independently recompute both completed runs from raw answers and gold."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import string
import sys

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
sys.path.insert(0, str(ROOT / '.vendor'))
from nltk.stem import PorterStemmer


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(value):
    text = str(value).lower()
    text = re.sub(r'(?<=\d)\.(?=\d)', 'DOTPLACEHOLDER', text).replace('_', 'UNDERSCOREPLACEHOLDER')
    text = re.sub(r'\b(a|an|the|and)\b', ' ', text)
    text = ''.join(' ' if c in string.punctuation else c for c in text)
    return ' '.join(text.replace('DOTPLACEHOLDER', '.').replace('UNDERSCOREPLACEHOLDER', '_').split())


protocol = read(OUTPUT / 'protocol.json')
stemmer = PorterStemmer()
reports = []
for method, root in [('gen_agents', Path(protocol['ga_control_root'])), ('memoryos', OUTPUT)]:
    assert read(root / 'execution_status.json')['status'] == 'complete'
    answers = rows(root / method / 'reader/responses.jsonl')
    gold = {r['id']: r for r in rows(root / 'inputs/gold.jsonl')}
    qs = rows(root / 'inputs/questions.jsonl')
    scored = {r['id']: r for r in rows(root / 'scored_questions.jsonl')}
    assert len(answers) == len({r['id'] for r in answers}) == len(gold) == len(qs) == len(scored) == 1711
    assert {r['id'] for r in answers} == set(gold) == set(scored) == {q['id'] for q in qs}
    correct, f1s, hits, references = 0, [], 0, 0
    for a in answers:
        g = gold[a['id']]
        pred, target = normalize(a['raw_response']), normalize(g['answer'])
        exact = pred == target
        pt, gt = pred.split(), target.split()
        overlap = sum((Counter(map(stemmer.stem, pt)) & Counter(map(stemmer.stem, gt))).values())
        f1 = Fraction(2 * overlap, len(pt) + len(gt)) if overlap else Fraction()
        correct += exact
        f1s.append(f1)
        s = scored[a['id']]
        assert s['raw_response'] == a['raw_response'] and s['answer'] == g['answer']
        assert s['em'] == exact and abs(s['f1'] - float(f1)) < 1e-12
        found = len(set(g['gold_ids']) & set(s['source_ids']))
        count = len(set(g['gold_ids']))
        assert s['gold_hits'] == found and s['gold_count'] == count
        hits += found
        references += count
    summary = read(root / 'final_results.json')['overall']
    macro_f1 = float(sum(f1s, Fraction()) / 1711)
    assert correct == summary['correct'] and correct / 1711 == summary['em']
    assert abs(macro_f1 - summary['f1']) < 1e-12
    assert hits / references == summary['source_recall_micro'] and references == 4109
    reports.append(dict(method=method, questions=1711, correct=correct, em=correct / 1711,
        f1=macro_f1, source_hits=hits, source_references=references, source_recall_micro=hits / references,
        response_sha256=digest(root / method / 'reader/responses.jsonl'),
        gold_sha256=digest(root / 'inputs/gold.jsonl'), questions_sha256=digest(root / 'inputs/questions.jsonl')))
assert reports[0]['gold_sha256'] == reports[1]['gold_sha256']
assert reports[0]['questions_sha256'] == reports[1]['questions_sha256']
audit = read(OUTPUT / 'audit.json')
assert audit['status'] == 'passed'
for path, expected in audit['artifacts_sha256'].items():
    assert digest(path) == expected, path
result = dict(status='passed', methods=reports, checked_artifacts=len(audit['artifacts_sha256']),
              checker_sha256=digest(Path(__file__)))
(OUTPUT / 'independent_metric_check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
