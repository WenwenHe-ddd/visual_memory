"""Run the Gallery experiment from its verified, immutable Python snapshots.

The original module location remains available for relative resource lookup;
executed code and traceback filenames come from the signed snapshot instead.
Shared working-tree edits are recorded, never reverted or used by this run.
"""
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/mem_gallery_qwen_seed42_text_methods_20261026_v2'
SNAPSHOTS = json.loads((OUTPUT / 'source_snapshot/manifest.json').read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SnapshotLoader(importlib.abc.Loader):
    def __init__(self, original, entry):
        self.original, self.entry = original, entry

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path = Path(self.entry['path'])
        source = path.read_bytes()
        assert hashlib.sha256(source).hexdigest() == self.entry['sha256'], str(path)
        module.__file__ = self.original
        module.__experiment_snapshot__ = dict(self.entry)
        exec(compile(source, str(path), 'exec'), module.__dict__)


class SnapshotFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        original = str(ROOT.joinpath(*fullname.split('.')).with_suffix('.py'))
        if original not in SNAPSHOTS:
            return None
        return importlib.util.spec_from_loader(fullname,
            SnapshotLoader(original, SNAPSHOTS[original]), origin=original)


FINDER = SnapshotFinder()
sys.meta_path.insert(0, FINDER)
experiment = importlib.import_module('scripts.evaluation.mem_gallery_qwen_text_methods_v2')


def verify(output, method):
    base = experiment.base
    assert Path(output).resolve() == OUTPUT
    p = base.read(output / method / 'protocol.json')
    assert p['experiment_signature'] == base.obj({k: v for k, v in p.items() if k != 'experiment_signature'})
    base.native.verify_vendor()
    assert FINDER in sys.meta_path
    for module in tuple(sys.modules.values()):
        original = getattr(module, '__file__', None)
        if original in SNAPSHOTS:
            assert getattr(module, '__experiment_snapshot__', None) == SNAPSHOTS[original], original
    differences = []
    for name, expected in p['input_sha256'].items():
        if name in SNAPSHOTS:
            entry = SNAPSHOTS[name]
            assert digest(entry['path']) == entry['sha256'] == expected, name
            current = digest(name)
            if current != expected:
                differences.append(dict(path=name, current_sha256=current,
                    executed_sha256=expected, executed_snapshot=entry['path']))
        else:
            assert digest(name) == expected, name
    base.freeze(output / 'snapshot_runtime.json', dict(
        policy='All snapshotted Python modules execute verified snapshot bytes; other inputs require exact original hashes.',
        loader_sha256=digest(__file__), sources=SNAPSHOTS,
        original_file_attribute='Retained only for relative data/vendor/model paths; code filenames identify snapshots.',
        completed_run_provenance='GA and MemoryOS loaded the original source before working-tree edits; '
            'the earlier hash/AST guard passed before generation. Their saved requests and outputs are reused unchanged.'))
    base.dump(output / 'snapshot_verification' / (str(time.time_ns()) + '_' + method + '.json'),
        dict(method=method, argv=sys.argv, differences=differences, status='passed'))
    return p


if __name__ == '__main__':
    experiment.base.verify = verify
    experiment.main()
