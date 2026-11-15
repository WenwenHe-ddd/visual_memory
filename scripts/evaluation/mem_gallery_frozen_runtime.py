"""Allow only AST-identical comment edits against the signed source snapshot.

Shared workspaces can receive inline comments while an experiment is running.
Every byte mismatch remains recorded. Executable changes are still rejected.
"""
import ast
from pathlib import Path

from scripts.evaluation import mem_gallery_qwen_text_methods_v2 as experiment


def verify(output, method):
    base = experiment.base
    p = base.read(output / method / 'protocol.json')
    assert p['experiment_signature'] == base.obj({k: v for k, v in p.items() if k != 'experiment_signature'})
    base.native.verify_vendor()
    snapshots = base.read(output / 'source_snapshot/manifest.json')
    exceptions = []
    for name, expected in p['input_sha256'].items():
        current = base.sha(name)
        if current == expected:
            continue
        assert name in snapshots and Path(name).suffix == '.py', name
        snapshot = Path(snapshots[name]['path'])
        assert base.sha(snapshot) == snapshots[name]['sha256'] == expected, name
        before = ast.dump(ast.parse(snapshot.read_text()), include_attributes=False)
        after = ast.dump(ast.parse(Path(name).read_text()), include_attributes=False)
        assert before == after, 'Executable source changed: ' + name
        exceptions.append(dict(path=name, frozen_snapshot=str(snapshot),
            expected_sha256=expected, current_sha256=current, ast_sha256=base.obj(before),
            equivalence='Identical Python AST including docstrings; only comments/formatting differ.'))
    proof = dict(policy='Exact hashes required, except AST-identical Python comment/formatting edits verified against immutable snapshots.',
        verifier_sha256=base.sha(Path(__file__)), exceptions=exceptions)
    base.freeze(output / 'source_comment_equivalence.json', proof)
    return p


if __name__ == '__main__':
    experiment.base.verify = verify
    experiment.main()
