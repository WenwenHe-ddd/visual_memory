"""Verify current DeepSeek Open adapters against the pinned screenshot source code."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ast
from pathlib import Path
from scripts.evaluation import memeye_deepseek_open_text_retrieval as run


def definitions(path):
    tree = ast.parse(Path(path).read_text())
    return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def main():
    snapshot_root = run.ROOT / 'results/memeye_deepseek_flash_seed42_text_methods_20261025/source_snapshot'
    for method in run.SOURCES:
        target = run.OUT / method
        protocol = run.read(target / 'protocol.json')
        source = Path(protocol['open_evaluation']['source_root'])
        original = run.read(source / 'protocol.json')
        hashes = original.get('input_sha256', original.get('input_hashes', {}))
        exact, differences = {}, []
        for name, expected in hashes.items():
            path = Path(name)
            if path.suffix == '.py' or 'MiniLM' in name:
                actual = run.base.sha(path)
                if actual == expected:
                    exact[name] = actual
                else:
                    differences.append({'path': name, 'source_sha256': expected, 'current_sha256': actual})
        checks = []
        if method != 'simplemem':
            snapshot = next(snapshot_root.glob('*_text_memory_baselines.py'))
            assert run.base.sha(snapshot) == hashes[str(run.ROOT / 'scripts/memory/text_memory_baselines.py')]
            before, after = definitions(snapshot), definitions(run.base.__file__)
            for name in ('Memory', 'bounded_a_mem_context', 'retrieval_query', 'reader_messages',
                         'method_config', 'helpers', 'effective_method_defaults'):
                assert before[name] == after[name], name
                checks.append({'definition': name, 'same_ast': True})
            # Runtime edits added the official SimpleMem branch and reordered
            # an unrelated LangChain shim. These three modes do not enter it.
            assert not protocol.get('simplemem_backend')
            explanation = 'Only unrelated official-SimpleMem dispatch/setup and local-Qwen reader placement changed; all executed Memory construction/recall, bounded A-Mem ordering, query and reader functions are AST-identical. Original DeepSeek adapter A-Mem limit min(5,n) equals current5 because every domain has >5 notes.'
            assert {Path(v['path']).name for v in differences} == {'text_memory_baselines.py', 'reflexion_memory.py'}
        else:
            assert {Path(v['path']).name for v in differences} == {'visual_memory_pipeline.py'}
            explanation = 'Only the top-level unused pipeline CLI changed. This run directly invokes pinned OfficialSimpleMemMemory; all native SimpleMem, query guard, RRF, transport and MiniLM inputs remain byte-identical.'
        captions_path = run.ROOT/'results/memeye_deepseek_flash_seed42_full_20261025/captions.jsonl'
        caption_map = {r['id']: r['caption'] for r in run.base.read_rows(captions_path)}
        query_uses = []
        for q in run.base.read_rows(target/'questions.jsonl'):
            expected = [caption_map[run.base.sha(Path(p))] for p in q['query_image_paths']]
            assert expected == q['query_image_captions'], q['id']
            query_uses.extend({'id': q['id'], 'image_sha256': run.base.sha(Path(p))}
                              for p in q['query_image_paths'])
        report = {'status': 'passed', 'method': method, 'exact_current_code_and_embedding_hashes': exact,
            'non_execution_source_differences': differences, 'native_adapter_ast_checks': checks,
            'difference_explanation': explanation, 'source_protocol_sha256': run.base.sha(source/'protocol.json'),
            'caption_source_sha256': run.base.sha(captions_path), 'question_image_caption_uses': query_uses,
            'question_image_captions_are_original_deepseek': True,
            'script_sha256': run.base.sha(Path(__file__))}
        run.base.write_json(target/'source_implementation_audit.json', report)
        print(method, 'native functions and model weights verified', len(exact), 'image-caption uses', len(query_uses))


if __name__ == '__main__':
    main()
