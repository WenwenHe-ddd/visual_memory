"""Self-describing, read-only online stores exported from existing offline memories."""
import hashlib
import json
from pathlib import Path
import shutil

from scripts.memory.caption_visual_master import read_json, sha, write_json

FORMAT = 'dialogue + newline + Memory description: + full caption'


def read_manifest(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'memory.json')
    if manifest.get('schema_version') != 1 or manifest.get('offline_method') not in ('natural', 'caption_visual_master'):
        raise ValueError('Unsupported memory.json schema or offline_method.')
    if manifest.get('leaf_text_format') != FORMAT:
        raise ValueError('Unsupported memory text format.')
    required = {'leaves'} | ({'graph', 'visual_sums'} if manifest['offline_method'] == 'caption_visual_master' else set())
    if not required <= manifest['files'].keys():
        raise ValueError('Memory manifest is missing required files.')
    for ref in manifest['files'].values():
        path = (directory / ref['path']).resolve()
        if not path.is_relative_to(directory) or sha(path) != ref['sha256']:
            raise ValueError(f'Memory file differs from its manifest: {ref["path"]}')
    return manifest


def leaf_truncations(protocol, leaves):
    """Preserve only explicitly recorded historical truncations, pinned to full text."""
    by_id = {leaf['image_id']: leaf for leaf in leaves}
    result = []
    for record in protocol.get('bge_encoding', {}).get('truncated_inputs', []):
        if record['id'] not in by_id:
            continue
        leaf = by_id[record['id']]
        text = leaf['dialogue'] + '\nMemory description: ' + leaf['caption']
        result.append({**record, 'text_sha256': hashlib.sha256(text.encode()).hexdigest()})
    return result


def export_store(source, destination, offline_method):
    """Package an existing domain's offline output; never use questions or gold labels."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Export to a fresh directory to preserve existing memory.')
    protocol_path = source / 'protocol.json'
    protocol = read_json(protocol_path)
    tracked = [protocol_path]
    graph_dir = source / 'memory'
    if offline_method == 'natural':
        tracked.append(source / 'inputs.json')
        raw_leaves = read_json(tracked[-1])['leaves']
    elif offline_method == 'caption_visual_master':
        tracked.extend(graph_dir / name for name in ('graph.json', 'master_sums.npy'))
        graph = read_json(graph_dir / 'graph.json')
        if graph['method'] != 'caption_ward_silhouette_visual':
            raise ValueError('Source is not a caption_visual_master offline memory.')
        raw_leaves = [{**n, 'image_id': n['id'], 'memory_id': n['id'].rsplit('#image', 1)[0]}
                      for n in graph['image_nodes']]
    else:
        raise ValueError('Unsupported offline method.')
    # Strip QA and unrelated experimental annotations from the portable text store.
    leaves = [{key: leaf[key] for key in ('image_id', 'memory_id', 'dialogue', 'caption')} for leaf in raw_leaves]
    domains = {leaf['image_id'].split('/', 1)[0] for leaf in leaves}
    if len(domains) != 1 or len({l['image_id'] for l in leaves}) != len(leaves):
        raise ValueError('Export requires one domain with unique leaf IDs.')
    hashes = {str(path): sha(path) for path in tracked}
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / 'leaves.json', leaves)
    files = {'leaves': 'leaves.json'}
    if offline_method == 'caption_visual_master':
        for key, filename in [('graph', 'graph.json'), ('visual_sums', 'master_sums.npy')]:
            shutil.copyfile(graph_dir / filename, destination / filename)
            files[key] = filename
    encoding = {key: protocol[key] for key in ('text_encoder', 'query_prefix')}
    encoding['allowed_leaf_truncations'] = leaf_truncations(protocol, leaves)
    if offline_method == 'caption_visual_master':
        encoding['image_encoder'] = protocol['image_encoder']
    write_json(destination / 'memory.json', {
        'schema_version': 1, 'offline_method': offline_method,
        'domain': next(iter(domains)), 'leaf_count': len(leaves), 'leaf_text_format': FORMAT,
        'files': {key: {'path': filename, 'sha256': sha(destination / filename)} for key, filename in files.items()},
        'encoding': encoding, 'source_sha256': hashes,
        'export_policy': 'existing offline memory; no new clustering; no QA data included',
    })
    for path, expected in hashes.items():
        if sha(Path(path)) != expected:
            raise ValueError('Offline source changed during export.')
    return read_manifest(destination)
