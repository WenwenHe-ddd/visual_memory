"""Opt-in M2A evidence compatibility policy, without changing frozen runs.

Pairs retain inclusive range semantics. Singletons and lists of three or more
IDs denote explicit individual messages; never widen such lists into a range.
"""
import json
from pathlib import Path
import sqlite3

POLICY = {
    'version': 'm2a-evidence-v1',
    'pair': 'inclusive start/end range, unchanged',
    'singleton': '[[n]] -> [[n,n]]',
    'long_list': '[[a,b,c,...]] -> [[a,a],[b,b],[c,c],...]',
    'invalid': 'reject invalid JSON, non-list groups, non-positive/non-integer IDs, reversed ranges and unknown raw IDs',
    'legacy_reads': 'normalize returned memory object only; do not rewrite source DB',
}


def normalize_evidence(value, valid_ids=None):
    """Return canonical ranges, or fail before changing persistent state."""
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, list):
        raise ValueError('Evidence must be a JSON list of ID groups, e.g. [[1,3],[5,5]]')
    result = []
    for group in data:
        if not isinstance(group, list) or not group:
            raise ValueError('Each evidence group must be a nonempty list of positive integer IDs')
        if any(type(n) is not int or n <= 0 for n in group):
            raise ValueError('Evidence IDs must be positive integers (not strings, floats or booleans)')
        if len(group) == 2:
            if group[0] > group[1]:
                raise ValueError('Evidence range start must not exceed end')
            result.append(list(group))
        else:
            result.extend([[n, n] for n in group])
    if valid_ids is not None:
        known = set(valid_ids)
        for start, end in result:
            if end - start + 1 > len(known) or any(n not in known for n in range(start, end + 1)):
                raise ValueError(f'Evidence range [{start},{end}] includes unknown raw message IDs')
    return result


def raw_ids(raw):
    with sqlite3.connect(raw.db_path) as db:
        return {r[0] for r in db.execute('SELECT msg_id FROM messages')}


def install_evidence_policy(output):
    """Wrap the native tool/store interfaces in this process; return cleanup.

The native tools, prompts and retrieval fusion remain in use. Tool failures are
returned to the existing agent loop so the model can correct its arguments.
"""
    from agent.agents.memory_manager import MemoryManagerTools
    from agent.stores import SemanticStore
    from scripts.memory.m2a_local import append_json

    log = Path(output) / 'evidence_validation.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    originals = {
        'add_tool': MemoryManagerTools.get_add_memory,
        'fetch_tool': MemoryManagerTools.get_fetch_raw_messages,
        'add_store': SemanticStore.add,
        'read_store': SemanticStore._get_memory_by_id,
    }

    def checked(value, operation, valid_ids=None):
        try:
            ranges = normalize_evidence(value, valid_ids)
        except (ValueError, TypeError) as exc:
            append_json(log, dict(operation=operation, status='rejected', original=value, error=str(exc)))
            raise
        data = json.loads(value) if isinstance(value, str) else value
        if ranges != data:
            append_json(log, dict(operation=operation, status='normalized', original=value, canonical=ranges))
        return ranges

    def add_tool(self):
        tool = originals['add_tool'](self)
        native = tool.func

        def invoke(text, image=None, image_caption=None, evidence_ids='[]'):
            try:
                ranges = checked(evidence_ids, 'add_memory', raw_ids(self.raw))
            except (ValueError, TypeError) as exc:
                return 'Error creating memory: ' + str(exc) + '. Use [[start,end], ...] referencing existing messages.'
            return native(text=text, image=image, image_caption=image_caption,
                          evidence_ids=json.dumps(ranges))

        tool.func = invoke
        return tool

    def fetch_tool(self):
        tool = originals['fetch_tool'](self)
        native = tool.func

        def invoke(id_ranges):
            try:
                ranges = checked(id_ranges, 'fetch_raw_messages', raw_ids(self.raw))
            except (ValueError, TypeError) as exc:
                return 'Error parsing ID ranges: ' + str(exc) + '. Use [[start,end], ...] referencing existing messages.'
            return native(id_ranges=json.dumps(ranges))

        tool.func = invoke
        return tool

    def add_store(self, memory):
        memory.evidence_ids = checked(memory.evidence_ids, 'semantic_add')
        return originals['add_store'](self, memory)

    def read_store(self, memory_id):
        memory = originals['read_store'](self, memory_id)
        memory.evidence_ids = checked(memory.evidence_ids, 'semantic_read:' + str(memory_id))
        return memory

    MemoryManagerTools.get_add_memory = add_tool
    MemoryManagerTools.get_fetch_raw_messages = fetch_tool
    SemanticStore.add = add_store
    SemanticStore._get_memory_by_id = read_store

    def cleanup():
        MemoryManagerTools.get_add_memory = originals['add_tool']
        MemoryManagerTools.get_fetch_raw_messages = originals['fetch_tool']
        SemanticStore.add = originals['add_store']
        SemanticStore._get_memory_by_id = originals['read_store']

    return cleanup
