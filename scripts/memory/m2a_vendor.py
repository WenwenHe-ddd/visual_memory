"""Build an audited runnable copy of the pinned M2A upstream source."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'scripts/vendor_m2a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_vendor():
    metadata = json.loads((VENDOR / 'UPSTREAM.json').read_text())
    changes = []
    for name, expected in metadata['sha256'].items():
        source = VENDOR / 'upstream' / name
        if sha(source) != expected:
            raise ValueError('Upstream source changed: ' + name)
        text = source.read_text()

        def replace(old, new, reason, count=None):
            nonlocal text
            found = text.count(old)
            if not found or (count is not None and found != count):
                raise ValueError(f'Patch mismatch {name}: {old[:70]} ({found})')
            text = text.replace(old, new)
            changes.append({'file': name, 'reason': reason, 'occurrences': found,
                            'old': old, 'new': new})

        if name in ('agent/agents/chat_agent.py', 'agent/agents/memory_manager.py'):
            replace('from langchain.tools import tool', 'from langchain_core.tools import tool',
                    'Use the core tool decorator without importing optional prebuilt agent dependencies.')
            replace('from tkinter import END', 'from langgraph.graph import END',
                    'Use the LangGraph terminal sentinel; tkinter.END is a nonterminal string.')
            replace('        workflow.add_node(END, lambda x: x)\n', '',
                    'A reserved LangGraph END sentinel must not be added as a node.')
        if name == 'agent/embeddings/text.py':
            replace('from langchain.embeddings.base import Embeddings',
                    'from langchain_core.embeddings import Embeddings',
                    'LangChain 1.x moved the Embeddings interface.')
        if name == 'agent/agents/memory_manager.py':
            replace('"configurable": {"thread_id": "1"}',
                    '"configurable": {"thread_id": "1"}, "recursion_limit": 100',
                    'Allow graph transitions for the explicit 15-tool-call budget.')
            replace('        self.config = config\n',
                    '        self.config = config\n        self.max_iteration = config.max_iteration\n',
                    'Honor the configured memory-manager tool-call limit.')
            replace('        response = self.llm.bind_tools([',
                    '        if state.iteration_count >= self.max_iteration:\n'
                    '            response = self.llm.invoke(messages)\n'
                    '            messages.append(response)\n'
                    '            content = (self.image_manager.format_msg_to_content(response.content)\n'
                    '                       if state.operation == "query" else response.content)\n'
                    '            return Command(update={"messages": messages, "response": content}, goto=END)\n'
                    '        response = self.llm.bind_tools([',
                    'Stop binding tools after the tool budget rather than leaving an open-ended loop.', 2)
            replace('if state.iteration_count == self.max_iteration:',
                    'if state.iteration_count > self.max_iteration:',
                    'Allow exactly the configured number of tool calls.')
            # Route tool errors back to the agent instead of terminating the graph without a response.
            begin = text.index('    def _exec_tool(')
            end = text.index('    def _fill_update_sys_prompt', begin)
            block = text[begin:end]
            patched = block.replace('return state',
                'return Command(update={"messages": state.messages, "iteration_count": state.iteration_count}, '
                'goto="handle_query" if state.operation == "query" else "handle_update")')
            replace(block, patched, 'Route failed tool calls back to the current operation.')
        if name == 'agent/agents/chat_agent.py':
            replace('if state.query_iteration == self.config.max_query_iteration:',
                    'if state.query_iteration > self.config.max_query_iteration:',
                    'Fix query tool-budget off-by-one.')
            replace('if state.update_iteration == self.config.max_update_iteration:',
                    'if state.update_iteration > self.config.max_update_iteration:',
                    'Fix update tool-budget off-by-one.')
            replace('max_tokens=8000,', 'max_tokens=self.config.max_chat_context,',
                    'Honor the configured chat context budget.')
            replace("                x['url'] = encode_image_to_base64(image_path)",
                    "                x = dict(x)\n                x['url'] = (image_path if image_path.startswith(('data:', 'http')) else encode_image_to_base64(image_path))",
                    'Do not reopen data URIs as local paths or mutate cached image blocks.')
        if name == 'agent/stores/image_manager.py':
            replace('        if not img_path_or_url.startswith("http"):\n'
                    '            img_path_or_url = encode_image_to_base64(img_path_or_url)\n'
                    '\n'
                    '        return self.format_to_msg_content(image=img_path_or_url, in_content=False)',
                    '        content = self.format_to_msg_content(image=img_path_or_url, in_content=False)\n'
                    '        for block in content:\n'
                    '            if block["type"] == "image" and not block["url"].startswith(("http", "data:")):\n'
                    '                block["url"] = encode_image_to_base64(block["url"])\n'
                    '        return content',
                    'Keep canonical local paths in the image-token registry; encode only the outgoing block.')
        if name == 'agent/stores/semantic.py':
            replace('        db.create_collection(\n            collection_name="memory",',
                    '        memory_schema.add_field(field_name="storage_vector", datatype=DataType.FLOAT_VECTOR, dim=2)\n'
                    '        db.create_collection(\n            collection_name="memory",',
                    'Milvus requires a vector field even for the metadata collection; this constant is never searched.')
            replace('            "text": memory.text,',
                    '            "storage_vector": [0.0, 0.0],\n            "text": memory.text,',
                    'Populate the unused metadata collection vector.')
            replace('max_length=1000', 'max_length=65535',
                    'Preserve generated text and evidence without truncating the upstream 1000-byte fields.')
            replace('max_length=100)', 'max_length=4096)',
                    'Support absolute MemEye image paths.')
            replace('            metric_type="IP",\n            params={"inverted_index_algo"',
                    '            metric_type="BM25",\n            params={"inverted_index_algo"',
                    'Milvus BM25-function indexes require the BM25 metric.')
            replace('text_vec = self._embed_text(text_content)',
                    'text_vec = self.text_embedder.embed_documents([text_content])[0]',
                    'Apply BGE query instructions only to queries, not stored documents.')
            replace('        self.db.delete("image_embs", [int(memory_id)])',
                    '        self.db.delete("image_embs", [int(memory_id)])\n        return True',
                    'Return success to the delete_memory tool after a successful deletion.')
        target = VENDOR / 'runtime' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_text() != text:
            target.write_text(text)
    report = {'upstream_revision': metadata['revision'], 'patches': changes,
              'runtime_sha256': {n: sha(VENDOR / 'runtime' / n) for n in metadata['sha256']}}
    (VENDOR / 'PATCHES.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    print(json.dumps(prepare_vendor(), indent=2))
