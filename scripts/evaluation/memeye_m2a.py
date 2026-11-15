"""Run the pinned M2A dual-agent core on a complete MemEye task with local models.

Construction and retrieval inputs exclude options, answers and annotated clues.
The final MCQ reader receives options only after agentic retrieval has ended.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.vendor'))
sys.path.insert(0, str(ROOT / '.vendor_m2a'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'

from scripts.memory.m2a_vendor import prepare_vendor, VENDOR, sha
from scripts.memory.m2a_local import LocalQwen, LocalEmbeddings, append_json, digest, safe_content
from scripts.memory.m2a_search_budget import (
    BudgetedLLM, SearchBudget, nonnegative_int, DEFAULT_MAX_ADDITIONAL_SEARCH_ROUNDS,
    EXHAUSTED_NOTICE_TEMPLATE)


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def prepare(args, patch_report):
    additional_rounds = nonnegative_int(getattr(
        args, 'max_additional_search_rounds', DEFAULT_MAX_ADDITIONAL_SEARCH_ROUNDS))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise ValueError('Use a fresh output root; existing experiments are immutable')
    from scripts.evaluation.m2a_inputs import load_task
    observations, all_qs, sources = load_task(args.dataset, args.task)
    qs = all_qs[:args.limit_questions] if args.limit_questions else all_qs
    if args.question_id:
        qs = [q for q in qs if q['id'] == args.question_id]
    if not qs:
        raise ValueError('No selected questions')
    if args.limit_rounds:
        observations = observations[:args.limit_rounds]
    write(out / 'observations.json', observations)
    write(out / 'retrieval_queries.json', [
        {'id': q['id'], 'question': q['question'], 'images': q['query_image_paths']}
        for q in qs])
    write(out / 'evaluation_labels.json', qs)
    download = json.loads((args.siglip / 'DOWNLOAD_MANIFEST.json').read_text())
    for name, expected in download['files'].items():
        if sha(args.siglip / name) != expected:
            raise ValueError('SigLIP2 checkpoint file changed: ' + name)
    protocol = {
        'method': 'M2A upstream dual-agent, BGE/Qwen benchmark adaptation',
        'dataset': args.dataset,
        'answer_format': 'open' if args.dataset == 'memgallery' else 'mcq',
        'reader_max_new_tokens': 256 if args.dataset == 'memgallery' else 16,
        'upstream': json.loads((VENDOR / 'UPSTREAM.json').read_text()),
        'patch_report': patch_report,
        'task': args.task, 'rounds': len(observations),
        'image_rounds': sum(bool(o['image']) for o in observations),
        'questions': len(qs), 'total_task_questions': len(all_qs),
        'is_complete_task': not args.limit_rounds and len(qs) == len(all_qs),
        'mandatory_initial_query': args.mandatory_query,
        'memory_reused_from': str(args.memory_source.resolve()) if args.memory_source else None,
        'seed': 42, 'do_sample': False, 'llm_path': str(args.qwen),
        'llm_config_sha256': sha(args.qwen / 'config.json'),
        'bge_path': str(args.bge), 'bge_dimension': 384,
        'bge_query_prefix': 'Represent this sentence for searching relevant passages: ',
        'bge_l2_normalized': True, 'bge_length_policy': '512-token model prefix truncation, audited per input',
        'siglip': download, 'siglip_dimension': 768, 'siglip_l2_normalized': True,
        'device': args.device, 'image_max_pixels': args.image_max_pixels,
        'max_new_tokens': args.max_new_tokens,
        'memory_manager_max_tools': 15, 'chat_max_query_tools': 5, 'chat_max_update_tools': 5,
        'max_additional_search_rounds': additional_rounds,
        'max_search_calls_per_question': 1 + additional_rounds,
        'search_budget_exhausted_instruction': EXHAUSTED_NOTICE_TEMPLATE.format(limit=1 + additional_rounds),
        'search_budget_policy': 'Per-question semantic search budget shared across all memory-manager queries; includes first search; raw fetches do not consume it',
        'recent_raw_context': 5, 'chat_context_budget': 8000,
        'retrieval': 'dense Top5 + BM25 Top5 -> RRF Top5; image Top3; final RRF k=60',
        'text_to_image_branch': 'disabled, matching upstream actual execution',
        'memory_input': 'all original dialogue rounds in chronological order; original pixels; no dataset or cached captions',
        'memeye_adaptations': [
            'One original user/assistant round is one raw observation; source IDs and timestamps are retained.',
            'Native images go to Qwen and SigLIP2; captions are generated by the memory manager.',
            'No options, gold labels, clues or reference answers in construction or retrieval.',
            'Read-only question graph avoids upstream chat() appending QA to raw storage.',
            'Final MCQ reader receives agent-retrieved multimodal content, agent answer and fixed options.',
            'One fixed option permutation from the existing local benchmark; not four-rotation paper EM.'],
        'sources': sources,
        'prepared_sha256': {n: sha(out / n) for n in
            ['observations.json', 'retrieval_queries.json', 'evaluation_labels.json']},
        'implementation_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
            Path(__file__), ROOT / 'scripts/memory/m2a_local.py', ROOT / 'scripts/memory/m2a_vendor.py',
            ROOT / 'scripts/memory/m2a_search_budget.py',
            ROOT / 'scripts/evaluation/m2a_inputs.py', ROOT / 'scripts/evaluation/mem_gallery_master_reader.py']},
        'packages': {n: importlib.metadata.version(n) for n in
            ['torch', 'transformers', 'sentence-transformers', 'langchain', 'langchain-core',
             'langchain-openai', 'langgraph', 'pymilvus', 'milvus-lite']}}
    if args.dataset == 'memgallery':
        protocol['memeye_adaptations'] = [
            'Original Gallery rounds and pixels with dates, round IDs and official image IDs; no supplied captions.',
            'Question plus original pixels; no supplied query caption, options, answer or clues enter retrieval.',
            'Read-only QA graph, followed by the shared Gallery open-answer reader (256 tokens).',
            'Shared official task instructions and normalized EM/F1/BLEU; no LLM judge.']
    if args.memory_source:
        source = args.memory_source
        if not (source / 'completion_manifest.json').exists() or args.limit_rounds:
            raise ValueError('Memory reuse requires a completed source and all source rounds')
        previous = json.loads((source / 'protocol.json').read_text())
        for key in ['task', 'llm_path', 'bge_path', 'siglip', 'seed', 'image_max_pixels', 'max_new_tokens']:
            if protocol[key] != previous[key]:
                raise ValueError('Frozen-memory configuration differs: ' + key)
        if observations != json.loads((source / 'observations.json').read_text()):
            raise ValueError('Frozen original observations differ')
        protocol['frozen_memory_source_signature'] = previous['signature']
        protocol['frozen_memory_inputs'] = {n: sha(source / n) for n in
            ['semantic_before_qa.json', 'semantic_after_qa.json', 'raw_before_qa.json',
             'raw_after_qa.json', 'construction.jsonl', 'image_manager.json']}
    protocol['signature'] = digest(protocol)
    write(out / 'protocol.json', protocol)
    for name, expected in protocol['implementation_sha256'].items():
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Implementation changed during preparation: ' + name)
        snapshot = out / 'code_snapshot' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(data)
    return observations, qs, protocol


def memory_snapshot(store):
    return sorted(store.db.query('memory', filter='', output_fields=[
        'id', 'text', 'image_caption', 'image_path', 'evidence_ids'], limit=16000), key=lambda x: x['id'])


def run(args):
    patches = prepare_vendor()
    sys.path.insert(0, str(VENDOR / 'runtime'))
    observations, qs, protocol = prepare(args, patches)
    out = args.output
    print(f'PREPARED {len(observations)} rounds / {len(qs)} questions', flush=True)
    from agent.config import ChatAgentConfig, MemoryManagerConfig
    from agent.stores import RawMessageStore, SemanticStore, ImageManager
    from agent.agents.memory_manager import MemoryManager, MemoryManagerTools
    from agent.agents.chat_agent import ChatAgent, ChatAgentState
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
    import torch
    torch.set_num_threads(8)
    started = time.time()
    print('Loading local BGE and SigLIP2', flush=True)
    emb = LocalEmbeddings(args.bge, args.siglip, out, args.device)
    emb.install()
    if args.memory_source:
        source = args.memory_source
        shutil.copy2(source / 'raw.db', out / 'raw.db')
        shutil.copytree(source / 'semantic.db', out / 'semantic.db')
        shutil.copy2(source / 'construction.jsonl', out / 'construction.jsonl')
        from pymilvus import MilvusClient
        from agent.stores import semantic as semantic_module
        semantic = SemanticStore.__new__(SemanticStore)
        semantic.db = MilvusClient(str(out / 'semantic.db'))
        for collection in ('memory', 'text_embs', 'image_embs'):
            semantic.db.load_collection(collection)
        semantic.text_embedder = semantic_module.TextEmbedding()
        semantic.multimodal_embedder = semantic_module.MultimodalEmbedder()
        semantic.log = []
    else:
        semantic = SemanticStore(str(out / 'semantic.db'))
    raw = RawMessageStore(str(out / 'raw.db'), reuse=bool(args.memory_source))
    images = ImageManager()
    if args.memory_source:
        images.load(str(args.memory_source / 'image_manager.json'))
    print('Loading local Qwen3-VL', flush=True)
    llm = LocalQwen(str(args.qwen), out, args.device, max_new_tokens=args.max_new_tokens,
                    image_max_pixels=args.image_max_pixels)
    original_search = semantic.hybrid_search
    search_budget = SearchBudget(protocol['max_additional_search_rounds'])
    agent_llm = BudgetedLLM(llm, search_budget)

    def traced_search(*a, **kw):
        if llm.stage == 'retrieval':
            search_budget.consume()
        result = original_search(*a, **kw)
        append_json(out / 'searches.jsonl', {'stage': llm.stage, 'sample_id': llm.sample_id,
            'args': a, 'kwargs': kw, 'results': [vars(m) for m in result]})
        return result

    semantic.hybrid_search = traced_search
    mm = MemoryManager(MemoryManagerTools(raw, semantic, images), raw, semantic,
                       agent_llm, images, MemoryManagerConfig(context_window=5, max_iteration=15))
    config = ChatAgentConfig(max_chat_context=8000, max_query_iteration=5, max_update_iteration=5)
    chat = ChatAgent(mm, raw, agent_llm, images, config, update_memory=True, update_only=True)
    chat.init_conversation(
        'You are an AI memory evolution agent with access to long-term memory. '
        'You receive chronological original conversation rounds between User and Assistant, '
        'including dates, round identifiers and images. Build persistent factual memories. '
        'You do not need to reply; respond with an empty string when no tool call is needed. '
        'Resolve relative times using the supplied dates and retain source evidence IDs.')
    try:
        llm.stage = 'construction'
        for i, obs in enumerate([] if args.memory_source else observations):
            llm.sample_id = obs['id']
            before_calls = llm.counter[0]
            if getattr(args, 'single_pass', False):
                from scripts.memory.m2a_single_pass import build_round
                construction = build_round(obs, raw, semantic, images, llm, out,
                    max_attempts=getattr(args, 'extraction_attempts', 1))
            else:
                chat.chat(user_text=obs['text'], user_image_path_or_url=obs['image'],
                          timestamp=datetime.fromisoformat(obs['timestamp']), role='conversation_round')
                construction = {'id': obs['id'], 'raw_id': chat.raw_messages[-1].msg_id}
            append_json(out / 'construction.jsonl', dict(construction,
                        calls=llm.counter[0] - before_calls, status='ok'))
            write(out / 'progress.json', {'stage': 'construction', 'completed': i + 1,
                                          'total': len(observations)})
            print(f'BUILD {i+1}/{len(observations)}', flush=True)
        snapshot = memory_snapshot(semantic)
        if args.memory_source and snapshot != json.loads((args.memory_source / 'semantic_before_qa.json').read_text()):
            raise ValueError('Copied semantic memories differ from the frozen source')
        write(out / 'semantic_before_qa.json', snapshot)
        raw.dump_db(str(out / 'raw_before_qa.json'))
        if args.memory_source and rows_raw(out) != rows_raw(args.memory_source):
            raise ValueError('Copied raw messages differ from the frozen source')
        raw_hash = sha(out / 'raw.db')
        semantic_hash = digest(snapshot)
        images.save(str(out / 'image_manager.json'))
        raw_to_source = {r['raw_id']: r['id'] for r in rows(out / 'construction.jsonl')}
        responses = []
        for i, q in enumerate(qs):
            search_budget.reset()
            llm.stage, llm.sample_id = 'retrieval', q['id']
            question_agent = ChatAgent(mm, raw, agent_llm, images, config, update_memory=False)
            question_agent.init_conversation(
                'You are a memory assistant answering questions about a past conversation. '
                'For every question you MUST call query_memory to examine stored memory. '
                'Return a concise factual answer grounded in retrieved evidence. '
                'Current conversation date: ' + observations[-1]['timestamp'])
            if len(q['query_image_paths']) > 1:
                raise ValueError('Question contains multiple images; upstream supports one')
            image = q['query_image_paths'][0] if q['query_image_paths'] else None
            content = question_agent._prepair_user_message(q['question'], image)
            query_state = ChatAgentState(
                messages=question_agent.chat_messages + [HumanMessage(content=content)],
                user_message=q['question'], user_image_path=image)
            if args.mandatory_query:
                # Enforce the original evaluator's MUST-query contract without exposing options.
                tool_args = {'text': q['question']}
                if image:
                    tool_args['image'] = images.image_to_image_token(image)
                query_state.messages.append(AIMessage(content='', tool_calls=[{
                    'name': 'query_memory', 'args': tool_args,
                    'id': 'mandatory_' + digest(q['id'])[:20], 'type': 'tool_call'}]))
                query_state = question_agent._query_memory(query_state)
            result = question_agent.graph.invoke(query_state, config={'recursion_limit': 100})
            contexts = []
            for message in result['messages']:
                if isinstance(message, ToolMessage):
                    contexts += (message.content if isinstance(message.content, list)
                                 else [{'type': 'text', 'text': message.content}])
            if args.dataset == 'memgallery':
                from scripts.evaluation.m2a_inputs import gallery_reader
                gallery = gallery_reader()
                reader_system = gallery.READER_SYSTEM
                answer_prompt = 'Question: ' + q['question'] + '\n' + gallery.TASK_INSTRUCTIONS.get(q['point'], '')
            else:
                reader_system = 'Return only A, B, C, or D.'
                answer_prompt = ('Answer the multiple-choice question using the retrieved evidence. '
                    'Return only one uppercase letter A, B, C, or D.\n'
                    f"Question: {q['question']}\n" + '\n'.join(f'{k}. {v}' for k, v in q['options'].items()))
            final_content = [{'type': 'text', 'text': 'Retrieved memory evidence:\n'}] + contexts
            final_content += [{'type': 'text', 'text': '\nMemory assistant answer:\n' + str(result['response'])}]
            if image:
                final_content += [{'type': 'text', 'text': '\nQuestion image:'}, {'type': 'image', 'url': image}]
            final_content += [{'type': 'text', 'text': answer_prompt}]
            llm.stage = 'reader'
            reader = __import__('copy').copy(llm)
            reader.max_new_tokens = protocol['reader_max_new_tokens']
            answer = reader.invoke([SystemMessage(reader_system),
                                    HumanMessage(final_content)]).content.strip()
            match = re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?', answer, flags=re.I)
            pred = match.group(1).upper() if match else None
            searches = [r for r in (rows(out / 'searches.jsonl') if (out / 'searches.jsonl').exists() else [])
                        if r['stage'] == 'retrieval' and r['sample_id'] == q['id']]
            ids = sorted({raw_to_source[rid] for search in searches for m in search['results']
                          for start, end in m['evidence_ids'] for rid in range(start, end + 1)
                          if rid in raw_to_source})
            record = {'id': q['id'], 'raw_answer': answer, 'predicted_choice': pred,
                'gold_choice': q.get('gold_choice'), 'correct': pred == q.get('gold_choice'),
                'status': 'ok' if pred else 'format_error', 'agent_answer': result['response'],
                'search_calls': len(searches), 'retrieved_source_ids': ids, 'gold_ids': q['gold_ids'],
                'gold_source_hits': len(set(ids) & set(q['gold_ids'])),
                'reader_context': safe_content(final_content), 'protocol_signature': protocol['signature']}
            if args.dataset == 'memgallery':
                scores = gallery.score_answer(answer, q['answer'])
                record.update(scores, correct=bool(scores['em']), status='ok' if answer else 'format_error',
                              answer=q['answer'], point=q['point'], predicted_choice=None)
            append_json(out / 'responses.jsonl', record)
            responses.append(record)
            write(out / 'progress.json', {'stage': 'qa', 'completed': i + 1, 'total': len(qs)})
            print(f'QA {i+1}/{len(qs)} correct={sum(r["correct"] for r in responses)}/{len(responses)}', flush=True)
        after = memory_snapshot(semantic)
        write(out / 'semantic_after_qa.json', after)
        raw.dump_db(str(out / 'raw_after_qa.json'))
        write(out / 'semantic_log.json', semantic.log)
        calls = rows(out / 'llm_calls.jsonl')
        evidence_invalid = [m['id'] for m in snapshot for a, b in json.loads(m['evidence_ids'])
                            if a > b or any(rid not in {r['msg_id'] for r in rows_raw(out)} for rid in range(a,b+1))]
        audit = {'question_ids_exact': [r['id'] for r in responses] == [q['id'] for q in qs],
                 'construction_ids_exact': [r['id'] for r in rows(out/'construction.jsonl')] == [o['id'] for o in observations],
                 'raw_db_frozen_during_qa': raw_hash == sha(out / 'raw.db'),
                 'semantic_memories_frozen_during_qa': semantic_hash == digest(after),
                 'invalid_evidence_memories': sorted(set(evidence_invalid)),
                 'questions_without_search': [r['id'] for r in responses if not r['search_calls']],
                 'questions_over_search_budget': [r['id'] for r in responses
                     if r['search_calls'] > protocol['max_search_calls_per_question']],
                 'llm_errors': sum(r['status'] != 'ok' for r in calls),
                 'llm_token_limits': sum(r['hit_token_limit'] for r in calls),
                 'mcq_format_errors': sum(r['status'] != 'ok' for r in responses),
                 'tool_error_messages': sum('Tool error:' in json.dumps(r['messages']) or
                    'Error creating memory:' in json.dumps(r['messages']) for r in calls)}
        write(out / 'audit.json', audit)
        summary = {'completed_questions': len(responses), 'correct': sum(r['correct'] for r in responses),
                   'accuracy': sum(r['correct'] for r in responses) / len(responses),
                   'semantic_memories': len(snapshot), 'image_memories': sum(bool(m['image_path']) for m in snapshot),
                   'source_hits': sum(r['gold_source_hits'] for r in responses),
                   'source_targets': sum(len(set(r['gold_ids'])) for r in responses),
                   'recall_label': 'union of source IDs across adaptive searches, not fixed Recall@K',
                   'llm_calls': len(calls), 'elapsed_seconds': time.time() - started,
                   'memory_reused_from': protocol['memory_reused_from'],
                   'mandatory_initial_query': args.mandatory_query,
                   'source_construction_calls': sum(r['calls'] for r in rows(out / 'construction.jsonl')),
                   'is_complete_task': protocol['is_complete_task'], 'audit': audit}
        if args.dataset == 'memgallery':
            summary['metrics'] = {k: sum(r[k] for r in responses) / len(responses)
                                  for k in ('em', 'f1', 'bleu', 'bleu_1', 'bleu_2')}
            summary['by_type'] = {point: {
                'questions': sum(r['point'] == point for r in responses),
                'correct': sum(r['correct'] for r in responses if r['point'] == point)}
                for point in sorted({r['point'] for r in responses})}
        write(out / 'summary.json', summary)
        write(out / 'completion_manifest.json', {'status': 'completed',
            'protocol_signature': protocol['signature'], 'artifacts': {
                p.name: sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json', '.jsonl')
                and p.name != 'completion_manifest.json'}})
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        write(out / 'failure.json', {'stage': llm.stage, 'sample_id': llm.sample_id,
                                   'error': repr(exc), 'elapsed_seconds': time.time() - started})
        write(out / 'semantic_log_partial.json', semantic.log)
        raise
    finally:
        semantic.close()


def rows_raw(out):
    return json.loads((out / 'raw_before_qa.json').read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['memeye', 'memgallery'], default='memeye')
    p.add_argument('--question-id', help='Select a single QA ID for an explicitly partial smoke test')
    p.add_argument('--task', default='Card_Playlog_Test')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit-questions', type=int)
    p.add_argument('--limit-rounds', type=int)
    p.add_argument('--extraction-attempts', type=int, choices=[1, 2, 3], default=1,
                   help='Maximum attempts for invalid single-pass extraction JSON; default one')
    p.add_argument('--memory-source', type=Path, help='Reuse a closed, completed run without reconstructing its memories')
    p.add_argument('--mandatory-query', action='store_true',
                   help='Execute query_memory(question, image) once before allowing the normal ChatAgent query loop')
    p.add_argument('--qwen', type=Path, default=ROOT / 'models/Qwen3-VL-8B-Instruct')
    p.add_argument('--siglip', type=Path, default=ROOT / 'models/siglip2-base-patch16-384')
    p.add_argument('--bge', type=Path, default=Path('/storage/hewenwen/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'))
    p.add_argument('--max-new-tokens', type=int, default=1200)
    p.add_argument('--max-additional-search-rounds', type=nonnegative_int,
                   default=DEFAULT_MAX_ADDITIONAL_SEARCH_ROUNDS,
                   help='Additional semantic searches per question after the first; default 2 (3 total), shared across both agents')
    p.add_argument('--image-max-pixels', type=int, default=262144)
    p.add_argument('--api-env-file', type=Path, help='Provider credential configuration for API entrypoints')
    p.add_argument('--api-cache-dir', type=Path, help='Exact-request cache for API entrypoints')
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
