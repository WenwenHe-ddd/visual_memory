"""Rebuild four pinned text-memory methods with DeepSeek seed42 captions/LLMs.

Each method runs in its own process because upstream transports use module
globals. No previous Qwen memory or reader responses enter these experiments.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.vendor'), str(ROOT)]
os.environ.update(HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                  LITELLM_LOCAL_MODEL_COST_MAP='True', ANONYMIZED_TELEMETRY='False')
from scripts.memory import text_memory_baselines as base
from scripts.memory.text_memory_transport import Broker
from scripts.evaluation.deepseek_text_memory_transport import APIModel, DEEPSEEK_MODEL, build_request, request_sha256

SOURCE = ROOT / 'results/memeye_deepseek_flash_seed42_full_20261025'
OUTPUT = ROOT / 'results/memeye_deepseek_flash_seed42_text_methods_20261025'
ENV = Path('/storage/hewenwen/NTU Decentralized Agent/.env')
NAMES = dict(gen_agents='Generative Agents', memoryos='MemoryOS',
             simplemem='SimpleMem (context ≤5)', a_mem='A-Mem (context ≤5)')


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def write_rows(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in data))
    temp.replace(path)


def freeze(path, value):
    if Path(path).exists():
        assert read(path) == value, str(path)
    else:
        dump(path, value)


def verify_caption_source():
    manifest = read(SOURCE / 'completion_manifest.json')
    for name, expected in manifest['artifacts_sha256'].items():
        p = Path(name)
        assert sha(p if p.is_absolute() else ROOT / p) == expected, name
    proto = read(SOURCE / 'inputs/protocol.json')
    for name, expected in proto['source_sha256'].items():
        assert sha(name) == expected, name
    cap = rows(SOURCE / 'captions.jsonl')
    assert len(cap) == len({r['id'] for r in cap}) == 437
    assert all(r['model'] == DEEPSEEK_MODEL and r['seed'] == 42 and r['caption'].strip() for r in cap)
    return proto, {r['id']: r['caption'] for r in cap}


def prepare(output):
    source, captions = verify_caption_source()
    vendor = base.verify_vendor()
    questions = rows(SOURCE / 'inputs/source_questions.jsonl')
    gold = rows(SOURCE / 'inputs/evaluation.jsonl')
    assert len(questions) == len(gold) == 371
    assert {q['id'] for q in questions} == {q['id'] for q in gold}
    tasks = sorted({q['task'] for q in questions})
    image_map = {str(Path(p).resolve()): r['id'] for r in rows(SOURCE/'inputs/images.jsonl') for p in r['source_paths']}
    observations, profiles = {}, {}
    verified_images = []
    sources = [SOURCE/'captions.jsonl', SOURCE/'completion_manifest.json', SOURCE/'inputs/protocol.json',
               SOURCE/'inputs/source_questions.jsonl', SOURCE/'inputs/evaluation.jsonl', SOURCE/'inputs/images.jsonl',
               SOURCE/'inputs/source_memories.jsonl', Path(__file__), Path(base.__file__),
               ROOT/'scripts/memory/text_memory_transport.py', ROOT/'scripts/memory/reflexion_memory.py',
               ROOT/'scripts/evaluation/deepseek_text_memory_transport.py',
               ROOT/'scripts/evaluation/deepseek_flash_seed42_audited_transport.py',
               ROOT/'scripts/evaluation/openai_chat_transport.py', base.VENDOR/'UPSTREAM.json']
    for task in tasks:
        path = ROOT/'data/MemEye/data/dialog'/f'{task}_Open.json'
        assert str(path) in source['source_sha256']
        data = read(path)
        sources.append(path)
        profiles[task] = data.get('character_profile') or {}
        observations[task] = []
        for session in data['multi_session_dialogues']:
            for turn in session['dialogues']:
                refs = turn.get('input_image') or []
                refs = [refs] if isinstance(refs, str) else refs
                current = []
                for image_path in refs:
                    absolute = (ROOT/'data/MemEye/data/image'/image_path).resolve()
                    key = image_map[str(absolute)]
                    assert sha(absolute) == key
                    current.append(captions[key])
                    verified_images.append((f'{task}/{turn["round"]}', key))
                observations[task].append(dict(id=turn['round'], session_id=session['session_id'],
                    timestamp=str(session.get('date') or '').strip(), user=turn.get('user',''),
                    assistant=turn.get('assistant',''), raw=dict(input_image=refs,
                        image_id=turn.get('image_id', []), image_caption=current)))
    assert len(verified_images) == 438
    assert sorted(verified_images) == sorted((m['id'],m['image_sha256']) for m in rows(SOURCE/'inputs/source_memories.jsonl'))
    public_questions = []
    for q in questions:
        public_questions.append({k:q[k] for k in ('id','task','question','options')} | {
            'query_image_captions':[captions[im['sha256']] for im in q['query_images']]})
    embedding = base.EMBEDDING/'snapshots'/(base.EMBEDDING/'refs/main').read_text().strip()
    sources.extend(p for p in embedding.rglob('*') if p.is_file())
    for method in NAMES:
        target = output / method
        obs = json.loads(json.dumps(observations))
        for task, data in obs.items():
            profile = profiles[task]
            name = str(profile.get('name') or (profile.get('primary_user') if method=='gen_agents' else '') or '').strip()
            for row in data:
                row['speaker'] = f'user ({name})' if name else 'user'
        defaults = base.effective_method_defaults(method, 5 if method=='a_mem' else None)
        protocol = dict(schema='deepseek_text_memory_v1', mode=method, model=DEEPSEEK_MODEL,
            provider='DashScope International', endpoint='https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions',
            seed=42, caption_seed=42, enable_thinking=False, tasks=tasks, questions=371,
            memory_rounds=sum(map(len,obs.values())), image_memory_leaves=438,
            source_memory=str(SOURCE), caption_sha256=sha(SOURCE/'captions.jsonl'),
            method_defaults=defaults, upstream=vendor, embedding_model_path=str(embedding),
            embedding_device='cuda:1', task_workers=2, api_workers=2, generation_batch_size=4,
            memory_max_new_tokens=2048, max_new_tokens=16, max_padded_batch_tokens=48000,
            reader_system_prompt=source['reader_system_prompt'], context_budget=128000,
            memory_sampling='upstream per-call temperatures, explicit wire seed42; unspecified output budget 2048',
            answer_sampling='temperature0, max_tokens16, explicit wire seed42, thinking=false',
            query_policy='question plus DeepSeek query-image caption; no options/gold for retrieval',
            answer_policy='existing text-method common MCQ prompt: system + assistant memory + user question/options; image captions, no native images',
            history_policy='all 848 public Open dialogue rounds, including 438 image-bearing memory leaves; same pinned adapter as prior Qwen',
            prior_generated_memory_reused=False, prior_reader_answers_reused=False,
            structured_output_policy='same local adapter: JSON schema as text instructions; preserve native parser/fallback and report invalid outputs',
            failure_policy='network/429/5xx: at most3 unchanged attempts; other API failure stops task; invalid/truncated final answer counted wrong without retry',
            context_memory_limit=5 if method in ('simplemem','a_mem') else None,
            context_limit_policy='original retrieval/expansion order; first ≤5 distinct memory units, preserve empty/short contexts, no refill' if method in ('simplemem','a_mem') else 'upstream configuration unchanged',
            retrieval_top_k=defaults.get('retrieve_k'),
            input_sha256={str(p):sha(p) for p in sources},
            observations_sha256=base.hash_object(obs), questions_sha256=base.hash_object(public_questions),
            gold_sha256=base.hash_object(gold))
        protocol['experiment_signature'] = base.hash_object(protocol)
        freeze(target/'protocol.json',protocol)
        freeze(target/'observations.json',obs)
        for filename, content in [('questions.jsonl',public_questions),('gold.jsonl',gold)]:
            if (target/filename).exists(): assert rows(target/filename)==content
            else: write_rows(target/filename,content)
    print(json.dumps(dict(prepared=True,methods=list(NAMES),questions=371,history_rounds=sum(map(len,observations.values())),caption_seed=42)))


def verify(root):
    p=read(root/'protocol.json')
    assert p['experiment_signature']==base.hash_object({k:v for k,v in p.items() if k!='experiment_signature'})
    for name, expected in p['input_sha256'].items(): assert sha(name)==expected, name
    base.verify_vendor()
    assert base.hash_object(read(root/'observations.json'))==p['observations_sha256']
    assert base.hash_object(rows(root/'questions.jsonl'))==p['questions_sha256']
    assert base.hash_object(rows(root/'gold.jsonl'))==p['gold_sha256']
    return p


def bounded_simplemem(result):
    selected, seen = [], set()
    for item in result.items:
        if item['id'] in seen: continue
        seen.add(item['id']); selected.append(item)
        if len(selected)==5: break
    bounded=replace(result,items=selected,expansion_candidates=[x for x in result.expansion_candidates if x in seen])
    detail=bounded.to_dict()
    detail.update(context_memory_ids=[x['id'] for x in selected],context_memory_count=len(selected),
        context_memory_limit=5, original_item_ids=[x['id'] for x in result.items],source_empty_preserved=not result.items)
    return bounded,detail


class Memory(base.Memory):
    def recall(self,query):
        if self.method=='simplemem':
            result=self.core.query(query,top_k=self.protocol['method_defaults']['retrieve_k'],auto_expand=True)
            result,detail=bounded_simplemem(result)
            return self.core.retriever.format_for_llm(result,include_instructions=False),detail
        if self.method=='a_mem':
            # Each domain has >5 stored notes; upstream primary retrieval K=5.
            # Preserve the existing interleaved link order and ID deduplication.
            keywords=self.core.generate_query(query)
            system=self.core.memory_system
            if not system.memories:
                return '',dict(context_memory_ids=[],context_memory_count=0,context_memory_limit=5,generated_keywords=keywords)
            context,detail=base.bounded_a_mem_context(system,keywords,self.core.retrieve_k,min(5,len(system.memories)))
            detail.update(context_memory_limit=5,generated_keywords=keywords,primary_top_k=self.core.retrieve_k)
            return context,detail
        return super().recall(query)


def run(output,method,task=None):
    root=output/method
    p=verify(root)
    questions=rows(root/'questions.jsonl')
    gold={x['id']:x for x in rows(root/'gold.jsonl')}
    observations=read(root/'observations.json')
    pending=[t for t in p['tasks'] if (task is None or task==t) and not (root/t/'complete.json').exists()]
    if not pending: return analyze(output,method)
    model=APIModel(root,ENV,p['api_workers'])
    model.torch.set_num_threads(2)
    broker=Broker(model,root,p,p['generation_batch_size'])
    encoder=base.install_runtime(broker,p)
    if not model.torch.cuda.is_available(): raise RuntimeError('GPU unavailable; requested MiniLM cuda:1')
    encoder.to(p['embedding_device'])
    dump(root/'generation_environment.json',model.environment|dict(embedding_device=str(encoder.device)))
    def one(domain):
        broker.local.owner=domain
        target=root/domain
        workdir=target/'runtime'
        if workdir.exists(): shutil.rmtree(workdir)
        workdir.mkdir(parents=True,exist_ok=True)
        memory=Memory(method,p,workdir)
        for i,observation in enumerate(observations[domain],1):
            if model.fatal_errors: raise RuntimeError('API failure: see audited api_attempts.jsonl')
            memory.store(observation)
            if model.fatal_errors: raise RuntimeError('Upstream swallowed API failure; task cannot be accepted')
            dump(target/'progress.json',dict(phase='ingest',done=i,total=len(observations[domain])))
            if i%10==0: print(f'{method} {domain}: ingest {i}/{len(observations[domain])}',flush=True)
        answers=[]
        for q in [x for x in questions if x['task']==domain]:
            query=base.retrieval_query(q)
            context,detail=memory.recall(query)
            context=str(context)
            if model.fatal_errors: raise RuntimeError('Upstream swallowed API failure; task cannot be accepted')
            messages=base.reader_messages(q,context,p)
            out=broker.request(messages,p['max_new_tokens'],kind='answer',owner=q['id'])
            raw=out['raw_response'].strip()
            prediction=raw if raw in ('A','B','C','D') and not out['hit_token_limit'] and not out.get('refusal') else None
            row=dict(q,**out,retrieval_query=query,retrieved_context=context,retrieval_detail=detail,
                reader_messages_sha256=base.hash_object(messages),predicted_choice=prediction,
                **{k:gold[q['id']][k] for k in ('gold_choice','gold_ids','x','y')},
                experiment_signature=p['experiment_signature'])
            row['correct']=prediction==row['gold_choice']
            answers.append(row)
            write_rows(target/'responses.jsonl',answers)
            dump(target/'progress.json',dict(phase='answer',done=len(answers),total=sum(x['task']==domain for x in questions)))
        memory.export(target)
        dump(target/'complete.json',dict(experiment_signature=p['experiment_signature'],questions=len(answers),responses_sha256=sha(target/'responses.jsonl')))
        print(f'{method} {domain}: complete {sum(r["correct"] for r in answers)}/{len(answers)}',flush=True)
    errors=[]
    try:
        with ThreadPoolExecutor(max_workers=p['task_workers']) as pool:
            jobs={pool.submit(one,t):t for t in pending}
            for job in as_completed(jobs):
                try: job.result()
                except Exception:
                    err=dict(task=jobs[job],traceback=traceback.format_exc())
                    errors.append(err);dump(root/'run_errors.json',errors)
                    print(err['traceback'],flush=True)
    finally: broker.close()
    analyze(output,method)
    if errors: raise RuntimeError(f'{len(errors)} task failures; see run_errors.json')


def analyze(output,method):
    root=output/method
    p=verify(root)
    all_rows=[]
    for task in p['tasks']:
        target=root/task
        if not (target/'complete.json').exists(): continue
        done=read(target/'complete.json')
        assert done['experiment_signature']==p['experiment_signature']
        assert done['responses_sha256']==sha(target/'responses.jsonl')
        all_rows.extend(rows(target/'responses.jsonl'))
    question_map={q['id']:q for q in rows(root/'questions.jsonl')}
    gold={g['id']:g for g in rows(root/'gold.jsonl')}
    assert len(all_rows)==len({r['id'] for r in all_rows})
    context_counts=Counter()
    for r in all_rows:
        q=question_map[r['id']]
        messages=base.reader_messages(q,r['retrieved_context'],p)
        assert base.hash_object(messages)==r['reader_messages_sha256']
        wire=build_request(messages,16,0)
        assert request_sha256(wire)==r['request_sha256']
        assert r['seed']==42 and r['returned_model']==DEEPSEEK_MODEL
        assert r['correct']==(r['predicted_choice']==gold[r['id']]['gold_choice'])
        assert r['retrieval_query']==base.retrieval_query(q)
        if method in ('a_mem','simplemem'):
            detail=r['retrieval_detail'];ids=detail['context_memory_ids']
            assert len(ids)==len(set(ids))==detail['context_memory_count']<=5
            context_counts[len(ids)]+=1
            if method=='simplemem':
                assert len(detail['items'])==len(ids)
                import re
                assert len(re.findall(r'^\[Context \d+\]$',r['retrieved_context'],flags=re.M))==len(ids)
    calls=[read(x) for x in (root/'calls').glob('*.json')]
    for c in calls:
        assert c['request']['seed']==42 and c['request']['experiment_signature']==p['experiment_signature']
        request=c['request'];out=c['output']
        wire=build_request(request['messages'],request['max_new_tokens'],request['temperature'])
        assert request_sha256(wire)==out['request_sha256']
        assert out['requested_model']==out['returned_model']==DEEPSEEK_MODEL and out['seed']==42
    n=len(all_rows);correct=sum(r['correct'] for r in all_rows)
    summary=dict(method=method,status='complete' if n==371 else 'partial',questions=n,expected_questions=371,
        correct=correct,accuracy=correct/n if n else None,model=DEEPSEEK_MODEL,seed=42,
        caption_sha256=p['caption_sha256'],context_counts=dict(context_counts),
        invalid_answers=sum(r['predicted_choice'] is None for r in all_rows),
        answer_token_limit_hits=sum(r['hit_token_limit'] for r in all_rows),
        unique_llm_calls=len(calls),memory_calls=sum(c['request']['kind']=='memory' for c in calls),
        memory_token_limit_hits=sum(c['output']['hit_token_limit'] for c in calls if c['request']['kind']=='memory'),
        invalid_structured_outputs=sum(c['output'].get('structured_output_valid') is False for c in calls),
        by_domain={t:dict(questions=sum(r['task']==t for r in all_rows),correct=sum(r['task']==t and r['correct'] for r in all_rows)) for t in p['tasks']},
        experiment_signature=p['experiment_signature'])
    write_rows(root/'qa_responses.jsonl',all_rows)
    dump(root/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    report(output)
    return summary


def report(output):
    data={name:read(output/name/'summary.json') for name in NAMES if (output/name/'summary.json').exists()}
    dump(output/'summary.json',data)
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>DeepSeek 四种记忆方法</title>',
        '<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:32px auto}td,th{padding:10px;border-bottom:1px solid #ccc}</style>',
        '<h1>DeepSeek V4.1 Flash · seed42 · MemEye</h1>',
        '<p>使用同一份 DeepSeek seed42 caption；四种方法的记忆构建、摘要、反思、查询改写和最终回答均调用 deepseek-v4.1-flash，所有请求显式 seed=42。记忆操作保留上游温度，最终回答 temperature=0、16 tokens。</p>',
        '<p>沿用既有文本方法评测：读取全部 848 轮对话（含 438 条图片记忆），查询不含选项/答案，题图使用 DeepSeek caption。与前述只索引图片记忆、直接输入题图的两种检索方案存在输入结构差异。</p>',
        '<p>Generative Agents 与 MemoryOS 保留原配置。SimpleMem 保留自适应检索/BM25/扩展，A-Mem 保留主检索 K=5 和链接扩展；最终按原顺序去重取至多 5 个记忆单元，不补齐空结果。SimpleMem 文本格式化仍保留上游细节文本 2000 字符上限。</p>',
        '<table><tr><th>方法</th><th>状态</th><th>正确数</th><th>准确率</th><th>上下文条数分布</th><th>无效回答</th></tr>']
    for name,s in data.items():
        acc=f'{s["accuracy"]:.4%}' if s['accuracy'] is not None else '—'
        parts.append(f'<tr><td>{html.escape(NAMES[name])}</td><td>{s["status"]}</td><td>{s["correct"]}/{s["questions"]}</td><td>{acc}</td><td>{html.escape(str(s["context_counts"]))}</td><td>{s["invalid_answers"]}</td></tr>')
    parts.append('</table><p>无效/截断回答保留并按错误计分；partial 表示实验尚未完整结束。</p></html>')
    (output/'report.html').write_text('\n'.join(parts))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['prepare','run','analyze'],required=True)
    parser.add_argument('--method',choices=list(NAMES))
    parser.add_argument('--task')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.stage=='prepare': prepare(args.output)
    elif args.stage=='run':
        if not args.method: parser.error('--method required')
        run(args.output,args.method,args.task)
    else:
        for method in ([args.method] if args.method else NAMES): analyze(args.output,method)


if __name__=='__main__': main()
