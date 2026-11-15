"""Audit native states, exact contexts, transport records and Gallery scores."""
import argparse
import ast
from collections import Counter,defaultdict
import csv
import html
import json
import math
from pathlib import Path

from scripts.evaluation import mem_gallery_gpt5_simplemem030 as run

read,rows,sha,obj,dump,freeze=run.read,run.rows,run.sha,run.obj,run.dump,run.freeze


def valid_output(out,p,temperature=None,allow_rejection=False):
    assert out['requested_model']=='gpt-5' and out['returned_model']==p['expected_model']
    assert out['seed']==42 and out['reasoning_effort']=='minimal'
    assert out['finish_reason']=='stop' and not out.get('refusal') and out['raw_response'].strip()


def independent_fusion(detail):
    """Recompute RRF independently of the production fusion implementation."""
    candidates=list(dict.fromkeys(detail['original_item_ids']));distinct={}
    for batch in detail['search_batches']:
        assert batch['channel'] in ('semantic','keyword','structured')
        assert batch['top_k']==(10 if batch['channel']=='semantic' else 5)
        key=batch['channel'],json.dumps(batch['query'],sort_keys=True,ensure_ascii=False)
        distinct.setdefault(key,list(dict.fromkeys(batch['result_ids'])))
    channel_lists={}
    for channel in ('semantic','keyword','structured'):
        lists=[ids for (ch,_),ids in distinct.items() if ch==channel]
        first=list(dict.fromkeys(identity for ids in lists for identity in ids));scores={}
        for identity in first:
            total=0.
            for ids in lists:
                if identity in ids:total+=1/(60+ids.index(identity)+1)
            scores[identity]=total
        channel_lists[channel]=sorted(first,key=lambda identity:-scores[identity])
    scores={identity:sum(1/(60+ids.index(identity)+1) for ids in channel_lists.values() if identity in ids) for identity in candidates}
    ranked=sorted(candidates,key=lambda identity:-scores[identity])
    assert channel_lists==detail['fusion']['channel_rankings']
    assert ranked==[r['id'] for r in detail['fusion']['ranked_candidates']]
    assert all(scores[r['id']]==r['score'] for r in detail['fusion']['ranked_candidates'])
    assert ranked[:5]==detail['context_memory_ids']


def summary(items):
    gold=[r for r in items if r['gold_count']]
    attributed=all(r['source_ids'] is not None for r in items)
    total=sum(r['gold_count'] for r in gold)
    hits=sum(r['gold_hits'] for r in gold) if attributed else None
    return dict(questions=len(items),correct=sum(int(r['em']) for r in items),
        **{k:sum(r[k] for r in items)/len(items) for k in ('em','f1','bleu_1','bleu_2','bleu')},
        gold_questions=len(gold),gold_links=total,gold_hits=hits,
        source_recall_micro=hits/total if attributed and total else None,
        source_recall_macro=sum(r['gold_hits']/r['gold_count'] for r in gold)/len(gold) if attributed and gold else None,
        context_counts=dict(Counter(r['context_count'] for r in items)),
        extra_knowledge_counts=dict(Counter(r['extra_knowledge_count'] for r in items)),
        provider_rejected_answers=sum(r['provider_rejected'] for r in items))


def audit(output):
    p=run.verify(output);method=p['mode'];root=output/'memory';artifacts={}
    for folder in ('memory','reader'):
        done=read(output/folder/'complete.json');assert done['status']=='complete' and done['experiment_signature']==p['experiment_signature']
        for name,digest in done['artifact_sha256'].items():assert sha(name)==digest,name
        artifacts.update(done['artifact_sha256']);artifacts[str(output/folder/'complete.json')]=sha(output/folder/'complete.json')
    qs=rows(output/'inputs/questions.jsonl');obs=read(output/'observations.json')
    assert sha(output/'observations.json')==sha(run.SOURCE/'observations.json')
    assert sha(output/'inputs/questions.jsonl')==sha(run.SOURCE/'inputs/questions.jsonl')
    notes,mappings,counts={},{},{}
    for task,original in obs.items():
        dest=root/task;stats=read(dest/'memory_stats.json');done=read(dest/'complete.json')
        assert done['stored_rounds']==stats['stored_rounds']==len(original)
        if method=='gen_agents':
            ns=read(dest/'final_memory.json');notes[task]=ns
            originals=[n for n in ns if n.get('source') is False];byid={n['dialogue_id']:n for n in originals}
            assert len(originals)==len(byid)==len(original)
            for row in original:assert byid[row['id']]['text']==run.round_text(row,row['speaker'],'assistant')
            for n in ns:
                if n.get('source') is not False:assert n['source'] and all(0<=i<n['counter_id'] for i in n['source'])
            counts[task]=dict(originals=len(originals),reflections=len(ns)-len(originals))
        elif method=='memoryos':
            mapping=defaultdict(list)
            for row in original:mapping[(run.user_text(row).strip(),row['assistant'].strip(),row['timestamp'].strip())].append(task+'/'+row['id'])
            assert all(len(v)==1 for v in mapping.values());mappings[task]=mapping
            directory=dest/'runtime/users/memorybench_user';short=read(directory/'short_term.json');middle=read(directory/'mid_term.json')
            pages=[page for session in middle['sessions'].values() for page in session['details']];covered=set()
            for page in [*short,*pages]:
                key=tuple(page[k].strip() for k in ('user_input','agent_response','timestamp'))
                assert len(mapping[key])==1;covered.update(mapping[key])
            assert covered=={task+'/'+r['id'] for r in original}
            counts[task]=dict(covered_rounds=len(covered),short_term_pages=len(short),mid_term_pages=len(pages),segments=len(middle['sessions']))
        else:
            ns=read(dest/'final_memory.json');notes[task]={n['entry_id']:n for n in ns};assert len(notes[task])==len(ns)
            windows=read(dest/'runtime/construction_windows.json');sources=read(dest/'runtime/dialogue_sources.json')
            assert {s['round_id'] for s in sources.values()}=={r['id'] for r in original}
            assert {i for w in windows for i in w['dialogue_ids']}=={int(i) for i in sources}
            assert {i for w in windows for i in w['entry_ids']}==set(notes[task])
            assert all(w['parsed_successfully'] for w in windows)
            cfg=read(dest/'runtime/resolved_config.json');assert cfg['llm_model']=='gpt-5' and cfg['actual_model']==p['expected_model'] and cfg['settings']==p['method_defaults']
            counts[task]=dict(rounds=len(original),entries=len(ns),windows=len(windows))
    retrieved=rows(root/'retrieval.jsonl');rmap={r['id']:r for r in retrieved}
    answers=rows(output/'reader/responses.jsonl');amap={a['id']:a for a in answers}
    manifest=rows(output/'reader/manifest.jsonl');payloads={m['record_id']:m for m in manifest}
    assert len(retrieved)==len(rmap)==len(answers)==len(amap)==len(payloads)==len(manifest)==len(qs)==1711
    assert set(rmap)==set(amap)==set(payloads)=={q['id'] for q in qs}
    reader_calls=rows(output/'reader/api_responses.jsonl');reader_outputs={r['id']:r['output'] for r in reader_calls}
    assert len(reader_calls)==len(reader_outputs)==1711
    if method=='simplemem':
        run.official.verify_installation()
        from simplemem.core.answer_generator import AnswerGenerator
        from simplemem.core.models.memory_entry import MemoryEntry
        formatter=object.__new__(AnswerGenerator)
    if method=='memoryos':helper=run.native.helpers('memoryos')
    evaluated=[]
    for q in qs:
        r,a=rmap[q['id']],amap[q['id']];detail=r['retrieval_detail'];sources=[];extra=0
        assert r['task']==q['task'] and r['retrieval_query']==q['retrieval_query']
        if method=='gen_agents':
            ids=detail['selected_memory_ids'];assert len(ids)==len(set(ids))<=10
            chosen=[notes[q['task']][i] for i in ids]
            full='\n'+'\n\n'.join(f'[{i}] '+n['text'] for i,n in enumerate(chosen))+'\n'
            context=' '.join(full.split(' ')[:4000])
            for i,n in enumerate(chosen):
                if n.get('source') is False and f'[{i}] '+n['text'] in context:sources.append(q['task']+'/'+n['dialogue_id'])
            count=len(ids)
        elif method=='memoryos':
            context=helper['_format_retrieval_context'](detail)
            for page in detail['retrieved_pages']:
                key=tuple(page[k].strip() for k in ('user_input','agent_response','timestamp'))
                matches=mappings[q['task']][key];assert len(matches)==1;sources.extend(matches)
            count=len(detail['retrieved_pages']);extra=sum(len(detail[k]) for k in ('retrieved_user_knowledge','retrieved_assistant_knowledge'))
        else:
            ids=detail['context_memory_ids'];assert len(ids)==len(set(ids)) and (len(ids)==5 or not ids and detail['source_empty_preserved'])
            ns=notes[q['task']];assert all({k:v for k,v in n.items() if k!='id'}==ns[n['id']] for n in detail['items'])
            context=formatter._format_contexts([MemoryEntry(**ns[i]) for i in ids]);independent_fusion(detail)
            count=len(ids);sources=None
        assert context==r['retrieved_context']
        payload=run.ga.reader_payload(q,context);assert payloads[q['id']]==payload
        saved=read(output/'reader/responses'/(obj(q['id'])+'.json'));out=saved['output']
        assert saved['payload']==payload and out==reader_outputs[q['id']]
        wire=run.images.wire_for(payload,p['max_new_tokens']);assert out['request_sha256']==run.transport.request_sha256(wire)
        assert a==dict(id=q['id'],**out,experiment_signature=p['experiment_signature'])
        assert saved['experiment_signature']==r['experiment_signature']==p['experiment_signature']
        valid_output(out,p,0,True)
        evaluated.append(dict(id=q['id'],task=q['task'],point=q['point'],raw_response=out['raw_response'],
            source_ids=sorted(set(sources)) if sources is not None else None,context_count=count,extra_knowledge_count=extra,
            provider_rejected=out.get('status')=='provider_rejection'))
    # Verify every memory operation against its actual wire request and raw response.
    api_calls=rows(root/'api_responses.jsonl');by_response={r['output']['response_id']:r for r in api_calls}
    assert len(api_calls)==len(by_response)
    paths=list((root/'calls').glob('*.json'))
    from scripts.evaluation.continue_deepseek_ga_format import instruction_for
    for path in paths:
        call=read(path);request,out=call['request'],call['output']
        assert obj(request)==path.stem and request['experiment_signature']==p['experiment_signature']
        assert request['kind']=='memory' and request['seed']==42
        wire=run.transport.build_request(request['messages'],request['max_new_tokens'],request['temperature'])
        assert out['request_sha256']==run.transport.request_sha256(wire)
        actual=by_response[out['response_id']];assert actual['request']==wire and actual['output']['raw_response']==out['raw_response']
        valid_output(out,p)
        assert out['input_tokens']+wire['max_completion_tokens']<=p['context_budget']
        if method=='gen_agents':
            instruction,kind=instruction_for(request['messages'][1:]);assert request['messages'][0]==dict(role='system',content=instruction)
            raw=out['raw_response'].strip()
            assert 1<=float(ast.literal_eval(raw))<=10 if kind=='rating' else len([s for s in raw.splitlines() if s.strip()])==2
    events=rows(root/'call_events.jsonl')
    for e in events:assert (root/'calls'/(e['key']+'.json')).exists() or (root/'format_failures'/e['key']).exists()
    # Gold labels are first opened after all generation, state and context audits.
    gold={g['id']:g for g in rows(output/'inputs/gold.jsonl')};assert set(gold)==set(amap)
    for r in evaluated:
        g=gold[r['id']];metrics=run.ga.gallery.score_answer(r['raw_response'],g['answer'])
        if r['provider_rejected']:metrics={k:0. for k in metrics}
        ids=set(g['gold_ids']);r.update(answer=g['answer'],**metrics,gold_ids=sorted(ids),gold_count=len(ids),
            gold_hits=len(ids & set(r['source_ids'])) if r['source_ids'] is not None else None)
        pt,gt=(run.ga.gallery.normalize_answer_universal(s).split() for s in (r['raw_response'],g['answer']))
        hits=sum((Counter(pt)&Counter(gt)).values());bleu=hits/len(pt)*min(1.,math.exp(1.-len(gt)/len(pt))) if pt and gt and hits else 0.
        assert abs(bleu-r['bleu_1'])<1e-12
    result=dict(status='complete',method=p['method'],model=p['expected_model'],seed=42,overall=summary(evaluated),
        by_task={t:summary([r for r in evaluated if r['task']==t]) for t in p['tasks']},
        by_type={t:summary([r for r in evaluated if r['point']==t]) for t in sorted({r['point'] for r in evaluated})},
        context_policy=p['context_policy'],source_recall_note=p.get('source_recall_policy','Credits fully rendered original source rounds/pages only; generated insights and knowledge do not create source hits.'),
        experiment_signature=p['experiment_signature'])
    run.write_rows(output/'scored_questions.jsonl',evaluated)
    with (output/'scored_questions.csv').open('w') as f:
        fields=['id','task','point','answer','raw_response','em','f1','bleu_1','bleu_2','bleu','context_count','extra_knowledge_count','gold_hits','gold_count','provider_rejected']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(evaluated)
    dump(output/'final_results.json',result)
    usage={}
    for stage,folder in [('memory',root),('answer',output/'reader')]:
        records=rows(folder/'api_responses.jsonl');outs=[r['output'] for r in records]
        for out in outs:valid_output(out,p,allow_rejection=stage=='answer')
        usage[stage]=dict(calls=len(outs),input_tokens=sum(o['input_tokens'] for o in outs),output_tokens=sum(o['generated_tokens'] for o in outs),
            finish_reasons=dict(Counter(o['finish_reason'] for o in outs)),attempts=dict(Counter(r['status'] for r in rows(folder/'api_attempts.jsonl'))))
    diagnostics={}
    for name in ('format_retry_events.jsonl','summary_normalization_events.jsonl'):
        if (root/name).exists():diagnostics[name]=len(rows(root/name))
    audit_record=dict(status='passed',questions=1711,source_rounds=3962,source_state_counts=counts,memory_calls=len(paths),
        actual_contexts_and_requests_verified=True,all_original_rounds_covered=True,independent_bleu1_checked=True,
        independent_rrf_questions=1711 if method=='simplemem' else None,api_usage=usage,diagnostics=diagnostics,
        experiment_signature=p['experiment_signature'])
    dump(output/'audit.json',audit_record)
    for f in [output/'protocol.json',output/'final_results.json',output/'audit.json',output/'scored_questions.jsonl',output/'scored_questions.csv']:
        artifacts[str(f)]=sha(f)
    freeze(output/'completion_manifest.json',dict(status='complete',artifacts_sha256=artifacts,experiment_signature=p['experiment_signature']))
    print(json.dumps(dict(method=p['method'],**result['overall'])),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=run.OUTPUT)
    args=parser.parse_args();audit(args.output.resolve())
