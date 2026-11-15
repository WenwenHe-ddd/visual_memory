"""MemEye MCQ answering with Top-15 captions and outcome-based similarity."""

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import collections
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.vendor'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib'))
import numpy as np

NAT=ROOT/'results/natural_observation'
RANK=ROOT/'results/mcq_fifth_non_target_rank'
OUT=ROOT/'results/mcq_top15_caption_qa'
SYSTEM='You answer multiple-choice questions about remembered images. Use the retrieved memory captions and any query image provided to select the best option. Return only one uppercase letter: A, B, C, or D. Do not explain your answer.'
TASK_NAMES={'Brand_Memory_Test':'Brand','Card_Playlog_Test':'Card','Cartoon_Entertainment_Companion':'Cartoon',
            'Home_Renovation_Interior_Design':'Home','Multi-Scene_Visual_Case_Archive_Assistant':'Cross-scene',
            'Outdoor_Navigation_Route_Memory_Assistant':'Outdoor','Personal_Health_Dashboard_Assistant':'Health',
            'Social_Chat_Memory_Test':'Social'}


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def jsonl(name,rows):
    (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))


def csvout(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    ranking_protocol=json.loads((RANK/'protocol.json').read_text())
    natural=json.loads((NAT/'protocol.json').read_text())
    for source in ranking_protocol['source_files']:assert sha(ROOT/source['path'])==source['sha256']
    for name,value in ranking_protocol['input_hashes'].items():assert sha(ROOT/name)==value
    memories=read(NAT/'memories.jsonl');memory={r['id']:r for r in memories}
    captions={r['id']:r['caption'] for r in read(NAT/'captions.jsonl')}
    texts={r['id']:' '.join(captions[i] for i in r['images']) for r in memories}
    rankings={r['id']:r for r in read(RANK/'retrieval_details.jsonl')}
    questions=[]
    for source in ranking_protocol['source_files']:
        p=ROOT/source['path'];task=p.stem;data=json.loads(p.read_text())
        for q in data['human-annotated QAs']:
            qid=task+'/'+q['question_id'];rank=rankings[qid]
            assert q['question']==rank['question']
            gids=list(dict.fromkeys(task+'/'+r for r in q['clue']))
            assert gids==rank['gold_ids']
            forms=q['options'];assert len(forms)==4
            assert all(set(f)=={'A','B','C','D','answer'} for f in forms)
            assert len({f[f['answer']] for f in forms})==1
            assert {f['answer'] for f in forms}<=set('ABCD')
            imgs=q.get('question_image',[])
            if isinstance(imgs,str):imgs=[imgs]
            paths=[str(ROOT/'data/MemEye/data/image'/i) for i in imgs]
            assert all(Path(p).is_file() for p in paths)
            top=rank['ranking'][:15];assert len(top)==15
            questions.append({'id':qid,'task':task,'question':q['question'],'x':q['point'][0][0],
                              'y':q['point'][1][0],'gold_ids':gids,'query_image_paths':paths,
                              'forms':forms,'top15_memory_ids':[r['id'] for r in top],
                              'top15_query_scores':[r['score'] for r in top]})
    assert len(questions)==len(rankings)==371
    rng=np.random.default_rng(42);counts=collections.Counter({l:0 for l in 'ABCD'})
    for task in TASK_NAMES:
        task_questions=[q for q in questions if q['task']==task];n=len(task_questions)
        labels=list('ABCD')*(n//4)
        for _ in range(n%4):
            least=min(counts[l]+labels.count(l) for l in 'ABCD')
            choices=[l for l in 'ABCD' if counts[l]+labels.count(l)==least]
            labels.append(str(rng.choice(choices)))
        rng.shuffle(labels)
        # Three source questions repeat a correct-letter position across variants.
        # Swap assigned labels within the task to keep original permutations and balance.
        for i,q in enumerate(task_questions):
            allowed={f['answer'] for f in q['forms']}
            if labels[i] not in allowed:
                donors=[j for j,other in enumerate(task_questions) if j!=i and labels[j] in allowed
                        and labels[i] in {f['answer'] for f in other['forms']}]
                assert donors,'No valid balanced original permutation assignment'
                j=int(rng.choice(donors));labels[i],labels[j]=labels[j],labels[i]
        for q,label in zip(task_questions,labels):
            form=next(f for f in q['forms'] if f['answer']==label)
            q['gold_choice']=label;q['gold_answer_text']=form[label]
            q['options']={l:form[l] for l in 'ABCD'};q.pop('forms')
            counts[label]+=1
    assert max(counts.values())-min(counts.values())<=1
    for q in questions:
        caption_lines=['Memory '+str(i+1)+': '+texts[mid] for i,mid in enumerate(q['top15_memory_ids'])]
        q['reader_text']='Retrieved memory captions (ranked by relevance):\n\n'+'\n\n'.join(caption_lines)+'\n\nQuestion:\n'+q['question']+'\n\nOptions:\n'+'\n'.join(l+'. '+q['options'][l] for l in 'ABCD')+'\n\nAnswer with one letter only.'
        payload={'system':SYSTEM,'user_text':q['reader_text'],
                 'query_images':[{'path':p,'sha256':sha(Path(p))} for p in q['query_image_paths']]}
        q['payload_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    source_paths=[NAT/n for n in ['protocol.json','memories.jsonl','captions.jsonl','embeddings.npz','images.jsonl']]
    source_paths += [RANK/n for n in ['protocol.json','retrieval_details.jsonl','query_embeddings.npz']]
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    protocol={'study':'MemEye Top-15 caption MCQ accuracy and target-to-five-non-target similarity',
              'status':'prepared','dataset':ranking_protocol['dataset'],'source_files':ranking_protocol['source_files'],
              'retriever':ranking_protocol['retriever'],'retriever_revision':ranking_protocol['retriever_revision'],
              'reader_model':natural['provenance']['caption_model'],'reader_model_path':natural['caption_model_path'],
              'retrieval_query':'original MCQ question text with BGE prefix; no options or query-image captions',
              'retrieval_scope':'all 438 original image-bearing rounds, restricted to original task, across all sessions',
              'retrieval_top_k':15,'reader_input':'original MCQ question, one original option permutation, Top-15 ordinary image captions in retrieval order, plus original query images for eight questions',
              'reader_metadata':'only numbered memory captions; no dialogue, dates, session IDs, character profile, gold choice, or clue',
              'reader_system_prompt':SYSTEM,'option_policy':'one original permutation per question; correct-letter counts balanced within tasks and globally using seed 42',
              'gold_choice_counts':dict(counts),'seed':42,'do_sample':False,'dtype':'bfloat16','max_new_tokens':16,
              'batch_size_text':8,'batch_size_query_images':4,'query_image_max_pixels':1638400,
              'primary_success':'parsed model choice equals the answer label for the presented option permutation',
              'failure':'valid parsed choice differs from the gold choice; format or generation errors are tracked separately',
              'accuracy_denominator':'all 371 questions; an invalid or failed response counts as incorrect in overall benchmark accuracy',
              'targets':'all original MCQ clue rounds, unchanged; used only for scoring and subsequent analysis',
              'non_targets':'first five after excluding ALL annotated targets from the full original query ranking; all five must be in the reader Top-15',
              'similarity':'cosine of cached normalized target and non-target BGE caption embeddings',
              'aggregation':'for each target, mean similarity to five non-targets; mean across targets within question; equal-weight mean across questions in QA-correct and QA-wrong groups',
              'similarity_exclusions':'fewer than five non-targets (five questions); format/generation errors; no exclusion based on target count or evidence recall',
              'ties':'reuse the same saved stable six-decimal score ordering for reader Top-15 and non-target selection',
              'interpretation':'descriptive association with QA correctness; no causal or encoding-missing diagnosis from group means alone',
              'input_hashes':source_hashes,'questions':len(questions),'execution':'local offline Qwen3-VL generation, resumable; one generation per question, no automatic failed-row retry'}
    signature=hashlib.sha256(json.dumps({k:v for k,v in protocol.items() if k!='status'},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    protocol['experiment_signature']=signature
    existing=read(OUT/'qa_responses.jsonl')
    assert all(r['experiment_signature']==signature for r in existing),'Existing inference configuration differs'
    jsonl('qa_manifest.jsonl',questions);dump('protocol.json',protocol)
    dump('status.json',{'state':'prepared','questions':371,'completed_responses':len(existing)})
    print(json.dumps({'questions':len(questions),'option_gold_counts':dict(counts),'query_image_questions':sum(bool(q['query_image_paths']) for q in questions)},indent=2),flush=True)


def run():
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    protocol=json.loads((OUT/'protocol.json').read_text());questions=read(OUT/'qa_manifest.jsonl')
    for path,value in protocol['input_hashes'].items():assert sha(ROOT/path)==value
    done=read(OUT/'qa_responses.jsonl');assert len(done)==len({r['id'] for r in done})
    qmap={q['id']:q for q in questions}
    assert all(r['experiment_signature']==protocol['experiment_signature'] and r['payload_sha256']==qmap[r['id']]['payload_sha256'] for r in done)
    pending=[q for q in questions if q['id'] not in {r['id'] for r in done}]
    if not pending:return
    torch.manual_seed(protocol['seed']);torch.set_num_threads(8)
    assert torch.cuda.is_available(),'Local GPU unavailable'
    print('Loading local Qwen3-VL-8B-Instruct on cuda:0',flush=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(protocol['reader_model_path'],dtype=torch.bfloat16,
                                                          attn_implementation='sdpa',local_files_only=True).to('cuda:0').eval()
    processor=AutoProcessor.from_pretrained(protocol['reader_model_path'],local_files_only=True)
    processor.tokenizer.padding_side='left'
    processor.image_processor.size={'shortest_edge':65536,'longest_edge':protocol['query_image_max_pixels']}
    context_limit=model.config.text_config.max_position_embeddings
    dump('generation_environment.json',{'torch':torch.__version__,'transformers':transformers.__version__,
        'gpu':torch.cuda.get_device_name(0),'context_limit':context_limit,'image_processor_size':processor.image_processor.size,
        'model_config_sha256':sha(Path(protocol['reader_model_path'])/'config.json')})
    completed=len(done)
    for has_image in [False,True]:
        selected=[q for q in pending if bool(q['query_image_paths'])==has_image]
        batch_size=protocol['batch_size_query_images'] if has_image else protocol['batch_size_text']
        for offset in range(0,len(selected),batch_size):
            batch=selected[offset:offset+batch_size];messages=[];paths=[];started=time.time()
            for q in batch:
                content=[{'type':'image'} for _ in q['query_image_paths']]+[{'type':'text','text':q['reader_text']}]
                paths.extend(q['query_image_paths'])
                messages.append([{'role':'system','content':protocol['reader_system_prompt']},{'role':'user','content':content}])
            rendered=[processor.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in messages]
            images=[Image.open(p).convert('RGB') for p in paths] if paths else None
            inputs=processor(text=rendered,images=images,padding=True,return_tensors='pt').to('cuda:0')
            lengths=inputs['attention_mask'].sum(dim=1).tolist()
            assert max(lengths)+protocol['max_new_tokens']<=context_limit,'Unexpected context truncation'
            with torch.inference_mode():
                generated=model.generate(**inputs,do_sample=False,max_new_tokens=protocol['max_new_tokens'],
                                          pad_token_id=processor.tokenizer.pad_token_id)
            tails=generated[:,inputs['input_ids'].shape[1]:]
            texts=processor.batch_decode(tails,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            with (OUT/'qa_responses.jsonl').open('a') as f:
                for q,text,tokens,length in zip(batch,texts,tails.tolist(),lengths):
                    raw=text.strip();match=re.fullmatch(r'(?:Answer\s*:\s*)?([ABCD])[.!]?',raw,flags=re.IGNORECASE)
                    eos=processor.tokenizer.eos_token_id
                    emitted=next((i+1 for i,t in enumerate(tokens) if t==eos),len(tokens))
                    record={'id':q['id'],'raw_response':raw,'predicted_choice':match.group(1).upper() if match else None,
                            'status':'ok' if match else 'format_error','input_tokens':int(length),'generated_tokens':emitted,
                            'hit_token_limit':emitted>=protocol['max_new_tokens'] and eos not in tokens,
                            'payload_sha256':q['payload_sha256'],'experiment_signature':protocol['experiment_signature']}
                    f.write(json.dumps(record,ensure_ascii=False)+'\n')
                    if not match:print('Format error: '+q['id']+' '+repr(raw),flush=True)
            completed+=len(batch)
            dump('status.json',{'state':'running','questions':len(questions),'completed_responses':completed})
            print(f'QA {completed}/{len(questions)}; batch seconds={time.time()-started:.1f}; max input tokens={max(lengths)}',flush=True)
            del inputs,generated,tails
    dump('status.json',{'state':'generation_complete','questions':len(questions),'completed_responses':completed})


def group(label,rows):
    eligible=[r for r in rows if r['similarity_eligible']]
    good=[r['target_non_target_similarity'] for r in eligible if r['qa_correct']]
    bad=[r['target_non_target_similarity'] for r in eligible if not r['qa_correct']]
    sm=float(np.mean(good)) if good else None;fm=float(np.mean(bad)) if bad else None
    return {'group':label,'questions':len(rows),'correct':sum(r['qa_correct'] for r in rows),
            'accuracy':sum(r['qa_correct'] for r in rows)/len(rows) if rows else None,
            'similarity_questions':len(eligible),'success_n':len(good),'failure_n':len(bad),
            'success_mean':sm,'failure_mean':fm,'failure_minus_success':fm-sm if good and bad else None}


def figure(name,title,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(16,6),gridspec_kw={'width_ratios':[1,4]})
    for ax,subset in zip(axes,[groups[:1],groups[1:]]):
        for i,r in enumerate(subset):
            for dx,arm,color in [(-.18,'success','#2a7e9b'),(.18,'failure','#d17b45')]:
                value=r[arm+'_mean'];n=r[arm+'_n'];x=i+dx
                if value is None:ax.text(x,.02,'N/A\nn=0',ha='center',va='bottom',fontsize=9,color='#666')
                else:
                    ax.bar(x,value,width=.34,color=color,label=('QA correct' if arm=='success' else 'QA wrong') if i==0 else None)
                    ax.text(x,value+.025,f'{value:.3f}\nn={n}',ha='center',va='bottom',fontsize=9)
        ax.set_xticks(range(len(subset)),['Overall' if r['group']=='all' else TASK_NAMES[r['group']] for r in subset])
        ax.set_ylim(0,1.11);ax.set_yticks(np.arange(0,1.01,.2));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Mean target-to-non-target caption cosine similarity');axes[0].legend(loc='upper left',fontsize=9)
    axes[0].set_title('Pooled questions');axes[1].set_title('Within each task');fig.suptitle(title)
    fig.text(.5,.02,'Groups use actual MCQ answers, not evidence recall. Reader: Top-15 captions; original query images retained for 8 questions.\nFor each question: average target-to-first-five-non-target cosine similarities. Every selected non-target is in the reader Top-15.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.085,1,.94))
    for ext in ['png','svg','pdf']:fig.savefig(OUT/(name+'.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def analyze():
    protocol=json.loads((OUT/'protocol.json').read_text());questions=read(OUT/'qa_manifest.jsonl')
    responses=read(OUT/'qa_responses.jsonl');response={r['id']:r for r in responses}
    assert len(responses)==len(response)==len(questions)==371
    assert set(response)=={q['id'] for q in questions}
    memories=read(NAT/'memories.jsonl');positions={m['id']:i for i,m in enumerate(memories)}
    captions={r['id']:r['caption'] for r in read(NAT/'captions.jsonl')}
    images={r['id']:r for r in read(NAT/'images.jsonl')}
    texts={m['id']:' '.join(captions[i] for i in m['images']) for m in memories}
    cache=np.load(NAT/'embeddings.npz');assert list(cache['memory_ids'])==[m['id'] for m in memories]
    emb=cache['memory'];rankings={r['id']:r for r in read(RANK/'retrieval_details.jsonl')}
    rows=[];targets=[];details=[]
    for q in questions:
        r=response[q['id']];assert r['payload_sha256']==q['payload_sha256']
        assert r['experiment_signature']==protocol['experiment_signature']
        payload={'system':protocol['reader_system_prompt'],'user_text':q['reader_text'],
                 'query_images':[{'path':p,'sha256':sha(Path(p))} for p in q['query_image_paths']]}
        assert hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==q['payload_sha256']
        ranking=rankings[q['id']]['ranking'];gold=set(q['gold_ids'])
        assert q['top15_memory_ids']==[v['id'] for v in ranking[:15]]
        others=[v for v in ranking if v['id'] not in gold][:5]
        has_five=len(others)==5
        if has_five:assert all(v['id'] in q['top15_memory_ids'] for v in others)
        eligible=has_five and r['status']=='ok'
        correct=r['status']=='ok' and r['predicted_choice']==q['gold_choice']
        row={'id':q['id'],'task':q['task'],'x':q['x'],'y':q['y'],'question':q['question'],
             **{'option_'+l:q['options'][l] for l in 'ABCD'},'gold_choice':q['gold_choice'],
             'gold_answer_text':q['gold_answer_text'],'predicted_choice':r['predicted_choice'],
             'raw_response':r['raw_response'],'response_status':r['status'],'qa_correct':correct,
             'has_query_image':bool(q['query_image_paths']),'gold_count':len(gold),
             'gold_in_top15':sum(mid in gold for mid in q['top15_memory_ids']),
             'top15_memory_ids':'|'.join(q['top15_memory_ids']),'gold_ids':'|'.join(q['gold_ids']),
             'non_target_ids':'|'.join(o['id'] for o in others),
             'non_target_ranks':'|'.join(str(o['rank']) for o in others),
             'target_non_target_similarity':None,'similarity_eligible':eligible,
             'similarity_exclusion_reason':'' if eligible else 'fewer_than_five_non_targets' if not has_five else r['status']}
        audit={'id':q['id'],'question':q['question'],'options':q['options'],'gold_choice':q['gold_choice'],
               'predicted_choice':r['predicted_choice'],'qa_correct':correct,'targets':[],
               'five_non_targets':[{'id':o['id'],'query_rank':o['rank'],'caption':texts[o['id']],
                                   'image_paths':[images[i]['path'] for i in memories[positions[o['id']]]['images']]} for o in others]}
        if has_five:
            gi=[positions[mid] for mid in q['gold_ids']];oi=[positions[o['id']] for o in others]
            sims=emb[gi]@emb[oi].T;per_target=sims.mean(axis=1)
            row['target_non_target_similarity']=float(per_target.mean())
            for i,mid in enumerate(q['gold_ids']):
                targets.append({'id':q['id'],'task':q['task'],'target_id':mid,'gold_count':len(gold),
                    'qa_correct':correct,'response_status':r['status'],'similarity_eligible':eligible,
                    'mean_similarity_to_five_non_targets':float(per_target[i])})
                audit['targets'].append({'id':mid,'caption':texts[mid],
                    'image_paths':[images[j]['path'] for j in memories[positions[mid]]['images']],
                    'mean_similarity_to_five_non_targets':float(per_target[i]),
                    'similarities':{o['id']:float(sims[i,j]) for j,o in enumerate(others)}})
        rows.append(row);details.append(audit)
    primary=[group('all',rows)]+[group(t,[r for r in rows if r['task']==t]) for t in TASK_NAMES]
    extra=[group('single_target',[r for r in rows if r['gold_count']==1]),group('multi_target',[r for r in rows if r['gold_count']>1]),
           group('text_only_questions',[r for r in rows if not r['has_query_image']]),group('query_image_questions',[r for r in rows if r['has_query_image']])]
    for field in ['x','y']:
        for value in sorted({r[field] for r in rows}):extra.append(group(value,[r for r in rows if r[field]==value]))
    audit={'responses':len(responses),'unique_response_ids':len(response),'missing_responses':0,
           'format_errors':sum(r['status']!='ok' for r in responses),'generation_token_limit_hits':sum(r['hit_token_limit'] for r in responses),
           'input_tokens_max':max(r['input_tokens'] for r in responses),'similarity_questions':sum(r['similarity_eligible'] for r in rows),
           'no_non_target_comparison':sum(len(a['five_non_targets'])<5 for a in details),
           'all_selected_non_targets_in_reader_top15':True,'actual_correct_choice_counts':dict(collections.Counter(q['gold_choice'] for q in questions)),
           'predicted_choice_counts':dict(collections.Counter(r['predicted_choice'] for r in responses)),
           'all_payload_hashes_and_signatures_match':True,'caption_regeneration_calls':0}
    summary={'main':primary,'other_groups':extra,'audit':audit}
    for path,value in protocol['input_hashes'].items():assert sha(ROOT/path)==value
    for source in protocol['source_files']:assert sha(ROOT/source['path'])==source['sha256']
    dump('summary.json',summary);csvout('per_question.csv',rows);csvout('per_target.csv',targets)
    csvout('group_summary.csv',primary+extra);jsonl('caption_audit.jsonl',details)
    figure('qa_success_failure_by_task','MemEye Top-15 caption MCQ: QA correct vs QA wrong',primary)
    single=[r for r in rows if r['gold_count']==1]
    figure('single_target_qa_comparison','Single-target MCQs: QA correct vs QA wrong',
           [group('all',single)]+[group(t,[r for r in single if r['task']==t]) for t in TASK_NAMES])
    def table(gs):
        text='<table><tr><th>分组</th><th>题数</th><th>准确率</th><th>相似度：答对题数</th><th>答错题数</th><th>答对均值</th><th>答错均值</th><th>答错－答对</th></tr>'
        for g in gs:
            values=[g['success_mean'],g['failure_mean'],g['failure_minus_success']]
            cells=''.join('<td>'+('无样本' if v is None else f'{v:.4f}')+'</td>' for v in values)
            text+=f'<tr><td>{html.escape(g["group"])}</td><td>{g["questions"]}</td><td>{g["accuracy"]:.1%}</td><td>{g["success_n"]}</td><td>{g["failure_n"]}</td>{cells}</tr>'
        return text+'</table>'
    report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MemEye Top-15 选择题实验</title>
<style>body{font:16px/1.7 system-ui;max-width:1400px;margin:24px auto;padding:20px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ddd;padding:7px}</style>
<h1>MemEye Top-15 caption 选择题：按真实作答正确性分组</h1>
<p>BGE-small-en-v1.5 用原始选择题题干检索同任务全部含图轮次，Qwen3-VL-8B-Instruct 读取排名前 15 的普通 caption、原始题干和一组原始选项作答。8 道附图问题保留原始查询图片。历史图片、对话、日期、人物档案、session ID、答案与 clue 不作为作答输入；caption 未重新生成。</p>
<p>每题只生成一次，选项用数据原有排列，seed 42 平衡正确字母的位置。成功＝模型选择与当前选项排列的正确答案一致，失败＝有效选择错误。不是按 Recall 或标注证据是否全部找齐分组。</p>
<p>相似度：排除全部原始标注目标后，取完整查询排名最靠前的五个非目标；这五个全部在模型看到的 Top-15 内。对每个目标求与五个非目标的平均 caption 余弦相似度，再对目标平均到题目，最后按答对/答错对题目平均。5 道题没有非目标，仍进入准确率分母，但不进入相似度比较。多目标题不因标注超过五条而排除。</p>
<h2>主要结果</h2><img src="qa_success_failure_by_task.png">'''+table(primary)+'''
<h2>单目标与其他分组</h2><img src="single_target_qa_comparison.png">'''+table(extra)+'''
<p>这些是描述性均值，没有作显著性检验。总体比较可能受到任务构成影响，应该同时查看任务内部结果。相似度是 BGE caption 表示中的语义相似度，不能仅凭组间方向就确定线索过载或 caption 缺失。</p>
<h2>审计</h2><pre>'''+html.escape(json.dumps(audit,ensure_ascii=False,indent=2))+'''</pre>
<p><a href="per_question.csv">逐题选择、正确性与相似度</a> · <a href="per_target.csv">逐目标相似度</a> · <a href="qa_manifest.jsonl">实际模型输入、选项与检索条目</a> · <a href="qa_responses.jsonl">原始模型输出</a> · <a href="caption_audit.jsonl">目标/非目标 caption 与图片路径</a> · <a href="protocol.json">实验设置</a></p></html>'''
    (OUT/'report.html').write_text(report)
    protocol['status']='complete';dump('protocol.json',protocol)
    dump('status.json',{'state':'complete','questions':371,'completed_responses':371,'correct':primary[0]['correct'],'accuracy':primary[0]['accuracy']})
    dump('artifact_manifest.json',{'source_files_unchanged':True,
        'outputs':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'}})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','analyze']);args=parser.parse_args()
    {'prepare':prepare,'run':run,'analyze':analyze}[args.stage]()
