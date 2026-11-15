"""Export completed, audited GPT-5 methods without changing signed results."""
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/memeye_gpt5_seed42_text_methods_20261025'
SOURCE=OUT/'caption_source'
NAMES={'gen_agents':'Generative Agents','memoryos':'MemoryOS','simplemem':'SimpleMem（最多5条）','a_mem':'A-Mem（最多5条）'}
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def snapshot_sources():
    paths=set()
    for method in NAMES:
        paths.update(Path(p) for p in read(OUT/method/'protocol.json')['input_sha256'] if Path(p).suffix=='.py')
    vendor=ROOT/'scripts/vendor_memeye_text'
    paths.update(vendor/name for name in read(vendor/'UPSTREAM.json')['files'])
    paths.add(vendor/'UPSTREAM.json')
    paths.update(ROOT/'scripts/evaluation'/name for name in [
        'continue_deepseek_text_methods_parallel.py','continue_deepseek_text_methods_streaming.py',
        'continue_deepseek_ga_format.py','continue_deepseek_memoryos_format.py','probe_deepseek_ga_format.py'])
    paths.update([Path(__file__),ROOT/'scripts/analysis/audit_gpt5_text_methods.py',ROOT/'scripts/tests/test_gpt5_text_methods.py'])
    target=OUT/'source_snapshot';target.mkdir(exist_ok=True)
    manifest={}
    for path in sorted(paths):
        digest=sha(path);copy=target/(digest[:16]+'_'+path.name)
        shutil.copyfile(path,copy);assert sha(copy)==digest
        manifest[str(path)]=dict(snapshot=str(copy.relative_to(OUT)),sha256=digest)
    dump(target/'manifest.json',manifest)
    sys.path.insert(0,str(ROOT/'.vendor'))
    from importlib.metadata import version,PackageNotFoundError
    versions={}
    for name in ['numpy','torch','transformers','sentence-transformers','openai','faiss-cpu','scikit-learn','rank-bm25','langchain-core','jsonschema']:
        try:versions[name]=version(name)
        except PackageNotFoundError:versions[name]=None
    dump(OUT/'software_environment.json',dict(python=sys.version,packages=versions))


def recall(method):
    root=OUT/method
    observations=read(root/'observations.json')
    known={task:{r['id'] for r in data} for task,data in observations.items()}
    data=[]
    for r in rows(root/'qa_responses.jsonl'):
        d=r['retrieval_detail']
        if method=='a_mem':ids=d['context_memory_ids']
        else:
            ids=[]
            for item in d['items']:
                tags=[t[len('round_id:'):] for t in item.get('tags',[]) if t.startswith('round_id:')]
                assert len(tags)==1,(r['id'],item['id'])
                ids.append(tags[0])
        assert len(ids)<=5 and all(i in known[r['task']] for i in ids)
        sources=[r['task']+'/'+i for i in ids]
        gold=r['gold_ids'];hits=len(set(sources)&set(gold))
        data.append(dict(id=r['id'],source_round_ids=sources,gold_ids=gold,hits=hits,gold_count=len(gold),recall=hits/len(gold)))
    hits=sum(x['hits'] for x in data);total=sum(x['gold_count'] for x in data)
    assert len(data)==371 and total==1041
    result=dict(method=method,definition='Top5 selected memory units mapped to original source-round IDs; no extra credit for generated mentions of other sources',
        questions=371,hits=hits,gold_references=total,micro=hits/total,macro=sum(x['recall'] for x in data)/371,
        questions_with_any_target=sum(x['hits']>0 for x in data),context_counts=dict(Counter(len(x['source_round_ids']) for x in data)),
        response_sha256=sha(root/'qa_responses.jsonl'))
    dump(root/'source_recall_at_5.json',result)
    (root/'source_recall_per_question.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data))
    return result


def main():
    results={}
    for method in NAMES:
        root=OUT/method;a=read(root/'external_audit.json');s=read(root/'summary.json')
        assert a['passed'] and s['status']=='complete' and s['questions']==371 and a['correct']==s['correct']
        for name,h in a['artifacts_sha256'].items():assert sha(root/name)==h,name
        results[method]=dict(s,external_audit_passed=True)
        if method in ('simplemem','a_mem'):results[method]['source_recall_at_5']=recall(method)
    dump(OUT/'final_results.json',results)
    with (OUT/'metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','correct','questions','accuracy','recall_at_5_micro','recall_at_5_macro','invalid_answers','memory_truncations'])
        for m,s in results.items():
            r=s.get('source_recall_at_5',{})
            w.writerow([m,s['correct'],371,s['accuracy'],r.get('micro',''),r.get('macro',''),s['invalid_answers'],s['memory_token_limit_hits']])
    by_method={m:{r['id']:r for r in rows(OUT/m/'qa_responses.jsonl')} for m in NAMES}
    with (OUT/'paired_questions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['question_id','gold_choice',*[x for m in NAMES for x in (m+'_answer',m+'_correct')]])
        for q in rows(SOURCE/'inputs/evaluation.jsonl'):
            w.writerow([q['id'],q['gold_choice'],*[x for m in NAMES for x in (by_method[m][q['id']]['predicted_choice'],by_method[m][q['id']]['correct'])]])
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>GPT-5 四种记忆方法</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto}td,th{padding:10px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style>',
        '<h1>GPT-5 · MemEye · seed42</h1><p>四组各371题。437张图的caption、记忆构建与最终回答全部重新调用官方GPT-5，所有请求seed42，推理minimal，温度不传。完成预算max(2048,原方法预算)，包含推理；最终回答预算2048。实际模型版本与caption版本逐次核对。</p>',
        '<table><tr><th>方法</th><th>正确数</th><th>准确率</th><th>Recall@5 micro</th><th>上下文数量</th><th>无效回答</th></tr>']
    for m,s in results.items():
        r=s.get('source_recall_at_5');rec=f'{r["micro"]:.4%}' if r else '不适用'
        parts.append(f'<tr><td>{NAMES[m]}</td><td>{s["correct"]}/371</td><td>{s["accuracy"]:.4%}</td><td>{rec}</td><td>{html.escape(str(s["context_counts"]) if s["context_counts"] else "原方法配置")}</td><td>{s["invalid_answers"]}</td></tr>')
    parts+=['</table><p>SimpleMem/A-Mem按原顺序去重后最多保留5条，不补齐空检索。Recall@5仅按选中记忆的原始来源轮次计算；摘要提及其他轮次不额外计入。Generative Agents原配置K10，MemoryOS多层组合，不标作Recall@5。</p>',
        '<p>四种方法输入完整848轮历史（含438条图片记忆），查询和回答使用题图caption。此前baseline/master仅索引438条图片记忆且回答时输入原始题图，输入口径不同。与DeepSeek相比，GPT-5还改变了温度接口和包含推理的完成预算，不能视为严格只换模型而全部采样设置相同。</p>',
        '<p>沿用正式DeepSeek实验的GA格式系统提示和MemoryOS无损摘要格式归一化。GA记忆格式失败最多3次相同请求；最终回答不因格式或正确性重试。原始输出及上游回退保留。seed42不保证服务端绝对复现。</p>',
        '<table><tr><th>方法</th><th>记忆截断数</th><th>结构化输出检查未通过</th></tr>']
    for m,s in results.items():parts.append(f'<tr><td>{NAMES[m]}</td><td>{s["memory_token_limit_hits"]}</td><td>{s["invalid_structured_outputs"]}</td></tr>')
    parts.append('</table><p>结构化检查仅涵盖明确声明schema的调用，并不等于缺失记忆数量。最终无效/截断回答全部计错。</p><p><a href="metrics.csv">指标CSV</a> · <a href="paired_questions.csv">逐题结果</a> · <a href="external_audit.json">独立审计</a></p>')
    for m in NAMES:parts.append(f'<details><summary>{NAMES[m]}：五项方法标注</summary><pre>'+html.escape(json.dumps(read(OUT/m/'method_annotations.json'),ensure_ascii=False,indent=2))+'</pre></details>')
    parts.append('</html>');(OUT/'report.html').write_text('\n'.join(parts))
    snapshot_sources()
    files=[OUT/n for n in ['final_results.json','metrics.csv','paired_questions.csv','report.html','external_audit.json','public_input_provenance.json','source_snapshot/manifest.json','software_environment.json']]
    files.append(SOURCE/'completion_manifest.json')
    for m in NAMES:
        root=OUT/m
        dump(root/'calls_manifest.json',{p.name:sha(p) for p in sorted((root/'calls').glob('*.json'))})
        for name in ['protocol.json','summary.json','method_annotations.json','external_audit.json','qa_responses.jsonl','calls_manifest.json','api_responses.jsonl','api_attempts.jsonl','call_events.jsonl','source_recall_at_5.json','source_recall_per_question.jsonl','format_retry_events.jsonl','summary_normalization_events.jsonl']:
            if (root/name).exists():files.append(root/name)
    dump(OUT/'completion_manifest.json',dict(status='complete',questions_per_method=371,seed=42,all_independent_audits_passed=True,artifacts_sha256={str(p.relative_to(OUT)):sha(p) for p in files}))
    print(json.dumps({m:dict(correct=s['correct'],questions=371,accuracy=s['accuracy']) for m,s in results.items()}))


if __name__=='__main__':main()
