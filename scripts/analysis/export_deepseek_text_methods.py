"""Export completed, audited DeepSeek methods without changing signed results."""
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/memeye_deepseek_flash_seed42_text_methods_20261025'
SOURCE=ROOT/'results/memeye_deepseek_flash_seed42_full_20261025'
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
    paths.update([Path(__file__),ROOT/'scripts/analysis/audit_deepseek_text_methods.py',ROOT/'scripts/tests/test_deepseek_text_methods.py'])
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
        p=OUT/method
        audit=read(p/'external_audit.json');summary=read(p/'summary.json')
        assert audit['passed'] and summary['status']=='complete' and summary['questions']==371
        assert audit['correct']==summary['correct']
        for name,expected in audit['artifacts_sha256'].items():assert sha(p/name)==expected,name
        results[method]=dict(summary,external_audit_passed=True)
        if method in ('simplemem','a_mem'):results[method]['source_recall_at_5']=recall(method)
    dump(OUT/'final_results.json',results)
    fields=['method','correct','questions','accuracy','context_memory_limit','recall_at_5_micro','recall_at_5_macro','invalid_answers','memory_invalid_structured_outputs','memory_truncations']
    with (OUT/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for m,s in results.items():
            r=s.get('source_recall_at_5',{})
            writer.writerow(dict(method=m,correct=s['correct'],questions=371,accuracy=s['accuracy'],context_memory_limit=5 if r else '',recall_at_5_micro=r.get('micro',''),recall_at_5_macro=r.get('macro',''),invalid_answers=s['invalid_answers'],memory_invalid_structured_outputs=s['invalid_structured_outputs'],memory_truncations=s['memory_token_limit_hits']))
    with (OUT/'paired_questions.csv').open('w',newline='') as f:
        by_method={m:{r['id']:r for r in rows(OUT/m/'qa_responses.jsonl')} for m in NAMES}
        writer=csv.writer(f);writer.writerow(['question_id','gold_choice',*[x for m in NAMES for x in (m+'_answer',m+'_correct')]])
        for q in rows(SOURCE/'inputs/evaluation.jsonl'):
            writer.writerow([q['id'],q['gold_choice'],*[x for m in NAMES for x in (by_method[m][q['id']]['predicted_choice'],by_method[m][q['id']]['correct'])]])
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>DeepSeek seed42 四种记忆方法</title>',
        '<style>body{font:16px/1.7 system-ui;max-width:1200px;margin:32px auto;padding:0 18px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
        '<h1>DeepSeek V4.1 Flash：四种记忆方法</h1>',
        '<p>四组均完成 MemEye 全部 371 题并通过独立核验。记忆库使用同一份 DeepSeek seed42 caption，四种方法均重新构建记忆；摘要、反思、关键词生成和最终回答均使用 DashScope International 的 deepseek-v4.1-flash。所有实际请求 seed=42，关闭 thinking；记忆操作保留上游温度，最终回答 temperature=0、max_tokens=16。</p>',
        '<table><tr><th>方法</th><th>正确数</th><th>准确率</th><th>最终记忆条数分布</th><th>无效回答</th></tr>']
    for m,s in results.items():
        counts=html.escape(str(s['context_counts'])) if s['context_counts'] else '保留原方法配置'
        parts.append(f'<tr><td>{NAMES[m]}</td><td>{s["correct"]}/371</td><td>{s["accuracy"]:.4%}</td><td>{counts}</td><td>{s["invalid_answers"]}</td></tr>')
    parts+=['</table><p>所有无效/截断最终回答均保留并计错，不因结果错误重试。SimpleMem/A-Mem 保留原检索与扩展顺序，按记忆ID去重后最多取5条，不补齐空检索。</p>',
        '<h2>Recall@5：可追溯到原始轮次的两种方法</h2><p>按最终选中记忆单元的原始来源轮次计算。SimpleMem 通过 round_id 标签映射，A-Mem 通过笔记ID映射；生成摘要提及其他轮次不会额外计为命中。Generative Agents 默认取10条，MemoryOS 为多层组合上下文，因此不把它们的结果标为 Recall@5。</p>',
        '<table><tr><th>方法</th><th>micro</th><th>macro</th><th>至少命中一个目标的题数</th></tr>']
    for m in ('simplemem','a_mem'):
        r=results[m]['source_recall_at_5']
        parts.append(f'<tr><td>{NAMES[m]}</td><td>{r["hits"]}/1041 ({r["micro"]:.4%})</td><td>{r["macro"]:.4%}</td><td>{r["questions_with_any_target"]}/371</td></tr>')
    parts+=['</table><h2>比较口径</h2><p>四种文本方法沿用此前 Qwen 实验的输入：全部848轮历史（含438条图片记忆），题图使用同模型caption；此前 baseline/master 仅索引438条图片记忆，回答模型还收到原始题图。这些输入结构不同，因此不能将跨方法准确率差异完全解释为记忆算法差异。</p>',
        '<h2>格式适配与运行异常</h2><p>Generative Agents 原评分提示在 DeepSeek 下出现长文输出；完整原始尝试已归档到 gen_agents_initial_format。正式结果对所有重要性评分/反思调用统一增加明确格式的 system 说明，评分范围和反思规则不变。格式失败最多3次完全相同请求重试，所有原始响应保留；最终回答提示未改变。</p><p>MemoryOS 在 Cartoon 出现单个 JSON 摘要对象而上游期待列表的类型错误。适配将单个对象套入列表，并将 summary 字段原文映射到 content，保留关键词；有效列表不变。请求、生成内容和最终回答均未改写。其他解析/回退沿用上游行为。以下为记忆操作记录：</p>',
        '<table><tr><th>方法</th><th>缓存调用数</th><th>记忆调用截断</th><th>原始结构化输出未通过检查</th></tr>']
    for m,s in results.items():parts.append(f'<tr><td>{NAMES[m]}</td><td>{s["unique_llm_calls"]}</td><td>{s["memory_token_limit_hits"]}</td><td>{s["invalid_structured_outputs"]}</td></tr>')
    parts+=['</table><p>结构化格式统计仅涵盖请求明确声明 JSON/schema 的调用；上游解析器可能进一步清理格式，因此该列不等于缺失的记忆数。未声明 schema 的生成不在此列中评价。</p><p>MemoryOS 的259次记忆输出截断中，230次来自原配置100-token的对话元摘要，29次来自2000-token的用户偏好分析；没有擅自加长预算或重生成。<a href="memoryos/memory_truncation_breakdown.json">操作级统计</a>。这与最终MCQ回答是否截断分别统计。</p><p>seed=42 是实际发送的请求参数，不保证服务端跨请求绝对复现。调度曾增加并行度并取消批量等待，所有调度调整均有独立记录；同轮完全相同请求可复用其原响应。</p>',
        '<p><a href="metrics.csv">指标CSV</a> · <a href="paired_questions.csv">逐题比较CSV</a> · <a href="external_audit.json">四组独立核验</a> · <a href="public_input_provenance.json">公开数据来源核验</a> · <a href="final_results.json">完整统计JSON</a></p>']
    for m in NAMES:
        parts.append(f'<details><summary>{NAMES[m]}：候选结构、查询、检索、编码、回答五项标注</summary><pre>'+html.escape(json.dumps(read(OUT/m/'method_annotations.json'),ensure_ascii=False,indent=2))+'</pre></details>')
    parts.append('</html>');(OUT/'report.html').write_text('\n'.join(parts))
    snapshot_sources()
    files=[OUT/name for name in ['final_results.json','metrics.csv','paired_questions.csv','report.html','external_audit.json','public_input_provenance.json','source_snapshot/manifest.json','software_environment.json']]
    files.extend(OUT/m/name for m in NAMES for name in ['summary.json','external_audit.json','method_annotations.json','protocol.json','streaming_amendment.json'])
    for method in NAMES:
        root=OUT/method
        dump(root/'calls_manifest.json',{p.name:sha(p) for p in sorted((root/'calls').glob('*.json'))})
        files.extend(root/name for name in ['calls_manifest.json','api_responses.jsonl','api_attempts.jsonl','call_events.jsonl'])
        for name in ['source_recall_at_5.json','source_recall_per_question.jsonl','format_amendment.json',
                     'format_retry_events.jsonl','memoryos_format_amendment.json','summary_normalization_events.jsonl',
                     'memory_truncation_breakdown.json']:
            if (root/name).exists():files.append(root/name)
    dump(OUT/'completion_manifest.json',dict(status='complete',questions_per_method=371,methods=list(NAMES),seed=42,
        caption_sha256=sha(SOURCE/'captions.jsonl'),all_independent_audits_passed=True,
        exporter_sha256=sha(Path(__file__)),artifacts_sha256={str(p.relative_to(OUT)):sha(p) for p in files}))
    print(json.dumps({m:dict(correct=s['correct'],questions=371,accuracy=s['accuracy'],context_counts=s['context_counts'],invalid_answers=s['invalid_answers']) for m,s in results.items()},ensure_ascii=False))


if __name__=='__main__':main()
