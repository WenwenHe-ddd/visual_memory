"""Seal all three completed methods and disclose the MemoryOS budget revision."""
from collections import Counter
import csv
import hashlib
from datetime import datetime,timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.evaluation import mem_gallery_deepseek_text_methods as run
from scripts.analysis.audit_mem_gallery_deepseek_text_methods import report


def finalize():
    output=Path(__file__).resolve().parent
    verified={};inputs={};results=[]
    for method in run.METHODS:
        folder=output/method;p=run.verify(folder)
        done=run.read(folder/'completion_manifest.json');assert done['status']=='complete'
        for name,digest in done['artifacts_sha256'].items():assert run.sha(name)==digest,name
        audit=run.read(folder/'audit.json');assert audit['status']=='passed'
        assert audit['questions']==1711 and audit['source_rounds']==3962
        result=run.read(folder/'final_results.json');assert result['overall']['questions']==1711
        reader_outputs=run.rows(folder/'reader/responses.jsonl')
        assert all(o['input_tokens']+p['max_new_tokens']<=p['context_budget'] for o in reader_outputs)
        inputs[method]={name:run.sha(folder/name) for name in ('observations.json','inputs/questions.jsonl','inputs/gold.jsonl')}
        verified[method]=dict(status='passed',artifacts=len(done['artifacts_sha256']),
            experiment_signature=p['experiment_signature'],memory_completion_floor=p['memory_output_floor'],
            memory_calls=audit['api_usage']['memory']['calls'],reader_calls=audit['api_usage']['answer']['calls'],
            maximum_reader_input_tokens=max(o['input_tokens'] for o in reader_outputs))
        results.append(dict(method=run.NAMES[method],**result['overall']))
    assert all(d==inputs['gen_agents'] for d in inputs.values())
    revision=run.read(output/'memoryos_budget_revision.json')
    for name,digest in revision['previous_artifacts_sha256'].items():
        assert run.sha(output/'attempts/memoryos_cap2048'/name)==digest,name
    extension=run.read(output/'memoryos/budget_extension_protocol.json')
    prefix=hashlib.sha256();prefix_lines=None
    with (output/'memoryos/memory/api_responses.jsonl').open('rb') as stream:
        for count,line in enumerate(stream,1):
            prefix.update(line)
            if prefix.hexdigest()==extension['old_api_responses_sha256']:
                prefix_lines=count;break
    assert prefix_lines is not None,'Original API response log prefix changed'
    for task,digest in extension['retained_complete_tasks'].items():
        assert run.sha(output/'memoryos/memory'/task/'complete.json')==digest,task
    report(output)
    details=run.rows(output/'simplemem/memory/retrieval.jsonl');channels=Counter()
    for row in details:
        batches=row['retrieval_detail']['search_batches']
        for channel in ('semantic','keyword','structured'):
            channels[channel+'_called']+=any(b['channel']==channel for b in batches)
            channels[channel+'_nonempty']+=any(b['channel']==channel and b['result_ids'] for b in batches)
        channels['reflection_additional_search']+=any(b['phase']=='reflection' for b in batches)
    overflow=run.read(output/'memoryos/audit.json')['budget_extension']
    note=('MemoryOS was rebuilt from scratch with an 8192-token memory-completion minimum after a 2048-token profile truncation. '
          'The failed attempt is preserved under attempts/memoryos_cap2048 and no generated state or responses from it were reused. '
          'A later 8192-token profile truncation triggered a documented continuation: complete scenarios and full-response caches were retained, '
          'incomplete states were rebuilt, and only truncated requests were retried with the same content and sampling at 32768 tokens. '
          f'This continuation discarded {overflow["discarded_truncated_completions"]} incomplete responses and accepted {overflow["expanded_complete_responses"]} complete expanded responses. '
          'The previous GPT-5 MemoryOS control used 2048 tokens. All three readers still use 2048 tokens. '
          'No completed memory or answer in the final runs may be truncated. Provider inspection failures, if any, count as zero-scored answers.')
    comparison=run.read(output/'comparison.json');comparison.update(memory_budget_note=note,simplemem_actual_channels=dict(channels))
    run.dump(output/'comparison.json',comparison)
    report_path=output/'report.html';report_path.write_text(report_path.read_text()+'<p>'+note+'</p>')
    fields=['method','questions','correct','em','f1','bleu_1','bleu_2','bleu','gold_hits','gold_links','source_recall_micro','provider_rejected_answers']
    with (output/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    old=run.read(output/'execution.json');run.freeze(output/'initial_driver_execution.json',old)
    run.dump(output/'execution.json',dict(status='complete',completed=datetime.now(timezone.utc).isoformat(),
        methods={m:'complete' for m in run.METHODS},memoryos_budget_revision=str(output/'memoryos_budget_revision.json')))
    run.dump(output/'postrun_verification.json',dict(status='passed',completed=datetime.now(timezone.utc).isoformat(),
        methods=verified,identical_inputs=inputs['gen_agents'],simplemem_actual_channels=dict(channels),
        memory_budget_note=note,memoryos_overflow_audit=overflow,memoryos_preserved_api_log_prefix_lines=prefix_lines,
        memoryos_retained_complete_scenarios=len(extension['retained_complete_tasks']),no_final_memory_or_answer_truncation=True))
    names=('comparison.json','comparison.csv','report.html','postrun_verification.json','execution.json',
           'initial_driver_execution.json','memoryos_budget_revision.json','simplemem_latest_verified.json',
           'validation.json','run_all.py','restart_memoryos_budget.py','finalize.py','memoryos/budget_extension_protocol.json')
    run.dump(output/'completion_manifest.json',dict(status='complete',
        artifacts_sha256={str(output/n):run.sha(output/n) for n in names},
        method_manifest_sha256={m:run.sha(output/m/'completion_manifest.json') for m in run.METHODS}))
    print('All methods and final comparison verified and sealed.',flush=True)


if __name__=='__main__':finalize()
