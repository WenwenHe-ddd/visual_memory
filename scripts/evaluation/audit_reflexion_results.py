"""Verify the complete state chain and compare archived answer artifacts without new inference."""

# Allow direct execution from any working directory.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from pathlib import Path

from scripts.memory.reflexion_memory import (ROOT, sha, hash_object, read_rows, write_json, upstream_helpers,
    make_memory, answer_messages, trial_text, reflection_prompt, apply_reflection)


def audit(target):
    protocol = json.loads((target/'protocol.json').read_text())
    for filename, digest in protocol['input_hashes'].items():
        assert sha(Path(filename)) == digest, filename
    questions = read_rows(target/'questions.jsonl')
    observations = json.loads((target/'observations.json').read_text())
    assert hash_object(questions) == protocol['prepared_questions_sha256']
    assert hash_object(observations) == protocol['observations_sha256']
    assert len(questions) == len({q['id'] for q in questions}) == 371
    responses = read_rows(target/'qa_responses.jsonl')
    manifest = read_rows(target/'qa_manifest.jsonl')
    answer = {r['id']:r for r in responses}
    qa = {q['id']:q for q in manifest}
    assert len(responses)==len(answer)==len(manifest)==len(qa)==len(questions)
    assert set(qa)==set(answer)=={q['id'] for q in questions}
    states = {q['id']:json.loads((target/'checkpoints'/(hash_object(q['id'])+'.json')).read_text()) for q in questions}
    assert len(list((target/'checkpoints').glob('*.json')))==371
    helpers = upstream_helpers()
    memories = {task:make_memory(rows, helpers) for task,rows in observations.items()}
    records=[]
    for q in questions:
        state=states[q['id']];memory=memories[q['task']]
        question,query,context,messages=answer_messages(q,memory,protocol,helpers)
        assert state['id']==q['id'] and state['task']==q['task']
        assert state['experiment_signature']==answer[q['id']]['experiment_signature']==protocol['experiment_signature']
        assert messages==state['answer_messages']==qa[q['id']]['answer_messages']
        assert hash_object(messages)==state['answer_payload_sha256']==answer[q['id']]['payload_sha256']==qa[q['id']]['payload_sha256']
        assert context==state['recalled_context'] and query==state['recall_query']
        assert question==state['question_with_options']
        assert state['insight_before']==memory.insight['global_insight']
        for k,v in state['answer'].items(): assert answer[q['id']][k]==v
        trial=trial_text(question,context,state['answer']['raw_response'],q['gold_choice'])
        prompt=reflection_prompt(memory,trial)
        assert prompt==state['reflection_prompt']
        assert hash_object([{'role':'user','content':prompt}])==state['reflection_payload_sha256']
        apply_reflection(memory,trial,prompt,state['reflection']['raw_response'])
        assert memory.insight['global_insight']==state['insight_after']
        visible=bool(state['insight_before']) and state['insight_before'] in context
        assert visible==state['prior_insight_fully_visible']
        saved_insight=memory.insight['global_insight']
        memory.insight['global_insight']=''
        context_without_insight=str(memory.recall({'text':query})).strip()
        memory.insight['global_insight']=saved_insight
        for generation,limit in [(state['answer'],protocol['max_new_tokens']), (state['reflection'],protocol['reflection_max_new_tokens'])]:
            assert generation['input_tokens']+limit <= protocol['context_budget']
        records.append({'id':q['id'],'task':q['task'],'correct':answer[q['id']]['predicted_choice']==q['gold_choice'],
            'prior_insight':bool(state['insight_before']),'prior_visible':visible,
            'prior_has_no_context_effect':bool(state['insight_before']) and context_without_insight==context,
            'answer_input_tokens':state['answer']['input_tokens'],'reflection_output_tokens':state['reflection']['generated_tokens']})
    audit_record={'n':371,'state_transitions_verified':371,'answer_payloads_verified':371,
        'reflection_payloads_verified':371,'input_hashes_verified':len(protocol['input_hashes']),
        'upstream_commit':protocol['upstream']['commit'],'feedback':'gold labels revealed only after each prediction; carried across subsequent questions of same task',
        'cross_task_state_resets':len(memories),'questions_with_prior_insight':sum(r['prior_insight'] for r in records),
        'questions_with_full_prior_insight_visible':sum(r['prior_visible'] for r in records),
        'questions_with_prior_insight_but_identical_context_without_it':sum(r['prior_has_no_context_effect'] for r in records),
        'mean_answer_input_tokens':sum(r['answer_input_tokens'] for r in records)/371,
        'mean_reflection_output_tokens':sum(r['reflection_output_tokens'] for r in records)/371,
        'by_task':{t:{'n':len(group:=[r for r in records if r['task']==t]),'prior_insight':sum(r['prior_insight'] for r in group),'prior_fully_visible':sum(r['prior_visible'] for r in group),'prior_has_no_context_effect':sum(r['prior_has_no_context_effect'] for r in group)} for t in memories}}
    write_json(target/'audit.json',audit_record)
    return protocol,questions,answer,audit_record


def compare(target,protocol,questions,answer):
    qmap={q['id']:q for q in questions}
    def counts(mapping):
        correct=sum(mapping[i]['predicted_choice']==qmap[i]['gold_choice'] for i in qmap)
        return {'n':len(qmap),'correct':correct,'accuracy':correct/len(qmap)}
    comparisons={'reflexion':{**counts(answer),'feedback':'gold after each answer'}}
    sources={'fc_t':ROOT/'results/visual_memory_pipeline/fc_t'}
    for name in ['caption','unique','shared_unique']:
        sources[name+'_top5']=ROOT/'results/mcq_top5_10_ablation/top5'/name
    sources['caption_top15']=ROOT/'results/mcq_top15_caption_qa'
    for name,p in sources.items():
        old_protocol=json.loads((p/'protocol.json').read_text())
        old_qrows=read_rows(p/'qa_manifest.jsonl');old_rrows=read_rows(p/'qa_responses.jsonl')
        old_q={q['id']:q for q in old_qrows};old_r={r['id']:r for r in old_rrows}
        assert len(old_qrows)==len(old_q)==len(old_rrows)==len(old_r)==371
        assert set(old_q)==set(old_r)==set(qmap)
        for key in ['reader_model','reader_model_path','reader_system_prompt','seed','do_sample','dtype','max_new_tokens']:
            assert old_protocol[key]==protocol[key],(name,key)
        for i,q in qmap.items():
            for key in ['task','question','options','gold_choice','gold_ids','query_image_paths']:
                assert q[key]==old_q[i][key],(name,i,key)
            assert old_r[i]['experiment_signature']==old_protocol['experiment_signature']
            payload={'system':old_protocol['reader_system_prompt'],'user_text':old_q[i]['reader_text'],
                     'query_images':[{'path':v,'sha256':sha(Path(v))} for v in old_q[i]['query_image_paths']]}
            assert hash_object(payload)==old_r[i]['payload_sha256']==old_q[i]['payload_sha256']
        comparisons[name]={**counts(old_r),'feedback':'none','source':str(p),
            'source_hashes':{f:sha(p/f) for f in ['protocol.json','qa_manifest.jsonl','qa_responses.jsonl']}}
    result={'interpretation':'descriptive comparison only: Reflexion has sequential gold feedback, 6000-word truncation, different context formatting, and captioned rather than native query images',
            'scoring':'one fixed MCQ permutation, original 371-question denominator; invalid answers count as wrong', 'methods':comparisons}
    write_json(target/'comparison_existing.json',result)
    return result


if __name__=='__main__':
    target=ROOT/'results/visual_memory_pipeline/reflexion'
    protocol,questions,answer,record=audit(target)
    result=compare(target,protocol,questions,answer)
    print(json.dumps({'audit':record,'comparison':{k:{f:v[f] for f in ['n','correct','accuracy']} for k,v in result['methods'].items()}},indent=2))
