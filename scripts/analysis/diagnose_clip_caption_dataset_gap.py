"""Descriptive audit of existing CLIP and caption runs; no new model inference."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/clip_caption_dataset_gap_20261103'
FILES = {}

def read(relative, lines=False):
    p=ROOT/relative
    data=p.read_bytes();FILES[str(p)]=hashlib.sha256(data).hexdigest()
    return [json.loads(l) for l in data.splitlines() if l.strip()] if lines else json.loads(data)


def main():
    prefix='results/mem_gallery_master_answer_20261021/'
    memories={r['id']:r for r in read(prefix+'inputs/memories.jsonl',True)}
    questions={r['id']:r for r in read(prefix+'inputs/questions.jsonl',True)}
    gold={r['id']:r for r in read(prefix+'inputs/gold.jsonl',True)}
    paired={r['id']:r for r in read(prefix+'reader/paired_results.json')}
    baseline={k:r['baseline_question_only'] for k,r in paired.items()}
    clip_root='results/mem_gallery_clip_image_dialogue_qwen_20261102/'
    manifest={r['id']:r for r in read(clip_root+'qa_manifest.jsonl',True)}
    clip={r['id']:r for r in read(clip_root+'per_question.jsonl',True)}
    assert set(gold)==set(questions)==set(baseline)==set(clip)==set(manifest)
    assert len(gold)==1711
    retrieval={}
    for visual in (True,False):
        total=b_hits=c_hits=0
        for k,g in gold.items():
            targets={mid for mid in g['gold_ids'] if memories[mid]['is_visual']==visual}
            total+=len(targets)
            b_hits+=len(targets&set(baseline[k]['top5_memory_ids']))
            c_hits+=len(targets&set(manifest[k]['top_memory_ids']))
        retrieval['visual_gold' if visual else 'text_only_gold']=dict(
            gold=total,caption_hits=b_hits,clip_hits=c_hits,
            caption_recall=b_hits/total,clip_recall=c_hits/total)
    groups={
        'all_questions':list(questions),
        'with_query_image':[k for k,q in questions.items() if q['query_image_paths']],
        'without_query_image':[k for k,q in questions.items() if not q['query_image_paths']],
        'all_gold_visual':[k for k,g in gold.items() if g['gold_ids'] and all(memories[mid]['is_visual'] for mid in g['gold_ids'])],
        'has_text_only_gold':[k for k,g in gold.items() if any(not memories[mid]['is_visual'] for mid in g['gold_ids'])],
    }
    splits={}
    for name,ids in groups.items():
        splits[name]=dict(questions=len(ids),caption={metric:sum(baseline[k][metric] for k in ids)/len(ids) for metric in ('em','f1','bleu_1')},
            clip={metric:sum(clip[k][metric] for k in ids)/len(ids) for metric in ('em','f1','bleu_1')})
    examples=[]
    for k,r in paired.items():
        if baseline[k]['em'] and not clip[k]['em'] and r['gold_ids'] and all(not memories[mid]['is_visual'] for mid in r['gold_ids']):
            examples.append(dict(id=k,question=questions[k]['question'],answer=r['answer'],caption_answer=baseline[k]['raw_response'],
                clip_answer=clip[k]['prediction'],evidence=[dict(id=mid,dialogue=memories[mid]['dialogue']) for mid in r['gold_ids']]))
    memeye={}
    for model,base_path,clip_path in [
        ('qwen','results/memeye_qwen_zscore_joint_search_20261024/paired_results.json','results/visual_memory_pipeline/top5/clip_image_dialogue'),
        ('deepseek','results/memeye_deepseek_flash_seed42_full_20261025/weighted_evaluation/paired_results.json','results/visual_memory_pipeline/top5/clip_image_dialogue_api/deepseek-v4.1-flash')]:
        base={r['question_id']:r for r in read(base_path)}
        ans={r['id']:r for r in read(clip_path+'/qa_responses.jsonl',True)}
        assert set(base)==set(ans) and len(base)==371
        wins=losses=bc=cc=0
        for k,r in base.items():
            new=ans[k]['predicted_choice']==r['gold_choice'];old=r['baseline_question_only_correct']
            wins+=new and not old;losses+=old and not new;bc+=old;cc+=new
        memeye[model]=dict(questions=len(base),caption_correct=bc,clip_correct=cc,caption_acc=bc/len(base),clip_acc=cc/len(base),
            wrong_to_correct=wins,correct_to_wrong=losses)
    result=dict(scope='Existing-run descriptive audit, not a matched causal ablation',
        gallery_candidate_rounds=len(memories),gallery_visual_rounds=sum(r['is_visual'] for r in memories.values()),
        gallery_gold_breakdown=retrieval,gallery_qwen_question_splits=splits,memeye_paired_changes=memeye,
        text_only_evidence_examples=examples[:5],source_sha256=FILES)
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/'diagnosis.json';dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(dest)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','text_only_evidence_examples')},ensure_ascii=False))

if __name__=='__main__':main()
