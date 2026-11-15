"""Independent post-run audit of the completed DeepSeek MemGallery judge results.

No API calls, no runner imports, and no writes to existing experiment inputs.
Only emits validation_final.json after every audit below has succeeded.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import string
from datetime import datetime, timezone

ROOT = Path('/storage/hewenwen/VisualMemory')
DEFAULT_OUTPUT = ROOT / 'results/memgallery_deepseek_flash_llm_judge_20261105'
GRADES = (0, 0.25, 0.5, 0.75, 1)
ANSWER_MODEL = 'deepseek-v4.1-flash'
JUDGE_MODEL = 'deepseek-v4.1-flash'
EXPECTED_EM = {'semantic_rag':770, 'ours1_ours2':785, 'gen_agents':374,
               'memoryos':622, 'simplemem':644, 'a_mem':678,
               'clip_image_dialogue':656, 'hippomm':427}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def unique(rows, key='id'):
    result = {row[key]: row for row in rows}
    check(len(result) == len(rows), f'Duplicate {key}')
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def normalize(value):
    # Independent implementation of the fixed official EM normalization.
    text = str(value).lower()
    text = re.sub(r'(?<=\d)\.(?=\d)', 'DOTPLACEHOLDER', text)
    text = text.replace('_', 'UNDERSCOREPLACEHOLDER')
    text = re.sub(r'\b(a|an|the|and)\b', ' ', text)
    text = ''.join(ch if ch not in string.punctuation else ' ' for ch in text)
    return ' '.join(text.replace('DOTPLACEHOLDER', '.').replace('UNDERSCOREPLACEHOLDER', '_').split())


def build_wire(row, template):
    fields = {key: row[key] for key in ('question', 'ground_truth', 'model_output')}
    prompt = re.sub(r'\{\{(question|ground_truth|model_output)\}\}',
                    lambda match: fields[match.group(1)], template)
    return {'model': JUDGE_MODEL,
            'messages': [{'role':'user','content':[{'type':'text','text':prompt}]}],
            'temperature':0,'seed':42,'max_tokens':1024,'n':1,'stream':False,
            'enable_thinking':False}


def parse_raw(text):
    cleaned = text.strip()
    fence = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned)
    value = json.loads(fence.group(1) if fence else cleaned)
    check(isinstance(value, dict), 'Judge output is not an object')
    check(type(value.get('score')) in (int,float) and value['score'] in GRADES,
          'Raw judge score is not an exact five-grade number')
    check(isinstance(value.get('reasoning'), str) and value['reasoning'].strip(),
          'Raw judge reasoning is empty')
    return value


def audit_score(row, request_hash, *, parsed_fields=True):
    check(row['request_sha256'] == request_hash, 'Score request hash differs')
    check(row['requested_model'] == row['returned_model'] == JUDGE_MODEL,
          'Actual judge model differs')
    check(row['finish_reason'] == 'stop' and not row.get('refusal'),
          'Judge refused or failed to finish')
    check(row.get('seed') == 42 and row.get('enable_thinking') is False,
          'Judge seed/thinking differs')
    check(not row.get('reasoning_content_present') and
          row.get('reasoning_tokens') in (None,0), 'Unexpected reasoning output')
    grade = parse_raw(row['raw_response'])
    if parsed_fields:
        check(row.get('score') == grade['score'] and row.get('reasoning') == grade['reasoning'],
              'Saved grade/reasoning differs from original raw judge JSON')
    return grade


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    summary = read_json(output/'summary.json')
    check(summary.get('status') == 'complete', 'Run is not complete; final validation was not written')
    protocol = read_json(output/'protocol.json')
    check(protocol['answer_model'] == ANSWER_MODEL, 'Protocol mislabels answer backbone')
    fixed = {'judge_model':JUDGE_MODEL,'temperature':0,'seed':42,'enable_thinking':False,
             'max_tokens':1024,'grades':list(GRADES),
             'judge_input_fields':['question','ground_truth','model_output'],
             'images_context_captions_sent_to_judge':False}
    for key,value in fixed.items():
        check(protocol.get(key) == value, f'Protocol judge setting differs: {key}')
    checked_hashes = {}
    hash_counts = {}
    for field in ('source_sha256','implementation_sha256','compatible_cache_sha256',
                  'canonical_question_cache_sha256'):
        mapping = protocol[field]
        for path,expected in mapping.items():
            actual = sha(path)
            check(actual == expected, f'Frozen {field} changed: {path}')
            checked_hashes[path] = actual
        hash_counts[field] = len(mapping)
    check(sha(protocol['manifest_path']) == protocol['manifest_sha256'], 'Methods manifest changed')
    check((output/'methods_manifest.json').read_bytes() == Path(protocol['manifest_path']).read_bytes(),
          'Frozen methods manifest differs from source')
    template = (output/'judge_prompt.txt').read_text()
    check(sha(output/'judge_prompt.txt') == protocol['official_prompt_sha256'], 'Judge prompt differs')
    for name,expected in summary['artifact_sha256'].items():
        check(sha(output/name) == expected, f'Final summary artifact hash differs: {name}')
    canonical = unique(read_rows(Path(protocol['canonical_question_cache_root'])/'requests.jsonl'))
    check(len(canonical) == 1711, 'Canonical cohort is not 1711')
    methods = unique(protocol['methods'])
    check(set(methods) == set(EXPECTED_EM), 'Wrong eight DeepSeek methods')
    requests = read_rows(output/'requests.jsonl')
    request_map = unique(requests,'request_sha256')
    question_rows = read_rows(output/'question_map.jsonl')
    check(len(question_rows) == 8*1711, 'Wrong method-question count')
    uses = defaultdict(list)
    by_method = defaultdict(list)
    for row in question_rows:
        by_method[row['method_id']].append(row)
        wire = build_wire(row,template)
        request_hash = digest(wire)
        check(row['request_sha256'] == request_hash, 'Question wire hash differs')
        check(request_map[request_hash]['request'] == wire, 'Question maps to wrong wire request')
        uses[request_hash].append({'method_id':row['method_id'],'id':row['id']})
    for request_hash,row in request_map.items():
        check(build_wire(row,template) == row['request'], 'Frozen wire differs from raw triple')
        check(digest(row['request']) == request_hash, 'Frozen wire hash differs')
        check(row['uses'] == uses[request_hash] and uses[request_hash], 'Frozen request uses differ')
    source_reports = []
    source_empty = []
    for method_id,method in methods.items():
        check(method['answer_model'] == ANSWER_MODEL, 'Method answer model differs')
        record_path = Path(method['records_path'])
        check(sha(record_path) == method['records_sha256'], 'Frozen method input changed')
        originals = unique(read_rows(record_path))
        method_questions = unique(by_method[method_id])
        check(set(originals) == set(method_questions) == set(canonical), 'Wrong per-method IDs')
        # Independently trace every frozen prediction to the original producer files.
        response_paths = [Path(path) for path in method['source_sha256']
                          if path.endswith('/responses.jsonl') or path.endswith('/qa_responses.jsonl')]
        check(len(response_paths) == (3 if method_id == 'a_mem' else 1),
              'Unexpected original response source count')
        source_rows = []
        source_file_records = {}
        for response_path in response_paths:
            subset = read_rows(response_path)
            if method_id in ('semantic_rag','ours1_ours2'):
                subset = [row for row in subset if row.get('condition') == method['source_condition']]
            source_file_records[response_path] = unique(subset)
            source_rows.extend(subset)
        source_responses = unique(source_rows)
        check(set(source_responses) == set(canonical), 'Original response coverage differs')
        correct = 0
        empty_count = 0
        rejected_count = 0
        returned_models = Counter()
        for qid,row in method_questions.items():
            original,canonical_row,response = originals[qid],canonical[qid],source_responses[qid]
            check({key:value for key,value in row.items() if key not in ('method_id','request_sha256')}
                  == original, 'Prepared judge input changed original record')
            for key in ('task','point','question','ground_truth'):
                check(row[key] == canonical_row[key], 'Canonical question/gold/category changed')
            response_text = response['answer'] if method_id == 'hippomm' else response['raw_response']
            response_model = response.get('model') if method_id == 'hippomm' else response.get('returned_model')
            rejected = response.get('status') == 'provider_rejection'
            rejected_count += rejected
            returned_models[str(response_model)] += 1
            if method_id == 'clip_image_dialogue' and rejected:
                check(response_model is None and response_text == '',
                      'Original rejected CLIP answer unexpectedly changed')
                source_protocol = read_json(Path(method['source_root'])/'protocol.json')
                check(source_protocol['model'] == ANSWER_MODEL, 'CLIP target model differs in protocol')
            else:
                check(response_model == ANSWER_MODEL, 'Source response is not DeepSeek')
            check(row['model_output'] == response_text, 'Original DeepSeek answer was modified')
            if 'source_returned_model' in row:
                check(row['source_returned_model'] == response_model, 'Source returned-model metadata differs')
            if 'source_status' in row:
                check(row['source_status'] == response.get('status','ok'), 'Source status metadata differs')
            if method_id == 'a_mem':
                response_file = Path(row['source_response_file'])
                check(response_file in response_paths, 'A-Mem per-question provenance not declared')
                check(source_file_records[response_file][qid] == response, 'A-Mem source split differs')
            em = int(normalize(response_text) == normalize(row['ground_truth']))
            check(row['em'] == em, 'Source EM does not reproduce')
            correct += em
            if response_text == '':
                empty_count += 1
                source_empty.append({'method_id':method_id,'id':qid,
                                     'source_status':response.get('status'),
                                     'request_sha256':row['request_sha256']})
        check(correct == method['expected_em_correct'] == EXPECTED_EM[method_id], 'Wrong source EM count')
        check(method['em'] == correct/1711, 'Wrong source EM fraction')
        expected_rejected = {'clip_image_dialogue':13, 'hippomm':14}.get(method_id, 0)
        check(empty_count == rejected_count == expected_rejected, 'Original rejection/empty-answer counts differ')
        source_reports.append({'method_id':method_id,'questions':1711,'correct':correct,
                               'original_deepseek_strings_verified':1711,'empty_answers':empty_count,
                               'provider_rejections':rejected_count,'source_model_counts':dict(returned_models),
                               'response_sources':[str(path) for path in response_paths]})
    check(Counter(r['method_id'] for r in source_empty) == {'clip_image_dialogue':13,'hippomm':14},
          'The original 27 rejected empty answers were not preserved')
    amem = methods['a_mem']
    check(amem['source_variant']['original_prompt_baseline'] is False and
          amem['source_variant']['modified_memory_calls'] == 1,
          'A-Mem original one-call memory-prompt variant provenance missing')
    amem_summary = read_json(Path(amem['source_root'])/'combined_results.json')
    check(amem_summary['original_prompt_baseline'] is False and
          amem_summary['prior_results_reused'] == 1657 and amem_summary['retry_questions'] == 54,
          'A-Mem frozen composite provenance differs')

    # Reconstruct the ordered compatible-cache selection independently.
    available = defaultdict(list)
    for cache_info in protocol['compatible_cache_sources']:
        cache_root = Path(cache_info['root'])
        cache_protocol = read_json(cache_root/'protocol.json')
        cache_requests = read_rows(cache_root/'requests.jsonl')
        cache_request_map = {row['request_sha256']:row for row in cache_requests}
        for row in cache_requests:
            check(digest(row['request']) == row['request_sha256'] and
                  row['request'] == build_wire(row,template), 'Cache wire reconstruction differs')
        if 'methods' in cache_protocol:
            cache_scores = read_rows(cache_root/'reused_scores.jsonl') + read_rows(cache_root/'scores.jsonl')
            unique(cache_scores,'request_sha256')
            entries = [(row['request_sha256'],row) for row in cache_scores]
        else:
            cache_scores = unique(read_rows(cache_root/'scores.jsonl'))
            entries = [(row['request_sha256'],cache_scores[row['id']]) for row in cache_requests]
        for request_hash,response in entries:
            audit_score(response,request_hash)
            available[request_hash].append({'root':str(cache_root),'response':response,
                                            'request':cache_request_map[request_hash]['request']})
    reused_rows = read_rows(output/'reused_scores.jsonl')
    new_rows = read_rows(output/'scores.jsonl')
    reused = unique(reused_rows,'request_sha256')
    new = unique(new_rows,'request_sha256')
    check(set(reused) == set(request_map)&set(available), 'Compatible cache selection differs')
    check(not set(reused)&set(new) and set(reused)|set(new) == set(request_map),
          'Missing, duplicate, or unexpected scored requests')
    all_scores = {**reused,**new}
    cache_reasoning_variants = 0
    for request_hash,row in reused.items():
        audit_score(row,request_hash)
        check(row['score_origin'] == 'baseline_cache', 'Reused score origin differs')
        check(not set(row)&{'id','question','ground_truth','model_output','task','point','em',
                           'source_record_id','method_id'}, 'Cache can overwrite original DeepSeek input')
        entries = available[request_hash]
        first = entries[0]
        check(row['source_root'] == first['root'], 'First compatible cache was not selected')
        for key in ('score','reasoning','raw_response','response_id'):
            check(row[key] == first['response'][key], 'Chosen cache response differs')
        check(len(row['reuse_sources']) == len(entries), 'Cache provenance count differs')
        for provenance,entry in zip(row['reuse_sources'],entries):
            check(provenance['root'] == entry['root'] and
                  provenance['source_response'] == entry['response'], 'Full cache provenance differs')
            check(entry['request'] == request_map[request_hash]['request'],
                  'Compatible cache reused for a different question/gold/DeepSeek prediction')
            check(entry['response']['score'] == row['score'], 'Compatible cache grade conflict')
            cache_reasoning_variants += entry['response']['reasoning'] != row['reasoning']
    for request_hash,row in new.items():
        audit_score(row,request_hash)
        check(row['score_origin'] == 'new_api', 'New score origin differs')

    attempts = read_rows(output/'attempts.jsonl')
    attempts_by_hash = defaultdict(list)
    for attempt in attempts:
        request_hash = attempt['request_sha256']
        check(request_hash in new, 'API attempt made for unknown or already cached request')
        previous = attempts_by_hash[request_hash]
        check(attempt['attempt'] == len(previous)+1, 'Attempt order is not consecutive')
        check(not previous or previous[-1]['status'] != 'ok', 'New attempt after prior success')
        check(not previous or previous[-1].get('retryable') is True, 'Retried a nonretryable failure')
        if attempt['status'] == 'ok':
            grade = audit_score(attempt,request_hash,parsed_fields=False)
            for key,value in attempt.items():
                if key not in ('attempt','started_at_unix','status','retryable'):
                    check(new[request_hash].get(key) == value, 'Success audit differs from saved response')
            check(new[request_hash]['score'] == grade['score'] and
                  new[request_hash]['reasoning'] == grade['reasoning'], 'Success JSON score differs')
        else:
            check(attempt['status'] == 'error' and attempt.get('error_kind'), 'Unclassified API error')
        previous.append(attempt)
    check(set(attempts_by_hash) == set(new), 'New score lacks original API attempt')
    for request_hash,events in attempts_by_hash.items():
        check(events[-1]['status'] == 'ok' and sum(e['status']=='ok' for e in events)==1,
              'New request lacks exactly one successful final attempt')
    failed = [row for row in attempts if row['status'] != 'ok']
    counts_expected = {'method_count':8,'total_method_questions':13688,
                       'total_scored_method_questions':13688,'unique_requests':len(requests),
                       'unique_requests_scored':len(requests),'unique_requests_missing':0,
                       'baseline_reused_unique_requests':len(reused),
                       'new_successful_unique_requests':len(new),'api_attempt_count':len(attempts),
                       'failed_api_attempt_count':len(failed)}
    for key,value in counts_expected.items():
        check(summary.get(key) == value, f'Final overall count differs: {key}')
    for key in ('prompt_tokens','completion_tokens','total_tokens'):
        check(summary['new_successful_request_usage'][key] ==
              sum((row.get('usage') or {}).get(key,0) or 0 for row in new_rows), 'API usage differs')
    check(protocol['unique_requests'] == len(requests) and
          protocol['compatible_cache_reused_unique_requests'] == len(reused) and
          protocol['new_unique_requests_expected'] == len(new), 'Protocol request counts differ')
    method_reports = unique(summary['methods'],'method_id')
    check(set(method_reports) == set(methods), 'Final summary methods differ')
    results = []
    for method_id,method in methods.items():
        report = method_reports[method_id]
        method_dir = output/'methods'/method_id
        check(read_json(method_dir/'summary.json') == report, 'Per-method summary differs')
        per_questions = unique(read_rows(method_dir/'per_question.jsonl'))
        inputs = unique(by_method[method_id])
        check(set(per_questions) == set(inputs), 'Per-method output coverage differs')
        scores = []
        for qid,source in inputs.items():
            score = all_scores[source['request_sha256']]
            expected = {**source,**score,'judge_status':'ok'}
            check(per_questions[qid] == expected, 'Per-question report differs from source/score mapping')
            scores.append(expected)
        total = sum(row['score'] for row in scores)
        distribution = Counter(row['score'] for row in scores)
        expected = {'status':'complete','questions_expected':1711,'questions_scored':1711,
                    'questions_missing':0,'source_em_correct':EXPECTED_EM[method_id],
                    'source_em':EXPECTED_EM[method_id]/1711,'score_sum':total,
                    'llm_judge':total/1711,'llm_judge_percent':100*total/1711,
                    'score_distribution':{str(g):distribution[g] for g in GRADES},'missing_ids':[],
                    'score_origin_question_counts':dict(Counter(row['score_origin'] for row in scores))}
        for key,value in expected.items():
            check(report.get(key) == value, f'Method aggregate differs: {method_id}/{key}')
        for field,output_key in [('point','by_question_type'),('task','by_scene'),('em','by_source_em')]:
            subsets = defaultdict(list)
            for row in scores:subsets[str(row[field])].append(row['score'])
            expected_groups = {key:{'questions_expected':len(values),'questions_scored':len(values),
                                    'mean_score':sum(values)/len(values)} for key,values in subsets.items()}
            check(report[output_key] == expected_groups, f'Method grouped aggregates differ: {output_key}')
        with (method_dir/'per_question.csv').open(encoding='utf-8-sig',newline='') as handle:
            csv_rows = list(csv.DictReader(handle))
        csv_by_id = unique(csv_rows)
        check(set(csv_by_id) == set(inputs), 'Per-question CSV coverage differs')
        for qid,row in csv_by_id.items():
            full = per_questions[qid]
            for key in ('question','ground_truth','model_output','reasoning','request_sha256'):
                check(row[key] == full[key], 'CSV altered source text or judge reasoning')
            check(float(row['score']) == full['score'], 'CSV score differs')
        results.append({'method_id':method_id,'em_correct':EXPECTED_EM[method_id],
                        'questions_scored':1711,'llm_judge':total/1711,
                        'score_sum':total,'score_distribution':expected['score_distribution']})
    empty_answer_report = []
    for item in source_empty:
        score = all_scores[item['request_sha256']]
        empty_answer_report.append({**item,'model_output':'','score':score['score'],
                                    'reasoning':score['reasoning'],'response_id':score['response_id']})
    final = {'passed':True,'status':'complete','validated_at_utc':datetime.now(timezone.utc).isoformat(),
             'answer_model':ANSWER_MODEL,'judge_model':JUDGE_MODEL,'grades':list(GRADES),
             'method_count':8,'questions_per_method':1711,'original_deepseek_inputs_verified':13688,
             'unique_wire_requests_verified':len(requests),'compatible_reused_scores_verified':len(reused),
             'new_api_scores_verified':len(new),'api_attempt_count':len(attempts),
             'failed_api_attempt_count':len(failed),
             'failed_api_attempt_kinds':dict(Counter(row['error_kind'] for row in failed)),
             'retried_unique_requests':sum(len(v)>1 for v in attempts_by_hash.values()),
             'maximum_total_attempts_per_request':max(map(len,attempts_by_hash.values()),default=0),
             'unresolved_api_errors':0,'hash_mapping_counts':hash_counts,
             'distinct_source_cache_implementation_files_rehashed':len(checked_hashes),
             'first_compatible_cache_selection_verified':True,
             'cached_question_gold_and_deepseek_prediction_exactly_match_entire_wire':True,
             'cache_alternative_reasonings_same_grade':cache_reasoning_variants,
             'all_original_deepseek_answers_preserved':True,'all_raw_five_grade_scores_reparsed':True,
             'original_provider_rejection_empty_answers':empty_answer_report,
             'a_mem_original_prompt_baseline':False,'a_mem_authorized_modified_memory_calls':1,'methods':results,
             'source_prediction_audit':source_reports,
             'summary_sha256':sha(output/'summary.json'),'protocol_sha256':sha(output/'protocol.json'),
             'audit_script':str(Path(__file__).resolve()),'audit_script_sha256':sha(__file__)}
    destination = output/'validation_final.json'
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(final,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(destination)
    print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
