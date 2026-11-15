"""Explicit, user-authorized missing-memory variant for one rejected round."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation import m2a_deepseek_recovery_run as entry
from scripts.evaluation import m2a_inputs
from scripts.evaluation import audit_memeye_m2a as standard_audit

TASK = 'Global_Travel_Sustainable_Fashion_Design'
OMITTED_ID = TASK+'/D6:11'
PARENT = ROOT/'results/m2a_single_pass_deepseek_20261114_recovery'
REJECTED = PARENT/'memgallery'/TASK/'attempt_003'
REJECTED_REQUEST = 'de65917a1d59e712f4bbe9ba74b84a289821d70c77fdaa3ec934cf4f6ec1b8a3'
VARIANT = 'DeepSeek M2A single-pass with one provider-rejected memory round omitted'
base = entry.baseline.single.base


def omission_record(observations):
    rejected = [o for o in observations if o['id'] == OMITTED_ID]
    if len(rejected) != 1:
        raise ValueError('Expected exactly one explicitly authorized omitted round')
    history = standard_audit.rows(REJECTED/'api_attempts.jsonl')
    match = [x for x in history if x['sample_id'] == OMITTED_ID
             and x.get('provider_error_code') == 'data_inspection_failed'
             and x['request_sha256'] == REJECTED_REQUEST]
    if len(match) != 1:
        raise ValueError('Missing historical rejection evidence')
    original = next(o for o in standard_audit.read(REJECTED/'observations.json') if o['id'] == OMITTED_ID)
    if original != rejected[0]:
        raise ValueError('Rejected observation changed')
    return dict(id=OMITTED_ID, reason='provider_data_inspection_failed',
                observation=rejected[0], rejection_request_sha256=REJECTED_REQUEST,
                rejection_log=str(REJECTED/'api_attempts.jsonl'),
                rejection_log_sha256=base.sha(REJECTED/'api_attempts.jsonl'),
                action='Omit entire round from raw, semantic and image memory; preserve original source and every QA')


def run(args):
    if args.dataset != 'memgallery' or args.task != TASK or args.limit_rounds or args.limit_questions or args.question_id:
        raise ValueError('Missing-memory variant must evaluate all 78 questions of the authorized task')
    original_prepare, original_hook = base.prepare, entry.protocol_hook

    def prepare(args, patches):
        full, questions, protocol = original_prepare(args, patches)
        omitted = omission_record(full)
        used = [o for o in full if o['id'] != OMITTED_ID]
        base.write(args.output/'original_observations.json', full)
        base.write(args.output/'memory_omissions.json', [omitted])
        base.write(args.output/'observations.json', used)
        protocol.update(rounds=len(used), original_rounds=len(full),
                        image_rounds=sum(bool(o['image']) for o in used),
                        memory_coverage_complete=False, question_coverage_complete=True,
                        missing_memory_ids=[OMITTED_ID], evaluation_variant=VARIANT)
        protocol['prepared_sha256'].update({n:base.sha(args.output/n) for n in
            ['observations.json','original_observations.json','memory_omissions.json']})
        protocol['sources'][omitted['rejection_log']] = omitted['rejection_log_sha256']
        return used, questions, protocol

    def hook(args, protocol):
        original_hook(args, protocol)
        protocol['method'] = VARIANT
        protocol['memory_input'] += '; omit only explicitly listed provider-rejected round from all memory stores'
        path = Path(__file__)
        name = str(path.relative_to(ROOT))
        protocol['implementation_sha256'][name] = base.sha(path)
        target = args.output/'code_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())

    base.prepare, entry.protocol_hook = prepare, hook
    try:
        return entry.run(args)
    finally:
        base.prepare, entry.protocol_hook = original_prepare, original_hook


def audit(out):
    """Audit the ordinary pipeline plus full source/QA and declared omission."""
    result = standard_audit.audit(out)
    full, questions, _ = m2a_inputs.load_task('memgallery', TASK)
    expected = [o for o in full if o['id'] != OMITTED_ID]
    protocol = standard_audit.read(out/'protocol.json')
    checks = {
        'full_original_history_retained': standard_audit.read(out/'original_observations.json') == full,
        'only_authorized_round_removed': standard_audit.read(out/'observations.json') == expected,
        'all_78_questions_unchanged': standard_audit.read(out/'evaluation_labels.json') == questions and len(questions) == 78,
        'omission_matches_rejection': standard_audit.read(out/'memory_omissions.json') == [omission_record(full)],
        'variant_disclosed': protocol.get('evaluation_variant') == VARIANT and
            protocol.get('missing_memory_ids') == [OMITTED_ID] and protocol.get('memory_coverage_complete') is False,
    }
    rejected = next(o for o in full if o['id'] == OMITTED_ID)
    raw = standard_audit.read(out/'raw_before_qa.json')
    semantic = standard_audit.read(out/'semantic_before_qa.json')
    checks['omission_absent_from_raw_and_semantic_memory'] = all(
        r['text'] != rejected['text'] and r['image_path'] != rejected['image'] for r in raw) and all(
        OMITTED_ID not in m['text'] and m['image_path'] != rejected['image'] for m in semantic)
    checks['omitted_image_not_registered'] = rejected['image'] not in standard_audit.read(out/'image_manager.json')['image_paths']
    checks['omitted_round_not_retrieved'] = all(OMITTED_ID not in a['retrieved_source_ids']
        for a in standard_audit.rows(out/'responses.jsonl'))
    calls = standard_audit.rows(out/'llm_calls.jsonl')
    checks['omitted_round_not_sent'] = all(c['sample_id'] != OMITTED_ID and
        all(i.get('path') != rejected['image'] for i in c.get('images', [])) for c in calls)
    result['errors'] += [k for k,v in checks.items() if not v]
    result.update(passed=not result['errors'], variant=VARIANT, memory_coverage_complete=False,
                  original_rounds=len(full), available_memory_rounds=len(expected), missing_memory_ids=[OMITTED_ID],
                  variant_checks=checks, original_complete_memory_protocol_passed=False)
    base.write(out/'memory_gap_audit.json', result)
    return result


if __name__ == '__main__':
    base.run = run
    base.main()
