import json
import unittest

from scripts.analysis.simplemem_source_attribution import messages, parse_and_validate
from scripts.analysis.simplemem_attribution_recall import duplicate_evidence_sources


class AttributionEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.job = {'id': 'task_window_group', 'task': 'private_task_label', 'sources': [
            {'alias': 'R1', 'kind': 'round', 'round_id': 'original-round', 'timestamp': '2024-01-01',
             'user': 'The dinosaur has blue eyes and a red hat.', 'assistant': '', 'caption': ''}],
            'targets': [{'alias': 'M1', 'entry_id': 'random-memory-uuid', 'memory': 'The dinosaur has blue eyes.',
                         'selected': True}]}
        self.value = [{'id': 'M1', 'status': 'resolved', 'evidence': [{'source': 'R1', 'field': 'user',
            'quote': 'The dinosaur has blue eyes'}], 'ambiguous_sources': [], 'note': ''}]

    def test_literal_evidence_accepted(self):
        result, issues = parse_and_validate(json.dumps(self.value), self.job)
        self.assertFalse(issues)
        self.assertEqual(result['M1']['evidence'][0]['source'], 'R1')

    def test_fabricated_evidence_rejected(self):
        self.value[0]['evidence'][0]['quote'] = 'The dinosaur has green eyes'
        _, issues = parse_and_validate(json.dumps(self.value), self.job)
        self.assertTrue(issues['M1'])

    def test_nonexistent_source_rejected(self):
        self.value[0]['evidence'][0]['source'] = 'R2'
        _, issues = parse_and_validate(json.dumps(self.value), self.job)
        self.assertTrue(issues['M1'])

    def test_missing_target_rejected(self):
        with self.assertRaises(ValueError):
            parse_and_validate('[]', self.job)

    def test_ambiguity_cannot_be_resolved(self):
        self.value[0]['ambiguous_sources'] = ['R1']
        _, issues = parse_and_validate(json.dumps(self.value), self.job)
        self.assertTrue(issues['M1'])

    def test_prompt_has_no_selection_or_benchmark_metadata(self):
        for pass_index in (1, 2):
            payload = messages(self.job, pass_index)
            content = json.loads(payload[1]['content'])
            self.assertEqual(set(content['target_memories'][0]), {'id', 'statement'})
            self.assertNotIn('round_id', content['original_sources'][0])
            text = json.dumps(payload)
            self.assertNotIn('private_task_label', text)
            self.assertNotIn('random-memory-uuid', text)
            self.assertNotIn('gold_ids', text)

    def test_repeated_quote_does_not_identify_unique_round(self):
        second = dict(self.job['sources'][0], alias='R2', round_id='second-round')
        job = dict(self.job, sources=[*self.job['sources'], second])
        self.assertEqual(duplicate_evidence_sources(self.value[0], job), {'R1': ['R1', 'R2']})

    def test_two_quotes_can_jointly_disambiguate_round(self):
        self.job['sources'][0]['assistant'] = 'This scene takes place beside the waterfall.'
        second = dict(self.job['sources'][0], alias='R2', round_id='second-round', assistant='Different scene.')
        job = dict(self.job, sources=[*self.job['sources'], second])
        self.value[0]['evidence'].append({'source': 'R1', 'field': 'assistant',
            'quote': 'This scene takes place beside the waterfall.'})
        self.assertEqual(duplicate_evidence_sources(self.value[0], job), {})


if __name__ == '__main__':
    unittest.main()
