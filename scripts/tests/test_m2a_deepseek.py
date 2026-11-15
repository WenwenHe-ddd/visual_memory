import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/'.vendor_m2a'),str(ROOT/'.vendor')]
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from scripts.memory import m2a_deepseek as adapter


class DeepSeekTests(unittest.TestCase):
    def test_original_pixels_and_native_tool_history(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'misnamed.jpg'
            Image.new('RGB',(4,4),'red').save(p,format='PNG')
            messages = [HumanMessage('Find it'), AIMessage(content='',tool_calls=[
                dict(name='fetch',args={'id':1},id='c1',type='tool_call')]),
                ToolMessage(content=[{'type':'text','text':'<image0>'},
                    {'type':'image','url':str(p)}],tool_call_id='c1')]
            schema = [{'type':'function','function':{'name':'fetch','parameters':{'type':'object','properties':{}}}}]
            wire, images = adapter.build_request(messages,schema,1200)
            self.assertEqual(wire['messages'][1]['tool_calls'][0]['function']['arguments'],'{"id": 1}')
            self.assertEqual(wire['messages'][2]['role'],'tool')
            self.assertEqual(wire['messages'][2]['tool_call_id'],'c1')
            self.assertIn('Tool image attachment 1',wire['messages'][2]['content'])
            self.assertEqual(wire['messages'][3]['role'],'user')
            data = wire['messages'][3]['content'][-1]['image_url']['url']
            self.assertTrue(data.startswith('data:image/png;base64,'))
            self.assertEqual(base64.b64decode(data.split(',',1)[1]),p.read_bytes())
            self.assertEqual(images[0]['bytes_sha256'],hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertNotIn(str(p),json.dumps(wire))
            self.assertFalse(wire['parallel_tool_calls'])
            self.assertFalse(wire['enable_thinking'])
            self.assertEqual(wire['seed'],42)

    def test_replay_uses_exact_request_and_logs_logical_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            def reply(config,wire):
                return dict(raw_response='{"facts":"x","image_caption":""}',tool_calls=[],
                    requested_model=adapter.MODEL,returned_model=adapter.MODEL,response_id='response1',
                    finish_reason='stop',refusal=None,input_tokens=12,output_tokens=10,
                    hit_token_limit=False,seed=42,enable_thinking=False,
                    request_sha256=adapter.provider.request_sha256(wire))
            with patch.object(adapter.provider,'load_config',return_value=None),patch.object(adapter,'chat',side_effect=reply) as chat:
                model = adapter.DeepSeek(None,Path(tmp))
                model.stage='construction';model.sample_id='round1'
                messages=[SystemMessage('Extract'),HumanMessage('User likes red')]
                first=model.invoke(messages);second=model.invoke(messages)
                self.assertEqual(first.content,second.content)
                self.assertEqual(chat.call_count,1)
                records=[json.loads(x) for x in (Path(tmp)/'llm_calls.jsonl').read_text().splitlines()]
                self.assertEqual([x['cache_replayed'] for x in records],[False,True])
                self.assertEqual([x['call'] for x in records],[1,2])
                model.invoke([SystemMessage('Extract'),HumanMessage('User likes blue')])
                self.assertEqual(chat.call_count,2)

    def test_length_prompt_does_not_mutate_input_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(adapter.provider,'load_config',return_value=None),patch.object(adapter,'chat') as chat:
                model=adapter.DeepSeek(None,Path(tmp),max_new_tokens=16)
                model.stage='reader'
                chat.return_value=dict(raw_response='A',tool_calls=[],returned_model=adapter.MODEL,
                    response_id='r',finish_reason='stop',refusal=None,hit_token_limit=False)
                messages=[SystemMessage('Pick an answer'),HumanMessage('Question')]
                model.invoke(messages)
                self.assertEqual(messages[0].content,'Pick an answer')
                self.assertIn(adapter.MCQ_LIMIT,chat.call_args.args[1]['messages'][0]['content'])

    def test_native_response_and_model_identity_validation(self):
        config=adapter.provider.APIConfig(model=adapter.MODEL,base_url=adapter.provider.DASHSCOPE_BASE_URL,
            api_key='FAKE',credential_variable='DEEPSEEKV4_API_KEY')
        schema=[{'type':'function','function':{'name':'search','parameters':{'type':'object','properties':{}}}}]
        wire,_=adapter.build_request([HumanMessage('Find the memory')],schema,1200)
        for model in (adapter.MODEL,'unexpected-model'):
            payload=dict(id='response1',model=model,choices=[dict(finish_reason='tool_calls',message=dict(
                content=None,tool_calls=[dict(id='call1',type='function',function=dict(name='search',arguments='{}'))]))],usage={})
            opener=MagicMock();response=opener.open.return_value.__enter__.return_value
            response.read.return_value=json.dumps(payload).encode();response.headers.get.return_value='request1'
            with patch('urllib.request.build_opener',return_value=opener):
                if model==adapter.MODEL:
                    result=adapter.chat(config,wire)
                    self.assertEqual(result['tool_calls'][0]['name'],'search')
                    self.assertEqual(result['finish_reason'],'tool_calls')
                else:
                    with self.assertRaises(adapter.provider.APIError):adapter.chat(config,wire)

    def test_truncated_response_is_logged_and_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(adapter.provider,'load_config',return_value=None),patch.object(adapter,'chat') as chat:
                chat.return_value=dict(raw_response='partial',tool_calls=[],returned_model=adapter.MODEL,
                    response_id='r',finish_reason='length',refusal=None,hit_token_limit=True)
                model=adapter.DeepSeek(None,Path(tmp));model.stage='construction'
                with self.assertRaises(ValueError):model.invoke([HumanMessage('Extract')])
                record=json.loads((Path(tmp)/'llm_calls.jsonl').read_text())
                self.assertEqual(record['status'],'error')
                self.assertTrue(record['hit_token_limit'])


if __name__=='__main__':unittest.main()
