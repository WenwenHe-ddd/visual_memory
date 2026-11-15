import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock,patch
from scripts.evaluation import gallery_deepseek_transport as api


class GalleryDeepSeekTransportTests(unittest.TestCase):
    def test_original_pixels_and_provider_settings(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            for fmt in ('JPEG','PNG','WEBP','GIF'):
                p=Path(tmp)/'misnamed.jpg';Image.new('RGB',(3,3),'red').save(p,format=fmt)
                data=p.read_bytes()
                payload=dict(record_id='not_sent',user_text='Caption.',query_images=[dict(path=str(p),sha256=hashlib.sha256(data).hexdigest())])
                wire=api.wire_for(payload)
                part=wire['messages'][-1]['content'][0]['image_url']
                self.assertEqual(base64.b64decode(part['url'].split(',',1)[1]),data)
                self.assertEqual(wire['model'],'deepseek-v4.1-flash')
                self.assertEqual(wire['seed'],42);self.assertEqual(wire['temperature'],0)
                self.assertIs(wire['enable_thinking'],False)
                self.assertNotIn('not_sent',json.dumps(wire))
                with self.assertRaises(ValueError):api.wire_for(dict(payload,answer='gold'))

    def test_http_route_and_thinking_guard(self):
        config=api.provider.APIConfig(model=api.MODEL,base_url=api.provider.DASHSCOPE_BASE_URL,api_key='FAKE',credential_variable='DEEPSEEKV4_API_KEY')
        wire=api.wire_for(dict(user_text='Question'))
        for reasoning in ('', 'unexpected thoughts'):
            data=dict(id='response',model=api.MODEL,choices=[dict(finish_reason='stop',message=dict(content='Answer',reasoning_content=reasoning))],usage={})
            opener=MagicMock();response=opener.open.return_value.__enter__.return_value
            response.read.return_value=json.dumps(data).encode();response.headers.get.return_value='request'
            with patch('urllib.request.build_opener',return_value=opener):
                if reasoning:
                    with self.assertRaises(api.provider.APIError):api.chat(config,wire)
                else:
                    result=api.chat(config,wire)
                    self.assertEqual(result['request_sha256'],api.request_sha256(wire))
                    self.assertEqual(opener.open.call_args.args[0].full_url,api.ENDPOINT)


if __name__=='__main__':unittest.main()
