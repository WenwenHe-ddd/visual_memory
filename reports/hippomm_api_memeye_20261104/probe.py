import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.evaluation import openai_gpt5_base_transport as gpt,deepseek_flash_seed42_audited_transport as deep
from scripts.memory import hippomm_core as core, hippomm_inputs as inputs
from scripts.evaluation.mem_gallery_master_reader import READER_SYSTEM
name=sys.argv[1];api=gpt if name=='gpt5' else deep
source=ROOT/('results/memeye_gpt5_seed42_text_methods_20261025/caption_source' if name=='gpt5' else 'results/memeye_deepseek_flash_seed42_full_20261025')
root=ROOT/'results/hippomm_original_controlled_20261104/memeye/offline'
e=inputs.read_lines(root/'events.jsonl')[0]
refs=inputs.unique(inputs.read_lines(root/'representative_images.jsonl'),'event_id')
h=inputs.sha(root/refs[e['event_id']]['file'])
cap=inputs.unique(inputs.read_lines(source/'captions.jsonl'))[h]
e.update(caption=cap['caption'],summary='')
prompt=core.summary_prompt(e,inputs.unique(inputs.read_lines(root/'dialogues.jsonl')))
payload={'system':READER_SYSTEM,'user_text':prompt,'query_images':[]}
request=api.build_request(payload,**({'seed':42} if name=='gpt5' else {'model':deep.DEEPSEEK_MODEL}),max_output_tokens=2048,image_detail='high')
response=api.chat_request(api.load_config('/storage/hewenwen/NTU Decentralized Agent/.env'),request,timeout=60)
response={k:v for k,v in response.items() if k not in ('request_sha256','system_fingerprint')}
Path(__file__).with_name(name+'_probe.json').write_text(json.dumps({'role':'summary','prompt':prompt,'image_paths':[],'response':response},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'backbone':name,'returned_model':response['returned_model'],'finish_reason':response['finish_reason'],'usage':response['usage']}))
