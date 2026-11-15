"""Keep the original MemEye GPT CUDA encoder boundary for seed stability."""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import subprocess
import sys
import types
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.evaluation import required_grid_common as c
from scripts.evaluation.required_grid_memeye_qwen_caption import gpu
SOURCE=c.ROOT/'scripts/evaluation/required_grid_memeye_gpt_caption.py'

def runtime(device):
    tree=ast.parse(SOURCE.read_text());counts={'cpu':0,'caption_methods':0}
    class Boundary(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in counts:
                counts[node.value]+=1
                return ast.copy_location(ast.Constant(device if node.value=='cpu' else 'caption_methods_cuda'),node)
            return node
    tree=ast.fix_missing_locations(Boundary().visit(tree))
    if counts!={'cpu':4,'caption_methods':2}:raise ValueError('Reviewed CUDA encoder/root boundary changed: '+str(counts))
    module=types.ModuleType('grid_gpt_cuda_caption');module.__file__=str(SOURCE);module.__package__='scripts.evaluation'
    exec(compile(tree,str(SOURCE),'exec'),module.__dict__)
    return module

def run(job,path):
    if job['dataset']!='memeye' or job['backbone']!='gpt5':raise ValueError('Wrong GPT CUDA handler')
    c.job_protocol(job,[Path(__file__),SOURCE,c.ROOT/'scripts/evaluation/required_grid_memeye_gpt_common.py'])
    with gpu(job) as device:
        c.freeze(Path(job['method_root'])/'runtime_device.json',dict(embedding_device=device,original_device_policy='CUDA float32 CLIP/BGE; original numerical recipe'))
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage','memory-retrieval','--device',device],check=True,cwd=c.ROOT)
    for phase in ('mcq','open-answers','judge','audit'):
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--job',str(path),'--stage',phase,'--device',device],check=True,cwd=c.ROOT)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--stage');p.add_argument('--device',default='cuda:1');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();j=c.read(a.job)
    if a.prepare_only:
        value=runtime(a.device);value.prepare(j);value.runtime(j);print(dict(status='prepared',id=j['id'],model_generations=0,device=a.device))
    elif a.stage:runtime(a.device).stage(j,a.stage)
    else:
        with c.lock(Path(j['method_root'])/'execution.lock',False):run(j,a.job)
