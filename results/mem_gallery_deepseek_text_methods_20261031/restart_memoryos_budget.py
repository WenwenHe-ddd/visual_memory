"""Archive the capped attempt and rebuild MemoryOS with an explicit 8192 cap."""
from datetime import datetime,timezone
import importlib.util
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.evaluation import mem_gallery_deepseek_text_methods as run


def prepare():
    base=Path(__file__).resolve().parent
    output=base/'memoryos';archive=base/'attempts/memoryos_cap2048'
    if archive.exists():
        p=run.verify(output);assert p['memory_output_floor']==8192;return
    p=run.verify(output)
    assert run.read(output/'execution.json')['status']=='failed'
    record=dict(reason='A native long-term user profile completion reached the 2048-token limit; no truncated state is retained in the replacement run.',
        previous_signature=p['experiment_signature'],previous_protocol_sha256=run.sha(output/'protocol.json'),
        old_memory_completion_floor=2048,new_memory_completion_floor=8192,
        reader_limit_unchanged=2048,all_original_inputs_and_native_retrieval_unchanged=True,
        prior_memory_or_answer_responses_reused=False,created=datetime.now(timezone.utc).isoformat(),
        previous_artifacts_sha256={str(f.relative_to(output)):run.sha(f) for f in output.rglob('*') if f.is_file()},
        script_sha256=run.sha(__file__))
    archive.parent.mkdir(parents=True,exist_ok=True);output.rename(archive)
    output.mkdir()
    for name in ('observations.json','inputs','source_snapshot'):
        src=archive/name
        if src.is_dir():shutil.copytree(src,output/name)
        else:shutil.copyfile(src,output/name)
    revision=base/'memoryos_budget_revision.json';run.freeze(revision,record)
    p.pop('experiment_signature');p['memory_output_floor']=8192
    p['memory_sampling']='Native temperatures and seed42/thinking=false; minimum completion budget8192 for memory construction after archived 2048-token truncation. Reader cap2048 unchanged. GPT-5 control used memory floor2048.'
    p['budget_revision']=str(revision)
    p['source_sha256'][str(revision)]=run.sha(revision)
    p['source_sha256'][str(Path(__file__).resolve())]=run.sha(__file__)
    p['experiment_signature']=run.obj(p);run.freeze(output/'protocol.json',p)
    run.verify(output)


if __name__=='__main__':
    prepare()
    spec=importlib.util.spec_from_file_location('gallery_driver',Path(__file__).with_name('run_all.py'))
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    sys.exit(0 if driver.run('memoryos') else 1)
