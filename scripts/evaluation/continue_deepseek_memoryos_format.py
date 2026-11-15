"""Lossless normalization of a single-object MemoryOS summary response.

DeepSeek returned {summary: text, keywords: [...]} where upstream expects a list
of {content: text, keywords: [...]}. Preserve the exact generated strings; change
only the container/key shape. Valid upstream lists are returned unchanged.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.vendor'),str(ROOT)]
import json
from concurrent.futures import ThreadPoolExecutor
from scripts.evaluation import memeye_deepseek_text_methods as experiment
from scripts.evaluation.continue_deepseek_text_methods_parallel import ParallelAPI
from scripts.evaluation.continue_deepseek_text_methods_streaming import StreamingBroker


def normalize_summaries(value):
    if isinstance(value,list) and all(isinstance(x,dict) for x in value):return value,False
    if isinstance(value,dict) and set(value)<= {'summary','content','keywords','theme'}:
        content=value.get('content',value.get('summary'))
        if isinstance(content,str) and isinstance(value.get('keywords',[]),list):
            item={k:v for k,v in value.items() if k!='summary'}
            item['content']=content
            return [item],True
    # Match upstream's existing fallback on invalid JSON; never invent content.
    return [],True


def main():
    root=experiment.OUTPUT/'memoryos'
    experiment.verify(root)
    amendment=dict(kind='lossless_structured_summary_normalization',
        mapping='single {summary:text,keywords:list} -> [{content:same_text,keywords:same_list}]',
        valid_lists_unchanged=True,api_requests_and_final_reader_unchanged=True,
        generated_content_not_rewritten=True,completed_domains_retained=True,
        resumed_domain='Cartoon_Entertainment_Companion',
        script_sha256=experiment.sha(__file__),base_protocol_sha256=experiment.sha(root/'protocol.json'),
        streaming_adapter_sha256=experiment.sha(ROOT/'scripts/evaluation/continue_deepseek_text_methods_streaming.py'))
    experiment.freeze(root/'memoryos_format_amendment.json',amendment)
    original=experiment.base.install_runtime
    def install(broker,protocol):
        encoder=original(broker,protocol)
        from memoryos_core import utils,updater
        generate=utils.gpt_generate_multi_summary
        def normalized(*args,**kwargs):
            result=generate(*args,**kwargs)
            fixed,changed=normalize_summaries(result['summaries'])
            if changed:
                broker.model.log('summary_normalization_events.jsonl',dict(
                    original=result['summaries'],normalized=fixed,input_sha256=experiment.base.hash_object(result['input'])))
                result=dict(result,summaries=fixed)
            return result
        utils.gpt_generate_multi_summary=normalized
        updater.gpt_generate_multi_summary=normalized
        return encoder
    experiment.base.install_runtime=install
    experiment.ThreadPoolExecutor=lambda max_workers:ThreadPoolExecutor(max_workers=8)
    experiment.APIModel=ParallelAPI
    experiment.Broker=StreamingBroker
    if (root/'run_errors.json').exists():
        (root/'run_errors.json').replace(root/'run_errors_before_format_adapter.json')
    experiment.run(experiment.OUTPUT,'memoryos',task='Cartoon_Entertainment_Companion')


if __name__=='__main__':main()
