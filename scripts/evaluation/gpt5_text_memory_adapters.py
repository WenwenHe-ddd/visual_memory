"""Reuse the formal DeepSeek run's format rules and independent scheduling.

Only generic broker/normalization helpers are used; the supplied APIModel is
the official GPT-5 transport. No DeepSeek credentials or requests are invoked.
"""
from scripts.evaluation.continue_deepseek_text_methods_streaming import StreamingBroker, StreamingFormatBroker
from scripts.evaluation.continue_deepseek_memoryos_format import normalize_summaries
from scripts.memory import text_memory_baselines as base

def make_broker(method,model,root,protocol):
    cls=StreamingFormatBroker if method=='gen_agents' else StreamingBroker
    return cls(model,root,protocol,8)

def install_runtime(method,broker,protocol):
    encoder=base.install_runtime(broker,protocol)
    if method=='memoryos':
        from memoryos_core import utils,updater
        generate=utils.gpt_generate_multi_summary
        def normalized(*args,**kwargs):
            result=generate(*args,**kwargs)
            fixed,changed=normalize_summaries(result['summaries'])
            if changed:
                broker.model.log('summary_normalization_events.jsonl',dict(
                    original=result['summaries'],normalized=fixed,input_sha256=base.hash_object(result['input'])))
                result=dict(result,summaries=fixed)
            return result
        utils.gpt_generate_multi_summary=normalized
        updater.gpt_generate_multi_summary=normalized
    return encoder
