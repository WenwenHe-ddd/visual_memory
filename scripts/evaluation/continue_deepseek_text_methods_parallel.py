"""Explicit scheduling-only amendment: eight independent domains per method.

Original signed algorithms, API bodies and same-run request caches are retained.
The amendment overrides only execution concurrency, not the experiment inputs.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.vendor'),str(ROOT)]
from concurrent.futures import ThreadPoolExecutor
from scripts.evaluation import memeye_deepseek_text_methods as experiment
from scripts.evaluation.deepseek_text_memory_transport import APIModel
from scripts.memory.text_memory_transport import Broker


class ParallelAPI(APIModel):
    def __init__(self,root,env_file,workers=2):
        super().__init__(root,env_file,workers=8)


class ParallelBroker(Broker):
    def __init__(self,model,root,protocol,batch_size=4):
        super().__init__(model,root,protocol,batch_size=8)


def main():
    method=sys.argv[1]
    assert method in experiment.NAMES
    root=experiment.OUTPUT/method
    protocol=experiment.verify(root)
    amendment=dict(kind='execution_concurrency_only',
        base_protocol_sha256=experiment.sha(root/'protocol.json'),
        base_experiment_signature=protocol['experiment_signature'],
        task_workers=8,api_workers=8,generation_batch_size=8,
        supersedes_runtime_fields=['task_workers','api_workers','generation_batch_size'],
        model_seed_temperature_prompts_budgets_retrieval_unchanged=True,
        same_run_cached_exact_requests_retained=True,
        script_sha256=experiment.sha(__file__))
    experiment.freeze(root/'execution_amendment.json',amendment)
    experiment.ThreadPoolExecutor=lambda max_workers: ThreadPoolExecutor(max_workers=8)
    experiment.APIModel=ParallelAPI
    experiment.Broker=ParallelBroker
    experiment.run(experiment.OUTPUT,method)


if __name__=='__main__':main()
