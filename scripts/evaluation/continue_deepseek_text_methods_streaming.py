"""Scheduling-only continuation removing cross-domain RPC batch barriers."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'.vendor'),str(ROOT)]
from concurrent.futures import ThreadPoolExecutor
import json
import queue
from scripts.evaluation import memeye_deepseek_text_methods as experiment
from scripts.evaluation.continue_deepseek_text_methods_parallel import ParallelAPI, ParallelBroker
from scripts.evaluation.continue_deepseek_ga_format import FormatBroker


class StreamingBroker(ParallelBroker):
    def _loop(self):
        def one(req,future):
            try:
                out=self.model.one(req['messages'],req['max_new_tokens'],req['temperature'])
                if req['response_format']:
                    try:
                        value=json.loads(out['raw_response'].strip())
                        schema=req['response_format'].get('json_schema',{}).get('schema')
                        if schema:
                            import jsonschema
                            jsonschema.validate(value,schema)
                        out['structured_output_valid']=True
                    except Exception as exc:
                        out['structured_output_valid']=False
                        out['structured_output_error']=str(exc)[:400]
                future.set_result(dict(request=req,output=out))
            except BaseException as exc:future.set_exception(exc)
        with ThreadPoolExecutor(max_workers=8) as pool:
            while not self.closed.is_set() or not self.pending.empty():
                try:req,future=self.pending.get(timeout=0.1)
                except queue.Empty:continue
                pool.submit(one,req,future)


class StreamingFormatBroker(FormatBroker):
    _loop=StreamingBroker._loop


def main():
    method=sys.argv[1];assert method in experiment.NAMES
    root=experiment.OUTPUT/method
    p=experiment.verify(root)
    amendment=dict(kind='independent_request_scheduling',task_workers=8,api_workers=8,
        earlier_semantic_configuration_unchanged=True,removes_batch_completion_barrier=True,
        exact_same_run_request_cache_retained=True,
        script_sha256=experiment.sha(__file__),base_protocol_sha256=experiment.sha(root/'protocol.json'),
        imported_adapter_sha256={name:experiment.sha(ROOT/'scripts/evaluation'/name) for name in [
            'continue_deepseek_text_methods_parallel.py','continue_deepseek_ga_format.py','probe_deepseek_ga_format.py']})
    experiment.freeze(root/'streaming_amendment.json',amendment)
    experiment.ThreadPoolExecutor=lambda max_workers:ThreadPoolExecutor(max_workers=8)
    experiment.APIModel=ParallelAPI
    experiment.Broker=StreamingFormatBroker if method=='gen_agents' else StreamingBroker
    experiment.run(experiment.OUTPUT,method)


if __name__=='__main__':main()
