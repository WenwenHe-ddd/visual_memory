"""Record provider content rejections as failures, without resubmitting them."""
from scripts.evaluation import gallery_deepseek_transport as strict
import time

MODEL, ENDPOINT = strict.MODEL, strict.ENDPOINT
request_sha256 = strict.request_sha256


def rejection_outcome(request_hash, elapsed_seconds=None):
    """This is an API failure record, never a synthetic model completion."""
    return dict(status='provider_rejection', provider_error_code='data_inspection_failed', http_status=400,
        raw_response='', requested_model=MODEL, returned_model=None,
        model_identity_kind='unavailable_on_rejection', resolved_model_revision=None,
        response_id=None, request_id=None, finish_reason='provider_rejection',
        refusal='provider_data_inspection_failed', system_fingerprint=None, usage={},
        input_tokens=0, generated_tokens=0, seed=42, temperature=0, enable_thinking=False,
        reasoning_tokens=None, reasoning_content_present=False, request_sha256=request_hash,
        hit_token_limit=False, elapsed_seconds=elapsed_seconds)


class Model(strict.Model):
    def invoke(self, request, stage, record_id=None):
        key = request_sha256(request)
        for attempt in range(1,4):
            if self.fatal_errors: raise RuntimeError('Stopped after fatal DeepSeek error; see api_attempts.jsonl')
            started=time.monotonic()
            try:
                out = strict.chat(self.config, request)
                self.log('api_attempts.jsonl',dict(stage=stage,id=record_id,request_sha256=key,
                    attempt=attempt,status='ok',response_id=out['response_id']))
                self.log('api_responses.jsonl',dict(stage=stage,id=record_id,output=out))
                return out
            except strict.provider.APIError as exc:
                failure=dict(stage=stage,id=record_id,request_sha256=key,attempt=attempt,status='error',
                    error_type=exc.kind,http_status=exc.http_status,retryable=exc.retryable,
                    provider_error_code=getattr(exc,'provider_error_code',None))
                self.log('api_attempts.jsonl',failure)
                if failure['provider_error_code']=='data_inspection_failed':
                    out=rejection_outcome(key,time.monotonic()-started)
                    self.log('api_responses.jsonl',dict(stage=stage,id=record_id,output=out))
                    return out
                if not exc.retryable or attempt==3:
                    with self.lock:self.fatal_errors.append(failure)
                    raise
                time.sleep(2*attempt)
