"""Run M2A single-pass construction and native test-time tools with DeepSeek."""
from functools import partial
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.evaluation import m2a_single_pass_run as single
from scripts.memory.m2a_deepseek import DeepSeek, MODEL, DEFAULT_ENV, provider


def protocol_hook(args, protocol):
    protocol.pop('llm_path')
    protocol.pop('llm_config_sha256')
    protocol.update(model=MODEL, llm_backend='deepseek_api', llm_provider='DashScope International',
        endpoint=provider.DASHSCOPE_BASE_URL+'/chat/completions',
        model_identity_kind='provider_alias',resolved_model_revision=None,
        enable_thinking=False,temperature=0,
        image_max_pixels=None,
        image_policy='Original file bytes, MIME sniffed; provider detail=high; no local Qwen image processor resizing',
        native_tool_policy='Native API function calls, parallel_tool_calls=false; images from tool results moved to a following user image message with attachment labels',
        api_retry_policy='At most three identical HTTP attempts for transient errors; no content-rejection rewriting or model substitution',
        api_cache_policy='Resume via exact request hashes including model, seed, images, prompts and tools; every logical call logged with cache_replayed',
        gallery_output_policy='Raw answers retained; deterministic internal image-token to official image-ID postprocessing reported separately')
    protocol['construction_policy'] = protocol['construction_policy'].replace('Qwen','DeepSeek')
    protocol['memory_input'] = 'Original chronological dialogue and original image bytes; fresh DeepSeek facts/captions; no supplied dataset captions'
    protocol['memeye_adaptations'] = [s.replace('Qwen','DeepSeek') for s in protocol['memeye_adaptations']]
    for path in [Path(__file__),ROOT/'scripts/memory/m2a_deepseek.py',Path(provider.__file__),
                 ROOT/'scripts/evaluation/openai_chat_transport.py']:
        name = str(path.relative_to(ROOT))
        protocol['implementation_sha256'][name] = single.base.sha(path)
        target = args.output/'code_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())


def run(args):
    factory = partial(DeepSeek,env_file=args.api_env_file or DEFAULT_ENV,cache_dir=args.api_cache_dir)
    return single.run(args,model_class=factory,protocol_hook=protocol_hook)


if __name__ == '__main__':
    single.base.run = run
    single.base.main()
