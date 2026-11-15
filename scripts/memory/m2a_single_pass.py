"""Append-only, one-call memory construction and explicit per-search retrieval limits."""
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import time

from scripts.memory.m2a_local import LocalQwen, append_json
from scripts.memory.m2a_length_prompt import LengthLimitedQwen

EXTRACTION_PROMPT = (
    'Extract a concise factual memory from this single original conversation round and its image, if any. '
    'Return ONLY a JSON object with exactly two string fields: "facts" and "image_caption". '
    'In facts use ONLY the dialogue text, not the image: retain names, preferences, actions, relationships, '
    'quantities, reasons and temporal details actually stated in the round. Use at most five short sentences '
    'and at most 100 English words. Distinguish the user statements from assistant suggestions; '
    'do not turn a suggestion into something the user did. Resolve relative dates only when unambiguous '
    'from the supplied date. In image_caption use ONE sentence of at most 40 English words to describe '
    'the main visible objects, appearance and identifying details. Describe repeated objects as a group; '
    'NEVER enumerate every object, card, repeated label, historical action or item in a visual sequence. '
    'Mention each fact or visual feature only once. Use an empty image_caption if no image is supplied. '
    'Do not infer missing history or invent facts. Do not generate IDs, evidence ranges, tool calls or markdown. '
    'Keep the combined facts and image_caption within 140 English words (or 220 Chinese characters). '
    'Always close the JSON object.'
)


class SinglePassQwen(LengthLimitedQwen):
    def invoke(self, messages):
        # Extraction has its own output schema and length instruction, without tool instructions.
        if self.stage == 'construction':
            if self.schemas:
                raise ValueError('Construction must not bind tools')
            return LocalQwen.invoke(self, messages)
        return super().invoke(messages)


def parse_extraction(content, has_image):
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        # A complete two-string object may end at EOS just before its last brace.
        # Only this exact syntax completion is allowed: never repair string values,
        # invent fields or accept output stopped by the hard generation token cap.
        if not content.lstrip().startswith('{') or content.rstrip().endswith('}'):
            raise
        value = json.loads(content + '}')
    if not isinstance(value, dict) or set(value) != {'facts', 'image_caption'}:
        raise ValueError('Extraction must have exactly facts and image_caption')
    if any(not isinstance(v, str) for v in value.values()):
        raise ValueError('Extraction fields must be strings')
    value = {k: v.strip() for k, v in value.items()}
    if not has_image:
        # Image presence is input metadata, not a model decision. A textual
        # discussion of an image must never become unsupported visual evidence.
        value['image_caption'] = ''
    if not any(value.values()):
        raise ValueError('Empty extracted memory')
    return value


def build_round(obs, raw, semantic, images, llm, out, max_attempts=1):
    from agent.stores.raw import RawMessage
    from agent.stores.semantic import SemanticMemory
    from langchain_core.messages import HumanMessage, SystemMessage
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError('Extraction allows one to three attempts')
    started = time.monotonic()
    raw_id = raw.append(RawMessage(0, datetime.fromisoformat(obs['timestamp']),
        'conversation_round', obs['text'], obs['image']))
    content = [{'type': 'text', 'text': obs['text']}]
    if obs['image']:
        images.image_to_image_token(obs['image'])
        content.append({'type': 'image', 'url': obs['image']})
    for attempt in range(1, max_attempts + 1):
        retry_instruction = ([] if attempt == 1 else [{'type': 'text', 'text': (
            f'FORMAT RETRY {attempt - 1}: Extract the original round above into exactly '
            '{"facts":"dialogue facts", "image_caption":"visual description or empty string"}. '
            'Start with { and end with }. Use double-quoted keys and string values. '
            'Do not continue the User/Assistant conversation or output prose outside JSON. '
            'Keep the original extraction length limits.')}])
        response = llm.invoke([SystemMessage(EXTRACTION_PROMPT), HumanMessage(content + retry_instruction)])
        try:
            value = parse_extraction(response.content, bool(obs['image']))
            break
        except ValueError as exc:
            append_json(out / 'extraction_retries.jsonl', dict(id=obs['id'], attempt=attempt,
                error=repr(exc), raw_response=response.content, will_retry=attempt < max_attempts))
            if attempt == max_attempts:
                raise
    discarded_caption = (parse_extraction(response.content, True)['image_caption']
                         if not obs['image'] else '')
    # Provenance comes only from the program, never model-generated identifiers.
    text = (f"[Source {obs['id']}; date {obs['timestamp']}; "
            f"session {obs['session_id']}; round {obs['round']}]\n" + value['facts'])
    memory_id = semantic.add(SemanticMemory(text=text, image_caption=value['image_caption'],
        image_path=obs['image'], evidence_ids=[[raw_id, raw_id]]))
    record = dict(id=obs['id'], raw_id=raw_id, memory_id=memory_id,
                  extraction_attempts=attempt,
                  elapsed_seconds=time.monotonic() - started,
                  json_repair=None if response.content.rstrip().endswith('}') else 'append_final_object_brace',
                  discarded_caption_without_image=discarded_caption)
    append_json(out / 'extractions.jsonl', dict(record, **value))
    return record


@dataclass(frozen=True)
class RetrievalLimits:
    dense: int = 5
    bm25: int = 5
    text_fused: int = 5
    image: int = 3
    final: int = 5
    rrf_k: int = 60

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError('All retrieval limits must be positive integers')


def search(store, limits, query_text=None, query_image_path=None, top_k=8):
    """The model-requested top_k is logged but cannot override the configured final limit."""
    from pymilvus import AnnSearchRequest, RRFRanker
    text_hits, image_hits = [], []
    if query_text:
        reqs = [AnnSearchRequest(data=[store._embed_text(query_text)], anns_field='text_dense',
                    param={'nprobe': 10}, limit=limits.dense),
                AnnSearchRequest(data=[query_text], anns_field='text_sparse',
                    param={}, limit=limits.bm25)]
        text_hits = [hit for hits in store.db.hybrid_search(collection_name='text_embs',
            reqs=reqs, ranker=RRFRanker(limits.rrf_k), limit=limits.text_fused) for hit in hits]
    if query_image_path:
        image_hits = [hit for hits in store.db.search(collection_name='image_embs',
            anns_field='image_dense', data=[store._embed_image(query_image_path)],
            limit=limits.image, search_params={'metric_type': 'IP'}) for hit in hits]
    ranked = store._rrf([text_hits, image_hits], k=limits.rrf_k)[:limits.final]
    store.log.append(dict(op='search', query_text=query_text, query_image_path=query_image_path,
        requested_top_k=top_k, limits=asdict(limits),
        text_candidate_ids=[h['id'] for h in text_hits], image_candidate_ids=[h['id'] for h in image_hits],
        final_ids=[mem_id for mem_id, _ in ranked]))
    return [store._get_memory_by_id(mem_id) for mem_id, _ in ranked]
