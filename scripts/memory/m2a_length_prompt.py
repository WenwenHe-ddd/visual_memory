"""Prompt-only output-length constraints for the repaired M2A entry point."""
from scripts.memory.m2a_local import LocalQwen

AGENT_LIMIT = (
    'Output length requirement: Keep this entire response within 200 English words '
    '(or 300 Chinese characters); preserve essential facts and IDs, avoid repetition, '
    'and when calling tools output only complete tool-call JSON with all braces and tags closed.'
)
OPEN_LIMIT = (
    'Output length requirement: Answer within 80 English words (or 120 Chinese characters), '
    'without explanation or repetition; follow the required answer format and retain requested facts or image IDs.'
)
MCQ_LIMIT = (
    'Output length requirement: Output exactly one uppercase letter: A, B, C, or D, '
    'with no explanation, punctuation or tool calls; if uncertain, choose the best-supported option.'
)
POLICY = {
    'version': 'm2a-output-length-prompt-v1',
    'agent_instruction': AGENT_LIMIT,
    'open_reader_instruction': OPEN_LIMIT,
    'mcq_reader_instruction': MCQ_LIMIT,
    'generation_limits': 'unchanged: agents 1200 tokens by default, MCQ 16, open reader 256',
    'scope': 'prompt instruction only, not a hard word-count guarantee or input-context fix',
}


def constrain_messages(messages, instruction):
    """Copy the system message; never mutate shared agent history/tool content."""
    from langchain_core.messages import SystemMessage
    result = list(messages)
    for index, message in enumerate(result):
        if message.type != 'system':
            continue
        content = message.content
        if isinstance(content, str):
            content = content if instruction in content else content + '\n\n' + instruction
        else:
            content = list(content)
            block = {'type': 'text', 'text': instruction}
            if block not in content:
                content.append(block)
        result[index] = message.model_copy(update={'content': content}, deep=True)
        break
    else:
        result.insert(0, SystemMessage(content=instruction))
    return result


class LengthLimitedQwen(LocalQwen):
    def invoke(self, messages):
        if self.stage == 'reader':
            instruction = MCQ_LIMIT if self.max_new_tokens == 16 else OPEN_LIMIT
        else:
            instruction = AGENT_LIMIT
        return super().invoke(constrain_messages(messages, instruction))
