"""A per-question semantic-search budget shared by both M2A agents."""

DEFAULT_MAX_ADDITIONAL_SEARCH_ROUNDS = 2
EXHAUSTED_NOTICE_TEMPLATE = (
    'The per-question semantic search budget is exhausted: {limit} searches have been used. '
    'Do not call search_semantic_memories or query_memory again, even if earlier instructions '
    'say to search or iterate. Use the evidence already retrieved to give your best supported '
    'answer. You may inspect existing evidence with an available raw-message fetch tool if needed.'
)


class _ExhaustedLLM:
    def __init__(self, llm, budget):
        self.llm, self.budget = llm, budget

    def invoke(self, messages, **kwargs):
        from scripts.memory.m2a_length_prompt import constrain_messages
        messages = constrain_messages(messages, EXHAUSTED_NOTICE_TEMPLATE.format(limit=self.budget.limit))
        return self.llm.invoke(messages, **kwargs)


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise ValueError('Additional search rounds must be nonnegative')
    return number


class SearchBudget:
    def __init__(self, max_additional_rounds=DEFAULT_MAX_ADDITIONAL_SEARCH_ROUNDS):
        self.limit = 1 + nonnegative_int(max_additional_rounds)
        self.reset()

    def reset(self):
        self.used = 0

    @property
    def exhausted(self):
        return self.used >= self.limit

    def consume(self):
        # This also guards multiple tool calls emitted in a single response.
        if self.exhausted:
            raise RuntimeError('Per-question semantic search budget exhausted')
        self.used += 1


class BudgetedLLM:
    """Remove search entry points when the shared budget is exhausted.

    Raw-message fetches remain available to the memory manager. Construction
    bypasses this policy; the final reader uses the original model directly.
    """

    def __init__(self, llm, budget):
        self.llm = llm
        self.budget = budget

    def invoke(self, *args, **kwargs):
        if self.llm.stage == 'retrieval' and self.budget.exhausted:
            return _ExhaustedLLM(self.llm, self.budget).invoke(*args, **kwargs)
        return self.llm.invoke(*args, **kwargs)

    def bind_tools(self, tools, **kwargs):
        if self.llm.stage == 'retrieval' and self.budget.exhausted:
            tools = [tool for tool in tools
                     if tool.name not in ('search_semantic_memories', 'query_memory')]
            bound = self.llm.bind_tools(tools, **kwargs) if tools else self.llm
            return _ExhaustedLLM(bound, self.budget)
        return self.llm.bind_tools(tools, **kwargs) if tools else self.llm
