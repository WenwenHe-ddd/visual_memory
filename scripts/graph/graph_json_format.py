"""Readable JSON for graph snapshots, with numeric embeddings kept inline."""
import json


def readable_dumps(value, indent: int = 2) -> str:
    if indent < 0:
        raise ValueError('indent must be nonnegative')

    def render(item, level):
        pad = ' ' * (indent * level)
        child_pad = ' ' * (indent * (level + 1))
        if isinstance(item, dict):
            if not item:
                return '{}'
            lines = [child_pad + json.dumps(key, ensure_ascii=False) + ': ' +
                     render(part, level + 1) for key, part in item.items()]
            return '{\n' + ',\n'.join(lines) + '\n' + pad + '}'
        if isinstance(item, list):
            if not item:
                return '[]'
            if all(type(part) in (int, float) for part in item):
                return json.dumps(item, ensure_ascii=False, allow_nan=False)
            lines = [child_pad + render(part, level + 1) for part in item]
            return '[\n' + ',\n'.join(lines) + '\n' + pad + ']'
        return json.dumps(item, ensure_ascii=False, allow_nan=False)

    return render(value, 0)
