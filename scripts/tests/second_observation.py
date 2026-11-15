"""Independent caption-to-caption similarity observation; no pipeline changes."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(path):
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() == '.txt':
        return [{'caption': text}]
    if path.suffix.lower() == '.jsonl':
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif path.suffix.lower() == '.json':
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = [rows]
    else:
        raise ValueError(f'Unsupported caption file: {path}; use .jsonl, .json or .txt')
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f'{path}: expected caption records (objects)')
    return rows


def select_captions(images, sources):
    if len(images) < 2:
        raise ValueError('Provide at least two images')
    if len(sources) not in (1, len(images)):
        raise ValueError('--captions requires one shared file or one file per image, in image order')
    images = [Path(p).expanduser().resolve(strict=True) for p in images]
    sources = [Path(p).expanduser().resolve(strict=True) for p in sources]
    if len(set(images)) != len(images):
        raise ValueError('Duplicate image paths; provide each path once')
    loaded = {path: (load_source(path), sha256(path)) for path in dict.fromkeys(sources)}
    selected = []
    for i, image in enumerate(images):
        source = sources[0] if len(sources) == 1 else sources[i]
        rows, source_hash = loaded[source]
        digest = sha256(image)
        matches = []
        for row in rows:
            hash_match = row.get('id') == digest or row.get('image_sha256') == digest
            raw_path = row.get('image_path', row.get('path'))
            path_match = False
            if isinstance(raw_path, str) and raw_path:
                p = Path(raw_path).expanduser()
                # Relative paths inside a caption file are relative to that file.
                path_match = (p if p.is_absolute() else source.parent / p).resolve() == image
            if hash_match or path_match:
                matches.append(row)
        # A single unkeyed caption can be explicitly paired with an image.
        if not matches and len(sources) == len(images) and len(rows) == 1:
            if not any(k in rows[0] for k in ('id', 'image_sha256', 'image_path', 'path')):
                matches = rows
        if len(matches) != 1:
            raise ValueError(f'{image}: expected exactly one matching caption in {source}, found {len(matches)}')
        row = matches[0]
        caption = row.get('caption')
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f'{image}: empty or non-text caption')
        if row.get('hit_token_limit'):
            raise ValueError(f'{image}: source caption hit its generation token limit')
        if row.get('image_sha256') and row['image_sha256'] != digest:
            raise ValueError(f'{image}: caption record image_sha256 does not match image content')
        selected.append({'index': i, 'image_path': str(image), 'image_sha256': digest,
                         'caption_file': str(source), 'caption_file_sha256': source_hash,
                         'caption_record_id': row.get('id'),
                         'caption_protocol_signature': row.get('protocol_signature'),
                         'caption': caption})
    return selected


def similarities(model, records, batch_size):
    import numpy as np

    texts = [record['caption'] for record in records]
    lengths = [len(model.tokenizer.encode(text, truncation=False)) for text in texts]
    too_long = [i for i, length in enumerate(lengths) if length > model.max_seq_length]
    if too_long:
        raise ValueError(f'Caption indices {too_long} exceed model limit {model.max_seq_length}; '
                         'refusing silent truncation')
    vectors = np.asarray(model.encode(texts, batch_size=batch_size,
                                     normalize_embeddings=True, show_progress_bar=False), dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != len(records) or not np.isfinite(vectors).all():
        raise ValueError('Invalid caption embeddings')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('Zero-length caption embedding')
    vectors = vectors / norms
    matrix = np.clip(vectors @ vectors.T, -1.0, 1.0)
    for record, length in zip(records, lengths):
        record['embedding_input_tokens'] = length
    return matrix.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', nargs='+', required=True, type=Path, help='Two or more image paths')
    parser.add_argument('--captions', nargs='+', required=True, type=Path,
                        help='One shared JSON/JSONL file, or one TXT/JSON/JSONL file per image')
    parser.add_argument('--model-path', type=Path, help='Local SentenceTransformer model; default: natural protocol retriever')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/second_observation')
    args = parser.parse_args()
    try:
        if args.batch_size < 1:
            raise ValueError('--batch-size must be positive')
        records = select_captions(args.images, args.captions)
        model_path = args.model_path
        if model_path is None:
            protocol = json.loads((ROOT / 'results/natural_observation/protocol.json').read_text())
            model_path = Path(protocol['retriever_model_path'])
        model_path = model_path.expanduser().resolve(strict=True)
        output = args.output_dir.expanduser().resolve()
        targets = [output / name for name in ('similarity.json', 'similarity_matrix.csv', 'pairs.csv')]
        if any(path.exists() for path in targets):
            raise ValueError(f'Results already exist in {output}; choose a new --output-dir')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        sys.path.insert(0, str(ROOT / '.vendor'))
        import torch
        import sentence_transformers
        torch.set_num_threads(8)
        model = sentence_transformers.SentenceTransformer(str(model_path), device=args.device, local_files_only=True)
        matrix = similarities(model, records, args.batch_size)
        pairs = [{'image_a': records[i]['image_path'], 'image_b': records[j]['image_path'],
                  'cosine_similarity': matrix[i][j]}
                 for i in range(len(records)) for j in range(i + 1, len(records))]
        result = {'method': 'caption-to-caption cosine similarity; no query prefix',
                  'model_path': str(model_path), 'device': args.device,
                  'sentence_transformers_version': sentence_transformers.__version__,
                  'model_max_seq_length': model.max_seq_length, 'truncated_inputs': 0,
                  'script_sha256': sha256(Path(__file__)),
                  'images': records, 'similarity_matrix': matrix, 'pairs': pairs}
        output.mkdir(parents=True, exist_ok=True)
        targets[0].write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        with targets[1].open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            labels = [r['image_path'] for r in records]
            writer.writerow(['image_path', *labels])
            writer.writerows([label, *row] for label, row in zip(labels, matrix))
        with targets[2].open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['image_a', 'image_b', 'cosine_similarity'])
            writer.writeheader()
            writer.writerows(pairs)
        for pair in pairs:
            print(f"{pair['cosine_similarity']:.6f}  {pair['image_a']} <-> {pair['image_b']}")
        print(f'Results: {output}')
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
