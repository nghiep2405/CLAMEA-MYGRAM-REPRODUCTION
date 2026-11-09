"""Validated PMF loader shared by training and stage B validation.

No graph NxN allocation, model import, normalization, or source-data mutation.
"""
from collections import Counter
import codecs
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import tempfile

import numpy as np

from .data_registry import REGISTRY, read_manifest, resolve_data_paths, sha256, write_json
from .Load import load_attr, load_relation, load_img_new
from .utils import read_raw_data


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype,
            ('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy._core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy.core.multiarray', 'scalar'): np._core.multiarray.scalar,
            ('numpy._core.multiarray', 'scalar'): np._core.multiarray.scalar,
            ('_codecs', 'encode'): codecs.encode,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unapproved class {module}.{name}')
        return allowed[module, name]


def array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(f'{array.dtype.str}:{array.shape}:'.encode())
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def vocab_hash(vocab):
    return hashlib.sha256(json.dumps(vocab, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def make_split(pairs, rate, rng):
    if not 0 < rate < 1:
        raise ValueError('rate must be strictly between zero and one')
    pairs = np.asarray(pairs, dtype=np.int64)
    order = np.arange(len(pairs), dtype=np.int64)
    rng.shuffle(order)  # Same permutation/RNG consumption as upstream shuffle(list).
    cut = int(len(pairs) * rate)
    arrays = {'train_indices': order[:cut], 'test_indices': order[cut:],
              'train_pairs': pairs[order[:cut]], 'test_pairs': pairs[order[cut:]]}
    meta = {'rate': rate, 'train_pairs': cut, 'test_pairs': len(pairs)-cut,
            'source_pairs_sha256': array_hash(pairs),
            'arrays_sha256': {key: array_hash(value) for key, value in arrays.items()},
            'shared_target_ids_train_test': sorted(set(arrays['train_pairs'][:, 1].tolist()) & set(arrays['test_pairs'][:, 1].tolist())),
            'shared_source_ids_train_test': sorted(set(arrays['train_pairs'][:, 0].tolist()) & set(arrays['test_pairs'][:, 0].tolist()))}
    meta['split_sha256'] = vocab_hash(meta['arrays_sha256'])
    return arrays, meta


def save_split(output_root, dataset, seed, rate, arrays, meta):
    folder = Path(output_root) / 'splits' / dataset
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'seed{seed}_rate{rate:g}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                raise FileExistsError(f'Split conflict: {path}')
    else:
        import os
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as stream:
            temp = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        try:
            os.link(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    document = {**meta, 'dataset': dataset, 'seed': seed, 'npz_sha256': sha256(path)}
    sidecar = path.with_suffix('.json')
    if sidecar.exists() and json.loads(sidecar.read_text(encoding='utf-8')) != document:
        raise FileExistsError(f'Split metadata conflict: {sidecar}')
    if not sidecar.exists():
        write_json(sidecar, document)
    return str(path.resolve())


@dataclass
class PMFData:
    dataset: str
    ent2id: dict
    pairs: list
    triples: list
    r_hs: dict
    r_ts: dict
    ids: list
    ordered_ids: list
    image_features: np.ndarray
    attribute_features: np.ndarray
    relation_features: np.ndarray
    image_masks: dict
    split: dict
    split_metadata: dict
    statistics: dict
    manifest: dict


def load_dataset(data_root, dataset, *, seed=2023, rate=0.3, rng=None, file_dir=None):
    directory, image_path, image_dim = resolve_data_paths(data_root, dataset, file_dir)
    manifest = read_manifest(data_root, dataset)  # Hash every input before unpickling.
    ent2id, pairs, triples, r_hs, r_ts, ids = read_raw_data(directory.as_posix())
    ordered_ids, mappings = [], []
    counts = {}
    for side in (1, 2):
        rows = [line.rstrip('\n').split('\t') for line in (directory / f'ent_ids_{side}').open(encoding='utf-8')]
        mapping = {int(row[0]): row[1] for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f'Duplicate entity ID in graph {side}')
        mappings.append(mapping)
        ordered_ids.append([int(row[0]) for row in rows])
    all_ids = ids[0] | ids[1]
    if ids[0] & ids[1] or all_ids != set(range(len(all_ids))) or len(ent2id) != len(all_ids):
        raise ValueError('Entity IDs must be disjoint/contiguous and URIs unique')
    n = len(all_ids)
    for side in (1, 2):
        for name in (f'ent_ids_{side}', f'triples_{side}', f'training_attrs_{side}'):
            with (directory / name).open(encoding='utf-8') as stream:
                counts[name] = sum(1 for _ in stream)
        with (directory / f'triples_{side}').open(encoding='utf-8') as stream:
            for line in stream:
                h, r, t = map(int, line.split())
                if h not in ids[side-1] or t not in ids[side-1] or r < 0:
                    raise ValueError(f'Invalid triple in graph {side}: {line.strip()}')
        with (directory / f'training_attrs_{side}').open(encoding='utf-8') as stream:
            if any(line.rstrip('\n').split('\t')[0] not in ent2id for line in stream):
                raise ValueError(f'Unknown attribute entity in graph {side}')
    if any(a not in ids[0] or b not in ids[1] for a, b in pairs):
        raise ValueError('Invalid alignment endpoints')
    counts['ill_ent_ids'] = len(pairs)
    # Exactly where upstream shuffles: text loaded, no other NumPy RNG draws yet.
    rng = np.random.RandomState(seed) if rng is None else rng
    split, split_meta = make_split(pairs, rate, rng)
    with image_path.open('rb') as stream:
        images = ArrayUnpickler(stream).load()
    if not isinstance(images, dict) or not images:
        raise ValueError('Image pickle must be a nonempty dictionary')
    for key, vector in images.items():
        if not isinstance(key, (int, np.integer)) or key not in all_ids:
            raise ValueError(f'Unknown image entity: {key!r}')
        if not isinstance(vector, np.ndarray) or vector.dtype != np.float32 or vector.shape != (image_dim,) or not np.isfinite(vector).all():
            raise ValueError(f'Invalid image dtype/shape/finite values for entity {key}')
    image_count = len(images)
    coverage = [sum(key in group for key in images)/len(group) for group in ids]
    image_features, masks = load_img_new(n, image_path, triples, images=images, rng=rng, return_metadata=True)
    del images
    if not np.isfinite(image_features).all() or image_features.shape != (n, image_dim):
        raise ValueError('Invalid imputed images')
    attributes, attribute_vocab = load_attr([str(directory/'training_attrs_1'), str(directory/'training_attrs_2')], n, ent2id, 1000, return_vocab=True)
    relations, relation_vocab = load_relation(n, triples, 1000, return_vocab=True)
    stats = {'status': 'passed', 'entities': [len(group) for group in ids], 'total_entities': n,
             'line_counts': counts, 'link_count': len(pairs), 'duplicate_links': len(pairs)-len(set(pairs)),
             'unique_left_links': len({a for a, b in pairs}), 'unique_right_links': len({b for a, b in pairs}),
             'duplicate_target_counts': {str(k): v for k, v in Counter(b for a, b in pairs).items() if v > 1},
             'ids_disjoint': True, 'ids_contiguous': True, 'invalid_triple_endpoints': 0,
             'invalid_alignment_endpoints': 0, 'unknown_attribute_entities': 0,
             'nonfinite_image_vectors': 0, 'unknown_image_entity_ids': 0,
             'image_entries': image_count, 'image_dim': image_dim, 'image_source_dtype': 'float32',
             'image_imputed_dtype': str(image_features.dtype), 'image_shape': list(image_features.shape),
             'image_coverage_per_graph': coverage, 'image_original_mask_sha256': array_hash(masks['original_mask']),
             'image_imputed_mask_sha256': array_hash(masks['imputed_mask']),
             'neighbor_mean_count': masks['neighbor_mean_count'], 'shared_random_count': masks['shared_random_count'],
             'attribute_shape': list(attributes.shape), 'relation_shape': list(relations.shape),
             'zero_attribute_rows': int(np.sum(~attributes.any(axis=1))),
             'zero_relation_rows': int(np.sum(~relations.any(axis=1))),
             'attribute_vocab_sha256': vocab_hash(attribute_vocab), 'relation_vocab_sha256': vocab_hash(relation_vocab),
             'attribute_vocab': attribute_vocab, 'relation_vocab': relation_vocab,
             'attribute_matrix_sha256': array_hash(attributes), 'relation_matrix_sha256': array_hash(relations),
             'split': split_meta}
    return PMFData(dataset, ent2id, pairs, triples, r_hs, r_ts, ids, ordered_ids,
                   image_features, attributes, relations, masks, split, split_meta, stats, manifest)
