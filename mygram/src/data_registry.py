"""Explicit PMF registry and stdlib provenance helpers (no ML imports)."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import warnings

ARCHIVE_BYTES = 1381232555
ARCHIVE_SHA256 = '2477b8af38d108fe4a66ebfaf460f4176b5b13b00c4082ffeb0637123f3059c1'
SOURCE_URL = 'https://drive.google.com/file/d/14CMfyg9q6ddnXGlxrj96RQZhbX9kwqJ6/view'
TEXT_FILES = ('ent_ids_1', 'ent_ids_2', 'triples_1', 'triples_2', 'ill_ent_ids', 'training_attrs_1', 'training_attrs_2')
REGISTRY = {
    'FBDB15K': ('FBDB15K/norm', 'pkls/FBDB15K_id_img_feature_dict.pkl', 4096),
    'FBYG15K': ('FBYG15K/norm', 'pkls/FBYG15K_id_img_feature_dict.pkl', 4096),
    'zh_en': ('DBP15K/zh_en', 'pkls/zh_en_GA_id_img_feature_dict.pkl', 2048),
    'ja_en': ('DBP15K/ja_en', 'pkls/ja_en_GA_id_img_feature_dict.pkl', 2048),
    'fr_en': ('DBP15K/fr_en', 'pkls/fr_en_GA_id_img_feature_dict.pkl', 2048),
}


def sha256(path):
    with Path(path).open('rb') as stream:
        return stream_hash(stream)


def stream_hash(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'Unsafe relative path: {relative}')
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f'Path escapes data root: {relative}')
    return path


def dataset_files(dataset):
    folder, image, _ = REGISTRY[dataset]
    return [f'{folder}/{name}' for name in TEXT_FILES] + [image]


def resolve_data_paths(data_root, dataset, file_dir=None):
    if dataset not in REGISTRY:
        raise ValueError(f'Unknown dataset {dataset!r}; choose {list(REGISTRY)}')
    root = Path(data_root).resolve()
    folder, image, dim = REGISTRY[dataset]
    directory = safe_path(root, folder)
    if file_dir is not None:
        warnings.warn('--file_dir is deprecated; use --data-root and --dataset', FutureWarning, stacklevel=2)
        if Path(file_dir).resolve() != directory:
            raise ValueError('--file_dir conflicts with explicit --data-root/--dataset')
    return directory, safe_path(root, image), dim


def read_manifest(root, dataset):
    manifest = json.loads((Path(root) / 'data_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('archive', {}).get('sha256') != ARCHIVE_SHA256 or manifest['archive'].get('bytes') != ARCHIVE_BYTES:
        raise ValueError('Manifest does not identify the verified PMF archive')
    for name in dataset_files(dataset):
        path = safe_path(root, name)
        record = manifest['files'][name]
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Data checksum conflict: {path}')
    return manifest


def write_json(path, value):
    """Atomic replace of a generated report; source inputs never use this helper."""
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    os.replace(temp, path)
