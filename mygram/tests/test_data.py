"""Stage B regression tests: preservation, trust boundaries and split persistence."""
import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from src import Load
from src.data import ArrayUnpickler, make_split, save_split
from src.data_registry import (REGISTRY, resolve_data_paths, safe_path, sha256,
                               ARCHIVE_BYTES, ARCHIVE_SHA256, dataset_files, read_manifest)
from scripts import prepare_pmf_data as prep

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = '8d6bd6f4ddcd807034e53d2b7fccede86b60343c'
UPSTREAM_FIXTURE = ROOT/'tests/fixtures/upstream_Load_selected.py'
UPSTREAM_FIXTURE_PROVENANCE = ROOT/'tests/fixtures/upstream_Load_selected.json'


def upstream_functions():
    provenance = json.loads(UPSTREAM_FIXTURE_PROVENANCE.read_text(encoding='utf-8'))
    if provenance['commit'] != UPSTREAM or sha256(UPSTREAM_FIXTURE) != provenance['fixture_sha256']:
        raise ValueError('Frozen upstream Load.py fixture provenance/hash mismatch')
    source = UPSTREAM_FIXTURE.read_text(encoding='utf-8')
    selected = {'load_attr', 'load_relation', 'load_img_new'}
    tree = ast.parse(source)
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in selected], type_ignores=[])
    env = {'np': np, 'pickle': pickle, 'Counter': Counter}
    exec(compile(functions, 'upstream_Load.py', 'exec'), env)
    return env


class DataTests(unittest.TestCase):
    def test_explicit_registry_and_alias_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertNotEqual(resolve_data_paths(tmp, 'FBDB15K')[1], resolve_data_paths(tmp, 'FBYG15K')[1])
            with self.assertRaises(ValueError):
                resolve_data_paths(tmp, 'norm')
            with self.assertWarns(FutureWarning):
                with self.assertRaises(ValueError):
                    resolve_data_paths(tmp, 'FBDB15K', Path(tmp)/'FBYG15K/norm')

    def test_paths_cannot_escape(self):
        for relative in ['../evil', '/absolute', 'C:/evil', 'pkls\\..\\evil', '.']:
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                safe_path(ROOT, relative)

    def test_pickle_rejects_non_numpy_globals(self):
        payload = pickle.dumps(Path('untrusted'))
        with self.assertRaises(pickle.UnpicklingError):
            ArrayUnpickler(io.BytesIO(payload)).load()

    def test_loader_rejects_modified_input_before_unpickling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = {}
            for name in dataset_files('zh_en'):
                path = safe_path(root, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
                records[name] = {'bytes': 7, 'sha256': sha256(path)}
            (root/'data_manifest.json').write_text(json.dumps({'archive': {'bytes': ARCHIVE_BYTES, 'sha256': ARCHIVE_SHA256}, 'files': records}), encoding='utf-8')
            read_manifest(root, 'zh_en')
            safe_path(root, dataset_files('zh_en')[0]).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum conflict'):
                read_manifest(root, 'zh_en')

    def test_imputation_matches_upstream_and_rng(self):
        # Duplicate edges, self loops and both directions must keep their multiplicity.
        images = {0: np.array([1, 3], np.float32), 1: np.array([5, 9], np.float32)}
        triples = [(0, 0, 2), (0, 1, 2), (2, 0, 1), (0, 2, 0), (3, 0, 2)]
        old = upstream_functions()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'images.pkl'
            path.write_bytes(pickle.dumps(images))
            before = sha256(path)
            np.random.seed(2023)
            expected = old['load_img_new'](5, path, triples)
            expected_next = np.random.random(8)
            rng = np.random.RandomState(2023)
            actual, masks = Load.load_img_new(5, path, triples, rng=rng, return_metadata=True)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(rng.random(8), expected_next)
            np.testing.assert_array_equal(actual[3], actual[4])  # Shared fallback, no cascade.
            np.testing.assert_array_equal(masks['original_mask'], [True, True, False, False, False])
            self.assertEqual(masks['neighbor_mean_count'], 1)
            self.assertEqual(masks['shared_random_count'], 2)
            self.assertEqual(before, sha256(path))

    def test_bow_ties_and_arrays_match_upstream(self):
        old = upstream_functions()
        triples = [(0, 9, 1), (1, 4, 2), (0, 6, 2)]
        rel, vocab = Load.load_relation(4, triples, 2, return_vocab=True)
        self.assertEqual(vocab, [9, 4])
        np.testing.assert_array_equal(rel, old['load_relation'](4, triples, 2))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'attrs'
            path.write_text('a\tz\ty\nb\tx\n', encoding='utf-8')
            args = ([str(path)], 3, {'a': 0, 'b': 1}, 2)
            attrs, vocab = Load.load_attr(*args, return_vocab=True)
            self.assertEqual(vocab, ['z', 'y'])
            np.testing.assert_array_equal(attrs, old['load_attr'](*args))

    def test_split_matches_list_rng_and_preserves_duplicate_targets(self):
        pairs = [(i, 100 + i//2) for i in range(27)]
        expected = list(pairs)
        rng = np.random.RandomState(2023)
        rng.shuffle(expected)
        next_expected = rng.normal(size=4)
        other = np.random.RandomState(2023)
        split, meta = make_split(pairs, .5, other)
        np.testing.assert_array_equal(np.concatenate([split['train_pairs'], split['test_pairs']]), expected)
        np.testing.assert_array_equal(other.normal(size=4), next_expected)
        self.assertTrue(meta['shared_target_ids_train_test'])
        self.assertEqual(len(split['train_pairs']) + len(split['test_pairs']), len(pairs))

    def test_split_reuse_and_conflict(self):
        pairs = [(i, i+100) for i in range(20)]
        split, meta = make_split(pairs, .2, np.random.RandomState(2023))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(save_split(tmp, 'fixture', 2023, .2, split, meta))
            before = path.stat().st_mtime_ns
            save_split(tmp, 'fixture', 2023, .2, split, meta)
            self.assertEqual(before, path.stat().st_mtime_ns)
            different, new_meta = make_split(pairs, .2, np.random.RandomState(2024))
            with self.assertRaises(FileExistsError):
                save_split(tmp, 'fixture', 2023, .2, different, new_meta)

    def test_archive_mismatch_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.zip'
            path.write_bytes(b'wrong')
            target = Path(tmp)/'target'
            with self.assertRaises(ValueError):
                prep.prepare(path, target)
            self.assertFalse(target.exists())

    def test_prepare_reuse_conflict_and_manifest_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root/'tiny.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('fixture/a', b'abc')
                z.writestr('unused', b'not extracted')
            target = root/'target'
            with patch.object(prep, 'ARCHIVE_BYTES', archive.stat().st_size), patch.object(prep, 'ARCHIVE_SHA256', sha256(archive)), patch.object(prep, 'REGISTRY', {'fixture': ('fixture', '', 2)}), patch.object(prep, 'dataset_files', return_value=['fixture/a']):
                prep.prepare(archive, target)
                self.assertFalse((target/'unused').exists())
                before = (target/'fixture/a').stat().st_mtime_ns
                manifest = target/'data_manifest.json'
                doc = json.loads(manifest.read_text(encoding='utf-8'))
                doc['datasets']['fixture']['status'] = 'passed'
                manifest.write_text(json.dumps(doc), encoding='utf-8')
                prep.prepare(archive, target)
                self.assertEqual(before, (target/'fixture/a').stat().st_mtime_ns)
                self.assertEqual(json.loads(manifest.read_text())['datasets']['fixture']['status'], 'passed')
                (target/'fixture/a').write_bytes(b'conflict')
                with self.assertRaises(FileExistsError):
                    prep.prepare(archive, target)
                self.assertEqual((target/'fixture/a').read_bytes(), b'conflict')


if __name__ == '__main__':
    unittest.main()
