import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ExperimentMatrixTests(unittest.TestCase):
    def setUp(self):
        self.matrix = json.loads(
            (ROOT/'configs/experiments.json').read_text(encoding='utf-8')
        )
        self.runs = self.matrix['experiments']

    def test_exact_required_matrix(self):
        self.assertEqual(len(self.runs), 38)
        self.assertEqual(len({item['id'] for item in self.runs}), 38)
        self.assertEqual(sum(item['profile'] == 'repaired_pmf' for item in self.runs), 37)
        self.assertEqual(sum(item['profile'] == 'author_code_compat' for item in self.runs), 1)
        self.assertEqual(
            {item['dataset'] for item in self.runs},
            {'FBDB15K', 'FBYG15K', 'zh_en', 'ja_en', 'fr_en'},
        )
        counts = {}
        for item in self.runs:
            counts[item['group']] = counts.get(item['group'], 0) + 1
        self.assertEqual(counts, {
            'table1': 6, 'table2': 3, 'table3': 6,
            'figure3': 12, 'figure4': 10, 'optimizer_control': 1,
        })

    def test_figure4_reuses_only_the_two_rate_point_two_baselines(self):
        self.assertEqual(self.matrix['reuse'], [
            {'id': 'figure4_fbdb_r020', 'source': 'table1_fbdb_r020', 'group': 'figure4'},
            {'id': 'figure4_fbyg_r020', 'source': 'table1_fbyg_r020', 'group': 'figure4'},
        ])
        figure4_rates = {}
        for item in self.runs:
            if item['group'] == 'figure4':
                figure4_rates.setdefault(item['dataset'], set()).add(item['rate'])
        self.assertEqual(figure4_rates['FBDB15K'], {.05, .10, .15, .25, .30})
        self.assertEqual(figure4_rates['FBYG15K'], {.05, .10, .15, .25, .30})


if __name__ == '__main__':
    unittest.main()
