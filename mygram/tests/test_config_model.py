import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.config import parse_args, resolved_dict
from src.Mfusion import MCD, MFusion
from src.loss import GRAM_Loss, InfoNCE_loss
from src.run import apply_optimizer_policy

ROOT = Path(__file__).resolve().parents[1]


class ConfigModelTests(unittest.TestCase):
    def test_baseline_config_and_profile_policies(self):
        repaired = parse_args(['--config', str(ROOT/'configs/repaired_pmf.json'), '--device', 'cpu'])
        compat = parse_args(['--config', str(ROOT/'configs/author_code_compat.json'), '--device', 'cpu'])
        self.assertEqual((repaired.structural_output_dim, repaired.joint_output_dim), (900, 1800))
        self.assertEqual(repaired.optimizer_reset_policy, 'once')
        self.assertEqual(compat.optimizer_reset_policy, 'every_epoch_after_il')
        self.assertEqual(repaired.weight_decay, 0.01)
        self.assertEqual((repaired.attention_dropout, repaired.output_dropout), (0.1, 0.1))
        self.assertEqual((repaired.eval_block_size, repaired.il_block_size), (1024, 1024))
        self.assertEqual(repaired.il_model_mode, 'train_compat')

    def test_json_unknown_key_and_profile_conflict_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.json'
            path.write_text(json.dumps({'invented_option': 1}), encoding='utf-8')
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_args(['--config', str(path)])
        with self.assertRaisesRegex(ValueError, 'requires optimizer reset policy'):
            parse_args(['--profile', 'author_code_compat', '--optimizer-reset-policy', 'once'])

    def test_device_has_no_implicit_fallback_and_legacy_alias_conflicts(self):
        cuda = parse_args(['--device', 'cuda:0'])
        self.assertEqual(cuda.device, 'cuda:0')
        self.assertTrue(cuda.cuda)
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            parse_args(['--data-root', str(ROOT/'data/mmkg'), '--dataset', 'FBDB15K', '--file_dir', str(ROOT/'data/mmkg/FBYG15K/norm')])
        from src.run import MyGram
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, 'No CPU fallback'):
                MyGram(cuda)

    def test_modality_flags_resolve_dimensions_and_legacy_meaning(self):
        args = parse_args(['--disable-image'])
        self.assertFalse(args.w_img)
        self.assertEqual(args.joint_output_dim, 1500)
        with self.assertWarns(FutureWarning):
            legacy = parse_args(['--w_img'])
        self.assertTrue(legacy.disable_image)
        with self.assertRaisesRegex(ValueError, 'At least one target modality'):
            parse_args(['--disable-image', '--disable-attribute', '--disable-relation'])

    def test_device_safe_loss_and_buffers(self):
        emb = torch.randn(6, 4, requires_grad=True)
        links = torch.tensor([[0, 3], [1, 4], [2, 5]])
        loss = InfoNCE_loss(torch.device('cpu'), temperature=.1)(emb, links)
        loss.backward()
        self.assertTrue(torch.isfinite(emb.grad).all())
        mcd = MCD(2, .7, .3, 0, 0, 0)
        self.assertIn('gamma', dict(mcd.named_buffers()))
        self.assertIn('gammas', dict(mcd.named_buffers()))
        gram = GRAM_Loss(temperature=.05, K=10, direction_scale=.5)
        self.assertEqual(gram.direction_scale, .5)

    def test_fusion_uses_configured_dropout_and_flattens(self):
        args = parse_args(['--device', 'cpu'])
        fusion = MFusion(args, modal_num=3)
        self.assertEqual(fusion.fusion_layer[0].attention.self.dropout.p, args.attention_dropout)
        self.assertEqual(fusion.fusion_layer[0].output.dropout.p, args.output_dropout)
        fusion.eval()
        out = fusion([torch.randn(4, 300) for _ in range(3)])
        self.assertEqual(tuple(out.shape), (4, 900))

    def test_optimizer_profile_state_policy(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        params = [{'params': [parameter]}]
        optimizer = torch.optim.AdamW(params, lr=.0005, weight_decay=.01)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        repaired = parse_args(['--device', 'cpu'])
        changed, event = apply_optimizer_policy(optimizer, params, repaired, 50)
        self.assertIs(changed, optimizer)
        self.assertEqual(event, 'learning_rate_changed')
        self.assertTrue(changed.state)
        self.assertEqual(changed.param_groups[0]['lr'], .0001)
        unchanged, event = apply_optimizer_policy(changed, params, repaired, 51)
        self.assertIs(unchanged, changed)
        self.assertIsNone(event)
        compat = parse_args(['--profile', 'author_code_compat'])
        reset, event = apply_optimizer_policy(optimizer, params, compat, 50)
        self.assertIsNot(reset, optimizer)
        self.assertEqual(event, 'optimizer_reset')
        self.assertFalse(reset.state)
        self.assertEqual(reset.param_groups[0]['lr'], .0001)
        self.assertEqual(reset.param_groups[0]['weight_decay'], .01)


if __name__ == '__main__':
    unittest.main()
