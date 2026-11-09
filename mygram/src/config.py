"""Canonical MyGram command-line/config resolution for reproducible runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

from .data_registry import REGISTRY, resolve_data_paths, write_json

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('repaired_pmf', 'author_code_compat')


def build_parser():
    parser = argparse.ArgumentParser(
        prog='python -m src.run',
        description='Run the repaired MyGram pipeline on verified PMF processed data.',
    )
    parser.add_argument('--config', type=Path, help='JSON object used as defaults; CLI values win')
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/mmkg')
    parser.add_argument('--dataset', choices=REGISTRY, default='FBDB15K')
    parser.add_argument('--device', default='cpu', help='cpu, cuda, or cuda:N; no implicit fallback')
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs')
    parser.add_argument('--profile', choices=PROFILES, default='repaired_pmf')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--rate', type=float)
    parser.add_argument('--file_dir', default=None, help=argparse.SUPPRESS)

    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Effective AdamW weight decay; upstream constructor default was 0.01')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--loss-batch-size', '--bsize', dest='loss_batch_size', type=int, default=3500)
    parser.add_argument('--eval-every', '--check_point', dest='eval_every', type=int, default=50)
    parser.add_argument('--enable-il', '--il', dest='enable_il', action='store_true', default=True)
    parser.add_argument('--disable-il', dest='enable_il', action='store_false')
    parser.add_argument('--il-start', '--il_start', dest='il_start', type=int, default=50)
    parser.add_argument('--il-step', '--semi_learn_step', dest='il_step', type=int, default=5)
    parser.add_argument('--csls-k', '--csls_k', dest='csls_k', type=int, default=3)
    parser.add_argument('--disable-csls', dest='csls', action='store_false', default=True)
    parser.add_argument('--eval-block-size', type=int, default=1024)
    parser.add_argument('--il-block-size', type=int, default=1024)
    parser.add_argument('--il-model-mode', choices=('train_compat', 'eval'), default='train_compat')
    parser.add_argument('--resume', type=Path, help='Resume an exact matching last/best checkpoint')
    parser.add_argument(
        '--max-epochs-this-process', type=int, default=0,
        help='Operational Colab limit; pause safely after this many epochs (0 disables)',
    )
    parser.add_argument(
        '--max-wall-minutes', type=float, default=0.0,
        help='Operational Colab limit; pause safely after an epoch once this time is reached',
    )
    parser.add_argument(
        '--allow-hardware-change', action='store_true',
        help='Allow resume after a recorded hardware/runtime identity change and log the migration',
    )
    parser.add_argument(
        '--gradient-checkpointing', action='store_true',
        help='Operational exact-memory mode: recompute encoder activations during backward',
    )

    parser.add_argument('--hidden-size', type=int, default=300)
    parser.add_argument('--rrgat-depth', type=int, default=2)
    parser.add_argument('--transformer-layers', dest='num_hidden_layers', type=int, default=2)
    parser.add_argument('--transformer-heads', dest='num_attention_heads', type=int, default=5)
    parser.add_argument('--intermediate-size', type=int, default=400)
    parser.add_argument('--bow-size', type=int, default=1000)
    parser.add_argument('--dropout', type=float, default=0.0, help='MCD/module dropout')
    parser.add_argument('--attention-dropout', type=float, default=0.1)
    parser.add_argument('--output-dropout', type=float, default=0.1)
    parser.add_argument('--mcd-k', type=int, default=2)
    parser.add_argument('--mcd-alpha', type=float, default=0.7)
    parser.add_argument('--mcd-beta', type=float, default=0.3)
    parser.add_argument('--mcd-self-loops', type=int, choices=(1, 2), default=2)
    parser.add_argument('--nce-temperature', '--tau', dest='nce_temperature', type=float, default=0.1)
    parser.add_argument('--gram-k', type=int, default=10)
    parser.add_argument('--gram-temperature', type=float, default=0.05)
    parser.add_argument('--gram-weight', type=float, default=0.1)
    parser.add_argument('--gram-direction-scale', type=float, default=0.5)
    parser.add_argument('--gram-modalities', choices=('available',), default='available')

    parser.add_argument('--disable-image', action='store_true')
    parser.add_argument('--disable-attribute', action='store_true')
    parser.add_argument('--disable-relation', action='store_true')
    parser.add_argument('--disable-mgd', action='store_true')
    parser.add_argument('--disable-gram', action='store_true')
    parser.add_argument('--optimizer-reset-policy', choices=('once', 'every_epoch_after_il'))
    # Old store_false flags are accepted only to expose their surprising meaning.
    parser.add_argument('--w_img', dest='legacy_disable_image', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--w_attr', dest='legacy_disable_attribute', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--w_rel', dest='legacy_disable_relation', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--w_name', dest='legacy_disable_name', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--w_char', dest='legacy_disable_char', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--w_gcn', dest='legacy_disable_structure', action='store_true', help=argparse.SUPPRESS)

    parser.add_argument('--print-config', action='store_true', help='Print resolved JSON and exit')
    parser.add_argument('--write-config', type=Path, help='Write resolved JSON atomically')
    return parser


def _config_defaults(argv):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=Path)
    known, _ = pre.parse_known_args(argv)
    if known.config is None:
        return {}, None
    path = known.config.resolve()
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Config must contain a JSON object: {path}')
    return value, path


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    defaults, config_path = _config_defaults(argv)
    parser = build_parser()
    allowed = {action.dest for action in parser._actions}
    unknown = sorted(set(defaults) - allowed)
    if unknown:
        parser.error(f'unknown config keys: {unknown}')
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.config = config_path
    return resolve(args, argv)


def resolve(args, argv=()):
    legacy = {
        'legacy_disable_image': 'disable_image',
        'legacy_disable_attribute': 'disable_attribute',
        'legacy_disable_relation': 'disable_relation',
    }
    for old, new in legacy.items():
        if getattr(args, old, False):
            warnings.warn(f'--{old.removeprefix("legacy_")} is deprecated and means --{new.replace("_", "-")}', FutureWarning)
            setattr(args, new, True)
    if getattr(args, 'legacy_disable_structure', False):
        raise ValueError('--w_gcn is deprecated and unsupported: structural encoding is required')
    if getattr(args, 'legacy_disable_name', False) or getattr(args, 'legacy_disable_char', False):
        warnings.warn('--w_name/--w_char are deprecated; name and character modalities are already disabled', FutureWarning)
    for name in list(vars(args)):
        if name.startswith('legacy_'):
            delattr(args, name)
    if args.rate is None:
        args.rate = 0.2 if args.dataset in ('FBDB15K', 'FBYG15K') else 0.3
    if not 0 < args.rate < 1:
        raise ValueError('--rate must be strictly between zero and one')
    if (args.seed < 0 or args.epochs <= 0 or args.loss_batch_size <= 0
            or args.eval_every <= 0 or args.eval_block_size <= 0 or args.il_block_size <= 0
            or args.il_start < 0 or args.il_step <= 0 or args.max_epochs_this_process < 0
            or args.max_wall_minutes < 0 or (args.csls and args.csls_k <= 0)):
        raise ValueError(
            'seed/IL start must be nonnegative and epoch, batch, block, IL step, '
            'and enabled CSLS k values must be positive'
        )
    if args.mcd_k <= 0 or args.gram_k <= 0:
        raise ValueError('MCD hop count and Gram K must be positive')
    if args.nce_temperature <= 0 or args.gram_temperature <= 0:
        raise ValueError('NCE and Gram temperatures must be positive')
    if args.gram_weight < 0 or args.gram_direction_scale < 0:
        raise ValueError('Gram weight and direction scale must be nonnegative')
    if args.hidden_size % args.num_attention_heads:
        raise ValueError('hidden size must be divisible by transformer heads')
    if args.bow_size != 1000:
        raise ValueError('Only BoW size 1000 is validated against upstream in stage B')
    if args.rrgat_depth != 2:
        raise ValueError('Only RRGAT depth 2 matches the repaired baseline')
    if all((args.disable_image, args.disable_attribute, args.disable_relation)):
        raise ValueError('At least one target modality is required')
    if args.optimizer_reset_policy is None:
        args.optimizer_reset_policy = 'once' if args.profile == 'repaired_pmf' else 'every_epoch_after_il'
    expected = 'once' if args.profile == 'repaired_pmf' else 'every_epoch_after_il'
    if args.optimizer_reset_policy != expected:
        raise ValueError(f'profile {args.profile} requires optimizer reset policy {expected}')
    if args.device != 'cpu' and args.device != 'cuda' and not args.device.startswith('cuda:'):
        raise ValueError('--device must be cpu, cuda, or cuda:N')
    directory, _, _ = resolve_data_paths(args.data_root, args.dataset, args.file_dir)
    args.data_root = Path(args.data_root).resolve()
    args.file_dir = str(directory)
    args.output_root = Path(args.output_root).resolve()
    if args.resume is not None:
        args.resume = Path(args.resume).resolve()
    args.structural_output_dim = args.hidden_size * (args.rrgat_depth + 1)
    active_modalities = 3 - sum((args.disable_image, args.disable_attribute, args.disable_relation))
    args.modality_output_dim = args.hidden_size * active_modalities
    args.joint_output_dim = args.structural_output_dim + args.modality_output_dim
    # Compatibility fields still used inside the model/training loop.
    args.hidden_units = ','.join([str(args.hidden_size)] * (args.rrgat_depth + 1))
    args.heads = ','.join(['2'] * args.rrgat_depth)
    args.attr_dim = args.hidden_size
    args.img_dim = args.hidden_size
    args.w_img, args.w_attr, args.w_rel = (not args.disable_image, not args.disable_attribute, not args.disable_relation)
    args.w_gcn, args.w_name, args.w_char = True, False, False
    args.structure_encoder, args.final_fusion_type, args.mr_fusion_type = 'Dualmodal-joint-LMF', 0, 7
    args.use_intermediate, args.use_project_head = 1, False
    args.cuda = args.device.startswith('cuda')
    args.check_point, args.bsize = args.eval_every, args.loss_batch_size
    args.il, args.semi_learn_step = args.enable_il, args.il_step
    args.tau, args.dist = args.nce_temperature, 2
    args.use_joint_loss, args.joint_use_nce = True, True
    args.use_ms_loss, args.inner_view_num = False, 0
    args.use_sheduler = args.use_sheduler_cos = False
    args.use_simple_lr = args.optimizer_reset_policy == 'once'
    args.sheduler_gamma, args.num_warmup_steps, args.num_training_steps = 0.98, 0, args.epochs
    args.ms_alpha, args.ms_beta, args.ms_base = 0.1, 40.0, 0.5
    args.word_embedding = 'glove'
    args.pred_name = f'{args.dataset}-r{args.rate:g}-s{args.seed}-{args.profile}'
    args.save_path = str(args.output_root)
    return args


def resolved_dict(args):
    omitted = {'print_config', 'write_config', 'resume'}
    value = {key: (str(item) if isinstance(item, Path) else item)
             for key, item in vars(args).items() if key not in omitted}

    # A research run may move between Google Drive and Colab's local SSD.  Paths,
    # device selection, and process chunk limits describe the runtime rather than
    # the experiment and therefore must not make a checkpoint semantically new.
    runtime_only = {
        'config', 'data_root', 'file_dir', 'output_root', 'save_path', 'device',
        'cuda', 'pred_name', 'max_epochs_this_process', 'max_wall_minutes',
        'allow_hardware_change',
        'gradient_checkpointing',
    }
    experiment_input = {
        key: item for key, item in value.items() if key not in runtime_only
    }
    experiment_json = json.dumps(
        experiment_input, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    )
    experiment_sha256 = hashlib.sha256(experiment_json.encode()).hexdigest()

    runtime_input = {
        key: value.get(key) for key in sorted(runtime_only) if key in value
    }
    runtime_input['experiment_sha256'] = experiment_sha256
    runtime_json = json.dumps(
        runtime_input, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    )
    value['experiment_sha256'] = experiment_sha256
    # Keep the established key as a semantic-hash alias for older callers.
    value['config_sha256'] = experiment_sha256
    value['runtime_sha256'] = hashlib.sha256(runtime_json.encode()).hexdigest()
    return value


def emit_resolved(args):
    value = resolved_dict(args)
    if args.write_config:
        write_json(args.write_config, value)
    return value
