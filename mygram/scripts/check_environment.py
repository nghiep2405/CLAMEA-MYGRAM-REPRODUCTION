"""Stage A preflight. No model/data mutation; failures produce JSON and exit 1."""
import argparse
import datetime
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"error": str(exc)}


def operations(torch, device):
    # Duplicate diagonal and isolated final node exercise coalescing/index_add.
    indices = torch.tensor([[0, 0, 1, 1, 2], [0, 0, 0, 1, 2]], device=device)
    values = torch.tensor([1., 1., .5, 2., 1.], device=device, requires_grad=True)
    x = torch.arange(12, dtype=torch.float32, device=device).reshape(4, 3).requires_grad_()
    a = torch.sparse_coo_tensor(indices, values, (4, 4)).coalesce()
    y = torch.sparse.mm(a, x)
    dense = a.to_dense()
    torch.testing.assert_close(y, dense @ x)
    grad = torch.autograd.grad(y.square().sum(), (x, values), retain_graph=True)
    ref = torch.autograd.grad((dense @ x).square().sum(), (x, values))
    for actual, expected in zip(grad, ref):
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual).all()
    logits = torch.tensor([.1, .2, .3, .4], device=device, requires_grad=True)
    unique = torch.tensor([[0, 0, 1, 2], [0, 1, 0, 2]], device=device)
    attention = torch.sparse.softmax(torch.sparse_coo_tensor(unique, logits, (4, 4)).coalesce(), dim=1)
    features = x.detach().clone().requires_grad_()
    messages = features[unique[1]] * attention.values().unsqueeze(1)
    out = torch.zeros_like(features).index_add(0, unique[0], messages)
    torch.testing.assert_close(out, attention.to_dense() @ features)
    out.square().sum().backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(features.grad).all()
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    return {"status": "passed", "checks": ["coo_coalesce", "sparse_mm_forward_backward_dense_oracle", "sparse_softmax_backward", "index_add_isolated_node_backward"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=ROOT / 'environment.json')
    parser.add_argument('--disk-path', type=Path, default=ROOT)
    parser.add_argument('--min-free-gib', type=float, default=2.0)
    args = parser.parse_args()
    if args.min_free_gib < 0:
        parser.error('--min-free-gib must be nonnegative')
    report = {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "python": {"version": sys.version, "executable": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix},
              "os": platform.platform(), "machine": platform.machine(), "device_requested": args.device,
              "packages": {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
              "imports": {}, "operations": {}, "model_smoke": "not_run", "backend": "native_torch_prerequisite_only",
              "nvidia_smi": command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv,noheader']),
              "cuda_toolkit_nvcc": command(['nvcc', '--version'])}
    for name in ['torch', 'numpy', 'scipy', 'tqdm', 'transformers', 'psutil']:
        try:
            importlib.import_module(name)
            report['imports'][name] = {"status": "passed"}
        except Exception as exc:
            report['imports'][name] = {"status": "failed", "error": repr(exc)}
    try:
        from transformers import apply_chunking_to_forward, get_cosine_schedule_with_warmup
        from transformers.activations import ACT2FN
        report['imports']['transformers_symbols'] = {"status": "passed"}
    except Exception as exc:
        report['imports']['transformers_symbols'] = {"status": "failed", "error": repr(exc)}
    try:
        import psutil
        report['ram'] = dict(psutil.virtual_memory()._asdict())
    except Exception as exc:
        report['ram'] = {"error": repr(exc)}
    usage = shutil.disk_usage(args.disk_path)
    report['disk'] = {"path": str(args.disk_path.resolve()), **dict(usage._asdict()), "minimum_free_bytes": int(args.min_free_gib * 2**30)}
    report['disk']['status'] = 'passed' if usage.free >= args.min_free_gib * 2**30 else 'failed'
    try:
        import torch
        report['torch'] = {"version": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        for device in dict.fromkeys(['cpu', args.device]):
            try:
                report['operations'][device] = operations(torch, device)
            except Exception as exc:
                report['operations'][device] = {"status": "failed", "error": repr(exc)}
    except Exception as exc:
        report['operations'][args.device] = {"status": "failed", "error": repr(exc)}
    passed = all(v['status'] == 'passed' for v in report['imports'].values()) and all(v['status'] == 'passed' for v in report['operations'].values()) and report['disk']['status'] == 'passed' and 'error' not in report['ram']
    report['requested_device_gate'] = 'passed' if passed else 'blocked'
    report['gate_A_cuda'] = 'passed' if passed and args.device.startswith('cuda') else 'blocked'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({"report": str(args.output.resolve()), "requested_device_gate": report['requested_device_gate'], "gate_A_cuda": report['gate_A_cuda'], "operations": report['operations']}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
