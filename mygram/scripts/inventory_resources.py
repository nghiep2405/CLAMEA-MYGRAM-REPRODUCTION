"""Read selected ZIP entries and estimate additional stage A storage without extraction."""
import datetime
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    prefixes = ['FBDB15K/norm', 'FBYG15K/norm', 'DBP15K/zh_en', 'DBP15K/ja_en', 'DBP15K/fr_en']
    text_names = ['ent_ids_1', 'ent_ids_2', 'triples_1', 'triples_2', 'ill_ent_ids', 'training_attrs_1', 'training_attrs_2']
    names = [f'{prefix}/{name}' for prefix in prefixes for name in text_names]
    names += [f'pkls/{name}_id_img_feature_dict.pkl' for name in ['FBDB15K', 'FBYG15K', 'zh_en_GA', 'ja_en_GA', 'fr_en_GA']]
    archive = ROOT / 'audit/sources/pmf_mmkg.zip'
    with zipfile.ZipFile(archive) as z:
        entries = [{"name": name, "bytes": z.getinfo(name).file_size, "compressed_bytes": z.getinfo(name).compress_size} for name in names]
    free = shutil.disk_usage(ROOT).free
    # Storage reserves, not measurements. CPU and CUDA environments are alternatives.
    # Two model+Adam checkpoints plus one atomic temporary file, 30M FP32 parameters.
    checkpoint = 30_000_000 * 4 * 3 * 3
    selected = sum(e['bytes'] for e in entries)
    # Initial snapshot reserved 1400 MiB; measured venv was 1,682,431,971 bytes.
    # Future estimates include margin above that observation.
    cpu = dict(wheels=350 * 2**20, environment=1800 * 2**20)
    cuda = dict(wheels=3 * 2**30, environment=6 * 2**30)
    plans = {}
    for backend, costs in [('cpu', cpu), ('cuda', cuda)]:
        total = selected + checkpoint + sum(costs.values()) + 2 * 2**30
        plans[backend] = {**costs, "selected_data_bytes": selected, "checkpoint_reserve_bytes": checkpoint,
                          "headroom_bytes": 2 * 2**30, "additional_bytes_budget": total,
                          "shortfall_bytes": max(0, total-free), "status": 'sufficient' if free >= total else 'blocked'}
    report = {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "archive": str(archive), "archive_bytes": archive.stat().st_size,
              "archive_hash_status": "not_rehashed_in_stage_A; stage_B_must_verify", "entries": entries,
              "selected_bytes": selected, "image_bytes": sum(e['bytes'] for e in entries if e['name'].startswith('pkls/')),
              "free_bytes_before_install": free, "target_root": str(ROOT), "plans": plans,
              "assumptions": ["All 40 selected files copied into a new data root; existing audit text retained.",
                              "Wheel/environment sizes are conservative reserves, not measured installation sizes.",
                              "Checkpoint reserve: 30M parameters x 4 bytes x (model + two Adam moments) x (last + best + temporary). Actual model size is not yet measured.",
                              "Budget covers one active run, not 38 retained run directories."]}
    output = ROOT / 'audit/stage_a_storage.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'entries'}, indent=2))


if __name__ == '__main__':
    main()
