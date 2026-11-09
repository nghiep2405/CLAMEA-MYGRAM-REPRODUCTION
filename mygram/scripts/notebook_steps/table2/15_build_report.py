# ============================================================
# VISUALIZE MYGRAM TABLE 2
# Đọc artifact trên Drive; không sửa run/checkpoint.
# ============================================================

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.lines import Line2D

STORE_ROOT = DRIVE_ROOT / "runs" / "experiments"
REPORT_ROOT = DRIVE_ROOT / "reports" / "table2"
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = ["zh_en", "ja_en", "fr_en"]
DIRECTIONS = ["l2r", "r2l"]
RATE = 0.3

DATASET_LABELS = {
    "zh_en": "ZH–EN",
    "ja_en": "JA–EN",
    "fr_en": "FR–EN",
}

DIRECTION_LABELS = {
    ("zh_en", "l2r"): "ZH → EN",
    ("zh_en", "r2l"): "EN → ZH",
    ("ja_en", "l2r"): "JA → EN",
    ("ja_en", "r2l"): "EN → JA",
    ("fr_en", "l2r"): "FR → EN",
    ("fr_en", "r2l"): "EN → FR",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(revision):
    hashes = revision.get("source_sha256")
    if not isinstance(hashes, dict) or not hashes:
        return None
    payload = json.dumps(
        hashes, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def is_table2_config(config):
    try:
        dataset = config.get("dataset")
        rate = round(float(config.get("rate")), 4)
        seed = int(config.get("seed"))
        epochs = int(config.get("epochs"))
    except (TypeError, ValueError):
        return False

    if dataset not in DATASETS or rate != RATE:
        return False
    if seed != 2023 or epochs != 1000:
        return False
    if config.get("profile") != "repaired_pmf":
        return False

    ablations = [
        "disable_image",
        "disable_attribute",
        "disable_relation",
        "disable_mgd",
        "disable_gram",
    ]
    return not any(bool(config.get(name, False)) for name in ablations)


def verify_small_manifest_files(run_dir, current):
    required = [
        "status.json",
        "summary.json",
        "COMPLETED.json",
        "config.resolved.json",
        "code_revision.json",
        "metrics.jsonl",
    ]
    files = current.get("files", {})

    for relative in required:
        path = run_dir / relative
        expected = files.get(relative)
        if not path.is_file():
            raise ValueError(f"Thiếu {relative}")
        if expected is None:
            raise ValueError(f"{relative} không có trong CURRENT.json")
        if path.stat().st_size != int(expected.get("bytes", -1)):
            raise ValueError(f"Sai kích thước {relative}")
        expected_hash = expected.get("sha256")
        if expected_hash and sha256_file(path) != expected_hash:
            raise ValueError(f"Sai SHA256 {relative}")


def load_table2_run(run_dir):
    config_path = run_dir / "config.resolved.json"
    if not config_path.is_file():
        return None

    config = read_json(config_path)
    if not is_table2_config(config):
        return None

    current = read_json(run_dir / "CURRENT.json")
    status = read_json(run_dir / "status.json")
    summary = read_json(run_dir / "summary.json")
    completed = read_json(run_dir / "COMPLETED.json")
    revision = read_json(run_dir / "code_revision.json")

    verify_small_manifest_files(run_dir, current)

    checks = {
        "CURRENT status": current.get("status") == "completed",
        "CURRENT epoch": int(current.get("epoch_next", -1)) == 1000,
        "status": status.get("status") == "completed",
        "status epochs": int(status.get("completed_epochs", -1)) == 1000,
        "summary epochs": int(summary.get("completed_epochs", -1)) == 1000,
        "COMPLETED status": completed.get("status") == "completed",
        "COMPLETED epoch": int(completed.get("epoch_next", -1)) == 1000,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Completion checks failed: {failed}")

    best_test = summary.get("best_test")
    if not isinstance(best_test, dict):
        raise ValueError("summary.json không có best_test")

    for direction in DIRECTIONS:
        values = best_test.get(direction)
        if not isinstance(values, dict):
            raise ValueError(f"Thiếu best_test.{direction}")
        for metric in ("hits@1", "hits@10", "mrr"):
            value = values.get(metric)
            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"Metric lỗi: {direction}.{metric}={value}")

    fingerprint = source_fingerprint(revision)
    if fingerprint is None:
        raise ValueError("code_revision thiếu source_sha256")

    return {
        "dataset": config["dataset"],
        "run_id": run_dir.name,
        "run_dir": run_dir,
        "best_epoch": int(summary["best_epoch"]) + 1,
        "best_test": best_test,
        "selection_rule": summary.get("selection_rule"),
        "experiment_hash": (
            config.get("experiment_sha256")
            or config.get("config_sha256")
        ),
        "source_fingerprint": fingerprint,
        "published_at_utc": current.get("published_at_utc"),
    }


candidates = {}
rejected = []

for run_dir in sorted(STORE_ROOT.iterdir()):
    if not run_dir.is_dir():
        continue
    try:
        record = load_table2_run(run_dir)
        if record is not None:
            candidates.setdefault(record["dataset"], []).append(record)
    except Exception as exc:
        try:
            config = read_json(run_dir / "config.resolved.json")
            if is_table2_config(config):
                rejected.append((run_dir.name, repr(exc)))
        except Exception:
            pass

for dataset, records in candidates.items():
    if len(records) > 1:
        raise RuntimeError(
            f"Nhiều run cho cùng Table 2 dataset {dataset}: "
            f"{[record['run_id'] for record in records]}"
        )

records_by_dataset = {
    dataset: records[0] for dataset, records in candidates.items()
}
missing = [dataset for dataset in DATASETS if dataset not in records_by_dataset]

print("\n" + "=" * 72)
print(f"TABLE 2 PROGRESS: {len(records_by_dataset)}/3 runs")
print("=" * 72)

for dataset in DATASETS:
    record = records_by_dataset.get(dataset)
    if record:
        print(
            f"✅ {dataset:5s} rate=0.3 "
            f"best_epoch={record['best_epoch']:4d} "
            f"source={record['source_fingerprint']} "
            f"run={record['run_id']}"
        )
    else:
        print(f"⏳ {dataset:5s} rate=0.3 chưa completed/valid")

if rejected:
    print("\n⚠️ Run bị loại:")
    for run_id, error in rejected:
        print(f"  - {run_id}: {error}")

source_ids = sorted({
    record["source_fingerprint"]
    for record in records_by_dataset.values()
})
source_consistent = len(source_ids) == 1

if source_consistent and source_ids:
    print("\n✅ Source fingerprint thống nhất:", source_ids[0])
else:
    print("\n⚠️ Source fingerprint không thống nhất:", source_ids)

rows = []
for dataset in DATASETS:
    record = records_by_dataset.get(dataset)
    if record is None:
        continue
    for direction in DIRECTIONS:
        metrics = record["best_test"][direction]
        rows.append({
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "direction": direction,
            "direction_label": DIRECTION_LABELS[(dataset, direction)],
            "rate": RATE,
            "hits@1": float(metrics["hits@1"]),
            "hits@10": float(metrics["hits@10"]),
            "mrr": float(metrics["mrr"]),
            "best_epoch": record["best_epoch"],
            "selection_rule": record["selection_rule"],
            "run_id": record["run_id"],
            "experiment_hash": record["experiment_hash"],
            "source_fingerprint": record["source_fingerprint"],
        })

long_df = pd.DataFrame(rows)
if long_df.empty:
    raise RuntimeError("Chưa có Table 2 run completed hợp lệ")

long_df = long_df.sort_values(["dataset", "direction"]).reset_index(drop=True)

table_df = long_df[[
    "dataset_label",
    "direction_label",
    "hits@1",
    "hits@10",
    "mrr",
    "best_epoch",
]].rename(columns={
    "dataset_label": "Dataset",
    "direction_label": "Direction",
    "hits@1": "Hits@1",
    "hits@10": "Hits@10",
    "mrr": "MRR",
    "best_epoch": "Best epoch",
})

display(
    table_df.style.format(
        {"Hits@1": "{:.2%}", "Hits@10": "{:.2%}", "MRR": "{:.2%}"}
    ).set_caption("MyGram Table 2 — rate 0.3, seed 2023")
)

all_three_present = not missing
scientifically_complete = all_three_present and source_consistent

if scientifically_complete:
    report_label = "complete"
    print("\n✅ TABLE 2 COMPLETE: 3/3 run, source thống nhất")
elif all_three_present:
    report_label = "three_runs_mixed_source"
    print("\n⚠️ Đủ 3/3 run nhưng source fingerprint không thống nhất")
else:
    report_label = f"interim_{len(records_by_dataset)}_of_3"
    print("\n⏳ TABLE 2 INTERIM; còn thiếu:", missing)

long_csv = REPORT_ROOT / "table2_long_latest.csv"
table_csv = REPORT_ROOT / "table2_latest.csv"
provenance_csv = REPORT_ROOT / "table2_provenance_latest.csv"
long_df.to_csv(long_csv, index=False)
table_df.to_csv(table_csv, index=False)
long_df[[
    "dataset", "run_id", "experiment_hash", "source_fingerprint",
    "best_epoch", "selection_rule"
]].drop_duplicates().to_csv(provenance_csv, index=False)
table_df.to_csv(REPORT_ROOT / f"table2_{report_label}.csv", index=False)

# ------------------------- Table PNG -------------------------
display_table = table_df.copy()
for column in ("Hits@1", "Hits@10", "MRR"):
    display_table[column] = display_table[column].map(
        lambda value: f"{100.0 * float(value):.2f}"
    )

fig, ax = plt.subplots(figsize=(12, 4.8))
ax.axis("off")
artist = ax.table(
    cellText=display_table.values,
    colLabels=display_table.columns,
    cellLoc="center",
    colLoc="center",
    loc="center",
)
artist.auto_set_font_size(False)
artist.set_fontsize(10)
artist.scale(1.0, 1.55)

for column_index in range(len(display_table.columns)):
    cell = artist[(0, column_index)]
    cell.set_facecolor("#1f4e78")
    cell.set_text_props(color="white", weight="bold")

for row_index in range(1, len(display_table) + 1):
    color = "#eaf2f8" if row_index % 2 else "#ffffff"
    for column_index in range(len(display_table.columns)):
        artist[(row_index, column_index)].set_facecolor(color)

state = "COMPLETE" if scientifically_complete else f"INTERIM {len(records_by_dataset)}/3"
source_note = (
    f"source={source_ids[0]}"
    if source_consistent and source_ids
    else "WARNING: mixed source fingerprints"
)
ax.set_title(
    f"MyGram Table 2 — {state}\n"
    f"Best-test metrics (%), rate 0.3, seed 2023, {source_note}",
    fontsize=14,
    weight="bold",
    pad=18,
)
fig.tight_layout()

table_png = REPORT_ROOT / "table2_latest.png"
fig.savefig(table_png, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(
    REPORT_ROOT / f"table2_{report_label}.png",
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
)
plt.show()

# ------------------------ Metrics PNG ------------------------
metric_specs = [("hits@1", "Hits@1"), ("hits@10", "Hits@10"), ("mrr", "MRR")]
colors = {"l2r": "#2f73b7", "r2l": "#c83a32"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
x = np.arange(len(DATASETS))
width = 0.34

for axis, (metric, label) in zip(axes, metric_specs):
    for direction_index, direction in enumerate(DIRECTIONS):
        values = []
        for dataset in DATASETS:
            match = long_df[
                (long_df["dataset"] == dataset)
                & (long_df["direction"] == direction)
            ]
            values.append(
                np.nan if match.empty else 100.0 * float(match.iloc[0][metric])
            )

        positions = x + (-width / 2 if direction == "l2r" else width / 2)
        bars = axis.bar(
            positions,
            values,
            width,
            color=colors[direction],
            label=direction.upper(),
        )
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    axis.set_title(label)
    axis.set_xticks(x, [DATASET_LABELS[d] for d in DATASETS])
    axis.set_ylabel(f"{label} (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", linestyle="--", alpha=0.35)

legend_handles = [
    Line2D([0], [0], color=colors[d], linewidth=8, label=d.upper())
    for d in DIRECTIONS
]
fig.suptitle(
    f"MyGram Table 2 metrics — {state}",
    fontsize=16,
    fontweight="bold",
    y=0.985,
)
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.925),
    ncol=2,
    frameon=False,
)
fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.84])

metrics_png = REPORT_ROOT / "table2_metrics_latest.png"
fig.savefig(metrics_png, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(
    REPORT_ROOT / f"table2_metrics_{report_label}.png",
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
)
plt.show()

print("\nFiles written:")
print("Long CSV      :", long_csv)
print("Table CSV     :", table_csv)
print("Provenance CSV:", provenance_csv)
print("Table PNG     :", table_png)
print("Metrics PNG   :", metrics_png)
