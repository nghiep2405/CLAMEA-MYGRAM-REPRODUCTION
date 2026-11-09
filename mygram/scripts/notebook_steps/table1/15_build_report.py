# ============================================================
# VISUALIZE MYGRAM TABLE 1
# Chạy lại cell này bất cứ lúc nào.
# Nó tự cập nhật từ 4/6 -> 5/6 -> 6/6.
# ============================================================

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


# ============================================================
# 1. ĐƯỜNG DẪN
# ============================================================

# Nếu DRIVE_ROOT đã được định nghĩa ở cell trước thì dùng lại.
# Nếu chưa, mặc định dùng thư mục MyGramRepro.
DRIVE_ROOT = Path(
    globals().get(
        "DRIVE_ROOT",
        "/content/drive/MyDrive/MyGramRepro"
    )
)

STORE_ROOT = DRIVE_ROOT / "runs" / "experiments"
REPORT_ROOT = DRIVE_ROOT / "reports" / "table1"
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

print("Experiment store:", STORE_ROOT)
print("Report output  :", REPORT_ROOT)

if not STORE_ROOT.exists():
    raise FileNotFoundError(
        f"Không tìm thấy experiment store: {STORE_ROOT}\n"
        "Hãy mount Drive và kiểm tra DRIVE_ROOT."
    )


# ============================================================
# 2. KHAI BÁO TABLE 1
# ============================================================

DATASETS = ["FBDB15K", "FBYG15K"]
RATES = [0.2, 0.5, 0.8]
DIRECTIONS = ["l2r", "r2l"]

DIRECTION_LABELS = {
    ("FBDB15K", "l2r"): "FB15K → DB15K",
    ("FBDB15K", "r2l"): "DB15K → FB15K",
    ("FBYG15K", "l2r"): "FB15K → YAGO15K",
    ("FBYG15K", "r2l"): "YAGO15K → FB15K",
}

EXPECTED_SLOTS = {
    (dataset, round(rate, 4))
    for dataset in DATASETS
    for rate in RATES
}


# ============================================================
# 3. HÀM ĐỌC VÀ KIỂM TRA ARTIFACT
# ============================================================

def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def source_fingerprint(code_revision):
    """
    Tạo fingerprint ổn định từ hash của các file source.
    """
    source_hashes = code_revision.get("source_sha256")

    if not isinstance(source_hashes, dict) or not source_hashes:
        return None

    payload = json.dumps(
        source_hashes,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()[:12]


def is_full_table1_config(config):
    """
    Chỉ nhận đúng sáu cấu hình full-model của Table 1.
    Loại bỏ pilot, ablation và optimizer control.
    """
    try:
        dataset = config.get("dataset")
        rate = round(float(config.get("rate")), 4)
        seed = int(config.get("seed"))
        epochs = int(config.get("epochs"))
    except (TypeError, ValueError):
        return False

    if dataset not in DATASETS:
        return False

    if rate not in {round(x, 4) for x in RATES}:
        return False

    if seed != 2023:
        return False

    if epochs != 1000:
        return False

    if config.get("profile") != "repaired_pmf":
        return False

    # Table 1 là full model, không phải ablation.
    ablation_flags = [
        "disable_image",
        "disable_attribute",
        "disable_relation",
        "disable_mgd",
        "disable_gram",
    ]

    if any(bool(config.get(flag, False)) for flag in ablation_flags):
        return False

    return True


def verify_small_manifest_files(run_dir, current):
    """
    Kiểm tra hash các metadata nhỏ cần cho Table 1.
    Không hash checkpoint lớn để cell visualize chạy nhanh.
    """
    required_files = [
        "status.json",
        "summary.json",
        "COMPLETED.json",
        "config.resolved.json",
        "code_revision.json",
        "metrics.jsonl",
    ]

    manifest_files = current.get("files", {})

    for relative in required_files:
        path = run_dir / relative

        if not path.is_file():
            raise ValueError(f"Thiếu file {relative}")

        expected = manifest_files.get(relative)

        if expected is None:
            raise ValueError(
                f"{relative} không có trong CURRENT.json"
            )

        expected_size = int(expected.get("bytes", -1))

        if path.stat().st_size != expected_size:
            raise ValueError(
                f"Sai kích thước {relative}"
            )

        expected_hash = expected.get("sha256")

        if expected_hash and sha256_file(path) != expected_hash:
            raise ValueError(
                f"Sai SHA256 {relative}"
            )


def load_completed_table1_run(run_dir):
    """
    Trả về metadata của một Table 1 run hợp lệ.
    Trả về None nếu thư mục không thuộc Table 1.
    Ném lỗi nếu đúng Table 1 nhưng artifact chưa hợp lệ.
    """
    config_path = run_dir / "config.resolved.json"

    if not config_path.is_file():
        return None

    config = read_json(config_path)

    if not is_full_table1_config(config):
        return None

    current = read_json(run_dir / "CURRENT.json")
    status = read_json(run_dir / "status.json")
    summary = read_json(run_dir / "summary.json")
    completed = read_json(run_dir / "COMPLETED.json")
    revision = read_json(run_dir / "code_revision.json")

    verify_small_manifest_files(run_dir, current)

    if current.get("status") != "completed":
        raise ValueError(
            f"CURRENT status={current.get('status')!r}"
        )

    if int(current.get("epoch_next", -1)) != 1000:
        raise ValueError(
            f"CURRENT epoch_next={current.get('epoch_next')}"
        )

    if status.get("status") != "completed":
        raise ValueError(
            f"status.json={status.get('status')!r}"
        )

    if int(status.get("completed_epochs", -1)) != 1000:
        raise ValueError(
            f"status completed_epochs="
            f"{status.get('completed_epochs')}"
        )

    if int(summary.get("completed_epochs", -1)) != 1000:
        raise ValueError(
            f"summary completed_epochs="
            f"{summary.get('completed_epochs')}"
        )

    if completed.get("status") != "completed":
        raise ValueError(
            f"COMPLETED status={completed.get('status')!r}"
        )

    if int(completed.get("epoch_next", -1)) != 1000:
        raise ValueError(
            f"COMPLETED epoch_next="
            f"{completed.get('epoch_next')}"
        )

    experiment_hash = (
        config.get("experiment_sha256")
        or config.get("config_sha256")
    )

    completed_hash = completed.get("experiment_sha256")

    if (
        completed_hash
        and experiment_hash
        and completed_hash != experiment_hash
    ):
        raise ValueError(
            "experiment hash trong config và COMPLETED không khớp"
        )

    best_test = summary.get("best_test")

    if not isinstance(best_test, dict):
        raise ValueError("summary.json không có best_test")

    for direction in DIRECTIONS:
        direction_metrics = best_test.get(direction)

        if not isinstance(direction_metrics, dict):
            raise ValueError(
                f"Thiếu best_test.{direction}"
            )

        for metric in ["hits@1", "hits@10", "mrr"]:
            value = direction_metrics.get(metric)

            if value is None or not np.isfinite(float(value)):
                raise ValueError(
                    f"Metric không hợp lệ: "
                    f"{direction}.{metric}={value}"
                )

    source_id = source_fingerprint(revision)

    if source_id is None:
        raise ValueError(
            "code_revision.json không có source_sha256"
        )

    return {
        "run_dir": run_dir,
        "run_id": run_dir.name,
        "dataset": config["dataset"],
        "rate": round(float(config["rate"]), 4),
        "seed": int(config["seed"]),
        "profile": config["profile"],
        "experiment_hash": experiment_hash,
        "source_fingerprint": source_id,
        "published_at_utc": current.get("published_at_utc"),
        "best_epoch_zero_based": int(summary["best_epoch"]),
        "best_epoch": int(summary["best_epoch"]) + 1,
        "selection_rule": summary.get("selection_rule"),
        "best_test": best_test,
    }


# ============================================================
# 4. QUÉT CÁC RUN TABLE 1
# ============================================================

slot_candidates = {}
rejected_runs = []

for run_dir in sorted(STORE_ROOT.iterdir()):
    if not run_dir.is_dir():
        continue

    if not (run_dir / "config.resolved.json").is_file():
        continue

    try:
        record = load_completed_table1_run(run_dir)

        if record is None:
            continue

        slot = (record["dataset"], record["rate"])
        slot_candidates.setdefault(slot, []).append(record)

    except Exception as exc:
        # Chỉ báo lỗi cho thư mục có vẻ là Table 1.
        try:
            config = read_json(
                run_dir / "config.resolved.json"
            )

            if is_full_table1_config(config):
                rejected_runs.append(
                    (run_dir.name, repr(exc))
                )
        except Exception:
            pass


# Không tự chọn giữa hai run trùng semantic config.
duplicate_slots = {
    slot: records
    for slot, records in slot_candidates.items()
    if len(records) > 1
}

if duplicate_slots:
    print("\n❌ Phát hiện nhiều run cho cùng cấu hình Table 1:")

    for slot, records in duplicate_slots.items():
        print(f"  {slot}:")
        for record in records:
            print("   -", record["run_id"])

    raise RuntimeError(
        "Không tự chọn run trùng. "
        "Hãy kiểm tra provenance trước."
    )


records_by_slot = {
    slot: records[0]
    for slot, records in slot_candidates.items()
}

completed_slots = set(records_by_slot)
missing_slots = sorted(EXPECTED_SLOTS - completed_slots)

print("\n" + "=" * 72)
print(
    f"TABLE 1 PROGRESS: "
    f"{len(completed_slots)}/{len(EXPECTED_SLOTS)} runs"
)
print("=" * 72)

for dataset in DATASETS:
    for rate in RATES:
        slot = (dataset, round(rate, 4))
        record = records_by_slot.get(slot)

        if record:
            print(
                f"✅ {dataset:8s} rate={rate:.1f} "
                f"best_epoch={record['best_epoch']:4d} "
                f"source={record['source_fingerprint']} "
                f"run={record['run_id']}"
            )
        else:
            print(
                f"⏳ {dataset:8s} rate={rate:.1f} "
                f"chưa completed/valid"
            )

if rejected_runs:
    print("\n⚠️ Table 1 run bị loại do artifact chưa hợp lệ:")

    for run_id, error in rejected_runs:
        print(f"  - {run_id}: {error}")


# ============================================================
# 5. KIỂM TRA SOURCE CONSISTENCY
# ============================================================

source_ids = sorted({
    record["source_fingerprint"]
    for record in records_by_slot.values()
})

source_consistent = len(source_ids) == 1

if source_consistent and source_ids:
    print(
        "\n✅ Các Table 1 run hiện có dùng cùng "
        f"source fingerprint: {source_ids[0]}"
    )
else:
    print(
        "\n⚠️ Các run đang dùng nhiều source fingerprint:"
    )
    for source_id in source_ids:
        print("  -", source_id)

    print(
        "Bảng vẫn được tạo để kiểm tra, nhưng chưa nên "
        "dùng làm Table 1 nghiên cứu chính thức."
    )


# ============================================================
# 6. TẠO DỮ LIỆU DẠNG LONG
# ============================================================

long_rows = []

for slot in sorted(records_by_slot):
    record = records_by_slot[slot]

    for direction in DIRECTIONS:
        metrics = record["best_test"][direction]

        long_rows.append({
            "dataset": record["dataset"],
            "direction": direction,
            "direction_label": DIRECTION_LABELS[
                (record["dataset"], direction)
            ],
            "rate": record["rate"],
            "hits@1": float(metrics["hits@1"]),
            "hits@10": float(metrics["hits@10"]),
            "mrr": float(metrics["mrr"]),
            "best_epoch": record["best_epoch"],
            "selection_rule": record["selection_rule"],
            "run_id": record["run_id"],
            "experiment_hash": record["experiment_hash"],
            "source_fingerprint":
                record["source_fingerprint"],
        })

long_df = pd.DataFrame(long_rows)

if long_df.empty:
    raise RuntimeError(
        "Chưa tìm thấy Table 1 run completed hợp lệ."
    )

long_df = long_df.sort_values(
    ["dataset", "direction", "rate"]
).reset_index(drop=True)


# ============================================================
# 7. TẠO TABLE 1 DẠNG WIDE
# ============================================================

lookup = {
    (
        row["dataset"],
        row["direction"],
        round(float(row["rate"]), 4),
    ): row
    for _, row in long_df.iterrows()
}

table_rows = []

for dataset in DATASETS:
    for direction in DIRECTIONS:
        row = {
            "Dataset": dataset,
            "Direction": DIRECTION_LABELS[
                (dataset, direction)
            ],
        }

        for rate in RATES:
            item = lookup.get(
                (dataset, direction, round(rate, 4))
            )

            for metric, label in [
                ("hits@1", "H@1"),
                ("hits@10", "H@10"),
                ("mrr", "MRR"),
            ]:
                column = f"r={rate:.1f} {label}"

                row[column] = (
                    float(item[metric])
                    if item is not None
                    else np.nan
                )

        table_rows.append(row)

table_df = pd.DataFrame(table_rows)

metric_columns = [
    column
    for column in table_df.columns
    if column not in ["Dataset", "Direction"]
]

styled_table = (
    table_df.style
    .format(
        {
            column: "{:.2%}"
            for column in metric_columns
        },
        na_rep="—",
    )
    .set_caption(
        "MyGram Table 1 — best test metrics, seed 2023"
    )
)

display(styled_table)


# ============================================================
# 8. XÁC ĐỊNH INTERIM HAY COMPLETE
# ============================================================

all_six_present = completed_slots == EXPECTED_SLOTS

scientifically_complete = (
    all_six_present
    and source_consistent
)

if scientifically_complete:
    report_label = "complete"
    print(
        "\n✅ TABLE 1 COMPLETE: đủ 6/6 run và "
        "cùng source fingerprint."
    )
elif all_six_present:
    report_label = "six_runs_mixed_source"
    print(
        "\n⚠️ Đủ 6/6 run nhưng source fingerprint "
        "không đồng nhất."
    )
else:
    report_label = (
        f"interim_{len(completed_slots)}_of_6"
    )
    print(
        f"\n⏳ TABLE 1 INTERIM: "
        f"{len(completed_slots)}/6 run hợp lệ."
    )

    print("Các run còn thiếu:")
    for dataset, rate in missing_slots:
        print(f"  - {dataset}, rate={rate:.1f}")


# ============================================================
# 9. LƯU CSV
# ============================================================

# Dữ liệu đầy đủ dạng long, giữ metric trong khoảng [0, 1].
long_csv = REPORT_ROOT / "table1_long_latest.csv"
long_df.to_csv(long_csv, index=False)

# Bảng wide dùng để đọc hoặc nhập vào spreadsheet.
wide_csv = REPORT_ROOT / "table1_latest.csv"
table_df.to_csv(wide_csv, index=False)

# Provenance riêng theo từng run.
provenance_columns = [
    "dataset",
    "rate",
    "run_id",
    "experiment_hash",
    "source_fingerprint",
    "best_epoch",
    "selection_rule",
]

provenance_df = (
    long_df[provenance_columns]
    .drop_duplicates()
    .sort_values(["dataset", "rate"])
)

provenance_csv = (
    REPORT_ROOT / "table1_provenance_latest.csv"
)
provenance_df.to_csv(provenance_csv, index=False)

# Giữ thêm bản theo trạng thái hiện tại.
table_df.to_csv(
    REPORT_ROOT / f"table1_{report_label}.csv",
    index=False,
)


# ============================================================
# 10. RENDER TABLE 1 THÀNH PNG
# ============================================================

display_df = table_df.copy()

for column in metric_columns:
    display_df[column] = display_df[column].map(
        lambda value: (
            "—"
            if pd.isna(value)
            else f"{100.0 * float(value):.2f}"
        )
    )

fig_width = max(18, 1.55 * len(display_df.columns))
fig_height = 1.8 + 0.62 * len(display_df)

fig, ax = plt.subplots(
    figsize=(fig_width, fig_height)
)

ax.axis("off")

table_artist = ax.table(
    cellText=display_df.values,
    colLabels=display_df.columns,
    cellLoc="center",
    colLoc="center",
    loc="center",
)

table_artist.auto_set_font_size(False)
table_artist.set_fontsize(9)
table_artist.scale(1.0, 1.55)

# Header
for column_index in range(len(display_df.columns)):
    cell = table_artist[(0, column_index)]
    cell.set_facecolor("#1f4e78")
    cell.set_text_props(
        color="white",
        weight="bold",
    )

# Xen kẽ màu dòng
for row_index in range(1, len(display_df) + 1):
    background = (
        "#eaf2f8"
        if row_index % 2 == 1
        else "#ffffff"
    )

    for column_index in range(len(display_df.columns)):
        table_artist[
            (row_index, column_index)
        ].set_facecolor(background)

title_state = (
    "COMPLETE"
    if scientifically_complete
    else f"INTERIM {len(completed_slots)}/6"
)

source_state = (
    f"source={source_ids[0]}"
    if source_consistent and source_ids
    else "WARNING: mixed source fingerprints"
)

ax.set_title(
    "MyGram Table 1 — "
    f"{title_state}\n"
    "Best-test metrics (%), seed 2023, "
    f"{source_state}",
    fontsize=14,
    weight="bold",
    pad=18,
)

fig.tight_layout()

latest_table_png = (
    REPORT_ROOT / "table1_latest.png"
)
versioned_table_png = (
    REPORT_ROOT / f"table1_{report_label}.png"
)

fig.savefig(
    latest_table_png,
    dpi=200,
    bbox_inches="tight",
)
fig.savefig(
    versioned_table_png,
    dpi=200,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# 11. BIỂU ĐỒ METRIC THEO TRAIN RATE
# Bản sửa: legend không chèn lên tiêu đề
# ============================================================

from matplotlib.lines import Line2D

metric_specs = [
    ("hits@1", "Hits@1"),
    ("hits@10", "Hits@10"),
    ("mrr", "MRR"),
]

colors = {
    "FBDB15K": "#2f73b7",
    "FBYG15K": "#c83a32",
}

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(16, 8),
    sharex=True,
    sharey=False,
)

for direction_index, direction in enumerate(DIRECTIONS):
    for metric_index, (metric, metric_label) in enumerate(
        metric_specs
    ):
        ax = axes[direction_index, metric_index]

        for dataset in DATASETS:
            values = []

            for rate in RATES:
                item = lookup.get(
                    (
                        dataset,
                        direction,
                        round(rate, 4),
                    )
                )

                values.append(
                    np.nan
                    if item is None
                    else 100.0 * float(item[metric])
                )

            ax.plot(
                RATES,
                values,
                marker="o",
                linewidth=2,
                markersize=7,
                color=colors[dataset],
                label=dataset,
                zorder=3,
            )

            # Tách vị trí nhãn số của hai dataset để tránh chồng nhau.
            if dataset == "FBDB15K":
                annotation_offset = (0, 8)
                vertical_alignment = "bottom"
            else:
                annotation_offset = (0, -8)
                vertical_alignment = "top"

            for rate, value in zip(RATES, values):
                if np.isfinite(value):
                    ax.annotate(
                        f"{value:.2f}",
                        xy=(rate, value),
                        xytext=annotation_offset,
                        textcoords="offset points",
                        ha="center",
                        va=vertical_alignment,
                        fontsize=8,
                        color="#222222",
                        zorder=4,
                    )

        ax.set_title(
            f"{direction.upper()} — {metric_label}",
            fontsize=12,
            pad=8,
        )

        ax.set_xticks(RATES)
        ax.set_xlabel("Train rate")
        ax.set_ylabel(f"{metric_label} (%)")
        ax.set_ylim(0, 100)

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.8,
            alpha=0.35,
            zorder=0,
        )


# Tạo legend riêng để kiểm soát chính xác vị trí.
legend_handles = [
    Line2D(
        [0],
        [0],
        color=colors[dataset],
        marker="o",
        linewidth=2,
        markersize=7,
        label=dataset,
    )
    for dataset in DATASETS
]


# Tiêu đề ở hàng trên cùng.
fig.suptitle(
    "MyGram Table 1 metrics by train rate — "
    f"{title_state}",
    fontsize=16,
    fontweight="bold",
    y=0.985,
)


# Legend nằm ở hàng riêng dưới tiêu đề.
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.945),
    ncol=2,
    frameon=False,
    fontsize=11,
    handlelength=2.2,
    columnspacing=2.5,
)


# Chừa vùng phía trên cho cả title và legend.
fig.tight_layout(
    rect=[0.01, 0.02, 0.99, 0.875],
    h_pad=2.0,
    w_pad=1.5,
)


latest_plot_png = (
    REPORT_ROOT
    / "table1_metrics_by_rate_latest.png"
)

versioned_plot_png = (
    REPORT_ROOT
    / f"table1_metrics_by_rate_{report_label}.png"
)

fig.savefig(
    latest_plot_png,
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    versioned_plot_png,
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()

print("✅ Đã lưu biểu đồ:", latest_plot_png)

# ============================================================
# 12. TỔNG KẾT
# ============================================================

print("\n" + "=" * 72)
print("FILES WRITTEN")
print("=" * 72)
print("Long CSV       :", long_csv)
print("Wide CSV       :", wide_csv)
print("Provenance CSV :", provenance_csv)
print("Table PNG      :", latest_table_png)
print("Metric PNG     :", latest_plot_png)

if scientifically_complete:
    print(
        "\n✅ Có thể dùng table1_complete.csv/png "
        "cho bước tổng hợp Giai đoạn I."
    )
else:
    print(
        "\nℹ️ Đây là kết quả tạm thời. "
        "Sau khi run 5 và 6 completed, chỉ cần chạy lại "
        "chính cell này; không cần sửa code."
    )

print(
    "\nSelection rule:",
    "best_test_hits@1_l2r. "
    "Các metric trong bảng được lấy từ summary.best_test."
)
