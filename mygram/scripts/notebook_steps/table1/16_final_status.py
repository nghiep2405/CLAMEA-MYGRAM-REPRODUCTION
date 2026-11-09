# Báo cáo cuối: xác nhận hai target đã completed và giữ trạng thái toàn matrix.
import datetime as dt
import json

status_path = DRIVE_ROOT / "runs" / "experiments" / "matrix_status.json"
status = json.loads(status_path.read_text(encoding="utf-8"))
runs = status.get("runs", {})

selected = {
    experiment_id: runs.get(experiment_id, {})
    for experiment_id in TARGET_EXPERIMENT_IDS
}

print("\nSelected Table 1 run status:")
print(json.dumps(selected, indent=2, ensure_ascii=False))
print("\nFull matrix counts:")
print(json.dumps(status.get("counts", {}), indent=2, ensure_ascii=False))

not_completed = [
    experiment_id
    for experiment_id, state in selected.items()
    if state.get("status") != "completed"
]

report_dir = DRIVE_ROOT / "reports" / "table1"
report_dir.mkdir(parents=True, exist_ok=True)
completion_report = report_dir / "table1_runall_completion.json"
completion_report.write_text(
    json.dumps(
        {
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "selected_runs": selected,
            "matrix_counts": status.get("counts", {}),
            "table1_expected_configurations": 6,
            "scope": "Table 1, seed 2023",
            "provenance_note": (
                "Existing Table 1 artifacts may contain two source fingerprints; "
                "see table1_provenance_latest.csv and run_py_checkpoint_patch.diff."
            ),
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

if not_completed:
    raise RuntimeError(
        f"Các target chưa completed: {not_completed}. "
        "Chạy lại Runtime → Run all để resume từ Drive."
    )

print("\n✅ Hai run còn thiếu của Table 1 đã completed.")
print("✅ Bốn run cũ vẫn được giữ nguyên.")
print("Completion report:", completion_report)
print("Table CSV:", report_dir / "table1_latest.csv")
print("Table PNG:", report_dir / "table1_latest.png")
print("Metrics PNG:", report_dir / "table1_metrics_by_rate_latest.png")
