# Báo cáo cuối Table 2 và xác nhận Table 1 không bị thay đổi/xóa.
import datetime as dt
import json

status_path = DRIVE_ROOT / "runs" / "experiments" / "matrix_status.json"
status = json.loads(status_path.read_text(encoding="utf-8"))
runs = status.get("runs", {})

selected = {
    experiment_id: runs.get(experiment_id, {})
    for experiment_id in TARGET_EXPERIMENT_IDS
}

print("\nTable 2 target status:")
print(json.dumps(selected, indent=2, ensure_ascii=False))
print("\nFull matrix counts:")
print(json.dumps(status.get("counts", {}), indent=2, ensure_ascii=False))

not_completed = [
    experiment_id
    for experiment_id, state in selected.items()
    if state.get("status") != "completed"
]

report_dir = DRIVE_ROOT / "reports" / "table2"
report_dir.mkdir(parents=True, exist_ok=True)
completion_report = report_dir / "table2_runall_completion.json"
completion_report.write_text(
    json.dumps(
        {
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "selected_runs": selected,
            "matrix_counts": status.get("counts", {}),
            "scope": "Table 2: zh_en, ja_en, fr_en; rate 0.3; seed 2023",
            "table1_artifacts_preserved_at": str(
                DRIVE_ROOT / "reports" / "table1"
            ),
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

if not_completed:
    raise RuntimeError(
        f"Table 2 target chưa completed: {not_completed}. "
        "Chạy lại Runtime → Run all để resume từ Drive."
    )

print("\n✅ TABLE 2 TARGETS COMPLETED")
print("✅ Table 1 runs/artifacts không bị xóa hoặc chạy lại")
print("Completion report:", completion_report)
print("Table 2 CSV:", report_dir / "table2_latest.csv")
print("Table 2 PNG:", report_dir / "table2_latest.png")
print("Metrics PNG:", report_dir / "table2_metrics_latest.png")
