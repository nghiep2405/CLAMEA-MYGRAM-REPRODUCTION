# CELL 21 — COMPLETE 9-RUN EVIDENCE IN THE EXISTING W&B VIEW-ONLY REPORT
# Paste THIS ENTIRE FILE into ONE new Python cell in MoLab/Marimo after Cell 20.
# Reads live W&B run configs, final metrics, 231-point validation histories,
# and logged artifacts. Adds/replaces ONE marked Markdown section in the
# EXISTING Report; preserves all pre-existing blocks, charts, and Run set.
# DRY_RUN first. No training, new runs, artifact mutations, or new reports.
# Requires: WANDB_API_KEY from MoLab Secrets; fs_drive from previous Drive cell.


def cell21_update_full_run_evidence():
    import base64
    import binascii
    import copy
    import hashlib
    import json
    import math
    import os
    import re
    import statistics
    import subprocess
    import sys
    import tempfile
    import time
    from collections import Counter, defaultdict
    from pathlib import Path
    from urllib.parse import unquote, urlparse

    # ========================== USER SETTINGS ===========================
    REPORT_URL = (
        "https://forge.coreweave.com/wandb/khnhg05/MLG-Seminar-CLAMEA/reports/"
        "CLAMEA-Reproduction-Experimental-Results-and-Self-Training-Analysis--"
        "VmlldzoxODI0MDc5NQ"
    )  # Deliberately NO accessToken. Do not embed sharing tokens in source.
    APPLY_CHANGES = True  # FIRST RUN: keep False.
    EXPECTED_SPEC_SHA256 = "fa8750de55ed11363f6113b9c367d63c68f0e694215cc5b0b9919136399820fb"  # Paste hash printed by the DRY RUN.
    EXPECTED_DATA_SHA256 = "befe75e0a8c5b928fb7babd44ca37d72c3644fe0117b9d06c76e62687a045c47"  # Paste hash printed by the DRY RUN.
    REQUIRE_FULL_HISTORY = True  # Require all epochs 70...300 for all 9 runs.
    REQUIRE_ARTIFACTS = True  # Require at least 1 logged artifact on each run.
    # ====================================================================

    ENTITY = "khnhg05"
    PROJECT = "MLG-Seminar-CLAMEA"
    GROUP = "CLAMEA-ProtocolB-Final-9Runs"
    BASE_URL = "https://api.forge.coreweave.com"
    BACKUP_ROOT = (
        "paper_method_runs/paper_method_20261023_160901_s66/diagnostics/"
        "cell21_wandb_report_backups"
    )
    MARKER = "<!-- CLAMEA_CELL21_FULL_RUN_EVIDENCE_V1 -->"
    ARMS = {
        "deterministic_method_on": "Self-training ON",
        "deterministic_method_off": "Self-training OFF",
        "deterministic_preserve_gold_on": "Preserve Gold + ON",
    }
    SEEDS = (66, 190, 1234)
    METRICS = {
        "raw": "final_test/raw",
        "csls10": "final_test/csls10",
    }
    METRIC_NAMES = (
        "hits_at_1_pct", "hits_at_5_pct", "hits_at_10_pct", "mrr", "mr"
    )
    # Explicit whitelist: never expose credentials or arbitrary user metadata
    # in a view-only Report; only the following Cell19 experiment fields.
    ALLOWED_CONFIG = {
        "source_run_id", "arm", "training_seed", "dataset", "split",
        "test_pairs", "visual_features", "cell15", "audit_id",
        "checkpoint_sha256", "historical_import", "training_replayed",
        "validation_history_imported",
    }

    def ensure(cond, message):
        if not cond:
            raise RuntimeError(message)

    def sha_obj(obj):
        serialized = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def norm_id(raw):
        raw = str(raw).strip()
        if raw.startswith("View:") and raw[5:].isdigit():
            b = raw.encode("ascii")
        else:
            ensure(re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", raw),
                   "Invalid W&B Base64 report ID")
            try:
                b = base64.b64decode(raw.rstrip("=") + "=" * (-len(raw.rstrip("=")) % 4),
                                     validate=True)
            except (ValueError, binascii.Error) as exc:
                raise RuntimeError("Cannot decode W&B Report ID") from exc
        ensure(b.startswith(b"View:") and b[5:].isdigit(),
               "W&B Report ID must decode to View:<number>")
        return base64.b64encode(b).decode("ascii")

    def safe_md(value):
        s = str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        s = s.replace("`", "'")
        return s

    def fmt(v, precision=6):
        return f"{float(v):.{precision}f}"

    def table(columns, rows):
        return ("| " + " | ".join(columns) + " |\n"
                + "| " + " | ".join(["---"] * len(columns)) + " |\n"
                + "\n".join("| " + " | ".join(safe_md(x) for x in row) + " |"
                            for row in rows))

    def as_float(mapping, key, lo=None, hi=None):
        ensure(key in mapping, f"Missing metric in W&B run summary: {key}")
        val = float(mapping[key])
        ensure(math.isfinite(val), f"Non-finite metric: {key}")
        if lo is not None:
            ensure(val >= lo, f"Metric {key} < {lo}")
        if hi is not None:
            ensure(val <= hi, f"Metric {key} > {hi}")
        return val

    ensure(bool(os.getenv("WANDB_API_KEY")),
           "WANDB_API_KEY not found in MoLab Secrets; do not paste the key into code.")
    url = urlparse(REPORT_URL)
    ensure(url.scheme == "https" and url.hostname in
           {"forge.coreweave.com", "app.forge.coreweave.com"},
           "REPORT_URL must be a W&B Forge HTTPS Report URL")
    parts = [unquote(x) for x in url.path.split("/") if x]
    positions = [i for i, x in enumerate(parts)
                 if x == "reports" and i >= 2
                 and parts[i-2:i] == [ENTITY, PROJECT]
                 and i+1 < len(parts) and "--" in parts[i+1]]
    ensure(len(positions) == 1,
           "Report URL must point to /khnhg05/MLG-Seminar-CLAMEA/reports/<title>--<id>")
    report_id = norm_id(parts[positions[0]+1].rsplit("--", 1)[1])
    clean_url = f"{url.scheme}://{url.netloc}{url.path}"
    os.environ["WANDB_BASE_URL"] = BASE_URL

    try:
        import wandb
        import wandb_workspaces
    except ImportError:
        print("[SETUP] Installing wandb and wandb-workspaces ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "wandb", "wandb-workspaces"])
        import wandb
        import wandb_workspaces

    from wandb_workspaces._graphql import execute_graphql
    from wandb_workspaces.reports.v2 import gql

    api = wandb.Api(overrides={"base_url": BASE_URL}, timeout=60)
    view_query = """
        query Cell21ReadReport($reportId: ID!) {
            view(id: $reportId) {
                id name type displayName description
                project {name entityName}
                spec
            }
        }
    """

    def get_report():
        result = execute_graphql(api, view_query, {"reportId": report_id})
        view = result.get("view")
        ensure(isinstance(view, dict) and bool(view.get("id")),
               "Report not found or not accessible by the W&B API key")
        ensure(norm_id(view["id"]) == report_id,
               "Server returned a different Report ID; abort")
        proj = view.get("project") or {}
        ensure(proj.get("name") == PROJECT and proj.get("entityName") == ENTITY,
               "Report belongs to the wrong project; abort")
        ensure(view.get("type") in ("runs", "runs/draft"),
               f"Unexpected Report visibility/type: {view.get('type')}; abort")
        raw = view.get("spec")
        spec = json.loads(raw) if isinstance(raw, str) else raw
        ensure(isinstance(spec, dict) and isinstance(spec.get("blocks"), list),
               "Report spec is not a recognized block structure")
        return view, spec

    print("[SOURCE] Reading W&B runs (no writes) ...", flush=True)
    project_runs = list(api.runs(f"{ENTITY}/{PROJECT}", per_page=50))
    expected_names = {f"{arm}_s{seed}" for arm in ARMS for seed in SEEDS}
    selected = [r for r in project_runs if str(r.name) in expected_names]
    ensure(len(selected) == 9,
           f"Expected 9 known CLAMEA final run names, found {len(selected)} "
           f"among {len(project_runs)} project runs. No Report changes.")
    for r in selected:
        group = getattr(r, "group", None)
        if group:  # Some Public API versions omit `group` on lazy Run objects.
            ensure(group == GROUP, f"Run {r.name} belongs to unexpected group: {group}")

    observed = []
    expected_pairs = {(arm, seed) for arm in ARMS for seed in SEEDS}
    for n, run in enumerate(selected, 1):
        cfg = dict(run.config)
        summary = dict(run.summary)
        arm = cfg.get("arm")
        seed = int(cfg.get("training_seed", -1))
        ensure((arm, seed) in expected_pairs,
               f"Unexpected run configuration: {run.name}: ({arm}, {seed})")
        ensure(str(run.name) == f"{arm}_s{seed}",
               f"Unexpected run name for {arm}, seed={seed}: {run.name}")
        ensure(str(run.state).lower() == "finished",
               f"Run {run.name} state is {run.state}, not finished")
        ensure(cfg.get("historical_import") is True and
               cfg.get("training_replayed") is False,
               f"Run {run.name} is not the documented retrospective import")
        ensure(int(cfg.get("test_pairs", -1)) == 8993,
               f"Unexpected test-pair count in config for {run.name}")
        assert cfg.get("source_run_id") == run.name
        assert cfg.get("validation_history_imported") is True
        checked = {"best_epoch": int(as_float(summary, "validation/best_epoch", 1, 300)),
                   "best_val_mrr": as_float(summary, "validation/best_raw_mrr", 0, 1),
                   "test_pairs": int(as_float(summary, "final_test/test_pairs", 1))}
        ensure(checked["test_pairs"] == 8993, "Test-pair count differs from 8993")
        for evaluation, prefix in METRICS.items():
            checked[evaluation] = {}
            for metric in METRIC_NAMES:
                lo, hi = ((0, 100) if metric.startswith("hits_") else
                          ((0, 1) if metric == "mrr" else (1, None)))
                checked[evaluation][metric] = as_float(summary, f"{prefix}/{metric}", lo, hi)

        # Collect the full validation curve, not an API sampling of history.
        hist_rows = list(run.scan_history(keys=["epoch", "validation/raw_mrr"],
                                          page_size=500))
        points = []
        for row in hist_rows:
            if row.get("epoch") is None or row.get("validation/raw_mrr") is None:
                continue
            ep = int(row["epoch"])
            value = float(row["validation/raw_mrr"])
            ensure(math.isfinite(value) and 0 <= value <= 1,
                   f"Bad Validation MRR in run {run.name}")
            points.append((ep, value))
        points.sort(key=lambda v: v[0])
        if REQUIRE_FULL_HISTORY:
            ensure([p[0] for p in points] == list(range(70, 301)),
                   f"Run {run.name}: expected 231 epoch points from 70 to 300, "
                   f"found {len(points)}; last={points[-1][0] if points else 'none'}")
        else:
            ensure(bool(points), f"Run {run.name} has no Validation MRR history")
        best_history = max(points, key=lambda v: v[1])
        ensure(best_history[0] == checked["best_epoch"] and
               math.isclose(best_history[1], checked["best_val_mrr"],
                            rel_tol=0, abs_tol=1e-8),
               f"Run {run.name}: best Validation metrics differ from history")
        checked["history"] = {"points": points, "n": len(points),
                              "last_epoch": points[-1][0],
                              "last_mrr": points[-1][1]}

        # Artifact metadata (no download). Report names/files are based on
        # what the W&B API actually returns, not assumed filenames.
        found_artifacts = []
        try:
            for artifact in run.logged_artifacts():
                file_names = None
                try:
                    file_names = sorted(str(x) for x in artifact.manifest.entries.keys())
                except Exception:
                    file_names = None  # Record lack of manifest verification.
                metadata = dict(getattr(artifact, "metadata", {}) or {})
                if str(artifact.type) == "evaluation-evidence" and metadata:
                    ensure(metadata.get("source_run_id") == run.name,
                           f"Artifact source_run_id mismatch in {run.name}")
                found_artifacts.append({
                    "name": str(artifact.name),
                    "type": str(artifact.type),
                    "files": file_names,
                    "final_json_sha256": str(metadata.get("final_json_sha256", "")),
                })
        except Exception as exc:
            if REQUIRE_ARTIFACTS:
                raise RuntimeError(
                    f"Cannot read logged artifact inventory for {run.name}: "
                    f"{type(exc).__name__}: {exc}") from exc
            print(f"[WARNING] Artifact API unavailable for {run.name}: {type(exc).__name__}")
        found_artifacts.sort(key=lambda a: (a["name"], a["type"]))
        if REQUIRE_ARTIFACTS:
            ensure(any(a["type"] == "evaluation-evidence" for a in found_artifacts),
                   f"Run {run.name}: no evaluation-evidence artifact found")

        # Keep only explicitly safe experiment configuration values.
        cfg_safe = {k: cfg[k] for k in sorted(cfg) if k in ALLOWED_CONFIG}
        observed.append({"name": str(run.name), "wandb_id": str(run.id),
                         "group": GROUP, "arm": arm, "seed": seed,
                         "state": str(run.state),
                         "created_at": str(getattr(run, "created_at", "") or ""),
                         "tags": sorted(str(t) for t in (getattr(run, "tags", []) or [])),
                         "config": cfg_safe, "metrics": checked,
                         "artifacts": found_artifacts})
        print(f"[RUN {n}/9] {run.name} | best_epoch={checked['best_epoch']} "
              f"| history={len(points)} | artifacts={len(found_artifacts)}", flush=True)

    order = {arm: i for i, arm in enumerate(ARMS)}
    observed.sort(key=lambda r: (order[r["arm"]], SEEDS.index(r["seed"])))
    actual_pairs = {(r["arm"], r["seed"]) for r in observed}
    ensure(actual_pairs == expected_pairs and len(actual_pairs) == 9,
           "The 9 expected (configuration, seed) combinations are not unique")
    ensure(len({r["wandb_id"] for r in observed}) == 9,
           "Duplicate W&B run IDs detected")

    def agg_metric(arm, evaluation, metric):
        vals = [r["metrics"][evaluation][metric] for r in observed if r["arm"] == arm]
        ensure(len(vals) == 3, f"Missing seeds for {arm}")
        d = 3 if metric.startswith("hits_") else (3 if metric == "mr" else 6)
        return f"{statistics.mean(vals):.{d}f} ± {statistics.stdev(vals):.{d}f}"

    def detailed_table(evaluation):
        rows = []
        for r in observed:
            metrics = r["metrics"][evaluation]
            rows.append([
                ARMS[r["arm"]], r["seed"],
                fmt(metrics["hits_at_1_pct"], 3),
                fmt(metrics["hits_at_5_pct"], 3),
                fmt(metrics["hits_at_10_pct"], 3),
                fmt(metrics["mrr"], 6), fmt(metrics["mr"], 3)
            ])
        return table(["Configuration", "Seed", "Hits@1 (%)", "Hits@5 (%)",
                      "Hits@10 (%)", "MRR", "MR"], rows)

    def aggregate_table(evaluation):
        return table(["Configuration", "Hits@1 (%)", "Hits@5 (%)", "Hits@10 (%)", "MRR", "MR"], [
            [label, *[agg_metric(arm, evaluation, m) for m in METRIC_NAMES]]
            for arm, label in ARMS.items()
        ])

    validation_table = table(
        ["Configuration", "Seed", "Best epoch", "Best Val RAW MRR",
         "Last epoch", "Val RAW MRR at last", "History points"],
        [[ARMS[r["arm"]], r["seed"], r["metrics"]["best_epoch"],
          fmt(r["metrics"]["best_val_mrr"], 6),
          r["metrics"]["history"]["last_epoch"],
          fmt(r["metrics"]["history"]["last_mrr"], 6),
          r["metrics"]["history"]["n"]] for r in observed]
    )

    run_metadata_table = table(
        ["Configuration", "Seed", "Run name", "Status", "W&B created at", "Tags"],
        [[ARMS[r["arm"]], r["seed"], r["name"], r["state"],
          r["created_at"], ", ".join(r["tags"])] for r in observed]
    )

    provenance_table = table(
        ["Configuration", "Seed", "W&B run ID", "Checkpoint SHA256"],
        [[ARMS[r["arm"]], r["seed"], r["wandb_id"],
          r["config"].get("checkpoint_sha256", "not recorded")]
         for r in observed]
    )

    artifact_rows = []
    for r in observed:
        a = [x for x in r["artifacts"] if x["type"] == "evaluation-evidence"]
        summary_artifacts = []
        for item in a:
            files = ", ".join(item["files"]) if item["files"] is not None else "manifest not readable"
            summary_artifacts.append(f"{item['name']} [{files}]")
        artifact_rows.append([ARMS[r["arm"]], r["seed"],
                              "; ".join(summary_artifacts) or "not available"])
    artifact_table = table(["Configuration", "Seed", "Logged evaluation artifact / filenames"],
                           artifact_rows)
    final_json_hashes = sorted({item.get("final_json_sha256", "")
                                for r in observed for item in r["artifacts"]
                                if item["type"] == "evaluation-evidence" and
                                item.get("final_json_sha256", "")})
    ensure(len(final_json_hashes) <= 1,
           "Evidence artifacts disagree on the final summary JSON SHA256; abort")
    final_json_note = ("`final_test_summary.json` SHA256 (from W&B artifact metadata): "
                       + ("`" + final_json_hashes[0] + "`" if final_json_hashes else
                          "not available in artifact metadata"))

    configs = [r["config"] for r in observed]
    all_keys = sorted(set().union(*(x.keys() for x in configs)))
    common_keys = []
    varying_keys = []
    for key in all_keys:
        vals = [x.get(key, "<MISSING>") for x in configs]
        if all(v == vals[0] for v in vals):
            common_keys.append((key, vals[0]))
        else:
            varying_keys.append(key)
    common_config_table = table(["Cell 19 configuration field", "Value (identical in 9 runs)"],
                                [[k, json.dumps(v, ensure_ascii=False)]
                                 for k, v in common_keys])
    varying_detail = "\n".join(
        f"- `{safe_md(r['name'])}`: "
        + ", ".join(f"{key}={json.dumps(r['config'].get(key, '<MISSING>'), ensure_ascii=False)}"
                    for key in varying_keys if key != "checkpoint_sha256")
        for r in observed
    )

    hashes = defaultdict(list)
    for r in observed:
        v = r["config"].get("checkpoint_sha256")
        if v:
            hashes[v].append(f"{ARMS[r['arm']]} (seed {r['seed']})")
    duplicated = [names for names in hashes.values() if len(names) > 1]
    duplicate_notice = "\n".join("- Identical checkpoint SHA256: " + ", ".join(names)
                                 for names in duplicated) if duplicated else "- No duplicate checkpoint SHA256 among runs with available hashes."

    notes = "\n".join([
        MARKER,
        "## Detailed Run Evidence — all 9 runs",
        "",
        "**Scope:** 3 experimental configurations × seeds 66, 190, and 1234; "
        "group `CLAMEA-ProtocolB-Final-9Runs`. All values below are read "
        "directly from W&B run summaries, configurations, history and artifacts "
        "at report update time. This section is intended for view-only reviewers "
        "who cannot open the underlying private run pages.",
        "",
        "### 3.1 Validation checkpoint selection and epoch trace",
        "",
        validation_table,
        "",
        "Full per-epoch Validation RAW MRR curves are shown in the **existing interactive chart** "
        "above. The history audit reads every logged point (epochs 70–300) from each run; "
        "the last-epoch score is **not** the checkpoint-selection score.",
        "",
        "### 3.2 Final Test — RAW (without CSLS)",
        "",
        detailed_table("raw"),
        "",
        "### 3.3 Final Test — CSLS-10",
        "",
        detailed_table("csls10"),
        "",
        "### 3.4 Mean ± sample standard deviation (3 seeds per configuration)",
        "",
        "**RAW:**",
        "",
        aggregate_table("raw"),
        "",
        "**CSLS-10:**",
        "",
        aggregate_table("csls10"),
        "",
        "### 3.5 W&B run metadata and recorded experiment configuration",
        "",
        run_metadata_table,
        "",
        "`W&B created at` is the **post-hoc import time**, not the original "
        "model training time. Run names and IDs identify the provenance of each "
        "record even though view-only viewers may not open private run pages.",
        "",
        "These are the configuration fields explicitly recorded by the historical "
        "W&B import. They are **not** a substitute for an exhaustive training-script "
        "hyperparameter manifest; consult the method section and original run evidence "
        "for settings not present in these fields.",
        "",
        common_config_table,
        "",
        "**Fields varying across runs (excluding checkpoint hash, listed below):**",
        "",
        varying_detail or "- None beyond checkpoint SHA256.",
        "",
        "### 3.6 Run IDs and checkpoint provenance",
        "",
        provenance_table,
        "",
        "### 3.7 Artifacts logged by each run",
        "",
        artifact_table,
        "",
        final_json_note,
        "",
        "These are artifact **names and manifests** recorded in W&B. "
        "With a view-only magic link, reviewers can read this inventory but "
        "may still need project permissions to open/download individual artifacts.",
        "",
        "### 3.8 Interpretation and limitations",
        "",
        "- The 9 W&B runs were **retrospectively imported** from previously completed "
        "MoLab experiments; their W&B upload timestamps are **not** training timestamps.",
        "- Checkpoint selection used **Validation RAW MRR**, never Test metrics. "
        "Final Test contains 8,993 entity pairs per run; RAW is primary and CSLS-10 secondary.",
        "- Results represent a **modified / method-level CLAMEA reproduction**, "
        "not proof of paper-faithful reproduction or original-paper performance.",
        "- Seed counts are small (n = 3/configuration). Differences in sample means "
        "alone do not establish statistical significance.",
        "- Historical evidence records one seeded stochastic VAE test draw per run; "
        "additional inference-sampling variability was not estimated here.",
        duplicate_notice,
        "- Run-level raw metrics, full Validation histories and artifact inventories "
        "remain in the W&B project. The report exposes curated evidence without "
        "granting unrestricted access to private run pages.",
    ])

    data_fingerprint = sha_obj({"records": observed, "content": notes})

    view_before, spec_before = get_report()
    original_blocks = spec_before["blocks"]
    original_grids = [b for b in original_blocks if isinstance(b, dict)
                      and b.get("type") == "panel-grid"]
    ensure(original_grids,
           "No existing panel-grid found: abort to protect report visualization")
    ensure(any(isinstance(b, dict) and b.get("type") == "markdown-block"
               and "CLAMEA_CELL20_FINAL_TEST_TABLE_V1" in str(b.get("content", ""))
               for b in original_blocks),
           "Cell 20's Final Test Markdown table is missing; abort to avoid wrong report")
    spec_after = copy.deepcopy(spec_before)
    blocks = spec_after["blocks"]
    existing = [i for i, b in enumerate(blocks)
                if isinstance(b, dict) and b.get("type") == "markdown-block"
                and MARKER in str(b.get("content", ""))]
    ensure(len(existing) <= 1,
           "Multiple Cell21 evidence sections found; abort to avoid ambiguous replacement")
    new_block = {"type": "markdown-block", "content": notes,
                 "children": [{"type": "text", "text": ""}]}
    if existing:
        blocks[existing[0]] = new_block
        action = "REPLACE CELL21 SECTION"
    else:
        blocks.append(new_block)  # Append: retain ALL existing blocks and content.
        action = "APPEND CELL21 SECTION"
    ensure([b for b in blocks if isinstance(b, dict) and b.get("type") == "panel-grid"]
           == original_grids, "Panel grid changed unexpectedly; abort")
    spec_hash = sha_obj(spec_before)
    print("\n[REPORT]", view_before.get("displayName"))
    print("[REPORT TYPE — PRESERVED]", view_before.get("type"))
    print("[RUN GROUP]", GROUP)
    print("[RUNS VERIFIED]", len(observed), "/9")
    print("[HISTORY POINTS]", sum(r["metrics"]["history"]["n"] for r in observed))
    print("[ARTIFACTS]", sum(len(r["artifacts"]) for r in observed))
    print("[PANEL GRIDS PRESERVED]", len(original_grids))
    print("[REPORT BLOCK ACTION]", action)
    print("[REPORT SPEC SHA256 BEFORE]", spec_hash)
    print("[RUN DATA SHA256]", data_fingerprint)
    print("[REPORT MARKDOWN CHARS]", len(notes))
    if not APPLY_CHANGES:
        print("[DRY RUN PASS] No changes made to W&B or Google Drive.")
        print("Paste BOTH SHA256 values above into EXPECTED_SPEC_SHA256 and "
              "EXPECTED_DATA_SHA256, set APPLY_CHANGES=True, then rerun.")
        return

    ensure(EXPECTED_SPEC_SHA256 == spec_hash and
           EXPECTED_DATA_SHA256 == data_fingerprint,
           "Report or source run data changed after preview. NO WRITE. "
           "Rerun DRY RUN and review both hashes.")
    try:
        drive = fs_drive
    except NameError as exc:
        raise RuntimeError("Missing fs_drive for mandatory Google Drive backup. "
                           "Rerun Drive connection cell. W&B unchanged.") from exc

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    work = Path(tempfile.mkdtemp(prefix="clamea_cell21_backup_"))
    backup_content = {
        "report_view_before": view_before,
        "report_spec_before": spec_before,
        "original_spec_sha256": spec_hash,
        "run_data_sha256": data_fingerprint,
        "run_evidence_snapshot": observed,
        "proposed_markdown": notes,
    }
    local = work / f"cell21_{report_id}_{stamp}_BEFORE.json"
    local.write_text(json.dumps(backup_content, ensure_ascii=False, indent=2,
                                allow_nan=False), encoding="utf-8")
    remote = f"{BACKUP_ROOT}/{report_id}/{local.name}"
    drive.makedirs(remote.rsplit("/", 1)[0], exist_ok=True)
    drive.put_file(str(local), remote)
    ensure(drive.exists(remote),
           "Google Drive backup could not be verified; W&B unchanged")
    print("[BACKUP VERIFIED ON DRIVE]", remote)

    latest_view, latest_spec = get_report()
    ensure(sha_obj(latest_spec) == spec_hash and
           latest_view.get("type") == view_before.get("type"),
           "Report changed during backup; W&B unchanged. Rerun DRY RUN.")
    ensure(latest_view.get("name"), "Report backend name missing; W&B unchanged")

    mutation_args = {
        "id": report_id,
        "name": latest_view["name"],
        "entityName": ENTITY,
        "projectName": PROJECT,
        "description": latest_view.get("description") or "",
        "displayName": latest_view.get("displayName") or "",
        "type": latest_view["type"],  # Crucial: DO NOT unpublish a shared Report.
        "spec": json.dumps(spec_after, ensure_ascii=False, separators=(",", ":")),
    }
    result = execute_graphql(api, gql.upsert_view, mutation_args)
    returned = (result.get("upsertView") or {}).get("view") or {}
    ensure(returned.get("id") and norm_id(returned["id"]) == report_id,
           "Unexpected W&B response after save. Inspect report and backup; do not rerun.")
    verified_view, verified_spec = get_report()
    verified_blocks = verified_spec["blocks"]
    matches = [b for b in verified_blocks if isinstance(b, dict)
               and b.get("type") == "markdown-block"
               and MARKER in str(b.get("content", ""))]
    ensure(len(matches) == 1 and matches[0]["content"] == notes,
           "Saved Report did not contain the expected complete section. "
           "Inspect report and backup before retrying.")
    ensure([b for b in verified_blocks if isinstance(b, dict)
            and b.get("type") == "panel-grid"] == original_grids,
           "Panel grids changed after save! Use backup for recovery.")
    ensure(verified_view.get("type") == view_before.get("type"),
           "Report publication state changed unexpectedly; inspect W&B.")
    print("[SAVE SUCCESS] Existing Report updated; server read-back verified.")
    print("[PRESERVED] Other report blocks, charts, Run set, and publication type.")
    print("[REPORT URL]", clean_url)
    print("[NEXT] Refresh the shared view-only URL in an incognito window, "
          "review 9-run details and check interactive chart.")


cell21_update_full_run_evidence()
