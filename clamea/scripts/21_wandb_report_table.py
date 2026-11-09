# CELL 20 V4 — SAFELY ADD A RENDERED MARKDOWN RESULTS TABLE TO AN EXISTING W&B REPORT
# Paste the complete script into ONE MoLab/Marimo Python cell.
# Does NOT train, evaluate, create runs, change artifacts, or create a new report.
# Prerequisites: WANDB_API_KEY secret; existing Report URL; fs_drive from earlier cells
# for a persistent, verified pre-edit backup. First run APPLY_CHANGES=False.


def cell20_update_existing_report():
    import base64
    import binascii
    import copy
    import hashlib
    import json
    import os
    import re
    import subprocess
    import sys
    import tempfile
    import time
    from pathlib import Path
    from urllib.parse import urlparse

    # ============================ USER SETTINGS =============================
    REPORT_URL = "https://forge.coreweave.com/wandb/khnhg05/MLG-Seminar-CLAMEA/reports/CLAMEA-Reproduction-Experimental-Results-and-Self-Training-Analysis--VmlldzoxODI0MDc5NQ"  # Share accessToken is deliberately omitted.
    APPLY_CHANGES = True  # Safe preview. Change to True ONLY after preview passes.
    EXPECTED_SPEC_SHA256 = "f489f525a6fbe8b44dc274f074fe6803191d27ae55afd83cf31bd1ea125b96bd"  # Copy this hash from the DRY RUN output before applying.
    SAVE_AS_DRAFT = True  # Keep the current report unpublished. Publish via W&B UI later.
    REMOVE_KNOWN_BROKEN_TEXT = True  # Only recognized text in section 2; fails on unknowns.
    # =======================================================================

    ENTITY = "khnhg05"
    PROJECT = "MLG-Seminar-CLAMEA"
    BASE_URL = "https://api.forge.coreweave.com"
    BACKUP_ROOT = "paper_method_runs/paper_method_20261023_160901_s66/diagnostics/cell20_wandb_report_backups"
    MARKER = "<!-- CLAMEA_CELL20_FINAL_TEST_TABLE_V1 -->"
    TABLE = "\n".join([
        MARKER,
        "| Configuration | RAW Hits@1 (%) | RAW MRR | CSLS-10 Hits@1 (%) | CSLS-10 MRR |",
        "| :-- | --: | --: | --: | --: |",
        "| Self-training ON | 6.913 ± 1.254 | 0.120639 ± 0.010747 | 7.610 ± 1.185 | 0.127392 ± 0.010001 |",
        "| **Self-training OFF** | **8.199 ± 0.952** | **0.135915 ± 0.016445** | **8.822 ± 1.064** | **0.141585 ± 0.017869** |",
        "| Preserve Gold + ON | 6.742 ± 1.752 | 0.117368 ± 0.020941 | 7.406 ± 1.894 | 0.124925 ± 0.023609 |",
        "",
        "*Mean ± sample standard deviation across 3 training seeds per configuration (66, 190, 1234). "
        "Hits@1 is in percent; MRR is in [0, 1]. Best checkpoints were selected with "
        "Validation RAW MRR. These are Final Test results from a modified CLAMEA reproduction.*",
    ])

    def ensure(condition, message):
        if not condition:
            raise RuntimeError(message)

    def sha_obj(value):
        data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def visible_text(block):
        # Extract only editable text children (never scrape PanelGrid/config contents).
        def rec(value):
            if isinstance(value, list):
                return " ".join(rec(item) for item in value)
            if isinstance(value, dict):
                # W&B text nodes are typically {"text": "..."}, WITHOUT a type.
                if isinstance(value.get("text"), str):
                    return value["text"]
                return rec(value.get("children", []))
            return ""
        return re.sub(r"\s+", " ", rec(block.get("children", []))).strip()

    def looks_like_next_section(block):
        t = visible_text(block)
        if block.get("type") == "heading" and t:
            return True
        return bool(re.match(r"^(?:[3-9]|[1-9]\d+)\.\s+\S", t))

    def is_known_broken_text(block):
        # Forge sometimes serializes an ordinary empty paragraph as "default"
        # rather than "paragraph". The old Cell20 v3 rejected such a harmless
        # block as unknown: (index, 'default', ''). Treat these two text types
        # equivalently, but NEVER consider non-text blocks safe to remove.
        if block.get("type") not in {"paragraph", "default"}:
            return False
        s = visible_text(block)
        if not s:
            # A blank text paragraph inserted by the rich-text editor.
            # Only clean sections whose other blocks are recognized.
            return True
        s = s.strip().replace("**", "").replace("__", "")
        parts = (
            "Self-training ON", "Self-training OFF", "Preserve Gold + ON",
            "RAW Hits@1", "RAW H@1", "RAW MRR", "CSLS-10 Hits@1", "CSLS-10 MRR",
            "Results are reported as mean", "Best checkpoints were selected",
            "Self-training OFF achieves", "| Configuration", "|---", "| :--",
            "| Self-training", "| Preserve Gold", "/mark", "\\|",
        )
        return s.startswith(parts)

    def transform_spec(original_spec):
        spec = copy.deepcopy(original_spec)
        blocks = spec.get("blocks")
        ensure(isinstance(blocks, list), "Report spec has no blocks array; abort.")
        ensure(sum(isinstance(b, dict) and b.get("type") == "panel-grid"
                   for b in blocks) >= 1,
               "Could not locate the existing panel grid; abort to protect charts/run set.")
        ensure(not any(MARKER in str(b.get("content", "")) for b in blocks),
               "Cell20 Markdown table is already in this report. No changes made.")

        heads = [i for i, b in enumerate(blocks)
                 if isinstance(b, dict)
                 and "2. Final Test Results" in visible_text(b)]
        ensure(len(heads) == 1,
               "Need exactly one existing '2. Final Test Results' section. "
               f"Found {len(heads)}; abort to avoid editing wrong place.")
        idx = heads[0]
        end = next((j for j in range(idx + 1, len(blocks))
                    if isinstance(blocks[j], dict) and looks_like_next_section(blocks[j])), len(blocks))
        region = blocks[idx + 1:end]
        to_remove, to_preserve = [], []
        for j, b in enumerate(region, idx + 1):
            if is_known_broken_text(b):
                to_remove.append(j)
            else:
                to_preserve.append((j, b.get("type"), visible_text(b)[:130]))
        if REMOVE_KNOWN_BROKEN_TEXT:
            # If paragraph text in section 2 cannot be identified, abort rather than delete.
            ensure(not to_preserve,
                   "Section 2 contains non-recognized content. NO CHANGE. "
                   f"Unrecognized blocks: {to_preserve}. "
                   "Set REMOVE_KNOWN_BROKEN_TEXT=False to insert the table without deleting any text.")
            del blocks[idx + 1:end]
        # Insert precisely after existing section heading, preserving all other blocks.
        blocks.insert(idx + 1, {
            "type": "markdown-block", "content": TABLE,
            "children": [{"type": "text", "text": ""}]
        })
        return spec, len(to_remove) if REMOVE_KNOWN_BROKEN_TEXT else 0

    ensure(os.getenv("WANDB_API_KEY"),
           "Missing WANDB_API_KEY in MoLab Secrets. Do not paste key into source.")
    ensure(REPORT_URL.strip(),
           "Paste the full W&B Report URL into REPORT_URL. "
           "It must contain '/reports/<title>--<report-id>'.")
    u = urlparse(REPORT_URL.strip())
    ensure(u.scheme == "https" and u.hostname in
           {"app.forge.coreweave.com", "forge.coreweave.com"},
           "Use the W&B Forge report HTTPS URL, not a project or run URL.")
    # Forge exposes both /wandb/<entity>/<project>/reports/<slug>--<id>
    # and /<entity>/<project>/reports/<slug>--<id>. Browser edit URLs may
    # have an extra suffix, e.g. /edit; query parameters are not needed here.
    # Print only the URL path -- never print query parameters/access tokens.
    from urllib.parse import unquote
    segments = [unquote(x) for x in u.path.split("/") if x]
    print("[REPORT URL PATH]", u.path, flush=True)
    print("[PATH SEGMENTS]", segments, flush=True)
    matches = [i for i, part in enumerate(segments) if part == "reports"]
    ensure(matches,
           "This does not appear to be a Report URL: no '/reports/' path. "
           "Open the actual Report (not the project or workspace), copy the "
           "entire URL from the browser address bar, and set REPORT_URL.")
    # Find reports immediately after the expected project identity.
    valid = [i for i in matches if i >= 2 and
             segments[i-2] == ENTITY and segments[i-1] == PROJECT and
             i+1 < len(segments) and "--" in segments[i+1]]
    ensure(len(valid) == 1,
           "The Report URL path does not identify the expected W&B project "
           "or report slug. Expected .../khnhg05/MLG-Seminar-CLAMEA/reports/"
           "<report-title>--<report-id>. See [PATH SEGMENTS] printed above. "
           "Do not use the project URL or an example with placeholders.")
    report_slug = segments[valid[0]+1]
    raw_report_id = report_slug.rsplit("--", 1)[1]
    ensure(raw_report_id.strip() and "<" not in raw_report_id and ">" not in raw_report_id,
           "The report ID is empty/a placeholder. Copy a real Report URL from W&B.")

    # W&B URL uses an unpadded Base64 global View ID; its Reports SDK restores
    # padding BEFORE GraphQL view(id:...). Example: VmlldzoxODI0MDc5NQ ->
    # VmlldzoxODI0MDc5NQ== (decoded: View:18240795).
    def normalize_view_id(raw):
        raw = str(raw).strip()
        if raw.startswith("View:") and raw[5:].isdigit():
            decoded = raw.encode("ascii")
        else:
            ensure(re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", raw) is not None,
                   "Invalid Base64 W&B View ID in Report URL / response.")
            padded = raw.rstrip("=") + "=" * (-len(raw.rstrip("=")) % 4)
            try:
                decoded = base64.b64decode(padded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise RuntimeError("Cannot decode W&B Report View ID.") from exc
        ensure(decoded.startswith(b"View:") and decoded[5:].isdigit(),
               "Decoded report ID is not a W&B 'View:<number>' identifier.")
        return base64.b64encode(decoded).decode("ascii")

    report_id = normalize_view_id(raw_report_id)
    print("[NORMALIZED REPORT ID]", report_id, flush=True)
    # The GraphQL helper only needs the padded Base64 View ID, not the UI path.
    os.environ["WANDB_BASE_URL"] = BASE_URL

    # Install in the MoLab environment; no changes to experiment files.
    try:
        import wandb
        import wandb_workspaces
    except ImportError:
        print("[SETUP] Installing wandb and wandb-workspaces...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "wandb", "wandb-workspaces"])
        import wandb
        import wandb_workspaces

    from wandb_workspaces._graphql import execute_graphql
    from wandb_workspaces.reports.v2 import gql

    # Raw report-spec editing avoids a lossy whole-report SDK block round-trip.
    # Uses the Reports SDK's own GraphQL helper / documented upsert query.
    api = wandb.Api()
    def get_view():
        payload = execute_graphql(api, gql.view_report, {"reportId": report_id})
        view = payload.get("view")
        ensure(isinstance(view, dict) and view.get("id"),
               "W&B server returned no report. Ensure the report was saved in Forge \n"
               "and your API key can access it.")
        ensure(normalize_view_id(view["id"]) == report_id,
               "W&B returned a different report ID; abort without editing.")
        ensure(view.get("project", {}).get("name") == PROJECT and
               view.get("project", {}).get("entityName") == ENTITY,
               "Report belongs to another project; abort.")
        data = view.get("spec")
        spec = json.loads(data) if isinstance(data, str) else data
        ensure(isinstance(spec, dict) and isinstance(spec.get("blocks"), list),
               "Cannot parse existing report spec.")
        return view, spec

    before_view, before_spec = get_view()
    before_hash = sha_obj(before_spec)
    after_spec, count_removed = transform_spec(before_spec)
    before_grids = [b for b in before_spec["blocks"]
                    if isinstance(b, dict) and b.get("type") == "panel-grid"]
    after_grids = [b for b in after_spec["blocks"]
                   if isinstance(b, dict) and b.get("type") == "panel-grid"]
    ensure(before_grids == after_grids,
           "Unexpected panel grid modification; abort.")
    print("[REPORT]", before_view.get("displayName"))
    print("[REPORT ID]", report_id)
    print("[SPEC SHA256 BEFORE]", before_hash)
    print("[BLOCKS]", len(before_spec["blocks"]), "->", len(after_spec["blocks"]))
    print("[PANEL GRIDS PRESERVED]", len(before_grids))
    print("[BROKEN TEXT BLOCKS TO REMOVE]", count_removed)
    print("[TABLE] MarkdownBlock (5 columns, 3 configurations) prepared")

    if not APPLY_CHANGES:
        print("[DRY RUN PASS] No changes made to W&B.")
        print("Copy the SPEC SHA256 above into EXPECTED_SPEC_SHA256, "
              "then set APPLY_CHANGES=True and re-run this cell.")
        return

    ensure(EXPECTED_SPEC_SHA256.strip() == before_hash,
           "Report changed since preview (or EXPECTED_SPEC_SHA256 missing). "
           "DRY RUN again. Nothing written.")
    # Persistent backup is mandatory before writing to W&B.
    try:
        drive = fs_drive  # Existing authenticated Drive connection from Cells 16-18.
    except NameError as exc:
        raise RuntimeError("fs_drive is missing. Re-run the Drive connection cell "
                           "before APPLY_CHANGES=True. No W&B changes made.") from exc

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    local_dir = Path(tempfile.mkdtemp(prefix="clamea_cell20_backup_"))
    backup_local = local_dir / f"report_{report_id}_{ts}_BEFORE.json"
    backup_local.write_text(json.dumps({"report": before_view,
                                        "original_spec": before_spec,
                                        "original_spec_sha256": before_hash},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
    backup_remote = f"{BACKUP_ROOT}/{report_id}/{backup_local.name}"
    drive.makedirs(backup_remote.rsplit("/", 1)[0], exist_ok=True)
    drive.put_file(str(backup_local), backup_remote)
    ensure(drive.exists(backup_remote),
           "Persistent backup not verified on Drive; nothing written to W&B.")
    print("[BACKUP VERIFIED ON DRIVE]", backup_remote)

    # Optimistic concurrency check immediately before mutation.
    current_view, current_spec = get_view()
    ensure(sha_obj(current_spec) == before_hash,
           "Report changed while backing up; abort without modifying W&B.")
    ensure(bool(current_view.get("name")),
           "Original report backend name is missing; abort.")

    update = {
        "id": report_id, "name": current_view["name"],
        "entityName": ENTITY, "projectName": PROJECT,
        "description": current_view.get("description") or "",
        "displayName": current_view.get("displayName") or "",
        "type": "runs/draft" if SAVE_AS_DRAFT else "runs",
        "spec": json.dumps(after_spec, ensure_ascii=False, separators=(",", ":")),
    }
    result = execute_graphql(api, gql.upsert_view, update)
    saved = result.get("upsertView", {}).get("view", {})
    ensure(saved.get("id") and normalize_view_id(saved["id"]) == report_id,
           "W&B returned an unexpected report ID. Inspect the Report before retrying.")
    verify_view, verify_spec = get_view()
    ensure(any(isinstance(b, dict) and b.get("type") == "markdown-block" and
               MARKER in b.get("content", "") for b in verify_spec["blocks"]),
           "W&B save returned but MarkdownBlock not found on read-back.")
    verify_grids = [b for b in verify_spec["blocks"]
                    if isinstance(b, dict) and b.get("type") == "panel-grid"]
    ensure(verify_grids == before_grids,
           "Post-save panel grids differ from backup! Stop; restore from the Drive backup.")
    print("[SAVE SUCCESS] Updated EXISTING W&B report, server read-back verified.")
    print("[PRESERVED] Existing charts, run set, title, and metadata.")
    print("[MODE]", "Draft: publish manually in W&B UI" if SAVE_AS_DRAFT else "Published")
    print("[REPORT URL]", f"{u.scheme}://{u.netloc}{u.path}")
    print("[NEXT] Refresh the Report page, inspect section 2, then publish from UI.")


cell20_update_existing_report()
