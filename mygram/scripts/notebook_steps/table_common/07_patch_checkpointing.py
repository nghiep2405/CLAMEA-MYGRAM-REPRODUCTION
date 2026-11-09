from pathlib import Path
import shutil
import py_compile

# ============================================================
# CONFIG
# ============================================================

RUN_PY = Path("/content/mygram/src/run.py")
BACKUP = Path("/content/mygram/src/run.py.before_checkpoint_optimization")

CHECKPOINT_EVERY = 10

# Marker để tránh patch 2 lần
PATCH_MARKER = "# MYGRAM_CHECKPOINT_OPTIMIZATION_V2"

# ============================================================
# CHECK SOURCE
# ============================================================

if not RUN_PY.exists():
    raise FileNotFoundError(
        f"Không tìm thấy {RUN_PY}\n"
        "Hãy chạy cell extract source trước."
    )

text = RUN_PY.read_text(
    encoding="utf-8",
    errors="strict",
)

if PATCH_MARKER in text:
    print("✅ run.py đã được patch trước đó.")
    print(f"Checkpoint interval: {CHECKPOINT_EVERY} epochs")

else:
    # --------------------------------------------------------
    # Backup source gốc
    # --------------------------------------------------------

    if not BACKUP.exists():
        shutil.copy2(RUN_PY, BACKUP)
        print(f"✅ Backup source gốc:")
        print(f"   {BACKUP}")

    # ========================================================
    # PATCH 1
    #
    # OLD:
    #   save last.pt EVERY epoch
    #
    # NEW:
    #   save every 10 epochs
    #   + always save final epoch
    # ========================================================

    old_checkpoint = """            self.save_training_checkpoint('last', epoch + 1)

            del joint_emb, gph_emb1, gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb
"""

    new_checkpoint = f"""            {PATCH_MARKER}
            epoch_next = epoch + 1
            checkpoint_every = {CHECKPOINT_EVERY}

            checkpoint_saved_this_epoch = False

            # Save định kỳ thay vì save last.pt ở mọi epoch.
            # Epoch cuối luôn được save.
            if (
                epoch_next % checkpoint_every == 0
                or epoch_next == self.args.epochs
            ):
                self.save_training_checkpoint(
                    'last',
                    epoch_next,
                )
                checkpoint_saved_this_epoch = True

            del joint_emb, gph_emb1, gph_emb, img_emb, rel_emb, att_emb, name_emb, char_emb
"""

    if text.count(old_checkpoint) != 1:
        raise RuntimeError(
            "Không tìm thấy đúng đoạn save checkpoint cần patch "
            "hoặc source không đúng version.\n"
            f"Số match: {text.count(old_checkpoint)}\n"
            "run.py chưa bị thay đổi."
        )

    text = text.replace(
        old_checkpoint,
        new_checkpoint,
        1,
    )

    # ========================================================
    # PATCH 2
    #
    # - Nếu stop/wall/chunk ở epoch không chia hết cho 10:
    #   save checkpoint trước khi pause.
    #
    # - Nếu epoch_next == total epochs:
    #   KHÔNG pause nữa -> đi xuống finalize -> completed.
    # ========================================================

    old_pause = """            if self.stop_requested or wall_limit_reached or epoch_limit_reached:
                reason = (
                    f'signal_{self.stop_signal}' if self.stop_requested
                    else 'max_wall_minutes' if wall_limit_reached
                    else 'max_epochs_this_process'
                )
                write_json(self.run_dir/'status.json', {
                    'status': 'paused', 'run_id': self.run_id,
                    'epoch_next': epoch + 1, 'reason': reason,
                })
                append_jsonl(self.run_dir/'events.jsonl', {
                    'event': 'run_paused', 'epoch': epoch,
                    'epoch_next': epoch + 1, 'reason': reason,
                })
                print(f'[run paused safely] epoch_next={epoch + 1} reason={reason}', flush=True)
                return 'paused'
"""

    new_pause = """            # Chỉ pause nếu training CHƯA hoàn tất.
            #
            # Trước đây nếu chunk kết thúc đúng epoch cuối,
            # epoch_next == self.args.epochs vẫn bị status=paused.
            # Điều kiện này cho phép vòng train kết thúc tự nhiên
            # và chạy phần finalize -> completed.
            if (
                (
                    self.stop_requested
                    or wall_limit_reached
                    or epoch_limit_reached
                )
                and epoch_next < self.args.epochs
            ):
                reason = (
                    f'signal_{self.stop_signal}' if self.stop_requested
                    else 'max_wall_minutes' if wall_limit_reached
                    else 'max_epochs_this_process'
                )

                # Nếu epoch hiện tại chưa thuộc checkpoint định kỳ,
                # vẫn phải save để resume an toàn.
                if not checkpoint_saved_this_epoch:
                    self.save_training_checkpoint(
                        'last',
                        epoch_next,
                    )
                    checkpoint_saved_this_epoch = True

                write_json(self.run_dir/'status.json', {
                    'status': 'paused',
                    'run_id': self.run_id,
                    'epoch_next': epoch_next,
                    'reason': reason,
                })

                append_jsonl(self.run_dir/'events.jsonl', {
                    'event': 'run_paused',
                    'epoch': epoch,
                    'epoch_next': epoch_next,
                    'reason': reason,
                })

                print(
                    f'[run paused safely] '
                    f'epoch_next={epoch_next} '
                    f'reason={reason}',
                    flush=True,
                )

                return 'paused'
"""

    if text.count(old_pause) != 1:
        raise RuntimeError(
            "Không tìm thấy đúng đoạn pause cần patch "
            "hoặc source không đúng version.\n"
            f"Số match: {text.count(old_pause)}\n"
            "run.py chưa được ghi."
        )

    text = text.replace(
        old_pause,
        new_pause,
        1,
    )

    # ========================================================
    # WRITE
    # ========================================================

    RUN_PY.write_text(
        text,
        encoding="utf-8",
    )

    # ========================================================
    # SYNTAX CHECK
    # ========================================================

    try:
        py_compile.compile(
            str(RUN_PY),
            doraise=True,
        )
    except Exception:
        # Nếu syntax có vấn đề -> restore ngay source gốc.
        shutil.copy2(BACKUP, RUN_PY)
        raise RuntimeError(
            "❌ Patch làm source không compile được. "
            "Đã tự restore run.py gốc."
        )

    print()
    print("✅ PATCH THÀNH CÔNG")
    print(f"File       : {RUN_PY}")
    print(f"Backup     : {BACKUP}")
    print(f"Checkpoint : mỗi {CHECKPOINT_EVERY} epochs")
    print()
    print("Thay đổi:")
    print("  ✓ Không save last.pt mỗi epoch")
    print(f"  ✓ Save định kỳ mỗi {CHECKPOINT_EVERY} epochs")
    print("  ✓ Save trước khi pause/stop nếu cần")
    print("  ✓ Save epoch cuối")
    print("  ✓ Epoch cuối đi tới completed, không bị paused")
    print("  ✓ Python syntax check passed")
