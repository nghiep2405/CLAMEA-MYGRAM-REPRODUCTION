# MyGram — code-only refactored project

Đây là bản mã nguồn đã tái cấu trúc cho quy trình tái lập MyGram trên máy cục bộ
và Google Colab. Refactor chỉ thay đổi cách tổ chức mã: thuật toán, cấu hình,
seed, data split, loss, checkpoint, evaluation và thứ tự thực nghiệm được giữ
nguyên.

Ba notebook là các entry point “Run All”. Hai cell đầu của mỗi notebook vẫn nằm
trực tiếp trong notebook vì chúng phải mount Drive và giải nén source bundle trước
khi bất kỳ module nào trong repository có thể được import. Các cell còn lại đã
được đưa ra 19 script và chạy trong shared namespace qua
`src/notebook_runner.py`.

Notebook không chứa output hoặc execution counter. Dataset, checkpoint, log,
metric, biểu đồ, báo cáo chạy và bundle sinh ra không nằm trong source tree.

## Folder chứa toàn bộ kết quả chạy và minh chứng chạy:
Source folder: <https://drive.google.com/drive/folders/1mE1PbWiXQdcLjXUpciQ0ZqgIy92DIk00?usp=sharing>

## Entry point notebook

| File | Chức năng |
| --- | --- |
| `notebooks/MyGram_Colab.ipynb` | Chuẩn bị Colab, chạy test/gate CUDA và chạy hoặc resume ma trận thực nghiệm. |
| `notebooks/table_runs/MyGram_Table1_RunAll.ipynb` | Chạy/resume các thực nghiệm Table 1 và dựng báo cáo Table 1. |
| `notebooks/table_runs/MyGram_Table2_RunAll.ipynb` | Chạy/resume các thực nghiệm Table 2 và dựng báo cáo Table 2 mà không xóa Table 1. |
| `notebook_workflows.json` | Manifest thứ tự cell, đường dẫn script và SHA-256 của cả ba workflow. |

## Module trong `src/`

| File | Chức năng |
| --- | --- |
| `src/artifact_store.py` | Publish, restore và xác minh snapshot run bền vững. |
| `src/checkpoint.py` | Lưu/đọc checkpoint, RNG state, identity và đối soát resume. |
| `src/config.py` | CLI/config schema, profile và hash semantic/runtime. |
| `src/data_registry.py` | Registry năm dataset, đường dẫn, hash và JSON helper. |
| `src/data.py` | Load PMF data, sinh/persist split và metadata dữ liệu. |
| `src/evaluation.py` | Đánh giá hai chiều, CSLS blockwise và ghi prediction. |
| `src/graph.py` | Xây graph input sparse cho RRGAT và MCD. |
| `src/iterative_learning.py` | Trạng thái và cập nhật iterative learning/MNN. |
| `src/Load.py` | Loader feature, relation, attribute, image và text kế thừa từ upstream. |
| `src/loss.py` | InfoNCE, Gram/MGD loss và objective alignment. |
| `src/Mfusion.py` | MCD, modality fusion và các lớp attention. |
| `src/models.py` | RRGAT và encoder đa phương thức MyGram. |
| `src/run.py` | Vòng đời train/evaluate/checkpoint của một run MyGram. |
| `src/utils.py` | Tiện ích sparse, sampling, ranking và dữ liệu upstream. |
| `src/notebook_runner.py` | Chạy workflow script trong namespace dùng chung của notebook. |
| `src/__init__.py` | Đánh dấu package `src`. |

## Script vận hành

| File | Chức năng |
| --- | --- |
| `scripts/run_mygram.py` | Entry point mỏng gọi `src.run`. |
| `scripts/run_experiments.py` | Lập lịch, khóa, chạy/resume và ghi trạng thái ma trận thực nghiệm. |
| `scripts/run_colab_gates.py` | Chạy các gate CUDA G3/G4/G6 với snapshot contract. |
| `scripts/run_stage_g_pilot.py` | Chạy pilot Stage G và đối soát checkpoint. |
| `scripts/prepare_pmf_data.py` | Xác minh archive và chuẩn bị dữ liệu PMF. |
| `scripts/validate_data.py` | Validate registry, split, coverage và dữ liệu đã chuẩn bị. |
| `scripts/colab_bootstrap.py` | Khôi phục/cache archive và chuẩn bị data trong Colab. |
| `scripts/sync_run_artifacts.py` | Publish hoặc restore artifact của một run. |
| `scripts/build_colab_bundle.py` | Đóng gói source ZIP sạch kèm manifest hash. |
| `scripts/verify_colab_bundle.py` | Kiểm tra tính toàn vẹn và nội dung bundle. |
| `scripts/check_mygram_dependencies.py` | Kiểm tra dependency bắt buộc của MyGram. |
| `scripts/check_colab_environment.py` | Kiểm tra môi trường Linux/CUDA trên Colab. |
| `scripts/check_environment.py` | Kiểm tra môi trường local và sparse primitives. |
| `scripts/check_stage_c.py` | Gate cấu hình, import và model initialization. |
| `scripts/check_stage_d.py` | Gate sparse graph và full-graph smoke. |
| `scripts/check_stage_e.py` | Gate loss, optimizer và ablation. |
| `scripts/check_stage_f.py` | Gate evaluation, IL và checkpoint. |
| `scripts/check_stage_g.py` | Gate tích hợp/full-graph trước long run. |
| `scripts/inventory_resources.py` | Kiểm kê tài nguyên và nguồn dữ liệu khả dụng. |
| `scripts/inventory_windows.ps1` | Kiểm kê môi trường Windows. |
| `scripts/lock_environment.py` | Ghi khóa phiên bản môi trường. |

## Script được tách từ notebook

Các file dưới đây là nội dung nguyên văn của code cell tương ứng trước refactor.
Tiền tố số giữ thứ tự thực thi.

| File | Chức năng |
| --- | --- |
| `scripts/notebook_steps/common/03_install_dependencies.py` | Cài dependency lock và chạy dependency checker. |
| `scripts/notebook_steps/common/04_check_environment.py` | Ghi báo cáo môi trường Colab. |
| `scripts/notebook_steps/common/05_prepare_data.py` | Chạy bước chuẩn bị dữ liệu dùng chung. |
| `scripts/notebook_steps/base/06_run_unit_tests.py` | Chạy test repository và dry-run matrix. |
| `scripts/notebook_steps/base/08_gate_g3.py` | Chạy gate G3. |
| `scripts/notebook_steps/base/09_gate_g4.py` | Chạy gate G4. |
| `scripts/notebook_steps/base/10_gate_g6.py` | Chạy pilot/resume gate G6. |
| `scripts/notebook_steps/base/12_run_matrix.py` | Chạy/resume một long run trong ma trận. |
| `scripts/notebook_steps/base/13_show_matrix_status.py` | Hiển thị trạng thái ma trận và điều kiện dừng phiên. |
| `scripts/notebook_steps/table_common/07_patch_checkpointing.py` | Áp dụng checkpoint optimization dùng chung cho Table 1/2. |
| `scripts/notebook_steps/table_common/08_record_patch_diff.py` | Lưu diff provenance của patch. |
| `scripts/notebook_steps/table_common/10_validate_patch.py` | Chạy test trên source sau patch. |
| `scripts/notebook_steps/table_common/12_clear_stale_lock.py` | Kiểm tra và xóa stale runner lock an toàn. |
| `scripts/notebook_steps/table1/13_run_missing_experiments.py` | Chạy/resume các run Table 1 còn thiếu. |
| `scripts/notebook_steps/table1/15_build_report.py` | Tổng hợp và dựng report Table 1. |
| `scripts/notebook_steps/table1/16_final_status.py` | Xác nhận trạng thái hoàn tất Table 1. |
| `scripts/notebook_steps/table2/13_run_experiments.py` | Chạy/resume nhóm thực nghiệm Table 2. |
| `scripts/notebook_steps/table2/15_build_report.py` | Tổng hợp và dựng report Table 2. |
| `scripts/notebook_steps/table2/16_final_status.py` | Xác nhận Table 2 và bảo toàn dữ liệu Table 1. |

## Cấu hình và môi trường

| File | Chức năng |
| --- | --- |
| `configs/repaired_pmf.json` | Profile thực nghiệm PMF đã sửa. |
| `configs/author_code_compat.json` | Profile tương thích hành vi code tác giả. |
| `configs/stage_g_smoke.json` | Cấu hình smoke/gate Stage G. |
| `configs/experiments.json` | Ma trận experiment và quy tắc tái sử dụng run. |
| `requirements.in` | Dependency trực tiếp. |
| `requirements/colab.lock` | Dependency khóa cho Colab. |
| `requirements/windows-cpu-py313.lock` | Dependency khóa cho Windows CPU/Python 3.13. |
| `run.sh` | Launcher shell tương thích quy trình cũ. |

## Test

| File | Phạm vi |
| --- | --- |
| `tests/test_refactor_structure.py` | Cú pháp, checksum, thứ tự workflow, notebook sạch output và shared state; không cần NumPy/Torch. |
| `tests/test_experiment_matrix.py` | Danh tính và số lượng experiment trong matrix. |
| `tests/test_colab_portability.py` | Portability checkpoint và artifact giữa runtime. |
| `tests/test_config_model.py` | Config/profile và model initialization. |
| `tests/test_data.py` | Registry, loader, split và data provenance. |
| `tests/test_sparse_graph.py` | Sparse graph và oracle tương đương. |
| `tests/test_loss_optimizer_ablation.py` | Loss, optimizer và modality ablation. |
| `tests/test_evaluation_il_checkpoint.py` | Evaluation, iterative learning và checkpoint/resume. |
| `tests/test_stage_g.py` | Smoke/integration của Stage G. |
| `tests/fixtures/upstream_Load_selected.py` | Bản hàm upstream dùng làm oracle. |
| `tests/fixtures/upstream_Load_selected.json` | Hash/metadata fixture upstream. |

Chạy test cấu trúc trong bất kỳ Python 3 hiện đại nào:

```powershell
python -m unittest tests.test_refactor_structure -v
```

Chạy toàn bộ test trong môi trường đã cài dependency lock:

```powershell
python -m unittest discover -s tests -v
```

## Chạy trên Colab

Tạo source bundle mới sau khi thay đổi mã:

```powershell
python scripts/build_colab_bundle.py
```

Upload `dist/mygram-source.zip`, `dist/source_bundle_manifest.json` và archive
PMF vào đúng vị trí Drive được khai báo trong notebook. Sau đó mở một trong ba
notebook, chọn GPU runtime và chạy từ trên xuống.

## Tài liệu còn lại

| File | Chức năng |
| --- | --- |
| `REFACTOR_INPUTS.md` | Nguồn và phạm vi snapshot đầu vào. |
| `plan_pipeline.md` | Kế hoạch tái lập theo Stage A–H. |
| `chuyen_sang_colab.md` | Thiết kế lưu trữ, resume và vận hành Colab. |
| `docs/implementation_decisions.md` | Các quyết định kỹ thuật cần giữ ổn định. |
| `provenance/source_bundle_manifest.json` | Bằng chứng hash của source bundle đầu vào trước refactor. |

Các báo cáo chạy Stage A–F và `environment.json` đã được loại khỏi source tree.
Chúng là output lịch sử, không phải đầu vào cần thiết để hiểu hoặc chạy mã.
