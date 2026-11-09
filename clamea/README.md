# CLAMEA refactored experiment pipeline

Thư mục này là bản tái cấu trúc của notebook CLAMEA ban đầu. Nội dung thực thi
của 23 code cell được giữ nguyên và tách thành 23 script theo đúng thứ tự. Notebook
`clamea.ipynb` chỉ điều phối các script trong cùng một global namespace, vì vậy các
biến dùng chung giữa các cell vẫn có hành vi như bản gốc.

Các output đã lưu và execution counter trong notebook đều được xóa. Dự án không
chứa checkpoint, biểu đồ, bảng kết quả hoặc artifact sinh ra từ các lần chạy.

## Folder chứa toàn bộ kết quả chạy và minh chứng chạy:
Source folder: <https://drive.google.com/file/d/1TPScgcrvMi227GN7pqLmQ2uEmOUKBdvC/view?usp=drive_link>

## Cấu trúc

- `clamea.ipynb`: entry point duy nhất để chạy toàn bộ pipeline theo thứ tự.
- `scripts/`: mã nguồn được tách từ 23 cell gốc; tên file có tiền tố số thể hiện
  thứ tự chạy.
- `src/clamea_refactor/runner.py`: helper thực thi từng script trong namespace
  dùng chung của notebook.
- `pipeline_manifest.json`: thứ tự, mô tả và SHA-256 của từng script sau khi tách.
- `tests/`: test cấu trúc, cú pháp, checksum, notebook sạch output và cơ chế chia
  sẻ namespace.

## Chức năng từng script

| File | Chức năng |
| --- | --- |
| `00_environment.py` | Import, hằng số, helper tái lập và thông tin môi trường. |
| `01_data_setup.py` | Kết nối Google Drive, xác minh, giải nén và kiểm tra split dữ liệu. |
| `02_upstream_patch.py` | Cài dependency, lấy commit CLAMEA đã ghim và áp dụng source patch. |
| `03_smoke_test.py` | Smoke test GPU, backward pass và kiểm tra exact-ranking. |
| `04_preflight_report.py` | Tạo và tải báo cáo preflight kỹ thuật lên Drive. |
| `05_training_runner.py` | Sinh standalone training runner cho các bước sau. |
| `06_training_execution.py` | Điều phối baseline training có khóa và khả năng resume. |
| `07_pseudo_label_audit.py` | Đánh giá chất lượng pseudo-label, không huấn luyện. |
| `08_evaluation_stability.py` | Kiểm tra độ ổn định checkpoint và phép đánh giá lặp. |
| `09_training_reproducibility.py` | Chẩn đoán khả năng tái lập bằng lượt chạy ngắn. |
| `10_deterministic_training.py` | Cô lập negative sampling xác định. |
| `11_controlled_self_training.py` | Chẩn đoán self-training xác định có kiểm soát. |
| `12_root_cause_isolation.py` | Cô lập nguyên nhân sai khác tái lập. |
| `13_graph_augmentation.py` | Chẩn đoán graph augmentation trong workspace độc lập. |
| `14_final_preflight.py` | Đóng băng và xác minh protocol tái lập cuối. |
| `15_dependency_check.py` | Kiểm tra FAISS và import source CLAMEA đã patch. |
| `16_stage5_training.py` | Chạy một Stage 5 training job cho mỗi lần gọi. |
| `17_integrity_audit.py` | Audit tính toàn vẹn của 9 run, không train/test. |
| `18_final_test_evaluation.py` | Đánh giá 9 checkpoint đã chọn bằng Validation trên Test. |
| `19_final_analysis.py` | Tổng hợp bảng, biểu đồ và diễn giải từ artifact đã đóng băng. |
| `20_wandb_history_upload.py` | Đăng các kết quả lịch sử đã có lên W&B. |
| `21_wandb_report_table.py` | Chèn bảng kết quả vào W&B report hiện hữu một cách an toàn. |
| `22_wandb_report_evidence.py` | Thêm bằng chứng đầy đủ của 9 run vào report hiện hữu. |

## Cách chạy

Mở `clamea.ipynb` từ thư mục này hoặc từ thư mục cha của repo, sau đó chạy các
cell theo thứ tự. Notebook tự tìm project root và thêm `src/` vào `sys.path`.
Các yêu cầu về GPU, Google Drive, secret và W&B vẫn giữ nguyên như mã gốc.

Không chạy trực tiếp từng file riêng lẻ nếu file đó phụ thuộc biến từ bước trước.
Notebook là cách chạy được hỗ trợ để bảo toàn thứ tự và shared state.

## Chạy test

Từ thư mục `clamea`:

```powershell
python -m unittest discover -s tests -v
```

Test không tải dữ liệu, không dùng GPU, không train mô hình và không gọi dịch vụ
ngoài. Chúng chỉ xác minh cấu trúc cùng tính toàn vẹn của bản refactor.

## Nguyên tắc bảo toàn logic

- Mỗi script là nội dung nguyên văn của code cell tương ứng trong notebook cũ.
- Thứ tự chạy được khóa trong `pipeline_manifest.json` và notebook.
- SHA-256 giúp phát hiện thay đổi ngoài ý muốn sau khi tách file.
- Helper chỉ thay thế cơ chế cell-to-cell của notebook; không thay thuật toán,
  hyperparameter, seed, dữ liệu, checkpoint selection hay quy trình đánh giá.

