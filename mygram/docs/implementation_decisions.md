# Nhật ký triển khai MyGram trên PMF processed data

## 2026-09-10 — Giai đoạn A

- Upstream và HEAD trước triển khai: `8d6bd6f4ddcd807034e53d2b7fccede86b60343c`.
- Trước chỉnh sửa có `audit/`, `mygram.pdf`, `plan_pipeline.md` chưa track. Giữ nguyên dữ liệu và các tệp này; không reset, không commit/push.
- Phạm vi A: kiểm kê, dependency lock và kiểm thử primitive. Không thay `src/` hoặc `run.sh`; diff mã mô hình so với upstream là rỗng.
- Chọn Windows native CPython 3.13.0 sẵn có, venv không kế thừa site-packages. Python MSYS2 3.11.6 không có ML dependencies, không dùng để cài wheel Windows CPython.
- Chọn torch 2.7.1, NumPy 2.2.5, SciPy 1.15.3, transformers 4.51.3. Đây là phiên bản của bản chạy lại, không phải môi trường tác giả. Wheel và dependency closure được resolver kiểm tra thực tế; lock lưu URL + SHA-256.
- Môi trường CPU là fallback được phép bởi A. CUDA 12.6/torch 2.7.1 là ứng viên có wheel CPython 3.13 Windows; chưa cài do ngân sách dung lượng. Không coi driver CUDA 13.1 là toolkit hoặc torch CUDA.
- `requirements.in` chỉ phục vụ backend native torch dự kiến sau C/D và preflight hiện tại. DGL và torch-scatter vẫn được upstream import; chưa sửa những import đó trong A. Không tuyên bố môi trường này import/khởi tạo toàn bộ MyGram được.
- Không cài torchvision/torchaudio: repo dùng vector ảnh có sẵn. Không cài h5py vào runtime chính: audit nguồn HDF5 là công việc riêng.
- Chưa thay thuật toán/protocol hoặc tối ưu mô hình. Sparse oracle trong preflight chỉ xác minh khả năng backend, chưa phải kiểm chứng MCD/RRGAT ở Gate D.
- Chưa chạy dataset end-to-end, smoke mô hình hoặc long run; tất cả ở trạng thái `not_run`.

Nguồn chính thức kiểm tra ngày 2026-09-10:

- https://pytorch.org/get-started/previous-versions/ (torch 2.7.1, CPU và CUDA 12.6 Windows).
- https://download.pytorch.org/whl/cu126/torch/ (wheel cp313-win_amd64 thực tồn tại).
- `audit/stage_a_cpu_resolution.json` chứa metadata và URL wheel từ PyPI/PyTorch cho toàn bộ dependencies.

## 2026-09-10 — Giai đoạn B

- Xác minh lại archive bằng size + SHA-256; giải nén đúng 40 entry vào `data/mmkg`.
  Không sửa raw MMKB/archive hoặc text audit. File đích tồn tại chỉ tái sử dụng khi hash khớp;
  mismatch dừng với conflict. Không giải nén các dataset khác trong ZIP.
- `src/data_registry.py` là bảng 5 key explicit. Không suy tên pickle từ basename `norm`.
  `src/data.py:load_dataset` dùng lại các hàm đọc text/BoW thực trong `src/Load.py` và `src/utils.py`.
  `src/run.py:init_data` đã chuyển sang loader này; entry point/model vẫn bị lỗi C/D gốc,
  nên chưa gọi đây là training end-to-end.
- `load_attr/load_relation` chỉ thêm tùy chọn trả ordered vocab. Giữ top 1000, stable tie
  order, dtype và giá trị; hash vocabulary và matrix được lưu. Không sửa các hàng zero BoW.
- `load_img_new` dùng NumPy whitelist unpickler và nhận dictionary đã được loader kiểm tra
  checksum/ID/shape/dtype. Chỉ thêm RNG injectable và metadata mask; thuật toán neighbor
  mean (hai hướng, có multiplicity) + một vector fallback chung giữ nguyên, không cascade
  từ ảnh vừa điền sang entity khác. Fixture so với hàm upstream đúng commit cho output
  và RNG kế tiếp giống hệt. Không sửa pickle nguồn.
- Ảnh nguồn float32. Upstream ghép vector fallback từ `np.random.normal` tạo array
  float64 khi có fallback: giữ hành vi này, training vẫn chuyển qua `torch.Tensor`
  float32 và normalize như trước. Không pad/cắt DBP15K 2048 chiều. `init_model` vốn đã
  truyền `self.img_features.shape[1]` vào `img_fc`, nên không cần sửa layer trong B.
- Split dùng permutation index với RandomState MT19937, trước draw cho imputation.
  Đối chiếu cả ordered pairs và RNG tiếp theo với `np.random.shuffle` trên list gốc.
  Training truyền global NumPy RNG đã seed để không làm thay đổi draw sequence phía sau;
  validator dùng RNG riêng. Artifact indices/pairs int64; training vẫn chuyển pairs int32.
- Lưu 19 split NPZ/JSON seed 2023 (8 rates/cross-KG + rate .3 trên 3 DBP15K), hash
  từng array và file. Paths không có profile/ablation: cùng dataset/rate/seed dùng cùng
  split; nếu nội dung khác dừng, không ghi đè. Hai profile có data policy giống nhau.
- FBDB15K duplicate targets và overlap giữ nguyên. Không chỉnh protocol để loại chúng.
- CLI validator nhận `--data-root`, `--dataset`/`--all`, `--device cpu`, `--output-root`,
  `--profile`, `--seed`, `--rate`; alias `--file_dir` chỉ được phép nếu khớp đường dẫn registry.
  Training mới nối các tham số data/split; CLI thiết bị/config/optimizer profile đầy đủ
  vẫn thuộc C/E, chưa tuyên bố hoạt động chỉ vì validator có flag tương ứng.
- Kiểm thử và giới hạn xem `docs/stage_b_report.md`; diff phần source so với upstream
  lưu trong `audit/stage_b_source.diff`. Không có metric nghiên cứu/checkpoint mô hình.

## 2026-09-10 — Giai đoạn C

- Xóa đúng ba initializer không tham gia forward/loss: `RRGAT.emb_atten_cat`,
  `MultiModalEncoderMrFusion.r_encoder`, `MultiModalEncoderMrFusion.joint_encoder`.
  Commit `82bf5ee887037c9696c33bbf276649ecb87e9f3a` xóa định nghĩa
  `MultiModalFusion` nhưng giữ lời gọi; không tìm thấy `InteractionMoE` trong lịch sử.
  Không dựng lớp thay thế. Việc bỏ `r_encoder` đổi RNG khởi tạo so với code lỗi;
  không tuyên bố bitwise fidelity với upstream.
- Giữ `rel_fc_dual` và phần value/output/feed-forward của Transformer layer cuối dù
  tiny backward chứng minh chúng không nhận gradient. `MFusion` chỉ dùng attention
  probabilities của layer cuối làm trọng số nên các phần đó không ảnh hưởng output.
  Giữ lại để tránh mở rộng patch kiến trúc/RNG; danh sách exact ở validation report.
- Sửa `MFusion` flatten ba modality `[N,3,300]` thành `[N,900]` trước khi concat
  structure `[N,900]`; code công bố cố concat tensor rank 2 với rank 3. Full config
  hiện có structural 900 và joint 1800.
- Bỏ wildcard imports và bare `except`; imports nội bộ dùng package-relative. DGL là
  optional import chỉ để C có thể import/init; yêu cầu DGL forward báo lỗi cụ thể.
  Thay dependency `torch_scatter.scatter_sum` bằng `index_add_` với output size N
  explicit để node không nhận message có hàng zero. Tiny gradient passed; đối chiếu
  oracle đầy đủ cho RRGAT vẫn thuộc Gate D.
- MCD `gamma/gammas` là registered buffers; DGL graph theo device của adjacency,
  không chuyển CPU rồi hard-code CUDA. InfoNCE label theo device của score. Type ID
  CUDA không dùng đã bỏ. Không còn `.cuda()`/`to('cuda')` trong `src/`.
- Tạo config resolver JSON+CLI. CLI override config; unknown key, invalid dimension,
  invalid device string, profile/policy mismatch và `--file_dir` conflict đều fail
  sớm. Thiết bị CUDA không có backend sẽ fail, không fallback CPU. Hai profile lưu
  riêng; `repaired_pmf` giảm LR đúng một lần và giữ optimizer, compatibility reset
  AdamW mỗi epoch từ index 50.
- Effective AdamW weight decay upstream là default constructor **0.01** vì cờ
  `--weight_decay=0` chưa được truyền. Bản sửa đặt/lưu 0.01 explicit trong cả profile.
- Cờ canonical modality/MGD/Gram và các hệ số Gram/MCD/dropout đã nối vào code.
  Name/char mặc định tắt và không nạp GloVe. Các cờ upstream không ảnh hưởng forward
  bị bỏ khỏi public parser; alias store-false cũ được ẩn và cảnh báo đúng semantics.
- `python -m src.run` là entry point chuẩn; `scripts/run_mygram.py` là launcher Windows;
  `run.sh` tự chuyển về repo root, tạo output/log directories và dùng CLI canonical.
  Resolved config được ghi trước khi khởi tạo training.
- Gate C kiểm tra 5 dataset thật và model cardinality/image dimension; không tạo graph
  N×N. Tiny forward/backward dùng `--disable-mgd` để cô lập phụ thuộc Gate D. Không
  coi đó là MCD/full-graph smoke hoặc kết quả nghiên cứu.

## 2026-09-10 — Giai đoạn D

- Thay `get_adjr2` dense và `MCD.build_graph` DGL bằng `src/graph.py`. Pipeline chỉ
  giữ edge/relation arrays, entity-average sparse COO và MCD normalized sparse COO.
  Hai operator là model buffers nên được cache và chuyển device đúng một lần cùng model.
- MCD tạo tập cạnh có hướng duy nhất từ hai chiều của mọi triple không phải self.
  Triple lặp, nhiều relation trên cùng pair và cạnh đảo không tăng trọng số MCD.
  Profile mốc giữ multiplicity hai self-loop/node: diagonal cũ cộng loop do DGL thêm.
  Degree được tính trước coalesce theo multiplicity; entry diagonal normalized là
  `2/degree`. Orientation operator là `[destination, source]`.
- MCD giữ recurrence `beta * A_norm @ h + alpha * h0`, gamma theo số hop và đúng vị trí
  x/edge/z dropout. Edge dropout được lấy một lần rồi dùng lại cho mọi hop/modality,
  giống một `g.local_scope()` của forward cũ.
- RRGAT dùng graph riêng: edge có hướng unique nhưng relation incidence giữ toàn bộ
  duplicate và nhiều relation, mỗi contribution có trọng số `1/count(edge)`. Relation
  đảo vẫn cộng offset bằng base relation size. `index_add_` có output size N explicit;
  test xác nhận node cô lập nhận hàng zero sau message passing.
- `src/Load.py:get_matrix` chỉ còn làm scipy oracle cho graph tối đa 4096 node; gọi lớn
  hơn dừng để tránh vô tình tạo dense N×N. `src/utils.py:get_adjr2` dừng với hướng dẫn
  dùng builder sparse, nhằm ngăn đường runtime cũ quay lại.
- Tiny oracle gồm triple lặp, hai relation trên cùng edge, cạnh đảo explicit, self-triple,
  node cô lập và graph một node. MCD/RRGAT output và gradient FP32 qua tolerance
  `atol=1e-6, rtol=1e-5`; graph RRGAT/entity averaging cũng khớp scipy reference.
- Kiểm kê 5/5 dataset không có dense graph N×N. Full `zh_en` CPU forward/backward qua
  với N=38.960, RRGAT E=259.545, MCD nnz=298.096, output 900/1800, 30 parameter tensors
  có gradient hữu hạn. Lần xác nhận gần nhất đo peak RSS process 7.485.583.360 bytes,
  tăng 7.213.830.144 bytes so với đầu pha full-graph (sampling 50 ms).
- Không cài DGL chỉ để đối chiếu phụ; native dense oracle độc lập đã qua. CUDA/VRAM
  forward, backward, evaluation và IL chưa đo vì torch hiện tại là CPU-only. Gate D
  chỉ xác nhận full graph model forward/backward, chưa phải epoch training, evaluator,
  IL, checkpoint hay metric nghiên cứu.

## 2026-09-10 — Giai đoạn E

- Giữ InfoNCE một chiều structure/joint left→right với positive theo vị trí hàng trong
  batch. Không force positive vào top-K và không đổi thành loss đối xứng.
- Gram vẫn nhận source structure trước RRGAT cùng target image/attribute/relation sau
  MCD. Với đủ modality, wrapper 4-vector và gradient khớp một bản copy độc lập của
  công thức công bố ở `atol=1e-6, rtol=1e-5`.
- Dùng `K_eff=min(K,B)`. Batch rỗng được `alignment_objective` đánh dấu skip trước khi
  gọi InfoNCE/Gram. B=1 trả loss hữu hạn và diagnostic không có negative hữu ích.
  Khi chỉ một số hàng có positive trong top-K, mean chỉ trên các hàng valid như code
  gốc; khi không hàng nào valid, trả `volumes.sum()*0` để giữ graph cho NCE backward.
- Volume giữ `sqrt(clamp(abs(det(G)), min=1e-8))`. Input FP16/BF16 được promote lên
  FP32 cho similarity, Gram determinant và softmax; FP64 input vẫn giữ FP64. Không có
  lý do đo được để chuyển riêng determinant sang FP64 ở profile mốc.
- Log từng batch gồm NCE, Gram raw một chiều, Gram sau direction scale, total, valid
  rows, batch rows, K_eff, zero-positive count, candidate modality và grad norm.
  Accumulator epoch chỉ giữ Python float, không giữ tensor graph.
- Training giữ một full-graph forward/epoch, cộng các batch mean, backward từng batch,
  và một optimizer step/epoch. `retain_graph=True` chỉ đến batch áp chót.
- `repaired_pmf` đổi LR 0,0005→0,0001 tại epoch index 50 trên cùng AdamW và giữ state.
  `author_code_compat` tạo AdamW mới mỗi epoch index ≥50, state rỗng và LR 0,0001.
  Test resume giả lập xác nhận state được giữ/reset đúng policy. Phát hiện và sửa việc
  param-group cũ có thể giữ khóa LR 0,0005 bằng cách clone group rồi override LR rõ ràng.
- `gram_modalities=available`: full dùng Gram 4×4; bỏ một modality dùng Gram 3×3.
  Candidate order cố định image→relation→attribute. Full + ba modality ablation + bỏ
  MGD + bỏ Gram đều qua tiny forward/loss/backward với gradient hữu hạn.
- Ba modality ablation là protocol được vận hành hóa theo kế hoạch vì upstream không
  cung cấp đường Gram đầy đủ cho Table 3; không gọi chúng là tái tạo chính xác Table 3.
  Gate E chưa chạy epoch trên dataset thật, evaluation, IL, checkpoint hay metric paper.

## 2026-09-10 — Giai đoạn F

- Evaluator giữ L2 normalization và score `1 - squared_l2_distance`. Candidate
  protocol là ordered `test_pairs`, ground truth theo vị trí đường chéo; duplicate ID
  không bị deduplicate. Tie chính xác dùng vị trí candidate thấp hơn.
- CSLS row/column top-k được tích lũy bằng block trên toàn candidate set trước khi
  rank; không ghép các neighborhood cục bộ. Dense oracle khớp block ở CSLS on/off,
  tie/duplicate fixtures và 96 cặp thật `zh_en` dùng PMF image features.
- Lưu l2r/r2l Hits@1/10/50, MR, MRR và prediction TSV. `last` tách khỏi `best_test`;
  best-test theo l2r Hits@1 để tương thích upstream và luôn ghi thiên lệch chọn trên test.
- IL dùng mutual nearest neighbor blockwise trên ordered pools. `train_compat` là mode
  mặc định; `no_grad` không đổi dropout. Test labels chỉ vào diagnostic sau selection,
  fixed test không đổi. Duplicate, pool rỗng và chu kỳ candidate/intersection/add có test.
- Run ID khóa dataset/rate/seed/profile/ablation/config hash. Artifact directory lưu
  resolved config, environment, data/code/split provenance, metrics/events/status,
  last/best checkpoint, predictions và summary. Run hoàn tất không bị overwrite.
- Checkpoint atomic lưu model, optimizer/scheduler, next epoch, best state, toàn bộ
  ordered IL state, config/hashes và RNG Python/NumPy/torch CPU/CUDA. Resume mismatch
  fail sớm. CPU continuous/resume khớp bitwise qua epoch 50 ở cả hai optimizer profile
  và replay giữa chu kỳ IL cho kết quả state giống hệt.
- Gate F qua 9 test riêng và toàn bộ 40 test. CUDA, full-dataset model evaluation,
  epoch training/IL end-to-end và metric nghiên cứu chưa chạy; các phần đó thuộc Gate G.
