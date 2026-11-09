# Kế hoạch  sửa và chạy lại MyGram với dữ liệu PMF

## 1. Mục tiêu và phạm vi

Sửa repository MyGram để chạy được pipeline nạp dữ liệu → encoder → fusion → loss → backward → iterative alignment → evaluation → lưu/resume → tổng hợp thực nghiệm, sử dụng **processed data từ PMF đã tải và kiểm tra**.

Kết quả phải được gọi là **MyGram chạy lại trên PMF processed data, với các sửa đổi được công bố**. Không khẳng định đây là bản tái lập chính xác số liệu paper hoặc bundle dữ liệu của tác giả MyGram.


### Phạm vi bắt buộc

- Sửa các lỗi đang chặn chạy và làm pipeline có thể kiểm thử trên CPU/CUDA.
- Dùng đủ 5 dataset đã kiểm tra, không tải lại từ đầu khi file còn nguyên vẹn.
- Giảm bộ nhớ bằng sparse graph và evaluation theo khối, giữ ngữ nghĩa được lựa chọn rõ ràng.
- Chạy smoke test thực sự trên dữ liệu thật trước các run 1000 epoch.
- Hỗ trợ các thí nghiệm MyGram trong Tables 1–3, Figures 3–4 và xuất dữ liệu phục vụ case study Figure 5.
- Lưu đầy đủ cấu hình, provenance, split, metrics, model, optimizer, RNG và pseudo-label để kiểm tra lại được.

### Ngoài phạm vi mặc định

- Huấn luyện lại toàn bộ các baseline khác trong paper.
- Tải lại ảnh thô, tự trích VGG hoặc biến vector DBP15K 2048 chiều thành vector 4096 chiều.
- Thêm LLM, MoE hoặc phương pháp mới để cải thiện kết quả.
- Tìm siêu tham số bằng test set để cố khớp số liệu paper.
- Deploy, push, tạo PR, gửi email/issue hoặc mua tài nguyên cloud.

## 2. Bối cảnh đã xác minh

Repository gốc: [HubuKG/MyGram](https://github.com/HubuKG/MyGram).
Commit làm mốc: `8d6bd6f4ddcd807034e53d2b7fccede86b60343c`.

Đọc các tệp sau trước khi triển khai, rồi đối chiếu với trạng thái code hiện tại:

- `audit/REPRODUCTION_AUDIT.md`
- `audit/processed_dataset_report.json`
- `audit/crosskg_provenance_report.json`
- `audit/pmf_archive_inventory.json`
- `src/run.py`, `src/models.py`, `src/Mfusion.py`, `src/loss.py`, `src/Load.py`, `src/utils.py`, `run.sh`
- `mygram.pdf` khi cần giải quyết một khác biệt giữa mô tả và implementation.

Nguồn processed data: [ninibymilk/PMF-MMEA](https://github.com/ninibymilk/PMF-MMEA), [archive Google Drive](https://drive.google.com/file/d/14CMfyg9q6ddnXGlxrj96RQZhbX9kwqJ6/view).

```text
Archive local: audit/sources/pmf_mmkg.zip
Size:          1381232555 bytes
SHA-256:       2477b8af38d108fe4a66ebfaf460f4176b5b13b00c4082ffeb0637123f3059c1
```

Các file text đã trích ở `audit/sources/processed/`; 5 pickle ảnh còn trong ZIP. Không suy luận layout ZIP từ tên archive: inventory đã cho thấy các thư mục `FBDB15K`, `FBYG15K`, `DBP15K`, `pkls` nằm trực tiếp trong archive.

| Dataset | Entity G1 | Entity G2 | Alignment pairs | Triples G1 | Triples G2 | Vector ảnh | Chiều ảnh |
|---|---:|---:|---:|---:|---:|---:|---:|
| FBDB15K | 14951 | 12842 | 12846 | 592213 | 99028 | 26281 | 4096 |
| FBYG15K | 14951 | 15404 | 11199 | 592213 | 122886 | 24638 | 4096 |
| zh_en | 19388 | 19572 | 15000 | 70414 | 95142 | 30035 | 2048 |
| ja_en | 19814 | 19780 | 15000 | 77214 | 93484 | 26480 | 2048 |
| fr_en | 19661 | 19993 | 15000 | 105998 | 115722 | 28032 | 2048 |

Đã kiểm tra các ID và vector, đồng thời thực thi các hàm đọc text/BoW của MyGram. **Chưa kiểm tra forward/backward/evaluation toàn mô hình**; việc này thuộc kế hoạch dưới đây.

Các lưu ý phải giữ trong báo cáo:

1. Không biết chắc PMF processed data có giống toàn bộ preprocessing của MyGram không.
2. DBP15K có ảnh 2048 chiều, khác 4096 chiều mô tả trong paper; không suy ra backbone chỉ từ kích thước.
3. FBDB15K có 12846 cặp nhưng chỉ 12842 target duy nhất. Seed 2023, rate 0.5 có target ID `15489`, `15490`, `24337` ở cả train và test qua các source khác nhau.
4. Ảnh processed cross-KG đã được đối chiếu bằng giá trị với MMKB gốc. Đây là chứng cứ nguồn ảnh, không phải chứng cứ toàn bộ protocol giống tác giả.

## 3. Quy tắc thực thi và quyết định mặc định


- Không sửa/xóa dữ liệu nguồn, không reset thay đổi của người dùng. Kiểm tra `git status` trước khi chỉnh sửa; tạo nhánh `codex/mygram-pmf-pipeline` nếu cần và phù hợp quyền hiện có.
- Lưu mốc upstream, diff và danh sách thay đổi có ảnh hưởng kết quả trong `docs/implementation_decisions.md`.
- Phân biệt sửa lỗi chạy, tối ưu tương đương về toán, và thay đổi protocol/thuật toán. Mỗi loại phải có bằng chứng riêng.
- Không có dữ liệu/metric/checkpoint thì ghi `not_run` hoặc `blocked` cùng lý do; không điền bằng số từ paper.
- Nếu vướng một giai đoạn, tiếp tục các phần độc lập còn làm được. Không đánh dấu toàn bộ nhiệm vụ hoàn tất chỉ vì code đã import được.
- Không tự tạo subagent theo mặc định; chỉ sử dụng nếu người dùng hoặc hướng dẫn áp dụng yêu cầu.

### 3.1 Hai profile hành vi, không trộn kết quả

Triển khai hai profile có config được lưu riêng:

| Thành phần | `repaired_pmf` — profile chính | `author_code_compat` — đối chứng implementation |
|---|---|---|
| Dataset | PMF nguyên ID và thứ tự cặp | Giống profile chính |
| LR đầu | 0.0005 theo run.sh | 0.0005 |
| Optimizer sau epoch index 50 | Giữ optimizer; đổi LR sang LR/5 đúng một lần | Tái tạo AdamW mỗi epoch từ index 50, như code công bố |
| Missing images | Giữ `load_img_new`: neighbor mean, fallback random chung | Giống profile chính |
| MCD self-loop | Giữ hai self-loop/node để tương đương implementation hiện tại | Giống profile chính |
| Gram inputs | `gph_emb1` trước RRGAT; 3 modality sau MCD | Giống profile chính |
| Gram scaling | Giữ `0.1 × (loss_one_direction/2)` | Giống profile chính |
| Positive ngoài top-K | Bỏ hàng thiếu positive; log coverage | Giống profile chính |
| Không hàng nào có positive | Gram đóng góp zero khả vi; NCE vẫn học | Giống sửa lỗi an toàn này; phải ghi đây là deviation tối thiểu |
| K lớn hơn batch | `K_eff=min(K,B)` | Giống sửa lỗi an toàn này |
| Evaluation chính | Candidate theo test pairs; best-test và last đều được báo | Giống profile chính |

`author_code_compat` là bản sửa tối thiểu để đối chiếu hành vi code, **không phải** bản gốc hoạt động nguyên trạng hoặc cam kết bitwise giống tác giả.

Mặc định chạy bộ thí nghiệm chính bằng `repaired_pmf`. Sau smoke test, chạy thêm một cặp đối chứng đầy đủ trên FBDB15K rate 0.2 seed 2023 giữa hai profile để tách ảnh hưởng của sửa optimizer. Không dùng kết quả đối chứng để chọn profile nào được báo cáo theo hướng đẹp hơn.

Các thay đổi như một self-loop, random ảnh theo paper, LR 0.005, Gram dùng hidden states, force-positive top-K, loss đối xứng, loại duplicate target hoặc đánh giá full graph đều là **thí nghiệm bổ sung**, không âm thầm áp dụng vào profile chính.

## 4. Giai đoạn A — kiểm kê tài nguyên và khóa môi trường

### A1. Kiểm kê lại máy ở thời điểm thực thi

Kiểm tra Python, package manager, CUDA/driver, GPU/VRAM, RAM, dung lượng ổ đĩa, các environment và container có sẵn. Audit trước thấy RTX 3060 Laptop 6 GiB, hai Python chưa có ML dependencies, WSL chỉ có `docker-desktop`, ổ C còn khoảng 1.8–1.9 GB sau tải dữ liệu. Đây là snapshot cũ, phải đo lại.

- Tính dung lượng cần từ các entry ZIP thực sự sẽ dùng, package wheels, environment và checkpoint dự kiến.
- Không giải nén toàn bộ archive gồm các dataset không cần. Không cài PyTorch CUDA khi chưa đủ dung lượng và headroom.
- Không xóa raw MMKB hoặc archive để lấy chỗ nếu chưa có chỉ định. Khi thiếu chỗ, hoàn thành sửa code/test nhẹ trước và báo chính xác dung lượng/đường dẫn cần bổ sung.
- Không coi `nvidia-smi` ghi CUDA 13.1 là CUDA toolkit hoặc torch CUDA đã được cài.

### A2. Môi trường thực thi

Ưu tiên môi trường Python cô lập có PyTorch CUDA tương thích máy. Không bắt buộc dùng WSL nếu Windows native đã đáp ứng được backend sau sửa. Không dùng distro nội bộ `docker-desktop` như Ubuntu làm việc.

Thiết kế backend MCD chính bằng `torch.sparse` ở giai đoạn D để bỏ phụ thuộc DGL khỏi đường chạy chính. Với RRGAT, ưu tiên phép scatter/index-add native tương đương nếu kiểm thử đạt; nếu giữ `torch-scatter`, khóa wheel tương thích. Không thay cả GNN bằng thư viện/kiến trúc khác.

Khi chọn phiên bản cụ thể, kiểm tra tài liệu chính thức và wheel thực có ở thời điểm thực thi. Chỉ ghi phiên bản là “đã kiểm thử” sau khi import và chạy sparse forward/backward trên máy thành công.

Đầu ra:

- `requirements.in`: dependencies thực sự được dùng.
- Lock môi trường tương ứng OS/backend đã chạy; lưu URL/index wheel CUDA nếu cần.
- `environment.json`: Python, torch, torch CUDA, GPU, driver, OS, các package version.
- `scripts/check_environment.py`: kiểm tra thiết bị, sparse matmul/backward, package imports, disk và RAM.

**Gate A:** có môi trường import thành công và chạy được phép toán GPU cần thiết trước smoke test CUDA. CPU test vẫn được thực hiện nếu GPU/environment chưa sẵn sàng.

## 5. Giai đoạn B — chuẩn hóa data registry và provenance

### B1. Viết `scripts/prepare_pmf_data.py`

- Nhận `--archive` và `--data-root`; kiểm tra size/SHA-256 archive so với mốc đã biết. Nếu khác thì dừng dùng mốc thống kê cũ và audit bản mới, không bỏ qua mismatch.
- Giải nén chọn lọc 7 file text/dataset và 5 pickle ảnh. Kiểm tra đường dẫn nằm trong root đích trước khi ghi. Chỉ đọc pickle có nguồn đã kiểm tra, giữ cách đọc giới hạn lớp NumPy như audit nếu khả thi.
- Tái sử dụng file đã có nếu checksum khớp. Nếu khác, báo conflict và tránh ghi đè mất dấu dữ liệu.
- Giữ nguyên entity IDs, relation IDs, thứ tự alignment pairs và nội dung ảnh. Không sort/deduplicate toàn bộ dữ liệu để “làm sạch”.
- Cho phép data root trên ổ có đủ dung lượng; không hard-code vị trí máy hiện tại trong code.
- Xuất `data_manifest.json`: source URL, archive hash, file hashes, shape/count, duplicate/overlap diagnostics và phiên bản script chuẩn bị.

Layout đích:

```text
<data-root>/
  FBDB15K/norm/
  FBYG15K/norm/
  DBP15K/zh_en/
  DBP15K/ja_en/
  DBP15K/fr_en/
  pkls/
    FBDB15K_id_img_feature_dict.pkl
    FBYG15K_id_img_feature_dict.pkl
    zh_en_GA_id_img_feature_dict.pkl
    ja_en_GA_id_img_feature_dict.pkl
    fr_en_GA_id_img_feature_dict.pkl
```

Mỗi thư mục dataset có `ent_ids_1`, `ent_ids_2`, `triples_1`, `triples_2`, `ill_ent_ids`, `training_attrs_1`, `training_attrs_2`.

### B2. Data registry

Tạo registry explicit, không suy ra dataset bằng substring hoặc basename `norm`:

| Key CLI | Thư mục tương đối | Pickle tương đối |
|---|---|---|
| `FBDB15K` | `FBDB15K/norm` | `pkls/FBDB15K_id_img_feature_dict.pkl` |
| `FBYG15K` | `FBYG15K/norm` | `pkls/FBYG15K_id_img_feature_dict.pkl` |
| `zh_en` | `DBP15K/zh_en` | `pkls/zh_en_GA_id_img_feature_dict.pkl` |
| `ja_en` | `DBP15K/ja_en` | `pkls/ja_en_GA_id_img_feature_dict.pkl` |
| `fr_en` | `DBP15K/fr_en` | `pkls/fr_en_GA_id_img_feature_dict.pkl` |

CLI thống nhất: `--data-root`, `--dataset`, `--device`, `--output-root`, `--profile`, `--seed`, `--rate`. Nếu giữ `--file_dir` cũ, báo rõ alias/deprecation và lỗi khi xung đột; không silently chọn một trong hai.

### B3. Validation trên loader thực

Kiểm tra đủ file, ID disjoint/contiguous, endpoint hợp lệ, ảnh float32/finite và số chiều theo registry. Tạo `img_fc` từ chiều ảnh thực; không pad/cắt vector DBP15K để giả 4096.

Giữ BoW top 1000 và tie order theo code/dữ liệu; mọi thay đổi về thứ tự vocab đều cần hash và ghi nhận. Các hàng zero BoW được thống kê, không tự coi là hỏng.

Missing-image strategy giữ chính xác cả hướng neighbor và thứ tự lấy RNG theo implementation. Tách mask ảnh gốc và ảnh imputed để báo coverage; không sửa pickle nguồn khi impute.

Tạo split ngay ở vị trí tương ứng code gốc, trước thao tác dùng NumPy RNG khác. Với compatibility, kiểm tra shuffle bằng seed 2023 khớp audit. Lưu ordered train/test pairs, train/test indices và hash. Mọi run ablation cùng dataset/rate/seed dùng cùng split cố định.

**Gate B:** 5 dataset qua validator bằng loader sau refactor; statistics và split mốc khớp audit. Duplicate targets được ghi rõ thay vì bị loại âm thầm.

## 6. Giai đoạn C — sửa khởi tạo, import và cấu hình

### C1. Sửa lỗi thiếu class

- Kiểm tra lại đường forward của `RRGAT.emb_atten_cat`, `MultiModalEncoderMrFusion.joint_encoder` và `r_encoder`.
- Loại các lời gọi `MultiModalFusion` và `InteractionMoE` nếu xác nhận chúng không tham gia forward/loss. Không sáng tác `InteractionMoE` mới. Ghi rõ quyết định và nguồn commit xóa class.
- Không xóa hàng loạt các module thừa khác trong cùng patch. Các parameter không dùng nhưng có random initialization có thể làm đổi RNG và embedding khởi tạo; ghi nhận khi phải bỏ, không hứa bitwise upstream fidelity.
- Test tất cả parameter được kỳ vọng học có gradient; lập danh sách các parameter thừa/gradient None có giải thích.

### C2. Imports và entry point

- Đổi wildcard imports/bare `except` che lỗi thành imports rõ ràng, tương đối trong package.
- Hỗ trợ entry point chuẩn `python -m src.run`; cập nhật `run.sh` gọi từ root bất kể cwd người dùng.
- Cung cấp launcher Python cross-platform; shell wrapper chỉ là tiện ích.
- Tạo output directory bằng `mkdir(parents=True, exist_ok=True)` trước khi mở log/prediction.
- Không nuốt ImportError rồi báo nhầm dataset error. Lỗi nêu đúng file/dependency/config.

### C3. Thiết bị và config

- Thay `.cuda()`/`g.to('cuda')` hard-code bằng device của tensor/model đã cấu hình.
- Dùng buffer cho các constant tensor cần theo model/device, hoặc scalar Python nếu phù hợp, ví dụ gamma MCD.
- Cờ modality rõ nghĩa: `--disable-image`, `--disable-attribute`, `--disable-relation`; ánh xạ cờ cũ có cảnh báo. Name/char mặc định tắt và không nạp GloVe.
- Bổ sung `--disable-mgd`, `--disable-gram`, `--gram-k`, `--gram-temperature`, `--gram-weight`, `--gram-direction-scale`, `--optimizer-reset-policy`, `--mcd-self-loops`, `--attention-dropout`, `--output-dropout`.
- Loại hoặc báo deprecated đối với các flag không có tác dụng; không để config hứa hành vi mà forward không dùng.

Config mốc đầy đủ: hidden 300, RRGAT depth 2, Transformer 2 layer/5 head/intermediate 400, BoW 1000, MCD k=2/alpha=0.7/beta=0.3, module dropout theo run.sh=0, attention/output dropout=0.1, batch loss=3500, NCE tau=0.1, Gram K=10/tau=0.05, outer Gram weight=0.1, direction scale=0.5, IL start index=50/step=5, CSLS k=3, 1000 epochs, eval mỗi 50 epoch, seed=2023.

Kiểm tra thực tế AdamW weight decay ở code gốc: không giả định CLI `--weight_decay=0` đã được dùng nếu constructor không truyền nó. Khi refactor phải khai báo và lưu giá trị effective rõ ràng, giữ giá trị effective upstream cho profile mốc.

**Gate C:** `--help`, resolve config, load 5 dataset và khởi tạo model được; cấu hình đầy đủ có structural output 900 chiều và joint output 1800 chiều.

## 7. Giai đoạn D — graph sparse và kiểm chứng tương đương

### D1. Bỏ dense adjacency N×N

Thay `get_adjr2`/`MCD.build_graph` bằng dựng edge list một lần từ triples:

1. Với triple h≠t, đưa cả h→t và t→h vào tập cạnh.
2. Deduplicate cạnh có hướng. Code hiện dùng `adj.nonzero()`, nên multiplicity của các triple không trở thành trọng số MCD.
3. Giữ hai self-loop/node ở profile mốc: diagonal từ `get_adjr2` cộng self-loop do DGL thêm.
4. Nếu coalesce self-loop thành một entry, value phải bằng 2; degree phải tính đúng multiplicity. Không normalize như thể chỉ có một loop.
5. Tạo hệ số symmetric degree normalization như `MCD.norm_adj`. Với torch sparse, kiểm tra rõ orientation ma trận `[destination, source]` để khớp message passing.
6. Cache normalized sparse operator cho mỗi dataset/device; không chuyển graph CPU↔GPU mỗi forward.

Tính lần lượt `H_l = beta × A_norm @ H_(l-1) + alpha × H_0`, chia gamma sau k bước, giữ dropout placement. Không làm mất gradient theo H_0 hoặc Linear projections.

Sử dụng dense reference **chỉ trên tiny graph** làm oracle độc lập. Nếu có DGL environment phù hợp, đối chiếu thêm DGL upstream; không bắt buộc cài DGL chỉ để có một dependency không còn dùng.

### D2. RRGAT và các graph khác

Giữ graph attention, relation reflection và relation index alignment của RRGAT. Nếu thay `torch_scatter.scatter_sum` bằng `index_add`, dùng output size N explicit và kiểm tra cả node không nhận message. Không áp dụng dedup MCD một cách tùy tiện lên relation triples/r_index của RRGAT.

Tối ưu các mảng dense trung gian trong `load_data_dual/get_matrix` nếu gây áp lực RAM, nhưng phải kiểm tra output sparse features/r_index/r_val khớp reference nhỏ. Không truyền đồng thời cả graph dense cũ và sparse mới vào model.

### D3. Tiêu chí tương đương

Tiny fixtures bao gồm duplicate triples, hai quan hệ trên cùng edge, reverse edge, self-triples, node cô lập, graph một node. So sánh outputs và gradients của MCD/RRGAT trước–sau tối ưu với cùng weights/dropout off. Mục tiêu FP32 ban đầu `atol=1e-6, rtol=1e-5`; nếu GPU sparse reduction cần tolerance khác phải giải thích bằng sai số đo được.

Kiểm tra bộ nhớ không xuất hiện tensor N×N của graph toàn bộ dataset. Ghi peak allocated/reserved VRAM ở forward, backward, evaluation và IL riêng.

Nếu vẫn OOM 6 GiB: ưu tiên giải phóng tensor thừa, block evaluation/Gram candidates, rồi checkpoint activation hoặc edge chunking có gradient test. Không detach tensor đang học. Không giảm hidden size, số hop hoặc batch contrastive âm thầm. Nếu vẫn không đủ, báo nhu cầu GPU khác dựa trên phép đo, không cam kết 6 GiB chắc chắn chạy được.

**Gate D:** kiểm thử tương đương qua; full graph sparse forward/backward được kiểm tra trước long run.

## 8. Giai đoạn E — loss, optimizer và ablation

### E1. InfoNCE và Gram cơ sở

- Giữ InfoNCE trên joint embedding normalized, một chiều, positives theo vị trí cặp của batch mốc.
- Giữ Gram trên source structure trước RRGAT và target modality sau MCD. Không đưa Transformer hidden states vào Gram dưới danh nghĩa sửa lỗi kỹ thuật.
- `K_eff=min(K,B)`, batch rỗng không gọi loss, B=1 được xử lý hữu hạn và có diagnostic vì không có negative hữu ích.
- Khi một phần positive ngoài top-K, chỉ mean trên các hàng valid như upstream và log tỷ lệ valid; khi không hàng valid, trả zero gắn computational graph để NCE backward bình thường.
- Xóa tính toán các modality phía trái hoặc target structure không dùng nếu việc xóa không thay đổi output/RNG.
- Giữ quy tắc volume `sqrt(clamp(abs(det(G)), min=1e-8))` cho mốc. Gram determinant và softmax phải được tính ở độ chính xác ít nhất FP32 khi sau này thử AMP.
- Test finite loss/grad với vectors gần phụ thuộc tuyến tính, duplicate targets, B<K và zero-valid positives. Nếu FP32 determinant không ổn định trên dữ liệu thật, thử FP64 riêng cho Gram nhỏ như thay đổi số học được ghi rõ; không đổi loss im lặng.

Log `loss_nce`, `loss_gram_raw_one_direction`, `loss_gram_scaled`, `loss_total`, `gram_valid_rows`, `gram_batch_rows`, `K_eff`, số batch Gram không có positive và grad norm. Không dùng force-positive mặc định.

### E2. Training step và optimizer

- `repaired_pmf`: khởi tạo AdamW một lần; đổi LR qua param groups đúng một lần khi epoch index đạt 50. Giữ optimizer moments và log sự kiện đổi LR.
- `author_code_compat`: mô phỏng rõ chính sách reset mỗi epoch sau ngưỡng; resume cũng phải áp dụng đúng chính sách này.
- Giữ semantics code hiện tại: full-graph forward một lần/epoch, tính loss theo các batch links, cộng gradient, optimizer step một lần/epoch. Upstream cộng các batch mean; không tự đổi thành mean toàn dataset hoặc optimizer step mỗi minibatch.
- Không tích lũy loss tensor có graph chỉ để logging; chuyển giá trị log sang scalar. `retain_graph` chỉ giữ khi còn batch loss dùng graph, giải phóng ở backward cuối.
- Phân biệt `loss_batch_size` và kích thước khối tính toán. Giảm `loss_batch_size` thay đổi negative pool/Gram candidates; gradient accumulation thông thường không tự khôi phục loss cũ.
- Nếu cần tiling loss, giữ nguyên tập negative của batch logic và test loss/grad với implementation không tile.

### E3. Modality ablation và component ablation

- Bỏ MGD: truyền modality Linear output thẳng vào fusion/Gram, vẫn giữ structural encoder.
- Bỏ Gram: không tính Gram; giữ NCE, batch/split và phần còn lại.
- Bỏ image/relation/attribute: không gọi `.shape` hoặc indexing trên None; fusion nhận đúng số modality còn lại; structural dimension giữ nguyên.
- Vì Gram gốc cần đủ 4 vector, Table 3 không có implementation ablation đầy đủ trong upstream. Mặc định triển khai `gram_modalities=available`: structure + các target modality còn lại, dùng Gram 3×3 khi bỏ một modality, giữ scale đã công bố. Với đủ modality phải khớp Gram 4×4 cũ.
- Candidate miner mốc là structure→image. Khi bỏ image, chọn cố định structure→relation; nếu relation cũng không có thì attribute. Không đọc ảnh bị ablate chỉ để chọn candidate. Log `gram_candidate_modality`.
- Gọi các run modality ablation là **ablation được vận hành hóa trong bản sửa**, không tuyên bố chính xác protocol Table 3 tác giả. Bổ sung sensitivity `disable_gram_when_missing` chỉ khi cần; không thay thế kết quả chính mà không ghi nhận.
- Chỉ hỗ trợ combination đã định nghĩa; thiếu tất cả target modalities thì fail config sớm, không tạo loss giả.

**Gate E:** loss/grad hữu hạn, optimizer policy đúng, full + 3 modality ablation + 2 component ablation đều qua forward/backward; mốc Gram đủ 4 modality khớp reference ở trường hợp reference xác định được.

## 9. Giai đoạn F — evaluation, iterative learning và checkpoint

### F1. Evaluation theo khối

- Giữ normalization, distance và công thức CSLS đang được dùng trong code; đọc lại `pairwise_distances` và `csls_sim` trước khi thay. Không thay metric bằng cosine trực tiếp vì tiện.
- Candidate mốc là ordered test pairs như upstream, bao gồm target trùng theo vị trí. Report phải ghi `candidate_protocol=test_pairs`.
- Tính CSLS top-k trung bình cho cả hai trục trên **toàn candidate set**, dùng block passes; không tính CSLS riêng từng block rồi ghép vì sai neighborhood.
- Tính rank dựa trên toàn bộ candidates; xuất Hits@1/10/50, MRR, MR cho l2r/r2l. Nếu có tie, định nghĩa rule deterministic và log; kiểm tra các trường hợp duplicate target riêng. Không hứa tie ordering cross-device giống `torch.sort` upstream.
- So sánh evaluator dense/reference trên tiny input và subset thật, cả CSLS bật/tắt và tie cases. Khi dataset có target trùng, báo vấn đề diagonal ground truth cùng kết quả, không dùng một rule alias mới mà vẫn gắn nhãn protocol cũ.
- Lưu `last` và `best_test` riêng. Best-test là protocol upstream và có thiên lệch lựa chọn; không gọi nó validation-best.

Có thể thêm sensitivity evaluation theo unique entity/nhóm equivalent và split theo connected components alignment để tránh target overlap, nhưng phải là protocol khác với manifest và denominators riêng. Không cần chặn toàn bộ run mốc để tự thay dataset.

### F2. Iterative learning

- Giữ mutual nearest neighbor trên hai pool non-train, lịch kiểm tra mỗi 5 epoch sau ngưỡng và lịch giao/thêm cặp như upstream. Dùng block distance để không giữ toàn ma trận.
- Giữ mode train/eval theo profile đã định nghĩa: code upstream không gọi `.eval()` riêng trước pseudo-label, nên không silently chuyển sang eval; expose/lưu `il_model_mode` và mặc định tương thích train mode. `no_grad` không tự tắt dropout.
- Lưu ordered pool IDs; không tái tạo thứ tự từ set khi resume. Chặn duplicate pair, handle pool rỗng, và ghi số lượng candidate/pseudo-label mỗi chu kỳ.
- Labels test chỉ được dùng cho diagnostic precision sau lựa chọn; không lọc pseudo-label bằng ground truth, không dùng nhãn held-out để tune threshold.
- Bảo toàn test set gốc khi thêm pseudo-label vào train trong transductive IL; giải thích protocol trong báo cáo.

### F3. Checkpoint/resume

Mỗi run có `run_id` gồm dataset/rate/seed/profile/ablation/config hash. Không overwrite run hoàn tất do trùng timestamp.

```text
outputs/<run_id>/
  config.resolved.json
  environment.json
  data_manifest.json
  code_revision.json
  split.npz
  metrics.jsonl
  events.jsonl
  status.json
  checkpoints/last.pt
  checkpoints/best_test.pt
  predictions/best_test_l2r.tsv
  predictions/last_l2r.tsv
  summary.json
```

Checkpoint chứa model, optimizer/scheduler nếu dùng, epoch kế tiếp, best metrics/epoch, Python/NumPy/torch CPU/CUDA RNG, current ordered train links, fixed test links, ordered non-train pools, `new_links` đang giao giữa các chu kỳ, effective config và hashes. Lưu atomic bằng temporary file trong cùng output filesystem rồi rename. Giữ tối đa last/best mặc định để tiết kiệm disk.

Khi resume kiểm tra config/hash, không tự load checkpoint khác dataset/profile. Kiểm tra resume tại epoch thường, sát mốc LR 50 và giữa chu kỳ pseudo-label. Trên CPU deterministic, so sánh run liên tục với run dừng/resume; trên CUDA dùng tolerance và công bố giới hạn nondeterminism nếu backend không deterministic.

**Gate F:** evaluation block đúng với oracle; IL không truy cập test labels để quyết định; resume bảo toàn trạng thái và không tạo metric/pseudo-label sai lịch.

## 10. Giai đoạn G — kiểm thử và profiling theo tầng

Không chạy ngay 1000 epoch. Mỗi gate thất bại phải được sửa hoặc báo giới hạn có bằng chứng trước bước phụ thuộc.

| Tầng | Công việc | Tiêu chí qua |
|---|---|---|
| G0 | Parse/import/config/data validation | Không lỗi import/path; 5 dataset đúng manifest |
| G1 | Tiny graph và synthetic links | Outputs/grad sparse tương đương; Gram edge cases; ablation hoạt động |
| G2 | Tiny deterministic training + resume | Loss hữu hạn; model cập nhật; checkpoint khôi phục đúng |
| G3 | Full graph thực, ít links, 1–2 epoch trên từng dataset | Forward/backward/eval thành công; có peak memory/timing |
| G4 | Full graph + batch logic 3500 hoặc train size nếu nhỏ hơn | Không OOM; loss, gradient và prediction hữu hạn |
| G5 | IL kiểm thử chức năng bằng lịch rút ngắn riêng | Đã thực thi tìm/giao/thêm pseudo-label và resume giữa chu kỳ |
| G6 | Pilot ~60 epoch FBDB15K rate 0.2 và zh_en rate 0.3 | Đi qua mốc LR/IL thực; log đủ, finite, checkpoint/eval đúng lịch |

G5 phải được gắn nhãn smoke config, không lấy metric của nó làm số nghiên cứu. Nếu đủ nguồn lực, pilot 100 epoch để đi qua cả chu kỳ thêm pseudo-label theo lịch gốc. Nếu chưa có cặp tự nhiên được thêm, dùng fixture kiểm thử riêng cho nhánh mutation, không ép cặp ground-truth vào run thật.

Profile G3/G4 trên full graph dù links ít; không kết luận GPU đủ từ graph đồ chơi. Ghi wall time/epoch và ước lượng runtime còn lại dựa trên đo thực tế. Chạy tuần tự trên GPU 6 GiB; không mở đồng thời nhiều training jobs.

## 11. Giai đoạn H — ma trận thực nghiệm

Tạo `configs/experiments.json` và `scripts/run_experiments.py` để sinh config, dry-run, chọn nhóm, chạy tuần tự, resume và bỏ qua run đã hoàn tất hợp lệ. Runner ghi `planned/running/completed/failed/blocked`, error summary và đường dẫn log. Run failed không tính như completed.

### H1. Kết quả chính, seed 2023, profile repaired_pmf

| Nhóm | Dataset | Train rate | Số run |
|---|---|---|---:|
| Table 1 | FBDB15K, FBYG15K | 0.2, 0.5, 0.8 | 6 |
| Table 2 | zh_en, ja_en, fr_en | 0.3 | 3 |

Mỗi run 1000 epoch, IL bật, loss batch 3500, CSLS k=3, eval mỗi 50 epoch. Không early-stop mặc định vì thay đổi budget/protocol upstream. Các lỗi NaN/OOM/disk đầy dừng run và giữ last checkpoint.

### H2. Thành phần và modality

| Nhóm | Thiết lập | Số run mới |
|---|---|---:|
| Table 3 | Hai cross-KG, rate 0.8, lần lượt bỏ relation/attribute/image | 6 |
| Figure 3 | Hai cross-KG × rates 0.2/0.5/0.8 × bỏ MGD hoặc Gram | 12 |
| Figure 4 | Hai cross-KG × rates 0.05/0.10/0.15/0.20/0.25/0.30, full model | 10 mới + 2 tái sử dụng H1 |

Tổng cấu hình repaired_pmf không trùng: **37 run** cho một seed. Tái sử dụng chỉ khi toàn bộ config/split/hash khớp, không chỉ vì cùng rate. Thêm 1 run `author_code_compat` FBDB15K 0.2 làm đối chứng optimizer; run repaired tương ứng đã nằm H1. Tổng mặc định: **38 long runs**, chưa tính pilot/smoke.

Không giảm budget hoặc bỏ một nhóm để báo “toàn bộ pipeline hoàn tất”. Nếu compute/disk không đáp ứng, runner vẫn chuẩn bị đủ manifest và báo chính xác các run còn pending; không tự thuê tài nguyên hoặc mua dịch vụ.

Sau bộ seed 2023, kế hoạch độ ổn định bổ sung là 9 cấu hình H1 cho seeds 2024 và 2025, tức 18 run thêm. Không gộp pilot hoặc profile khác vào mean/std. Đây là phần mở rộng, không bắt buộc cho bộ 38 run mặc định.

### H3. Case study

- Tìm entity Shanghai trong mapping bằng URI/alias thực tế, xác minh cặp target có trong dataset và test split được chọn.
- Dùng checkpoint của run đầy đủ phù hợp; xuất top candidates, ranks, scores và modality coverage.
- Nếu Shanghai nằm trong train, không đổi split để đạt hình đẹp. Chọn một test case khác và ghi rõ, hoặc chạy một case protocol riêng.
- Không tự viết rank của MEAformer/PMF/IBMEA nếu chưa có prediction tương ứng. Số từ paper nếu dùng phải gắn nguồn, tách khỏi số chạy lại.

## 12. Giai đoạn I — tổng hợp và cách đọc kết quả

Viết `scripts/summarize_results.py` tạo bảng từ `summary.json/metrics.jsonl`, không parse thủ công một vài dòng log tốt nhất.

Mỗi hàng tối thiểu gồm dataset, rate, seed, profile, ablation, input image dim, split hash, candidate protocol, selection rule, best epoch, completed epochs, l2r/r2l Hits@1/10/MRR, runtime, peak VRAM, status và run_id.

Xuất:

- `results/main_results.json` và bảng Markdown tương ứng Tables 1–2.
- `results/modality_ablation.md`, ghi quy tắc Gram 3×3/candidate fallback.
- `results/component_ablation.md`, `results/low_resource.md`.
- `results/optimizer_control.md` cho hai profile.
- `results/case_study.md` và prediction files có ID/URI.
- `results/reproduction_report.md`: khác biệt data/code/paper, môi trường, cách chạy, limitations, bảng failed/pending.

Nếu so với paper, lưu số paper trong một bảng tham chiếu có nguồn riêng. Báo chênh lệch Hits@1 bằng **điểm phần trăm**; không dùng dấu % mơ hồ. Không suy diễn chênh lệch là do một module khi cùng lúc thay dataset/protocol/optimizer.

Các phát hiện duplicate target và best-test selection phải xuất hiện trong limitations. Không thay chúng bằng tuyên bố chung “random seed khác” để giải thích mọi sai lệch.

## 13. Danh sách thay đổi dự kiến theo file

| File/nhóm | Trách nhiệm |
|---|---|
| `src/run.py` | CLI/config, device, training policy, IL, orchestration, checkpoint |
| `src/Load.py` | Registry/path resolution hoặc gọi module data mới; validation/imputation |
| `src/models.py` | Bỏ lời gọi class thiếu không dùng, sparse graph input, modality None handling |
| `src/Mfusion.py` | MCD torch sparse, buffer/device, fusion active modalities, dropout config |
| `src/loss.py` | Gram edge cases, logging components, available-modality Gram, giữ reference 4-vector |
| `src/utils.py` | Bỏ dense graph khỏi đường chạy, block distances/CSLS/ranking hoặc tách evaluation module |
| `src/data.py`, `src/evaluation.py`, `src/checkpoint.py` nếu cần | Tách các trách nhiệm để kiểm thử, tránh viết lại model không cần thiết |
| `run.sh` | Wrapper gọi entry point chuẩn, không hard-code tuyệt đối |
| `scripts/prepare_pmf_data.py` | Giải nén chọn lọc và manifest |
| `scripts/check_environment.py`, `scripts/validate_data.py` | Preflight môi trường/dữ liệu |
| `scripts/run_experiments.py`, `scripts/summarize_results.py` | Matrix runner/resume/report |
| `configs/` | Profile mốc, smoke, matrix với overrides explicit |
| `tests/` | Tests có ý nghĩa về toán, gradient, split, Gram, rank, IL và resume |
| `docs/implementation_decisions.md`, README | Hướng dẫn chạy cùng các deviation đã chọn |

## 14. Hợp đồng lệnh cần cung cấp sau khi triển khai

```bash
python scripts/check_environment.py --device cuda:0
python scripts/prepare_pmf_data.py --archive audit/sources/pmf_mmkg.zip --data-root data/mmkg
python scripts/validate_data.py --data-root data/mmkg --all
python -m pytest tests
python -m src.run --config configs/smoke.json --dataset FBDB15K --data-root data/mmkg --device cuda:0
python scripts/run_experiments.py --matrix configs/experiments.json --group main --dry-run
python scripts/run_experiments.py --matrix configs/experiments.json --group all --resume
python scripts/summarize_results.py --output-root outputs --report-root results
```

Runner matrix phải lưu hoặc nhận data root/device/output root rõ ràng; chạy từ cwd khác vẫn resolve config/path nhất quán. README có lệnh cho môi trường thực sự đã kiểm thử, không chỉ một shell command giả định Linux.

## 15. Checklist nghiệm thu cuối cùng

- [ ] Mốc upstream và mọi thay đổi có ảnh hưởng thực nghiệm được ghi lại.
- [ ] PMF archive/file hashes và statistics của 5 dataset được xác minh.
- [ ] Không còn class thiếu, path hard-code hoặc `.cuda()` ngoài config.
- [ ] Không tạo dense graph N×N; sparse forward và gradients được đối chiếu oracle.
- [ ] Gram B<K/zero-valid/near-singular và modality ablation chạy được.
- [ ] Hai optimizer profile được kiểm thử và không trộn kết quả.
- [ ] Evaluator CSLS block dùng toàn candidate set, rank/ties có test.
- [ ] IL và resume qua cả mốc LR và giữa chu kỳ pseudo-label.
- [ ] Full-graph smoke trên cả 5 dataset, có peak VRAM và thời gian thực đo.
- [ ] 38 long runs mặc định có trạng thái rõ ràng; chỉ báo hoàn tất thực nghiệm khi tất cả required runs hoàn tất.
- [ ] Có best-test/last metrics, checkpoint, config, environment, split, predictions và manifest.
- [ ] Báo cáo phân biệt kết quả tự chạy với số paper và nêu DBP15K 2048 chiều, FBDB duplicate targets, best-test selection, Gram ablation extension.
- [ ] README có đường chạy lại từ data archive tới kết quả, không phụ thuộc trạng thái ngầm của phiên làm việc.

Nếu chưa hoàn tất training vì tài nguyên, báo riêng: **implementation đã kiểm thử đến gate nào**, **dataset nào chạy end-to-end**, **run nào đã hoàn tất**, **run nào còn thiếu** và **điều kiện cần để tiếp tục**. Không đánh đồng loader chạy được với toàn pipeline đã thành công.
