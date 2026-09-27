# Audit ngẫu nhiên — hướng dẫn

File của bạn: `er_pairs_negative_audit.csv`

Đây là mẫu độc lập khoảng 100 cặp khách sạn. Câu hỏi **giống hệt** bài ghép
cặp, không phải bài QC khách sạn.

## Nguyên tắc

**Bạn sẽ không thấy nhãn có sẵn, và điều đó là cố ý.** Chấm từng dòng như thể
chưa ai từng chấm. Không cố cân bằng số `1` và `0`.

## Câu hỏi

Với mỗi dòng: **`name_a` và `name_b` có phải cùng một khách sạn vật lý không?**

Điền cột `label`: `1` = cùng một khách sạn, `0` = hai khách sạn khác nhau.

Quy tắc đầy đủ: `annotation/GUIDELINES.md` mục 2.1. Bản tiếng Việt ngắn:
`HUONG-DAN.md` mục 1. Tóm tắt:

- Khác **thành phố** thì luôn là `0`, kể cả trùng thương hiệu.
- Khác **hạng / thương hiệu con** (`Grand` vs `Luxury`, `Premier`) là `0`.
- Khác **số chi nhánh** (`X` vs `X 2`) là `0`.
- Chỉ khác dấu, phiên âm, từ chỉ loại hình, thứ tự từ thì là `1`.
- Khác **loại hình thật sự** (`Hotel` vs `Villa`) là `0`, trừ khi tra được đó là
  cùng một cơ sở.

## Khi không chắc

Tra tên kèm thành phố trên nền tảng ở cột `platforms_*`, dán link vào `notes`.
Nếu vẫn không xác minh được thì ghi `0` và nêu lý do — đừng đoán thành `1`.

## Xong rồi

Giữ nguyên tên file, gửi lại, rồi người điều phối chạy:

```bash
python annotation_packets.py merge
python er_evaluate.py
```
