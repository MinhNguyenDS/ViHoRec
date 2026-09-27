# Xác minh của người — hướng dẫn

File của bạn: `er_pairs_human_verification.csv`

## Vì sao chỉ có chừng này dòng

Đây không phải mẫu ngẫu nhiên. Đây là **toàn bộ những dòng mà các con số công bố
phụ thuộc vào**: mọi cặp mà thuật toán có gộp, mọi cặp được kết luận là trùng, và
mọi cặp mà các bộ chấm tự động không thống nhất. Các dòng còn lại không chống đỡ
kết luận nào nên không cần bạn xem.

## Nguyên tắc quan trọng nhất

**Bạn sẽ không thấy nhãn của máy, và điều đó là cố ý.** Nếu biết máy chấm gì,
bạn sẽ vô thức chấm theo, và con số "người đồng ý với máy bao nhiêu phần trăm"
sẽ mất hết ý nghĩa. Hãy chấm như thể chưa ai từng chấm.

Nhãn của bạn **thắng** nhãn máy trong mọi trường hợp.

## Câu hỏi

Với mỗi dòng: **`name_a` và `name_b` có phải cùng một khách sạn vật lý không?**

Điền cột `label`: `1` = cùng một khách sạn, `0` = hai khách sạn khác nhau.

Quy tắc quyết định đầy đủ nằm ở `annotation/GUIDELINES.md` mục 2.1, và bản tiếng
Việt ngắn gọn ở `HUONG-DAN.md` mục 1. Tóm tắt:

- Khác **thành phố** thì luôn là `0`, kể cả trùng thương hiệu.
- Khác **hạng / thương hiệu con** (`Grand` vs `Luxury`, `Premier`) là `0`.
- Khác **số chi nhánh** (`X` vs `X 2`) là `0`.
- Chỉ khác dấu, phiên âm, từ chỉ loại hình, thứ tự từ thì là `1`.
- Khác **loại hình thật sự** (`Hotel` vs `Villa`) là `0`, trừ khi tra được đó là
  cùng một cơ sở.

## Khi không chắc

Tra tên kèm thành phố trên nền tảng ghi ở cột `platforms_*`, dán link vào
`notes`. Nếu vẫn không xác minh được thì ghi `0` và nêu lý do — đừng đoán thành
`1`, vì như vậy sẽ cộng điểm cho thuật toán ở một ca chưa được chứng minh.

Có vài dòng sẽ khó thật sự; đó chính là lý do chúng nằm trong danh sách này.
Ghi rõ suy nghĩ vào `notes` cho những dòng đó.

## Xong rồi

Giữ nguyên tên file, gửi lại, rồi người điều phối chạy:

```bash
python annotation_packets.py merge
python er_evaluate.py
```

Kết quả merge sẽ báo bạn đồng ý với máy bao nhiêu dòng và lật ngược dòng nào.
