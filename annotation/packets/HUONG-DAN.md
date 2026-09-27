# Hướng dẫn gán nhãn ViHoRec

Mỗi người nhận **một file riêng** trong thư mục này. Không mở file của người
khác và không trao đổi đáp án trong lúc làm — độ tin cậy của nghiên cứu phụ
thuộc vào việc hai người phán đoán độc lập.

Mở file bằng Excel hoặc Google Sheets. File đã lưu dạng UTF-8 có BOM nên tiếng
Việt hiển thị đúng. Chỉ điền hai cột `label` và `notes`, **không sửa các cột
khác và không đổi thứ tự dòng**.

---

## 1. `er_pairs_annotatorN.csv` — ghép cặp khách sạn

Câu hỏi cho mỗi dòng: **`name_a` và `name_b` có phải cùng một khách sạn không?**

Điền cột `label`:

- `1` — cùng một khách sạn
- `0` — hai khách sạn khác nhau

### Ghi `1` khi hai tên chỉ khác nhau ở:

- dấu tiếng Việt hoặc phiên âm: `Phú Quốc` / `Phu Quoc`
- từ chỉ loại hình ở đầu hay cuối: `Khách sạn X` / `X Hotel` / `Khu nghỉ dưỡng X`
- thứ tự từ hoặc dấu câu: `Phan Thiet Ocean Dunes Resort` / `Khu nghỉ dưỡng Ocean Dunes Phan Thiết`
- cách viết tên thương hiệu: `Mövenpick Resort Phan Thiet` / `Khu nghỉ dưỡng Movenpick Phan Thiết`

### Ghi `0` khi:

- **khác thành phố**, kể cả cùng thương hiệu: `Raon Hotel` Đà Nẵng vs `Raon Hotel` Quy Nhơn
- khác **hạng hoặc thương hiệu con**: `Mường Thanh Grand` vs `Mường Thanh Luxury`;
  `Majestic` vs `Majestic Premium`
- khác **số chi nhánh**: `La Cactus Hotel` vs `La Cactus Hotel 2`
- khác **loại hình thật sự** trong cùng khu: `Raon Hotel` vs `Raon Villa`;
  `Sea Links Beach` (khách sạn) vs `Sea Links Beach Villas`

### Khi không chắc

Tra tên khách sạn kèm tên thành phố trên chính nền tảng ghi ở cột `platforms_a`
/ `platforms_b`, rồi dán link vào `notes`. Nếu vẫn không xác minh được, ghi `0`
và ghi rõ lý do vào `notes` — không đoán bừa thành `1`.

Cột thành phố lấy từ dữ liệu thu thập nên đôi khi trống hoặc ghi chung chung.
Thành phố trống nghĩa là **không rõ**, không được coi là trùng khớp.

### Vì sao có nhiều cặp rõ ràng khác nhau?

Mẫu được rút theo tầng, cố ý trộn cả cặp dễ lẫn và cặp ngẫu nhiên. Tỉ lệ `1`
thấp là bình thường. Hãy chấm từng dòng độc lập, đừng cố cân bằng số `1` và `0`.

---

## 2. `qc_hotels_annotatorN.csv` — kiểm tra mục khách sạn

Câu hỏi: **mục này có mô tả đúng một khách sạn có thật tại địa điểm ghi kèm không?**

- `1` — tên chỉ đúng một cơ sở lưu trú có thật tại thành phố đó
- `0` — tên ở mức chuỗi/thương hiệu chung có thể trỏ tới nhiều cơ sở, hoặc địa
  điểm mâu thuẫn với tên, hoặc tên bị cắt cụt

---

## 3. Sau khi làm xong

Gửi lại file đã điền, giữ nguyên tên file. Người điều phối chạy:

```bash
python annotation_packets.py merge
```

Lệnh này gộp nhãn vào bảng gốc. Nhãn cuối cùng của mỗi dòng lấy theo **đa số**
phiếu. Những dòng **không thống nhất tuyệt đối** được ghi ra file
`*_adjudication.csv`; người điều phối điền cột `adjudicated` cho các dòng đó rồi
chạy `merge` lần nữa. Nhãn `adjudicated` luôn thắng phiếu đa số.

Quy trình đầy đủ và định nghĩa từng tiêu chí: `annotation/GUIDELINES.md`.
