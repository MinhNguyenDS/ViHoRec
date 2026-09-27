"""Split annotation work into per-annotator packets and merge the labels back.

Independence is the property that makes Cohen's kappa meaningful, so each
annotator receives a file containing only their own label column: no other
annotator's labels, no similarity score, and no pipeline decision. Rows are
shuffled per annotator as well, which keeps a shared scroll position from
turning into a shared judgement when two people work side by side.

``merge`` writes the packets back into the master sheets, reports coverage and
disagreement, and emits an adjudication file holding only the contested rows.
Running ``merge`` again picks the adjudicated labels up.

Any panel size works. With two annotators a disagreement leaves no gold label
and must be adjudicated. With three or more, a majority vote already resolves
every item, but items that were not unanimous are still routed to the
adjudicator: a 2-1 split is precisely where a shared blind spot surfaces, and
that matters most when the annotators are language models.

``verify`` builds a much smaller blind packet holding only the rows a headline
number actually rests on, for a human to label from scratch. Their labels
override the panel, and the merge step reports how often the two agreed.

``audit`` draws a uniform random sample of unanimous panel negatives that are
*not* in the verification packet. The annotator-facing sheet is still blind:
it does not say the panel voted 0, because that would anchor the labels.

Usage:
    python annotation_packets.py split --annotators 3
    python annotation_packets.py merge
    python annotation_packets.py verify
    python annotation_packets.py audit --n 100
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

import config as C
from er_evaluate import _clean_labels, agreement_report, annotator_columns, resolve_gold

PACKET_DIR = C.OUT_ANNOTATION / "packets"
MAX_ANNOTATORS = 5
DEFAULT_ANNOTATORS = C.N_ANNOTATORS

STUDIES = {
    "er_pairs": {
        "master": "er_pairs_to_annotate.csv",
        "id_col": "pair_id",
        "show": [
            "pair_id",
            "name_a", "city_a", "platforms_a",
            "name_b", "city_b", "platforms_b",
        ],
        "question": "1 = cùng một khách sạn, 0 = hai khách sạn khác nhau",
    },
    "qc_hotels": {
        "master": "qc_sample_to_annotate.csv",
        "id_col": "record_id",
        "show": ["record_id", "hotel_id", "name", "location"],
        "filter": lambda df: df["record_type"] == "hotel",
        "question": "1 = đúng một khách sạn có thật tại địa điểm này, 0 = không phải",
    },
}


def _load_master(spec: dict) -> pd.DataFrame:
    path = C.OUT_ANNOTATION / spec["master"]
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "filter" in spec:
        df = df[spec["filter"](df)].copy()
    return df


def cmd_split(n_annotators: int = DEFAULT_ANNOTATORS) -> dict:
    PACKET_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for study, spec in STUDIES.items():
        master = _load_master(spec)
        if master.empty:
            out[study] = {"skipped": f"{spec['master']} not found or empty"}
            continue
        cols = [c for c in spec["show"] if c in master.columns]
        files = []
        for i in range(1, n_annotators + 1):
            packet = master[cols].sample(frac=1.0, random_state=100 + i).reset_index(drop=True)
            packet["label"] = ""
            packet["notes"] = ""
            path = PACKET_DIR / f"{study}_annotator{i}.csv"
            packet.to_csv(path, index=False, encoding="utf-8-sig")
            files.append(str(path))
        out[study] = {"n_rows": int(len(master)), "packets": files}
    (PACKET_DIR / "HUONG-DAN.md").write_text(_instructions(), encoding="utf-8")
    out["instructions"] = str(PACKET_DIR / "HUONG-DAN.md")
    out["n_annotators"] = n_annotators
    return out


def _present_slots(study: str) -> list[int]:
    """Annotator slots that actually have a packet file on disk."""
    return [
        i for i in range(1, MAX_ANNOTATORS + 1)
        if (PACKET_DIR / f"{study}_annotator{i}.csv").exists()
    ]


def _strata(study: str, master: pd.DataFrame, id_col: str) -> pd.Series:
    key_path = C.OUT_ANNOTATION / "er_pairs_key.csv"
    if study != "er_pairs" or not key_path.exists():
        return pd.Series("", index=master.index, dtype="string")
    key = pd.read_csv(key_path).set_index("pair_id")["stratum"]
    return master[id_col].map(key).astype("string").fillna("")


def load_bearing_rows(study: str, master: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Rows whose label a headline number actually rests on.

    Three groups, and they overlap heavily:

    * every pair in an exhaustively annotated stratum where a matcher merged —
      precision and the matcher-vs-matcher comparison are computed entirely
      from these, with no sampling weight;
    * every positive, since a wrong `1` anywhere moves recall;
    * every item the panel did not agree on.

    Everything else is a unanimous negative carrying no claim on its own, and
    is better handled by a random audit than by exhaustive review.
    """
    ann_cols = annotator_columns(master)
    labels = pd.DataFrame({c: _clean_labels(master[c]) for c in ann_cols})
    gold = resolve_gold(master)
    strata = _strata(study, master, id_col)

    reasons: dict[int, list[str]] = {i: [] for i in master.index}
    for i in master.index[strata == "algo_merge"]:
        reasons[i].append("exhaustive_stratum")
    for i in master.index[gold == 1]:
        reasons[i].append("positive")
    contested = labels.notna().all(axis=1) & labels.nunique(axis=1).gt(1)
    for i in master.index[contested]:
        reasons[i].append("panel_disagreed")

    picked = [i for i, r in reasons.items() if r]
    out = master.loc[picked].copy()
    out["why"] = ["+".join(reasons[i]) for i in picked]
    out["stratum"] = strata.loc[picked]
    return out


def cmd_verify() -> dict:
    """Build a blind packet of the load-bearing rows for a human.

    The packet deliberately hides the model labels. Showing them would anchor
    the adjudicator, and the headline the study needs — how often a person
    agrees with the panel — would then be unmeasurable.
    """
    PACKET_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for study, spec in STUDIES.items():
        master_path = C.OUT_ANNOTATION / spec["master"]
        if not master_path.exists():
            continue
        master = pd.read_csv(master_path)
        id_col = spec["id_col"]
        if not annotator_columns(master):
            out[study] = {"status": "no labels yet; run the panel first"}
            continue

        rows = load_bearing_rows(study, master, id_col)
        if rows.empty:
            out[study] = {"status": "nothing load-bearing yet"}
            continue

        cols = [c for c in spec["show"] if c in master.columns]
        packet = rows[cols].sample(frac=1.0, random_state=C.RANDOM_SEED).reset_index(drop=True)
        packet["label"] = ""
        packet["notes"] = ""
        packet_path = PACKET_DIR / f"{study}_human_verification.csv"
        packet.to_csv(packet_path, index=False, encoding="utf-8-sig")

        # Why each row was selected stays with the coordinator, not the annotator.
        key_path = PACKET_DIR / f"{study}_human_verification_key.csv"
        rows[[id_col, "why", "stratum"]].to_csv(key_path, index=False, encoding="utf-8")

        out[study] = {
            "packet": str(packet_path),
            "coordinator_key": str(key_path),
            "n_rows": int(len(rows)),
            "by_reason": rows["why"].value_counts().to_dict(),
        }
    (PACKET_DIR / "HUONG-DAN-XAC-MINH.md").write_text(_verify_instructions(), encoding="utf-8")
    out["instructions"] = str(PACKET_DIR / "HUONG-DAN-XAC-MINH.md")
    return out


def cmd_audit(n: int = 100) -> dict:
    """Blind random sample of unanimous panel negatives, outside the verify set.

    A clean audit of *n* items bounds the panel's residual error rate at
    roughly 3/*n* (95% rule of three). The coordinator key records that the
    rows were drawn from unanimous negatives; the packet given to the person
    does not.
    """
    PACKET_DIR.mkdir(parents=True, exist_ok=True)
    spec = STUDIES["er_pairs"]
    master_path = C.OUT_ANNOTATION / spec["master"]
    if not master_path.exists():
        return {"er_pairs": {"status": f"{spec['master']} not found"}}

    master = pd.read_csv(master_path)
    id_col = spec["id_col"]
    ann_cols = annotator_columns(master)
    if not ann_cols:
        return {"er_pairs": {"status": "no labels yet; run the panel first"}}

    labels = pd.DataFrame({c: _clean_labels(master[c]) for c in ann_cols})
    gold = resolve_gold(master)
    rated_by_all = labels.notna().all(axis=1)
    unanimous = rated_by_all & labels.nunique(axis=1).eq(1)
    negative = gold == 0

    bearing = load_bearing_rows("er_pairs", master, id_col)
    held_out = set(bearing[id_col].astype(str)) if len(bearing) else set()
    pool = master[unanimous & negative & ~master[id_col].astype(str).isin(held_out)]
    if pool.empty:
        return {"er_pairs": {"status": "no unanimous negatives left outside the verify packet"}}

    n_take = min(int(n), len(pool))
    sampled = pool.sample(n=n_take, random_state=C.RANDOM_SEED + 17)
    cols = [c for c in spec["show"] if c in master.columns]
    packet = sampled[cols].sample(frac=1.0, random_state=C.RANDOM_SEED + 18).reset_index(drop=True)
    packet["label"] = ""
    packet["notes"] = ""
    packet_path = PACKET_DIR / "er_pairs_negative_audit.csv"
    packet.to_csv(packet_path, index=False, encoding="utf-8-sig")

    strata = _strata("er_pairs", sampled, id_col)
    key = sampled[[id_col]].copy()
    key["stratum"] = strata.values
    key["panel_gold"] = 0
    key_path = PACKET_DIR / "er_pairs_negative_audit_key.csv"
    key.to_csv(key_path, index=False, encoding="utf-8")

    instr_path = PACKET_DIR / "HUONG-DAN-AUDIT.md"
    instr_path.write_text(_audit_instructions(), encoding="utf-8")
    return {
        "er_pairs": {
            "packet": str(packet_path),
            "coordinator_key": str(key_path),
            "n_rows": n_take,
            "pool_size": int(len(pool)),
            "held_out_load_bearing": int(len(held_out)),
        },
        "instructions": str(instr_path),
    }


def _read_packet(study: str, i: int, id_col: str) -> pd.DataFrame | None:
    path = PACKET_DIR / f"{study}_annotator{i}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "label" not in df.columns:
        return None
    return df[[id_col, "label", "notes"]].rename(columns={"label": f"annotator_{i}"})


def cmd_merge() -> dict:
    out = {}
    for study, spec in STUDIES.items():
        master_path = C.OUT_ANNOTATION / spec["master"]
        if not master_path.exists():
            continue
        master = pd.read_csv(master_path)
        id_col = spec["id_col"]
        merged_any = False

        for i in _present_slots(study):
            packet = _read_packet(study, i, id_col)
            if packet is None:
                continue
            merged_any = True
            labels = packet.set_index(id_col)[f"annotator_{i}"]
            master[f"annotator_{i}"] = (
                master[id_col].map(labels).astype("string").fillna("")
            )
            notes = packet.set_index(id_col)["notes"]
            mapped_notes = master[id_col].map(notes).astype("string").fillna("")
            existing = master.get("notes", pd.Series("", index=master.index)).astype("string").fillna("")
            master["notes"] = (existing + " " + mapped_notes).str.strip()

        if not merged_any:
            out[study] = {"status": "no packets found"}
            continue

        # Human labels win over the panel, from either route: the tie-break
        # sheet, or the blind verification packet built by `verify`.
        human = pd.Series(pd.NA, index=master.index, dtype="string")
        adj_path = PACKET_DIR / f"{study}_adjudication.csv"
        if adj_path.exists():
            adj = pd.read_csv(adj_path)
            if "adjudicated" in adj.columns:
                human = master[id_col].map(adj.set_index(id_col)["adjudicated"]).astype("string")

        n_audited = 0
        audit_path = PACKET_DIR / f"{study}_negative_audited.csv"
        if not audit_path.exists():
            audit_path = PACKET_DIR / f"{study}_negative_audit.csv"
        if audit_path.exists():
            aud = pd.read_csv(audit_path)
            if "label" in aud.columns:
                a = master[id_col].map(aud.set_index(id_col)["label"]).astype("string")
                n_audited = int(_clean_labels(a).notna().sum())
                human = a.where(a.notna() & (a != ""), human)

        verify_path = PACKET_DIR / f"{study}_human_verification.csv"
        n_verified = 0
        if verify_path.exists():
            ver = pd.read_csv(verify_path)
            if "label" in ver.columns:
                v = master[id_col].map(ver.set_index(id_col)["label"]).astype("string")
                n_verified = int(_clean_labels(v).notna().sum())
                human = v.where(v.notna() & (v != ""), human)
        master["adjudicated"] = human.fillna("")

        ann_cols = annotator_columns(master)
        labels = pd.DataFrame({c: _clean_labels(master[c]) for c in ann_cols})
        rated_by_all = labels.notna().all(axis=1) & (len(ann_cols) > 0)
        # Any item the panel did not agree on unanimously, whatever the panel size.
        contested = rated_by_all & labels.nunique(axis=1).gt(1)

        pending = master[contested]
        if "adjudicated" in master.columns:
            pending = pending[_clean_labels(pending["adjudicated"]).isna()]
        if len(pending):
            cols = [c for c in spec["show"] if c in master.columns]
            sheet = pending[cols + ann_cols + ["notes"]].copy()
            sheet["adjudicated"] = ""
            sheet.to_csv(adj_path, index=False, encoding="utf-8-sig")
        elif adj_path.exists() and not contested.any():
            adj_path.unlink()

        master.to_csv(master_path, index=False, encoding="utf-8")
        out[study] = {
            "master": str(master_path),
            "n_rows": int(len(master)),
            # Rows the annotators were actually shown; the rest are machine-checked.
            "n_in_scope": int(len(_load_master(spec))),
            "annotators": ann_cols,
            "labelled_by_all": int(rated_by_all.sum()),
            "not_unanimous": int(contested.sum()),
            "awaiting_adjudication": int(len(pending)),
            "adjudication_sheet": str(adj_path) if len(pending) else None,
            "human_verified": n_verified,
            "negative_audited": n_audited,
            "human_vs_panel": _human_vs_panel(master, id_col, labels),
            "agreement": agreement_report(master),
        }
    return out


def _human_vs_panel(master: pd.DataFrame, id_col: str,
                    labels: pd.DataFrame) -> dict | None:
    """How often the human agreed with the model panel, on blind labels only.

    This is the number that justifies calling the panel's output a ground
    truth. It is only meaningful because the verification packet hides the
    model labels from the annotator.
    """
    human = _clean_labels(master["adjudicated"]) if "adjudicated" in master else None
    if human is None or not human.notna().any() or labels.empty:
        return None

    n_ones, n_zeros = labels.eq(1).sum(axis=1), labels.eq(0).sum(axis=1)
    panel = pd.Series(pd.NA, index=master.index, dtype="Int64")
    panel[n_ones > n_zeros] = 1
    panel[n_zeros > n_ones] = 0

    both = human.notna() & panel.notna()
    if not both.any():
        return None
    agree = both & (human == panel)
    overturned = master.loc[both & (human != panel), id_col].astype(str).tolist()
    return {
        "n_compared": int(both.sum()),
        "agreed": int(agree.sum()),
        "agreement_rate": round(float(agree.sum() / both.sum()), 4),
        "overturned_by_human": overturned,
    }


def _instructions() -> str:
    return """# Hướng dẫn gán nhãn ViHoRec

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
"""


def _verify_instructions() -> str:
    return """# Xác minh của người — hướng dẫn

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
"""


def _audit_instructions() -> str:
    return """# Audit ngẫu nhiên — hướng dẫn

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
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_split = sub.add_parser("split", help="create per-annotator packets")
    p_split.add_argument(
        "--annotators", type=int, default=DEFAULT_ANNOTATORS,
        choices=range(2, MAX_ANNOTATORS + 1), metavar=f"2..{MAX_ANNOTATORS}",
        help="panel size; an odd number lets a majority vote resolve every item",
    )
    sub.add_parser("merge", help="merge packets back and build the adjudication sheet")
    sub.add_parser("verify", help="blind packet of the load-bearing rows for a human")
    p_audit = sub.add_parser(
        "audit",
        help="blind random sample of unanimous panel negatives",
    )
    p_audit.add_argument("--n", type=int, default=100, help="audit size (default 100)")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.cmd == "split":
        result = cmd_split(args.annotators)
    elif args.cmd == "verify":
        result = cmd_verify()
    elif args.cmd == "audit":
        result = cmd_audit(args.n)
    else:
        result = cmd_merge()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
