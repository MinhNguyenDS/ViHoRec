"""Text normalisation helpers for hotel entity resolution across the three sites.

The original pipeline assigned a hotel id with ``LabelEncoder(NameHotel)``, i.e.
exact string matching, so the *same* hotel written slightly differently on two
sites (accents, "Khach san" vs "Khách sạn", trailing city, punctuation) was
counted as two different entities. These helpers build a canonical key so that
cross-site duplicates can be detected and quantified.
"""

from __future__ import annotations

import re
import unicodedata

# Generic words that carry no discriminative signal for a hotel name.
_STOPWORDS = {
    "khach", "san", "khachsan", "hotel", "hotels", "khu", "nghi", "duong",
    "resort", "resorts", "spa", "the", "and", "&", "villa", "villas",
    "homestay", "guesthouse", "house", "inn", "motel", "boutique",
}


def strip_accents(text: str) -> str:
    """Remove Vietnamese diacritics (NFKD decomposition + drop combining marks)."""
    if not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    no_marks = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return no_marks.replace("đ", "d").replace("Đ", "D")


def canonical_hotel_key(name: str) -> str:
    """Return an order-independent canonical token key for a hotel name.

    Lower-cased, accent-free, punctuation-free, with generic hotel stopwords
    removed and remaining tokens sorted so that word-order variants collapse.
    """
    base = strip_accents(str(name)).lower()
    base = re.sub(r"[^a-z0-9\s]", " ", base)
    tokens = [t for t in base.split() if t and t not in _STOPWORDS]
    return " ".join(sorted(set(tokens)))


def canonical_location(loc: str) -> str:
    """Normalise a location/city string to an accent-free lower key."""
    return re.sub(r"\s+", " ", strip_accents(str(loc)).lower()).strip()
