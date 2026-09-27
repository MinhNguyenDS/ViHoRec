"""Text normalisation and entity resolution for hotel names across the three sites.

The original pipeline assigned a hotel id with ``LabelEncoder(NameHotel)``, i.e.
exact string matching, so the *same* hotel written slightly differently on two
sites (accents, "Khach san" vs "Khách sạn", trailing city, punctuation) was
counted as two different entities. ``canonical_hotel_key`` fixed that by
collapsing names to an order-independent token key.

That key over-merged in two ways, both found by pairwise inspection:

* It ignored the city, so one brand operating in several cities became a single
  hotel. ``RAON Hotel`` (Quy Nhơn), ``Raon Hotel`` (Đà Nẵng) and ``Raon Villa``
  (Đà Lạt) shared one id carrying 139 interactions.
* It discarded property-type words as stopwords, so ``X Hotel`` and ``X Villa``
  in the same city became indistinguishable.

``resolve_hotel_entities`` addresses both while keeping the merges that matter.
The type word cannot simply be kept in the key: ``Khách sạn Pullman Vũng Tàu``
and ``Pullman Vung Tau`` are the same hotel, and only one of them states a type.
Types are therefore compared for *compatibility* rather than equality, and an
unstated type absorbs into the stated one.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

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


# Crawl-time placeholders written in place of a missing reviewer display name.
# These are NOT people: collapsing them into one pseudonym creates a single
# artificial "user" that absorbs hundreds of unrelated reviews.
_PLACEHOLDER_NAMES = {
    "khong ten", "khong te", "an danh", "no name", "noname", "anonymous",
    "unknown", "guest", "khach", "user", "n a", "na", "nan", "none", "null",
    "", "-", "--",
}


# --------------------------------------------------------------------------
# Property type
# --------------------------------------------------------------------------

# Ordered longest-first so that "khu nghi duong" wins over "nghi".
_TYPE_MARKERS: tuple[tuple[str, str], ...] = (
    ("khu nghi duong", "resort"),
    ("guest house", "guesthouse"),
    ("nha nghi", "guesthouse"),
    ("khach san", "hotel"),
    ("biet thu", "villa"),
    ("can ho", "apartment"),
    ("aparthotel", "apartment"),
    ("apartment", "apartment"),
    ("condotel", "apartment"),
    ("guesthouse", "guesthouse"),
    ("homestay", "homestay"),
    ("hostel", "hostel"),
    ("resort", "resort"),
    ("villas", "villa"),
    ("villa", "villa"),
    ("motel", "motel"),
    ("hotel", "hotel"),
)

UNSPECIFIED = "unspecified"

# Whole-word matching is required: "village" contains "villa", and treating it
# as a villa marker made Pilgrimage Village look like a type conflict.
_TYPE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rf"\b{re.escape(marker)}\b"), label) for marker, label in _TYPE_MARKERS
)


def property_type(name: str) -> str:
    """Accommodation type stated in a hotel name, or ``unspecified``.

    A name that states two conflicting types (rare, e.g. "Hotel & Resort") is
    reported as ``unspecified`` so that it stays compatible with both rather
    than splitting a real property in half.
    """
    base = strip_accents(str(name)).lower()
    base = re.sub(r"[^a-z0-9\s]", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    found = {label for pattern, label in _TYPE_PATTERNS if pattern.search(base)}
    return found.pop() if len(found) == 1 else UNSPECIFIED


def _discriminative_tokens(name: str, city: str = "") -> str:
    """Name tokens with generic words, type words and the city removed.

    The city is dropped because it is carried separately in the entity key;
    without this, ``Mira Hotel`` and ``Mira Hotel Quy Nhon`` would not merge.
    If removing the city empties the key, the city tokens are kept so that
    hotels named only after their location remain distinguishable.
    """
    base = strip_accents(str(name)).lower()
    base = re.sub(r"[^a-z0-9\s]", " ", base)
    tokens = [t for t in base.split() if t and t not in _STOPWORDS]
    city_tokens = set(strip_accents(str(city)).lower().split())
    stripped = [t for t in tokens if t not in city_tokens]
    return " ".join(sorted(set(stripped or tokens)))


def resolve_hotel_entities(
    names: pd.Series, locations: pd.Series
) -> pd.DataFrame:
    """Assign a canonical entity key to each (name, city) pair.

    Two listings merge when they share a city and the same discriminative
    tokens, and their stated property types are compatible. Within one
    ``(city, tokens)`` group:

    * zero or one stated type -> a single entity (an unstated type absorbs);
    * several stated types -> one entity per type, and unstated listings join
      the most frequent stated type, ties broken alphabetically for determinism.

    Returns a frame with ``base_key``, ``property_type`` and ``entity_key``,
    indexed like the inputs.
    """
    df = pd.DataFrame(
        {
            "name": names.astype(str),
            "city": locations.astype(str).map(canonical_location),
        }
    )
    df["property_type"] = df["name"].map(property_type)
    df["tokens"] = [
        _discriminative_tokens(n, c) for n, c in zip(df["name"], df["city"])
    ]
    df["base_key"] = df["city"] + " | " + df["tokens"]

    resolved_type: dict[tuple[str, str], str] = {}
    for base, grp in df.groupby("base_key"):
        stated = grp.loc[grp["property_type"] != UNSPECIFIED, "property_type"]
        distinct = sorted(set(stated))
        if len(distinct) <= 1:
            single = distinct[0] if distinct else UNSPECIFIED
            for t in grp["property_type"].unique():
                resolved_type[(base, t)] = single
        else:
            # Deterministic fallback for listings that never state a type.
            counts = stated.value_counts()
            fallback = sorted(counts[counts == counts.max()].index)[0]
            for t in grp["property_type"].unique():
                resolved_type[(base, t)] = fallback if t == UNSPECIFIED else t

    df["resolved_type"] = [
        resolved_type[(b, t)] for b, t in zip(df["base_key"], df["property_type"])
    ]
    df["entity_key"] = df["base_key"] + " | " + df["resolved_type"]
    return df[["base_key", "property_type", "resolved_type", "entity_key"]]


def is_placeholder_name(name: str) -> bool:
    """True when a display name is a missing-value placeholder, not an identity.

    Comparison is accent- and punctuation-insensitive so that ``"Không tên"``
    matches ``"khong ten"``; an earlier exact-string check missed the accented
    form and reported a 0% missing rate for the name field.
    """
    key = strip_accents(str(name)).lower()
    key = re.sub(r"[^a-z0-9\s]", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key in _PLACEHOLDER_NAMES
