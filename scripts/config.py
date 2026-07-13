"""Shared configuration and path resolution for the ViHoRec dataset-release pipeline.

All scripts read the original crawled files from the existing project ``data``
directory and write reproducible artifacts into ``dataset_release/``.
Paths are resolved relative to this file so the pipeline runs from anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

# dataset_release/scripts/config.py -> repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Original crawled data shipped with the project.
def _find_report_dir() -> Path:
    for cand in REPO_ROOT.rglob("data_raw"):
        if (cand / "data_content_based_raw.csv").exists():
            return cand.parent.parent  # .../data/data_raw -> .../Report...-Master
    raise FileNotFoundError("Could not locate crawled data under repo root.")


_REPORT_DIR = _find_report_dir()
DATA_DIR = _REPORT_DIR / "data"
RAW_DIR = DATA_DIR / "data_raw"
DATASET_DIR = DATA_DIR / "dataset"

# Raw per-site rating files (schema: CustomerName, Address, NameHotel, Rating, Date).
SITE_FILES = {
    "booking": RAW_DIR / "data_booking.csv",
    "traveloka": RAW_DIR / "data_travel.csv",
    "ivivu": RAW_DIR / "data_ivivu.csv",
}

# Merged interaction file with assigned IDs (IDuser, CustomerName, Location, IDhotel, ...).
MERGED_RATING_FILE = RAW_DIR / "data_crawl_rating.csv"
# Content-based hotel metadata (309 hotels, 11 attributes).
CONTENT_RAW_FILE = RAW_DIR / "data_content_based_raw.csv"

# Outputs of this pipeline.
RELEASE_DIR = REPO_ROOT / "dataset_release"
OUT_RELEASE = RELEASE_DIR / "release"
OUT_REPORTS = RELEASE_DIR / "reports"
OUT_ANNOTATION = RELEASE_DIR / "annotation"

for _d in (OUT_RELEASE, OUT_REPORTS, OUT_ANNOTATION):
    _d.mkdir(parents=True, exist_ok=True)


# LaTeX paper directory (holds acl_latex.tex + the Image/ figure folder). The
# project was relocated from DS300/ to ViHoRec/; auto-detect the live copy so
# figure scripts always write to the folder the paper actually compiles from.
def _find_paper_dir() -> Path:
    for name in ("ViHoRec", "DS300"):
        cand = REPO_ROOT / name
        if (cand / "acl_latex.tex").exists() or (cand / "Image").is_dir():
            return cand
    return REPO_ROOT / "ViHoRec"


PAPER_DIR = _find_paper_dir()
PAPER_IMG_DIR = PAPER_DIR / "Image"
PAPER_IMG_DIR.mkdir(parents=True, exist_ok=True)

# Valid domain ranges used by consistency checks.
RATING_MIN, RATING_MAX = 0.0, 10.0
DATE_MIN, DATE_MAX = "2010-01-01", "2024-12-31"

# Reproducibility.
RANDOM_SEED = 42

# HMAC secret for pseudonymisation. Override in production via environment
# variable and keep it OUT of the public release (see anonymize.py / DATASHEET).
DEFAULT_SALT = "vihorec-public-demo-salt-v1"
PSEUDONYM_SALT = os.environ.get("VIHOREC_SALT", DEFAULT_SALT)
