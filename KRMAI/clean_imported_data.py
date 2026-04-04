#!/usr/bin/env python3
import re
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARCHIVE_DIR = ROOT / "noise_archive"

# Lines repeated across many pages are usually menu/footer scaffolding.
COMMON_LINE_THRESHOLD = 40
MIN_USEFUL_LINES = 8
MIN_USEFUL_CHARS = 450

NOISE_PATTERNS = [
    r"^Source File:",
    r"^Careers$",
    r"^IQAC$",
    r"^NAAC$",
    r"^DSW$",
    r"^Alumni$",
    r"^Foundry$",
    r"^LMS$",
    r"^ERP$",
    r"^Library$",
    r"^Contact Us$",
    r"^Apply Now$",
    r"^Admissions$",
    r"^FAQs$",
    r"^Placements$",
    r"^Feedback$",
    r"^Blogs$",
    r"^Ombudsperson$",
    r"^360.? Virtual Tour$",
    r"^Sohna Road, Gurugram, Haryana",
    r"^01148884888",
    r"^\[email protected\]$",
    r"^Student Portal for Grievance",
]

NOISE_REGEXES = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]


def is_imported_file(text: str) -> bool:
    return text.startswith("Source File:")


def keep_line(line: str, common_counts: Counter) -> bool:
    s = line.strip()
    if not s:
        return False

    for rx in NOISE_REGEXES:
        if rx.search(s):
            return False

    # Drop short, repetitive menu/footer labels seen on many pages.
    if common_counts.get(s, 0) >= COMMON_LINE_THRESHOLD and len(s) <= 80:
        return False

    # Drop very long repeated nav blobs copied from page headers.
    if common_counts.get(s, 0) >= COMMON_LINE_THRESHOLD and len(s) > 80:
        return False

    return True


def extract_title(lines):
    for line in lines[:6]:
        if line.lower().startswith("title:"):
            return line.split(":", 1)[1].strip()
    return ""


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(DATA_DIR.glob("*.txt"))
    imported_files = []
    doc_lines = {}

    for path in txt_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not is_imported_file(text):
            continue
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        imported_files.append(path)
        doc_lines[path] = lines

    common_counts = Counter()
    for lines in doc_lines.values():
        common_counts.update(set(lines))

    cleaned_count = 0
    archived_count = 0

    for path, lines in doc_lines.items():
        title = extract_title(lines)
        kept = [ln for ln in lines if keep_line(ln, common_counts)]

        # Remove immediate duplicate lines after filtering.
        deduped = []
        prev = None
        for ln in kept:
            if ln != prev:
                deduped.append(ln)
            prev = ln

        body = "\n".join(deduped).strip()

        if len(deduped) < MIN_USEFUL_LINES or len(body) < MIN_USEFUL_CHARS:
            shutil.move(str(path), str(ARCHIVE_DIR / path.name))
            archived_count += 1
            continue

        parts = []
        if title:
            parts.append(title)
            parts.append("=" * len(title))
            parts.append("")
        parts.append(body)
        path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
        cleaned_count += 1

    print(f"total_txt={len(txt_files)}")
    print(f"imported_txt={len(imported_files)}")
    print(f"cleaned_in_place={cleaned_count}")
    print(f"archived_low_signal={archived_count}")
    print(f"archive_dir={ARCHIVE_DIR}")


if __name__ == "__main__":
    main()
