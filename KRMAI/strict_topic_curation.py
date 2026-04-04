#!/usr/bin/env python3
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARCHIVE_DIR = ROOT / "noise_archive"
STRICT_ARCHIVE_DIR = ARCHIVE_DIR / "strict_non_topic"

# Keep university handcrafted summaries and strongly topic-relevant corpus.
KEEP_EXACT = {
    "krmu_key_updates.txt",
    "krmu_admissions.txt",
    "krmu_fee_structure.txt",
    "krmu_placements.txt",
    "krmu_scholarships.txt",
}

KEEP_KEYWORDS = [
    "admission",
    "admissions",
    "phd",
    "fee",
    "fees",
    "scholar",
    "placement",
    "recruit",
    "payment",
    "financial-assistance",
    "open-applications",
]


def is_topic_file(filename: str) -> bool:
    lower = filename.lower()
    if lower in KEEP_EXACT:
        return True
    return any(keyword in lower for keyword in KEEP_KEYWORDS)


def safe_move(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.move(str(src), str(dst))
        return dst

    if src.read_bytes() == dst.read_bytes():
        src.unlink()
        return dst

    stem = dst.stem
    suffix = dst.suffix
    counter = 1
    while True:
        candidate = dst.with_name(f"{stem}__dup{counter}{suffix}")
        if not candidate.exists():
            shutil.move(str(src), str(candidate))
            return candidate
        counter += 1


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    STRICT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    moved_to_data = 0
    moved_to_strict_archive = 0

    # Recover topic-relevant docs from archive to active data set.
    for path in sorted(ARCHIVE_DIR.glob("*.txt")):
        if not is_topic_file(path.name):
            continue
        safe_move(path, DATA_DIR / path.name)
        moved_to_data += 1

    # Keep active data tightly focused on key query topics.
    for path in sorted(DATA_DIR.glob("*.txt")):
        if is_topic_file(path.name):
            continue
        safe_move(path, STRICT_ARCHIVE_DIR / path.name)
        moved_to_strict_archive += 1

    final_active = len(list(DATA_DIR.glob("*.txt")))
    final_archive = len(list(ARCHIVE_DIR.glob("*.txt")))
    final_strict_archive = len(list(STRICT_ARCHIVE_DIR.glob("*.txt")))

    print(f"moved_topic_from_archive_to_data={moved_to_data}")
    print(f"moved_non_topic_from_data_to_strict_archive={moved_to_strict_archive}")
    print(f"active_data_txt={final_active}")
    print(f"noise_archive_root_txt={final_archive}")
    print(f"noise_archive_strict_txt={final_strict_archive}")


if __name__ == "__main__":
    main()