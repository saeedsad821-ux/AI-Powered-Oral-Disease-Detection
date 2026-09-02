"""Independent M2 verification: full-decode audit of every image in valid classes."""

import sys
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "Oral Diseases"
VALID = ["Calculus", "Data caries", "Gingivitis", "hypodontia", "Mouth Ulcer",
         "Tooth Discoloration"]
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

bad = []            # (path, error)
empty_dirs = []
unsupported = []
counts = Counter()

for cls in VALID:
    cls_dir = DATASET / cls
    if not cls_dir.is_dir():
        print("MISSING CLASS DIR:", cls)
        continue
    files = sorted(f for f in cls_dir.rglob("*") if f.is_file())
    for f in files:
        if f.suffix.lower() not in EXT:
            unsupported.append(f)
            continue
    subdirs_empty = [str(d) for d in cls_dir.rglob("*") if d.is_dir() and not any(d.iterdir())]
    empty_dirs.extend(subdirs_empty)
    for f in files:
        if f.suffix.lower() not in EXT:
            continue
        try:
            with Image.open(f) as im:
                im.load()          # force full decode
            counts[cls] += 1
        except (UnidentifiedImageError, OSError, ValueError) as e:
            bad.append((str(f), repr(e)))

total = sum(counts.values())
print(f"== FULL DECODE AUDIT ==")
print(f"checked (fully decoded): {total}")
print(f"corrupted/undecodable  : {len(bad)}")
print(f"unsupported formats    : {len(unsupported)}")
print(f"empty subdirectories   : {len(empty_dirs)}")
print("per class:", dict(counts))
for p, e in bad[:5]:
    print("BAD:", p, e)
for p in unsupported[:5]:
    print("UNSUPPORTED:", p)
for p in empty_dirs[:5]:
    print("EMPTY DIR:", p)

print("AUDIT_STATUS:", "PASS" if not bad and not unsupported and not empty_dirs else "FAIL")
