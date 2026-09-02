"""M2 verification: replicate the exact split algorithm, prove exclusivity,
reproducibility, and cross-check class weights vs saved JSON + notebook output."""

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "Oral Diseases"
CLASS_MAP = {"Calculus": "Calculus", "Data caries": "Caries",
             "Gingivitis": "Gingivitis", "Mouth Ulcer": "Ulcers",
             "Tooth Discoloration": "Tooth Discoloration",
             "hypodontia": "Hypodontia"}
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(d):
    return [p for p in Path(d).rglob("*") if p.suffix.lower() in EXT and p.is_file()]


def build_file_list():
    records = []
    for label, (actual_dir, disease) in enumerate(CLASS_MAP.items()):
        for path in sorted(list_images(DATASET / actual_dir)):
            records.append((str(path), label))
    return records


def stratified_split(records, val_ratio=0.15, test_ratio=0.15, seed=42):
    rng = random.Random(seed)
    by_class = {}
    for path, label in records:
        by_class.setdefault(label, []).append((path, label))
    train, val, test = [], [], []
    for label, items in sorted(by_class.items()):
        shuffled = items[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        test.extend(shuffled[:n_test])
        val.extend(shuffled[n_test:n_test + n_val])
        train.extend(shuffled[n_test + n_val:])
    return train, val, test


records = build_file_list()
assert len(records) == 12320, f"record count mismatch: {len(records)}"

# --- determinism / reproducibility ------------------------------------------
tr1, va1, te1 = stratified_split(records, seed=42)
tr2, va2, te2 = stratified_split(records, seed=42)
tr3, va3, te3 = stratified_split(records, seed=7)
same1 = [r[0] for r in tr1] == [r[0] for r in tr2] and \
        [r[0] for r in va1] == [r[0] for r in va2] and \
        [r[0] for r in te1] == [r[0] for r in te2]
diff_seed = [r[0] for r in tr1] != [r[0] for r in tr3]

# --- exclusivity: no path in two splits; counts add up -----------------------
all_paths = [r[0] for r in records]
assert len(all_paths) == len(set(all_paths)), "duplicate path in records!"
tr_set, va_set, te_set = {r[0] for r in tr1}, {r[0] for r in va1}, {r[0] for r in te1}
overlap = (tr_set & va_set) | (tr_set & te_set) | (va_set & te_set)
assert not overlap, f"split overlap: {len(overlap)} paths"
assert len(tr_set) + len(va_set) + len(te_set) == len(records)

# --- class counts per split --------------------------------------------------
def counts(recs):
    return Counter(l for _, l in recs)

names = list(CLASS_MAP.values())
ct, cv, ctest = counts(tr1), counts(va1), counts(te1)
print("== SPLIT TABLE (replicated, seed 42) ==")
print(f"{'class':<22}{'total':>7}{'train':>8}{'val':>7}{'test':>8}{'train%':>9}{'val%':>7}{'test%':>8}")
for k in range(6):
    n = len([r for r in records if r[1] == k])
    print(f"{names[k]:<22}{n:>7}{ct[k]:>8}{cv[k]:>7}{ctest[k]:>8}"
          f"{ct[k]/n*100:>8.1f}%{cv[k]/n*100:>6.1f}%{ctest[k]/n*100:>7.1f}%")
print(f"{'TOTAL':<22}{len(records):>7}{len(tr1):>8}{len(va1):>7}{len(te1):>8}"
      f"{len(tr1)/len(records)*100:>8.1f}%{len(va1)/len(records)*100:>6.1f}%{len(te1)/len(records)*100:>7.1f}%")

# --- class weights (train split) ----------------------------------------------
N, C = len(tr1), 6
weights = {names[k]: round(N / (C * ct[k]), 4) for k in range(6)}
saved = json.loads((ROOT / "reports" / "class_weights_final.json").read_text(encoding="utf-8"))
print("\n== CLASS WEIGHTS vs saved JSON ==")
for k in range(6):
    w = weights[names[k]]
    s = saved[names[k]]
    flag = "OK" if abs(w - s) < 1e-4 else "MISMATCH"
    print(f"{names[k]:<22}recomputed={w:>8.4f}  saved={s:>8.4f}  {flag}")

summary = json.loads((ROOT / "reports" / "preprocessing_summary.json").read_text(encoding="utf-8"))
print("\n== preprocessing_summary.json ==")
print(json.dumps(summary, indent=2)[:1500])

print("\nREPRODUCIBLE (same seed, identical assignment):", same1)
print("SEED CHANGES ASSIGNMENT:", diff_seed)
print("OVERLAP BETWEEN SPLITS:", len(overlap))
print("VERDICT:", "PASS" if same1 and diff_seed and not overlap else "FAIL")
