# Milestone 2 Verification Report — Preprocessing & TensorFlow Data Pipeline

**Project:** AI-Powered Oral Disease Detection System
**Scope:** `notebooks/02_Preprocessing.ipynb` + reports + deliverables
**Date:** 2026-08-01 · **Method:** independent re-execution of every check (scripts in `scripts/verify_*.py`)

---

## 1. Dataset Integrity — ✅ PASS

| Check | Result | Evidence |
| --- | --- | --- |
| Total images match EDA | ✅ 12,320 | Independent count: 1296+2601+2349+1251+2806+2017 = 12,320 (matches `analysis_summary.json` and M1 report) |
| No corrupted images | ✅ 0 corrupted | Independent full-decode audit (`verify_dataset.py`): every file opened with PIL + `.load()` — 0 failures |
| No empty folders | ✅ none | Recursive scan of all 6 class dirs — no empty subdirectories |
| No unsupported formats | ✅ none | All files `.jpg`/`.jpeg`; excluded YOLO folder (1,542 imgs + 1,510 txt + 1 yaml) correctly excluded |
| No missing labels | ✅ none | Every image lives in exactly one class folder; label = folder (CLASS_MAP) |
| Statistics consistent | ✅ | Cleaning report (12,320 checked / 0 removed), summary JSON, and notebook outputs agree |

## 2. Train / Validation / Test Split — ✅ PASS

Independent replication of the exact algorithm (`verify_split.py`, seed 42):

| class | total | train | val | test | train% | val% | test% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Calculus | 1,296 | 908 | 194 | 194 | 70.1 | 15.0 | 15.0 |
| Caries | 2,601 | 1,821 | 390 | 390 | 70.0 | 15.0 | 15.0 |
| Gingivitis | 2,349 | 1,645 | 352 | 352 | 70.0 | 15.0 | 15.0 |
| Ulcers | 2,806 | 1,964 | 421 | 421 | 70.0 | 15.0 | 15.0 |
| Tooth Discoloration | 2,017 | 1,411 | 303 | 303 | 70.0 | 15.0 | 15.0 |
| Hypodontia | 1,251 | 875 | 188 | 188 | 69.9 | 15.0 | 15.0 |
| **TOTAL** | **12,320** | **8,624** | **1,848** | **1,848** | 70.0 | 15.0 | 15.0 |

- ✅ Mutually exclusive: intersection of all three split path-sets = **0**
- ✅ No duplicate images: all 12,320 paths unique; per-split sets disjoint
- ✅ Fixed seed: `random.Random(42)` in `stratified_split`; same-seed re-run → byte-identical assignment
- ✅ Reproducible: seed change (42→7) changes the assignment → seed is genuinely controlling the split
- ✅ Persisted for M3: `reports/split_partition.npz` (12,320 paths + labels, overlap-free) + `reports/split_metadata.json`

## 3. TensorFlow Data Pipeline — ✅ PASS

Order verified in notebook cell 5 (code):

```
from_tensor_slices → map(load+preprocess, AUTOTUNE) → cache → shuffle(2048)
→ map(augment, AUTOTUNE)  [train only] → batch(32) → prefetch(AUTOTUNE)
```

- ✅ `cache()` after decode+resize: images decoded once, reused every epoch (8,624×224×224×3 float32 ≈ 4.1 GB RAM trade-off, acceptable)
- ✅ `shuffle()` after cache: order re-randomized every epoch, augmentations re-sampled
- ✅ `batch(32)` + `prefetch(AUTOTUNE)`: measured batch shape `(32,224,224,3)` / `(32,)`
- ✅ Augmentation applied **only** when `training=True`; `val_ds`/`test_ds` built with `training=False` — untouched
- ✅ Efficiency: `num_parallel_calls=AUTOTUNE` on decode and augment maps

## 4. Image Preprocessing — ✅ PASS (issue found & fixed)

| Step | Status |
| --- | --- |
| RGB conversion (`decode_image channels=3`) | ✅ |
| Resize 224×224 (bilinear) | ✅ |
| float32 cast | ✅ |
| Normalization | ✅ corrected |

**❌ Issue found (fixed):** the pipeline normalized to **[0,1]** (`/255.0`), but the installed Keras `EfficientNetB0` **expects float inputs in [0,255]** — its internal `Rescaling(scale=1/255, offset=0)` + ImageNet `Normalization` layers do the normalization themselves. Feeding [0,1] would be divided by 255 a second time. Verified three ways:
1. Model docstring: *"EfficientNet models expect their inputs to be float tensors of pixels with values in the [0-255] range"* + `preprocess_input` is a documented no-op.
2. First layers of a built model: `InputLayer → rescaling(1/255) → normalization(ImageNet)`.
3. Feature-correlation test: outputs of identical batches in [0,1] vs [0,255] correlate at **0.958** (≠1) — the double-normalization measurably distorts features.

**Fix applied:** pipeline keeps float32 in **[0,255]** (no `/255`); display code maps back to [0,1] only for plotting. Notebook rebuilt + re-executed (0 errors), all deliverables updated.

## 5. Data Augmentation — ✅ PASS (issue found & fixed)

| Transform | Clinical rationale | Artifacts risk | Strength verdict |
| --- | --- | --- | --- |
| Horizontal flip | Mirrored oral anatomy is still valid anatomy | None (symmetry) | ✅ keep 50% |
| Rotation ±15° | Camera tilt during screening | Mild border fill (nearest) | ✅ keep 0.26 rad |
| Zoom 0.9–1.1× | Distance variation camera↔mouth | None meaningful | ✅ keep 0.1 |
| Translation ±10% | Patient/camera misalignment | Border fill | ✅ keep 0.1 |
| Brightness ±0.2 | Clinic lighting inconsistency | Saturation risk | ✅ keep, now correct range |
| Contrast ±0.2 | Exposure differences across cameras | None (mean-preserving) | ✅ keep 0.8–1.2 |

**❌ Issue found (fixed):** `RandomBrightness(0.2)` was applied to [0,1] images while its default `value_range=(0,255)` computes deltas of `0.2×255 ≈ ±51` — measured output on mid-gray images reached **46.4** (expected ~0.5±0.2). Post-clip this saturated every pixel to 0/1, destroying information. Fixed by keeping images in [0,255]: measured 76.7–177.9 (expected 76.5–178.5) ✅. Contrast is scale-invariant (mid-gray unchanged) and was already safe. Augment chain output confirmed within [0,255] after clip. **Train-only** confirmed.

## 6. Class Imbalance — ✅ PASS

- Formula correct: `w_c = N_train / (C · n_c)`, computed on **training split only**
- Recomputed independently → matches `class_weights_final.json` **exactly to 4 dp**
- Values reasonable: Ulcers 0.7318 … Hypodontia 1.6427 (spread 2.24:1, mirrored by inverse weights)
- No oversampling/undersampling — zero data duplication, trivial reproducibility
- No improvement required

## 7. Performance Review — ⚠ measured on 1-core CPU

Benchmarked on this machine (`scripts/bench_cpu.py`): **1 CPU core, no GPU**.

| Metric | Measured / estimate |
| --- | --- |
| Train step (batch 32, fwd+bwd) | **1.59 s best** (1 core, EffNetB0@224) |
| Per-image | 49.7 ms |
| Epoch (8,624 train imgs) | **~7.2 min** → 10 epochs ≈ 1.2 h, 15 epochs ≈ 1.8 h |
| GPU memory, batch 32 | EffNetB0 ≈ 21 MB weights + ~0.5 GB activations (fp32) + Adam ≈ negligible |
| GPU speed (reference) | RTX 3060 8 GB ≈ 3–5 ms/step (~1 min/epoch); RTX 3090 ≈ 1–2 ms/step (~30 s/epoch) |
| **Recommended batch size** | **CPU: 32** (measured) · **8 GB GPU: 64** · **12 GB GPU: 128** |

Bottlenecks: (1) decode+resize on 1 core — mitigated by `cache()` after epoch 1; (2) model compute dominates on CPU; (3) `prefetch(AUTOTUNE)` already hides pipeline latency. GPU memory is not a constraint at bs32–128. Mixed precision (`float16`) would roughly halve step time on GPU (recommended for M3 if GPU available).

## 8. Reproducibility — ✅ PASS

| Requirement | Status |
| --- | --- |
| Fixed random seed | ✅ seed 42 (split) + `tf.keras.utils.set_random_seed` (M2 execution) |
| Saved class mapping | ✅ `CLASS_MAP` in notebook + `split_metadata.json` |
| Saved split metadata | ✅ `split_partition.npz` + `split_metadata.json` (new, added this review) |
| Deterministic execution | ✅ same-seed re-run produces identical split; notebook re-executed twice → identical outputs |
| Reproducible pipeline | ✅ builder (`build_preprocess_notebook.py`) + executor (`execute_notebook.py`) scripts version the pipeline |

## 9. Deliverables Consistency — ✅ PASS

Automated key-number scan (12,320 · 8,624 · 1,848 · seed 42 · batch 32 · 224×224 · weights 0.73–1.64 · [0,255]) across notebook, README, report, viva prep, and PowerPoint: **all consistent**. The only "missing" hits are (a) M1-era documents that legitimately predate the split, and (b) seed 42 present in notebook code + metadata JSON rather than prose. PowerPoint rebuilt with corrected [0,255] text (5 slides, notes on all, no out-of-bounds shapes).

---

## Verdict

| Section | Status |
| --- | --- |
| 1. Dataset integrity | ✅ PASS |
| 2. Split | ✅ PASS |
| 3. TF pipeline | ✅ PASS |
| 4. Preprocessing | ✅ PASS (1 issue fixed) |
| 5. Augmentation | ✅ PASS (1 issue fixed) |
| 6. Class imbalance | ✅ PASS |
| 7. Performance | ⚠ INFO (1-core CPU measured) |
| 8. Reproducibility | ✅ PASS |
| 9. Deliverables | ✅ PASS |

**Warnings (non-blocking):**
- ⚠ Dataset folder names are inconsistently cased (`hypodontia`, `Data caries`) — handled via explicit `CLASS_MAP`; consider renaming for hygiene.
- ⚠ `cache()` holds ~4.1 GB RAM of preprocessed images — fine here, but reduce `cache()` to `cache('')` file-backed if RAM is constrained.
- ⚠ Training on this 1-core CPU will take ~7 min/epoch; plan 10–15 epochs (1.2–1.8 h) or run M3 on GPU.

**Issues:** 2 found during review, both **fixed and re-verified** (EffNetB0 [0,255] contract; RandomBrightness range). Both are now covered by empirical tests and documented in the report/viva pitfalls.

## Overall score: **9.5 / 10**

**Approval status: ✅ READY FOR MODEL TRAINING**
