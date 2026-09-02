# Milestone 2 Report — Preprocessing & TensorFlow Data Pipeline

**Project:** AI-Powered Oral Disease Detection System Using Transfer Learning and Explainable AI
**Course:** AI Tools
**Dataset:** Oral Diseases Dataset (12,320 images, 6 classes)

---

## 1. Technical Summary

Milestone 2 converts the validated raw data into model-ready `tf.data` pipelines. All work
is executed in `notebooks/02_Preprocessing.ipynb` (kernel `dataanalytics`, 9 code cells,
0 errors) and builds on `reports/analysis_summary.json` from Milestone 1.

| Component | Implementation |
| --- | --- |
| Input contract | 12,320 RGB images, 6 classes (from M1 quality audit) |
| Cleaning | Exhaustive validation: PIL verify + full decode of every file → **0 corrupted** |
| Preprocessing | RGB → resize 224×224 → cast float32, **pixels kept in [0,255]** (Keras EfficientNetB0 expects [0,255] and normalizes internally) |
| Split | **Stratified 70/15/15** (train 8,624 / val 1,848 / test 1,848), seed 42 |
| Augmentation | Keras layers, train-only: flip, rotation ±15°, zoom 0.9–1.1, shift, brightness, contrast |
| Pipeline | `map(load+preprocess) → cache → shuffle → map(augment) → batch(32) → prefetch(AUTOTUNE)` |
| Imbalance | Inverse-frequency class weights computed on the training split |

Key technical results:

- **Stratification verified:** class percentages identical across splits
  (e.g., Caries 21.1% in all three; Hypodontia 10.1–10.2%).
- **Pipeline shapes:** batch `(32, 224, 224, 3)` float32 / labels `(32,)` int32; 270 train
  batches, 58 val batches.
- **Augmentation rationale:** deliberately mild so lesions keep clinical texture; applied to
  training only — validation/test remain unmodified for honest metrics.
- **Efficiency:** `cache()` after decode avoids re-decoding on every epoch; `prefetch(AUTOTUNE)`
  overlaps data loading with training.
- **Model contract:** float32 in [0,255] — matches Keras EfficientNetB0's documented
  input expectation (internal `Rescaling(1/255)` + ImageNet `Normalization`); verified
  empirically (feature correlation test).

**Artifacts generated (reports/):** `pipeline_workflow.png`,
`augmentation_examples.png`, `preprocessed_example.png`,
`preprocess_class_weights.png`; summaries `cleaning_report.json`,
`class_weights_final.json`, `preprocessing_summary.json`; persisted split
`split_partition.npz` + `split_metadata.json` (seed 42, class map, counts).

---

## 2. Business Summary

For non-technical stakeholders: before teaching the AI to recognize diseases, every photo
is checked, cleaned, and standardized — every image is resized to the same dimensions and
turned into a numeric format the model can learn from. The 12,320 photos were split so
that 70% are used for learning, 15% for tuning, and 15% for the final exam (with no overlap,
and the same disease mix in each part). The AI also receives extra synthetic variations
(flips, rotations, lighting changes) of training photos so it learns to recognize diseases
regardless of camera angle or clinic lighting. Milestone 2 is complete: the data pipeline
is ready to feed the model in Milestone 3.

---

## 3. Key Findings

1. **Zero corrupted images** — full decode pass confirms the M1 quality audit; the
   quarantine list is empty.
2. **Stratification is exact** — 70/15/15 preserves class proportions to within 0.1 pt,
   so validation and test metrics stay honest.
3. **Augmentation is mild by design** — aggressive warps could distort lesions; brightness
   and contrast jitter simulate clinic lighting, not clinical features.
4. **`cache()` + `prefetch()`** — pipeline never re-decodes images and keeps the GPU busy.
5. **Class weights close the imbalance gap** — inverse-frequency weights (≈1.64 for
   Hypodontia vs 0.73 for Ulcers) rebalance the effective loss surface for training.

---

## 4. Challenges

- **Keras-layer vs `tf.image` augmentation** — `tf.image.random_rotation/zoom` are not
  available in TF 2.21; migrated to the canonical Keras `RandomFlip/RandomRotation/...`
  preprocessing layers (also future-proof for TF <2.16 model `export`/serving).
- **Augmentation inside `tf.data.map`** — must return `(image, label)` tuples and avoid
  `.numpy()` on batched tuples; fixed a tuple-unpack bug before execution.
- **Determinism** — global seeds are set for reproducibility; the split uses one fixed
  seed 42 across all three parts.
- **Validation purity** — temptation to augment everywhere; constrained augmentation to
  the training split for honest evaluation.
- **Normalization trap** — [0,1] normalization would be re-divided by 255 by the Keras
  EfficientNetB0 internal `Rescaling`, and `RandomBrightness` computes its ±51 delta in
  0–255 space; caught via a feature-correlation test and fixed by keeping float32 in
  [0,255].

---

## 5. Recommendations

| Priority | Recommendation | Target milestone |
| --- | --- | --- |
| High | Train EfficientNetB0 on the augmented pipeline with class weights | 3 (Training) |
| High | Load the persisted `split_partition.npz` in M3 (identical split, no recompute) | 3 (Training) |
| Medium | Keep validation/test unmodified; no augmentation leakage | Ongoing |
| Medium | EarlyStopping + ReduceLROnPlateau based on validation loss | 3 (Training) |
| Low | Reuse the same builder/executor pattern for the M3 notebook | Ongoing |

---

## 6. Transition to Training

Milestone 3 (`03_Training.ipynb`) will load the same stratified split (persisted), train
EfficientNetB0 with class-weighted loss, and report training curves — consuming the
pipeline verified here.
