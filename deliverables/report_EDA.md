# Milestone 1 Report — Exploratory Data Analysis

**Project:** AI-Powered Oral Disease Detection System Using Transfer Learning and Explainable AI
**Course:** AI Tools
**Dataset:** Oral Diseases Dataset (12,320 images, 6 classes)

---

## 1. Technical Summary

The Oral Diseases dataset was discovered automatically and validated before analysis.
The six valid class folders (Calculus, Caries, Gingivitis, Ulcers, Tooth Discoloration,
Hypodontia) contain **12,320 RGB images**; a seventh folder with mixed YOLO-annotated
content (1,542 images) was **excluded** because its labels are noisy and inconsistent.

| Metric | Value |
| --- | --- |
| Total images | 12,320 |
| Number of classes | 6 |
| Images per class (min/max) | 1,251 (Hypodontia) / 2,806 (Ulcers) |
| Mean / median per class | 2,053 / 2,183 |
| Standard deviation | 659 |
| Imbalance ratio | 2.24:1 |

Key technical results:

- **Distribution:** Ulcers 22.8%, Caries 21.1%, Gingivitis 19.1%, Tooth Discoloration 16.4%, Calculus 10.5%, Hypodontia 10.2%.
- **Dimensions:** resolution and aspect ratio vary; median sampled resolution low → resize to 224x224 required.
- **Quality audit (150-image sample per class):** 0 corrupted images, 0 within-class duplicate filenames, 0 unsupported files, 0 empty folders inside valid classes.
- **Class weights:** inverse-frequency weights range 0.73 (Ulcers) to 1.64 (Hypodontia).

**Artifacts generated (reports/):** `class_distribution.png`, `class_pie.png`, `sample_images.png`,
`resolution_distribution.png`, `aspect_ratio_distribution.png`, `format_distribution.png`,
`class_weights.png`, `quality_report.json`, `analysis_summary.json`, `class_statistics.csv`.

---

## 2. Business Summary

For non-technical stakeholders: the project screens photographs of the mouth for six
common oral diseases. This milestone proved the dataset is **fit for AI training**:
clean folders, no broken images, and enough examples per condition. The only challenge
is that some conditions (Hypodontia, Calculus) have fewer photos than others, which the
model will compensate for by weighting rare conditions more heavily during learning.
Next, the data will be prepared and an EfficientNetB0 model will be trained to detect
each condition automatically, with heatmap explanations showing why the model made
each decision.

---

## 3. Key Findings

1. **Clean data, ready to model:** zero corruption, zero duplicates, zero unsupported files.
2. **Moderate imbalance (2.24:1):** minority classes need explicit mitigation.
3. **Variable image geometry:** standardization to 224x224 is mandatory.
4. **Excluded folder:** mixed YOLO-annotated data would corrupt label semantics.
5. **Reproducibility:** seeded sampling (seed 42) and machine-readable JSON reports.

---

## 4. Challenges

- **Class imbalance** — Hypodontia (10.2%) risks being under-learned.
- **Resolution variance** — needs resizing without losing clinical texture.
- **Kaggle-style filename collisions** — cross-class name duplicates are benign but must not be used for deduplication.
- **Noisy YOLO folder** — exclusion decision must be documented and defended.

---

## 5. Recommendations

| Priority | Recommendation | Target milestone |
| --- | --- | --- |
| High | Stratified 70/15/15 train/val/test split | 2 (Preprocessing) |
| High | Inverse-frequency class weights | 3 (Training) |
| Medium | Mild augmentation: rotation, flip, zoom, brightness (train only) | 2 (Preprocessing) |
| Medium | EfficientNetB0 preprocessing (RGB, per-channel normalization) | 2 (Preprocessing) |
| Low | Log every preprocessing decision for auditability | Ongoing |

---

## 6. Transition to Preprocessing

Milestone 2 (`02_Preprocessing.ipynb`) will implement: folder-to-split stratification,
resize + normalization pipeline, augmentation, and TensorFlow `tf.data` batch loaders —
using `reports/analysis_summary.json` produced here as the single source of truth.
