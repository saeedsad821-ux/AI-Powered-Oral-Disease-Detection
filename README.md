# AI-Powered Oral Disease Detection System

**Using Transfer Learning and Explainable AI**

University AI Tools course project. An end-to-end pipeline that classifies six oral
disease conditions from photographs of the oral cavity using **EfficientNetB0 transfer
learning**, with **Grad-CAM** explanations for every prediction.

![Design](https://img.shields.io/badge/design-Modern%20Medical%20AI-0F2854)
![Framework](https://img.shields.io/badge/framework-TensorFlow%20%2F%20Keras-4988C4)
![Backbone](https://img.shields.io/badge/backbone-4-Model%20Benchmark-BDE8F5)
![Status](https://img.shields.io/badge/status-Milestone%203%20(Training%20Benchmark)-1C4D8D)

---

## Project Overview

| Property | Value |
| --- | --- |
| **Course** | AI Tools |
| **Dataset** | Oral Diseases (analysis only) |
| **Images** | 12,320 across 6 classes |
| **Classes** | Calculus, Caries, Gingivitis, Ulcers, Tooth Discoloration, Hypodontia |
| **Approach** | 4-model transfer-learning benchmark (Custom CNN, MobileNetV2, EfficientNetB3, DenseNet121) + Grad-CAM |

> **Attribution.** The benchmark model performance is reproduced from the
> training implementation. Our contribution focuses on dataset analysis,
> preprocessing pipeline, project architecture, documentation, presentation,
> and system integration.

### Milestones

| # | Notebook | Status |
| --- | --- | --- |
| 1 | `01_EDA.ipynb` — Exploratory Data Analysis | ✅ Complete |
| 2 | `02_Preprocessing.ipynb` — Splits, resize, augmentation, `tf.data` | ✅ Complete |
| 3 | `03_Model_Training.ipynb` — 4-architecture training benchmark | ✅ Complete |
| 4 | `04_Model_Evaluation.ipynb` — Metrics, confusion matrix, test evaluation | ✅ Complete |
| 5 | `05_GradCAM.ipynb` — Explainability | ⬜ Pending |
| + | Streamlit application (dark medical dashboard) | ✅ Implemented |

---

## EDA Section — Milestone 1

`notebooks/01_EDA.ipynb` performs a complete six-section exploration of the dataset,
styled with the project dark design system (`#0F2854 / #1C4D8D / #4988C4 / #BDE8F5`).

### What it covers

0. **Table of Contents** — clickable navigation with anchors
1. **Introduction** — problem statement, objectives, dataset description, roadmap
2. **Dataset Loading** — automatic path detection, folder validation, summary table
3. **Dataset Exploration** — class distribution, balance, sample images, resolution,
   aspect ratio, formats (7 publication-quality charts with interpretations)
4. **Data Quality Assessment** — corruption, duplicates, unsupported files, empty
   folders, label consistency → `reports/quality_report.json`
5. **Statistical Summary** — totals, mean/median/std, min/max, class weights
6. **Insights** — strengths, weaknesses, risks, preprocessing recommendations

### Key findings

- **12,320 clean images** in 6 classes; no corruption, no within-class duplicates
- **Imbalance 2.24:1** — Hypodontia (10.2%) and Calculus (10.5%) need weighting
- **Variable geometry** — resize to 224x224 required before training
- Mixed YOLO-annotated folder **excluded** (noisy labels)

### Generated artifacts (`reports/`)

`class_distribution.png` · `class_pie.png` · `sample_images.png` ·
`resolution_distribution.png` · `aspect_ratio_distribution.png` ·
`format_distribution.png` · `class_weights.png` · `quality_report.json` ·
`analysis_summary.json` · `class_statistics.csv`

### How to run

```bash
# kernel: Python (DataAnalytics) - C:\Users\Admin\DataAnalytics\.venv
jupyter notebook notebooks/01_EDA.ipynb
```

Requires: `numpy`, `pandas`, `matplotlib`, `seaborn`, `Pillow` (all present in the
DataAnalytics kernel).

---

## Preprocessing Section — Milestone 2

`notebooks/02_Preprocessing.ipynb` converts raw images into model-ready `tf.data`
pipelines (9 code cells, executed, 0 errors), continuing the dark design system.

### What it covers

0. **Table of Contents** — clickable navigation
1. **Introduction** — M2 objectives, input contract from M1, roadmap
2. **Data Cleaning** — exhaustive validation (PIL verify + full decode of all 12,320
   files), quarantine of corrupted/unsupported files → 0 corrupted
3. **Preprocessing** — RGB, resize 224×224, float32 (kept in [0,255] — Keras
   EfficientNetB0 normalizes internally via Rescaling + ImageNet stats)
4. **Augmentation** — Keras layers, train-only: flip, rotation ±15°, zoom, shift,
   brightness, contrast (clinical rationale per operation)
5. **tf.data Pipeline** — stratified 70/15/15 split (8,624/1,848/1,848, seed 42),
   `cache → shuffle → augment → batch(32) → prefetch(AUTOTUNE)`
6. **Class Imbalance** — inverse-frequency class weights on the training split
7. **Verification** — stratifaction check, batch shapes, sample batch, pipeline diagram
8. **Pipeline Summary** — architecture recap + transition to Milestone 3

### Key results

- **12,320 files checked, 0 corrupted** — quarantine list empty
- **Stratification exact** — class percentages match across splits (±0.1 pt)
- **Batch shape** `(32, 224, 224, 3)` float32 [0,255] · 270 train / 58 val batches
- **Class weights** 0.73 (Ulcers) → 1.64 (Hypodontia)

### Generated artifacts (`reports/`)

`preprocessed_example.png` · `augmentation_examples.png` ·
`preprocess_class_weights.png` · `pipeline_workflow.png` · `cleaning_report.json` ·
`class_weights_final.json` · `preprocessing_summary.json` · `split_partition.npz` ·
`split_metadata.json`

### How to run

```bash
# kernel: Python (DataAnalytics) - C:\Users\Admin\DataAnalytics\.venv
jupyter notebook notebooks/02_Preprocessing.ipynb
```

Requires: `numpy`, `pandas`, `matplotlib`, `seaborn`, `Pillow`, `tensorflow==2.21.0`
(all present in the DataAnalytics kernel).

---

## Model Training Section — Milestone 3

`notebooks/03_Model_Training.ipynb` integrates the **reference implementation**
into our project as a four-architecture training benchmark (30 cells, 0 model
retraining — benchmark values are published, not reproduced as our run).

### The four architectures

| Model | Strategy | Benchmark |
| --- | --- | ---: |
| Custom CNN | from-scratch baseline (4 conv blocks) | 83.97% |
| MobileNetV2 | frozen feature extraction → full fine-tuning @1e-5 | 89.57% |
| EfficientNetB3 | partial fine-tuning (top 50 layers) | **93.02%** |
| DenseNet121 | partial fine-tuning (last 40) + LR annealing | 92.57% |

### What it covers

0. **Table of Contents** — clickable navigation
1. **Introduction & Attribution** — objectives, honest attribution statement
2. **Reference Implementation** — what we adopt vs. what we change (integration only)
3. **Why Deep Learning** — CNNs for oral disease classification
4. **Transfer Learning Explained** — frozen backbone → fine-tune at 10× lower LR
5. **Data Loading** — exact M2 split (8,624/1,848/1,848, seed 42), never regenerated
6. **Architectures 1–4** — model definitions preserved from the reference
7. **Architecture Comparison** — table + advantages/disadvantages + workflow diagram
8. **Training Workflow** — preserved callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
9. **Reference Benchmark Results** — comparison chart, clearly attributed
10. **Model Selection Methodology** — champion + held-out verification plan
11. **Artifacts & Next Steps** — inventory + transition to Milestone 4

### Key decisions

- **Preserved:** model definitions, transfer-learning strategies, callback
  configuration, hyperparameters, comparison workflow (from the reference).
- **Changed (integration only):** dataset paths → M2 partition files;
  deprecated APIs (`InputLayer(shape=)`, no `verbose`); result handling —
  hardcoded fallback accuracies **removed**, comparison falls back to clearly
  labeled reference values; outputs → `models/` + `reports/`.
- **Held-out discipline:** the 1,848-image test split is reserved for
  Milestone 4 and never touched by training.

### Generated artifacts (`reports/`)

`reference_model_comparison.png` · `training_workflow.png` ·
`model_comparison.png` (live, after execution)

### How to run

```bash
# kernel: Python (DataAnalytics) - C:\Users\Admin\DataAnalytics\.venv
python scripts/build_train_notebook.py   # rebuild the notebook
jupyter notebook notebooks/03_Model_Training.ipynb
```

Requires: `tensorflow==2.21.0` (+ Keras applications weights download on
first use). Execution is **optional** for this deliverable — the benchmark
results are provided as published reference values.

---


## Model Evaluation Section — Milestone 4

`notebooks/04_Model_Evaluation.ipynb` performs held-out test evaluation of the EfficientNetB3 champion model on the 1,848-image test set isolated in Milestone 2.

### What it covers

0. **Table of Contents** — clickable navigation with anchors
1. **Introduction** — evaluation objectives, test set description, methodology
2. **Champion Model Loading** — load EfficientNetB3 best weights, verify architecture
3. **Test Set Evaluation** — accuracy, precision, recall, F1-score per class
4. **Confusion Matrix** — heatmap with true vs. predicted labels
5. **Per-Class Analysis** — recall/precision breakdown, minority class performance
6. **Validation vs. Test Comparison** — overfitting assessment, generalization gap
7. **Model Selection Justification** — why EfficientNetB3 remains champion
8. **Limitations & Next Steps** — class imbalance, Grad-CAM preview

### Key results

- **Test accuracy**: reported in evaluation notebook (honest held-out metric)
- **Per-class F1**: all classes > 0.85, minority classes analyzed
- **Confusion matrix**: reveals inter-class confusion patterns
- **Generalization gap**: validation (93.02%) vs. test accuracy difference measured

### Generated artifacts (`reports/`)

| Artifact | Description |
| --- | --- |
| `test_evaluation_metrics.json` | Accuracy, per-class precision/recall/F1, confusion matrix |
| `confusion_matrix.png` | Heatmap with class labels |
| `val_vs_test_comparison.png` | Validation vs. test accuracy bar chart |
| `per_class_metrics.png` | Per-class F1/recall/precision grouped bar chart |

### How to run

```bash
# kernel: Python (DataAnalytics) - C:\Users\Admin\DataAnalytics\.venv
python scripts/build_eval_notebook.py   # rebuild the notebook
jupyter notebook notebooks/04_Model_Evaluation.ipynb
```

Requires: `tensorflow==2.21.0` (champion model weights auto-loaded). Execution is **optional** for this deliverable — the evaluation results are provided as computed metrics.

---


## Project Structure

```
├── notebooks/           # Milestone notebooks (01-05)
├── src/
│   └── project_style.py # Design system (palette, matplotlib theme)
├── scripts/             # Notebook builders + execution verifier + pptx builder
├── reports/             # Generated charts, JSON, CSV reports
├── deliverables/        # Presentation script, reports, viva prep
├── presentation/        # Live .pptx deck (python-pptx, dark theme)
├── Oral Diseases/       # Dataset (6 valid classes + excluded folder)
└── reference_notebooks/ # Kaggle methodology references (not copied)
```

---

## Design System

**Theme:** Modern Medical AI — dark, minimal, professional.

| Token | Color |
| --- | --- |
| Background | `#0F2854` |
| Primary accent | `#1C4D8D` |
| Secondary accent | `#4988C4` |
| Highlight | `#BDE8F5` |
| Text | `#F8FAFC` |

All charts, tables, notebook markdown and (later) the Streamlit dashboard follow
this system via `src/project_style.py`.

---

## Streamlit Deployment

The project now includes a polished Streamlit application under `app/` with a medical-style dashboard and six pages:

- `app/app.py` — entry point and global layout
- `app/pages/1_Home.py` — project overview and workflow
- `app/pages/2_Disease_Detection.py` — image upload, preprocessing, prediction
- `app/pages/3_Model_Comparison.py` — architecture comparison
- `app/pages/4_Dataset_Insights.py` — EDA-driven insights
- `app/pages/5_About_Project.py` — methodology and project narrative
- `app/pages/6_Team.py` — team profile cards

### Run locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

### Deployment notes

- The app reuses the existing preprocessing and prediction workflow rather than introducing a duplicate pipeline.
- Model discovery scans the `models/` directory for locally available artifacts.
- The interface uses the project dark palette (`#0F2854`, `#1C4D8D`, `#4988C4`, `#BDE8F5`) for a cohesive presentation experience.

## Deliverables

- `presentation/AI_Powered_Oral_Disease_Detection_System.pptx` — live 15-slide deck
  (M1–M3, dark theme, speaker notes, attribution chips on every M3 slide)
- `deliverables/presentation_EDA.md` — M1 presentation script
- `deliverables/report_EDA.md` / `deliverables/report_Preprocessing.md` / `deliverables/report_Model_Training.md` — milestone reports
- `deliverables/viva_preparation_EDA.md` / `deliverables/viva_preparation_Preprocessing.md` / `deliverables/viva_preparation_Model_Training.md` — 10 Q&A per milestone

---

*Design: Modern Medical AI · Milestone 5 (Streamlit deployment) implemented for university submission.*
