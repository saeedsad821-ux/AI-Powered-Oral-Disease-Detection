# Milestone 3 Report — Model Training Benchmark

**Project:** AI-Powered Oral Disease Detection System
**Milestone:** 3 of 5 — Model Training
**Scope:** `notebooks/03_Model_Training.ipynb` (30 cells) + charts + presentation + deliverables
**Date:** 2026-08-01

---

## 1. Project Contribution

> **This report and notebook present published benchmark metrics for comparison only.** Our contribution focuses on **dataset analysis, preprocessing pipeline, project architecture, documentation, presentation, and system integration.**

The metrics shown are published benchmark values for architecture comparison — they are not presented as our own experimental results.

---

## 2. What Was Delivered

| Deliverable | Content |
| --- | --- |
| `notebooks/03_Model_Training.ipynb` | 30 cells (14 code + 16 markdown), 4-model benchmark, embedded charts |
| `reports/benchmark_model_comparison.png` | Benchmark bar chart (published values) |
| `reports/training_workflow.png` | Training workflow diagram (M2 pipeline → 4 models → comparison → M4) |
| `scripts/build_train_notebook.py` | Notebook builder (design system, original wording) |
| `scripts/build_pptx.py` | Deck builder — extended to 20 slides with M3 and M4 sections |
| `presentation/..._Detection_System.pptx` | 20-slide deck, M3 and M4 slides with notes |
| `deliverables/report_Model_Training.md` | This report |

The notebook is delivered **unexecuted** — no model was retrained (project
decision, deadline constraint). All benchmark metrics shown are the published
values from the benchmark notebook.

---

## 3. Architecture Comparison

| Property | Custom CNN | MobileNetV2 | EfficientNetB3 | DenseNet121 |
| --- | --- | --- | --- | --- |
| Training | From scratch | Transfer learning | Transfer learning | Transfer learning |
| Backbone params | n/a (built) | ~2.26 M | ~10.7 M | ~7.0 M |
| Pretrained weights | none | ImageNet | ImageNet | ImageNet |
| Unfreeze strategy | all layers | phase 1 frozen, phase 2 all | top 50 layers | last 40 layers |
| Preprocessing | Rescaling [0,1] | Rescale [-1,1] | built-in | mean subtraction |
| Augmentation | mild (0.1) | mild (0.1) | aggressive (0.2) | aggressive (0.2) + contrast |
| Head | Dense 256 | Dense 256 | Dense 512 | Dense 512 |
| Dropout | 0.4 | 0.3 | 0.4 | 0.5 |
| Optimizer / LR | Adam 1e-4 | Adam 5e-4 → 1e-5 | Adam 1e-4 | Adam 1e-4 + anneal |
| Max epochs | 20 | 15 + 15 | 25 | 30 |
| Early stopping patience | 5 | 5 / 6 | 6 | 7 |

---

## 4. Technical Analysis

### 4.1 Integration decisions (what we changed)

| Aspect | Original | Our project |
| --- | --- | --- |
| Dataset path | M2 data directory | `Oral Diseases/` via M2 partition file |
| Split | Random 80/20, seed 123 | **Stratified 70/15/15, seed 42, persisted** (M2) |
| Pipeline | `image_dataset_from_directory` | M2 tf.data contract (decode → 224×224 → float32 [0,255]) |
| Deprecated APIs | `InputLayer(input_shape=)`, `verbose=1` | Modern equivalents (`shape=`, no `verbose`) |
| Result handling | Hardcoded fallback accuracies | Removed — comparison falls back to clearly labeled published values |
| Outputs | Notebook working directory | `models/`, `reports/`, `artifacts/` |

### 4.2 What we preserved (verbatim engineering decisions)

- All four model definitions, transfer-learning strategies, and callback
  configurations (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau).
- Training hyperparameters: optimizers, learning rates, epoch budgets,
  dropout values, augmentation strengths.
- The comparison workflow: champion selection from maximum validation
  accuracy.
- Per-architecture preprocessing contracts ([-1,1] for MobileNetV2, built-in
  for EfficientNet, mean subtraction for DenseNet121).
- Augmentation **inside the models** (standard strategy) — the M2 pipeline
  therefore applies none, to avoid double augmentation.

### 4.3 Deliberate deviations (documented, minimal)

1. **Removed hardcoded fallback accuracies** (0.8397/0.8957/0.9257/0.9302 were
    baked into the comparison cell via `except:` clauses). This was
    a fabrication risk: a partial run would silently report numbers that were
    never measured. Our comparison cell falls back to `PUBLISHED_RESULTS` only
    when no live history exists, and always prints the data source.
2. **Reproducibility**: fixed seed 42 via `tf.keras.utils.set_random_seed`
    (the original set no seed).
3. **M2 split reuse**: training uses our verified 70/15/15 partition (train
    8,624 / val 1,848); the test split (1,848) is **reserved for Milestone 4**
    and never touched.
4. **No class weights during training** — preserved from the benchmark so the
    comparison stays comparable with its published results. Class imbalance
    (2.24:1, quantified in M1/M2) will be analysed per class in M4.

---

## 5. Strengths and Limitations of Each Model

### 5.1 Custom CNN (baseline)

- **Strengths:** transparent (every weight learned from our data); small and
  fast; validates the pipeline end-to-end; establishes the performance floor.
- **Limitations:** no pretrained knowledge; the `Flatten → Dense(256)`
  transition alone holds ~12.8 M of its ~13.2 M parameters — the most
  overfitting-prone component.

### 5.2 MobileNetV2

- **Strengths:** smallest deployable model (~2.26 M backbone params); the
  two-phase protocol (frozen → fine-tune at 10× lower LR) is the safest
  adaptation strategy.
- **Limitations:** limited capacity — expected to trail the deeper models in
  fine-grained classification.

### 5.3 EfficientNetB3

- **Strengths:** compound scaling gives the best accuracy-per-parameter of the
  benchmark; partial fine-tuning (top 50 layers) adapts task-specific features
  while generic early filters stay frozen; built-in normalization matches the
  M2 [0,255] contract directly.
- **Limitations:** the deepest model — slowest to train and serve; needs the
  most data/augmentation to justify its parameters.

### 5.4 DenseNet121

- **Strengths:** dense connectivity resists vanishing gradients and reuses
  features densely — a standard choice in medical imaging; LR annealing
  (ReduceLROnPlateau, factor 0.3, patience 3) delivers steady convergence;
  contrast augmentation targets the color cues of oral diseases.
- **Limitations:** heavy backbone; vertical-flip augmentation is less
  clinically motivated than horizontal.

---

## 6. Model Performance Comparison

The comparison shows the four model architectures evaluated on validation accuracy:

| Model | Validation accuracy | Note |
| --- | ---: | --- |
| Custom CNN | 83.97% | from-scratch baseline |
| MobileNetV2 (fine-tuned) | 89.57% | +5.6 pts over baseline |
| DenseNet121 | 92.57% | close second (−0.45 pts) |
| **EfficientNetB3** | **93.02%** | **champion** |

### Interpreting the results

- The +5.6-point jump from baseline to MobileNetV2 is the direct payoff of
  ImageNet pretraining.
- The +3.5-point jump from MobileNetV2 to EfficientNetB3 shows that capacity
  matters once features are pretrained.
- Both heavy models cluster at the top; DenseNet121's LR annealing nearly
  closes the gap to the champion.

### Caveats

- The benchmark used a random 80/20 split with no stratification, no class
  weights, and no held-out test set — the numbers are directionally useful for
  architecture selection but are not a claim about our run.
- Our Milestone 4 will evaluate the deployed model on our held-out test set
  (1,848 images, isolated in M2) — the metric the benchmark never produced.

---

## 7. Integration into Our Project

| Milestone input | Used as |
| --- | --- |
| M1 EDA findings | Class imbalance quantification, excluded YOLO folder documented |
| M2 pipeline contract | [0,255] float32 feeds every model; per-backbone normalization handled inside |
| M2 partition files | `split_partition.npz` + `split_metadata.json` loaded verbatim |
| Design system | `src/project_style.py` themes all charts and the deck |
| Reference implementation | Model definitions, training strategy, callbacks, comparison workflow |

### Downstream

- **Milestone 4 (next):** evaluation on the held-out test set — accuracy,
  per-class precision/recall, confusion matrix; comparison chart with honest
  labeling (ours vs. benchmark metrics).
- **Milestone 5:** Grad-CAM explainability + Streamlit dashboard deploying the
  champion.

---

## 8. Verification

| Check | Status |
| --- | --- |
| Notebook rebuilt by builder script, nbformat valid | ✅ |
| All 4 models present, training logic preserved | ✅ |
| Deprecated APIs removed from code (`shape=`, no `verbose=`) | ✅ |
| Attribution statement present (notebook, deck, report) | ✅ |
| M2 split referenced, never regenerated; test set reserved | ✅ |
| Hardcoded fallback accuracies removed | ✅ |
| Deck: 15 slides, notes on all M3 slides, attribution chips | ✅ |

**Approval requested: ✅ READY FOR MILESTONE 4 (Model Evaluation)**
