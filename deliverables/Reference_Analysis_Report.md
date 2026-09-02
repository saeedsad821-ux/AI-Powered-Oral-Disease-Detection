# Reference Notebook Technical Analysis & Integration Plan

**Project:** AI-Powered Oral Disease Detection System
**Reference:** `reference_notebooks/ai-powered-oral-disease-diagnostic-system.ipynb` (Kaggle, "AI-Powered Oral Disease Diagnostic System")
**Scope:** Engineering analysis only — no project files modified, no training started
**Date:** 2026-08-01

> **Purpose of this document.** The reference notebook is treated strictly as a technical
> reference. Nothing from it is copied; the concepts below are rewritten to fit our
> project's architecture, folder structure, and design system. This document produces the
> five requested outputs: (1) Technical Analysis Report, (2) Component Mapping Table,
> (3) Refactoring Plan, (4) Integration Roadmap, (5) Recommended Implementation Order.

---

## Part 1 — Technical Analysis Report

### 1.1 Project architecture

| Aspect | Reference notebook | Our project |
| --- | --- | --- |
| Form | Single monolithic notebook (18 cells), papermill-orchestrated on Kaggle (2× T4 GPU) | Modular: 5 milestone notebooks + `src/` modules + `scripts/` builders |
| Structure | Setup → EDA-lite → data loading → 4 model cells → comparison → conclusion | M1 EDA → M2 Preprocessing (verified) → M3 Training → M4 Evaluation → M5 Grad-CAM + Streamlit |
| Code organization | 100% inline, no modules, no functions beyond `build_custom_cnn()` / `plot_history()` | Notebooks import shared code from `src/`, build via `scripts/build_*.py` |
| Reproducibility | No seeds set; split regenerated implicitly by Keras seed 123 | Seed 42 everywhere; persisted split partition, class map, weights |

### 1.2 Model pipeline (reference)

```
12,320 images (7 folders) ── class_names filter ── 6 clean classes (12,320 = 9,856 + 2,464)
        │
        ▼
image_dataset_from_directory (224×224, batch 32, seed 123, 80/20 random split)
        │  cache → shuffle(1000) → prefetch(AUTOTUNE)
        ▼
Custom CNN (baseline, 20 epochs, Adam 1e-4)              → val_acc ≈ 83.97%
MobileNetV2 (frozen 15 ep, then full fine-tune 15 ep @1e-5) → val_acc ≈ 89.57%
EfficientNetB3 (top-50 unfrozen, 25 ep, Adam 1e-4)       → val_acc ≈ 93.02% ← champion
DenseNet121 (last-40 unfrozen, 30 ep, Adam 1e-4 + RLROP) → val_acc ≈ 92.57%
        │
        ▼
Champion bar chart → "deploy best_efficientnet.keras via Gradio"
```

### 1.3 Data loading strategy

- `tf.keras.utils.image_dataset_from_directory` with an explicit `class_names` list —
  this is the notebook's best data-engineering idea: it silently excludes the noisy
  `Caries_Gingivitus_..._yolo_annotated-Dataset` folder from training.
- 80/20 random split, seed 123, no stratification, no class weights (imbalance 2.24:1 ignored).
- `cache().shuffle(1000).prefetch(AUTOTUNE)` — correct performance pattern.

### 1.4 Transfer learning implementation

- **MobileNetV2:** base frozen (head only) then **all** base layers unfrozen at 1e-5 —
  a rescue learning rate that protects pretrained weights. Correct preprocessing:
  `Rescaling(1./127.5, offset=-1)` (MobileNetV2 expects [-1, 1]).
- **EfficientNetB3:** partial unfreeze (top 50 layers), head `GAP → BN → Dense(512) → Dropout(0.4)`.
  Correctly notes built-in rescaling (recent Keras EfficientNet models normalize internally).
- **DenseNet121:** partial unfreeze (last 40 layers), `Lambda(densenet.preprocess_input)` —
  correct per-architecture preprocessing, plus `RandomContrast(0.2)` with sound clinical
  rationale (teeth/gum color differences).
- **Common pattern:** GlobalAveragePooling (no Flatten), one dense layer + dropout,
  softmax head, Adam, sparse categorical crossentropy.

### 1.5 Training workflow

- One cell per model: build → compile → fit → checkpoint; 4 independent training runs.
- Callbacks: `EarlyStopping` (patience 5–7, `restore_best_weights`), `ModelCheckpoint`
  (`save_best_only`, monitor `val_accuracy`), and — only for DenseNet — `ReduceLROnPlateau`
  (factor 0.3, patience 3, min_lr 1e-6).
- Histories live only in memory; no CSV logs, no timing, no config persistence.

### 1.6 Evaluation methodology

- **Top-1 validation accuracy only**, compared across the 4 models in a bar chart with a
  highlighted champion.
- **Critical flaw:** the 20% validation split is used both for early stopping/checkpoints
  *and* for final model selection — there is no held-out test set. Reported numbers are
  optimistically biased and unreproducible from the notebook alone (see 1.9).
- No confusion matrix, no precision/recall per class, no error analysis, no clinical
  interpretability of misclassifications.

### 1.7 Explainability

- **None.** No Grad-CAM, SHAP, or saliency. The notebook ends at deployment. This is a
  gap we already close by design in Milestone 5 (`05_GradCAM.ipynb` + Streamlit).

### 1.8 Deployment workflow

- Conceptual only: "export `.keras`, integrate into a Gradio web app for dentists".
- No serving code, no pre/post-processing contract, no size/throughput analysis, no
  confidence-threshold or disclaimer handling. Our project's Streamlit plan (M5+) is
  strictly stronger.

### 1.9 Strengths

1. **Noise reduction via `class_names`** — clean, deterministic exclusion of the mixed folder.
2. **Baseline → transfer learning progression** — the custom CNN establishes a floor
   (~84%) before pretrained models are introduced; good scientific method.
3. **Per-architecture preprocessing** — each backbone gets its documented input contract.
4. **Two-phase MobileNetV2 training** — feature extraction, then fine-tune at 10× lower LR.
5. **Partial unfreezing** — "top 50" / "last 40" layers targeted adaptation, avoiding
   destructive full-weights updates.
6. **`ReduceLROnPlateau`** (DenseNet) — sensible annealing (factor 0.3, patience 3).
7. **Augmentation adapted to model** — mild for MobileNetV2, aggressive for EfficientNetB3,
   color-aware (contrast) for DenseNet.
8. **`ModelCheckpoint(save_best_only)`** everywhere.

### 1.10 Weaknesses

1. **No held-out test set** — validation double-role (selection + final score) inflates results.
2. **Hardcoded fallback accuracies** — the comparison cell contains `except: acc_custom = 0.8397`
   fallbacks; if a run is skipped the report silently fabricates numbers. Not scientific.
3. **Class imbalance ignored** — no class weights, no stratified split; minority classes
   (Hypodontia 10.2%, Calculus 10.5%) are under-learned and unreported.
4. **No seeds** — runs are not bit-reproducible.
5. **No artifact persistence** — histories, configs, and summaries vanish with the session;
   the notebook cannot be audited later.
6. **Only accuracy measured** — meaningless for a medical screening tool (no sensitivity/
   specificity/precision/recall per class).
7. **No explainability** — clinicians get a black box.
8. **Full-unfreeze MobileNetV2** — 2.2 M ImageNet params trained at 1e-5 over the whole
   network; the partial-unfreeze pattern used for the other models is strictly safer.
9. **Custom CNN head `Flatten → Dense(256)`** — 12.8 M parameters (96% of the 13.2 M total)
   in one FC layer; GAP would cut this to ~0 and reduce overfitting.
10. **Inconsistent model names in code vs. text** (`pretrained_model_1` vs `history_mobilenet`).

### 1.11 Best engineering practices (worth adopting conceptually)

- Baseline model to quantify transfer-learning gains.
- Frozen-then-fine-tune two-phase protocol with 10× LR reduction.
- Targeted unfreezing (freeze early generic layers, adapt top task-specific blocks).
- Per-architecture input contracts (rescale [-1,1] / preprocess_input / built-in).
- LR annealing via `ReduceLROnPlateau` combined with early stopping + best-weight checkpoint.
- Champion-selection comparison chart (as *reporting*, not as training control).

### 1.12 Deprecated / problematic APIs

| Item | Issue | Our replacement |
| --- | --- | --- |
| `layers.InputLayer(input_shape=...)` | Deprecated → `shape=` (warning observed in the notebook's own output) | `tf.keras.Input(shape=...)` (already used in our M3 builder) |
| `ReduceLROnPlateau(verbose=1)` | `verbose` deprecated in TF ≥ 2.16 callbacks | Omit `verbose` (already omitted in our builder) |
| `ImageDataGenerator` | Imported, never used (dead import) | Not imported at all |
| `tf.keras.optimizers.Adam` | Legacy alias (`keras.optimizers.Adam` preferred) | `tf.keras.optimizers.Adam` acceptable; new namespace if TF ≥ 2.16 |
| Random 80/20 split | No stratification, seed-only control | Persisted stratified `split_partition.npz` (M2) |

### 1.13 Performance bottlenecks

- **RAM cache** — `cache()` holds the full preprocessed set in memory (~3–4 GB); our M2
  verified the same trade-off and documented `cache('')` (file-backed) as fallback.
- **Flatten + Dense(256)** — 12.8 M-parameter bottleneck in the baseline CNN.
- **Full fine-tune of MobileNetV2** — slowest stage for the least gain.
- **Kaggle T4s unused** — no `MirroredStrategy`, no mixed precision, despite 2 GPUs.
- **Single-core CPU reality on our machine** — M2 benchmark: ~7.2 min/epoch for EffNetB0
  at batch 32 (reference's EfficientNetB3 at 224 px would be far worse — another argument
  for our B0 choice).

---

## Part 2 — Component Mapping Table

Only genuinely reusable engineering concepts are mapped. Duplicate code, hardcoded
numbers, and dead imports are intentionally ignored.

| # | Reference component | Concept extracted | Our project module | Required changes |
| --- | --- | --- | --- | --- |
| C1 | `class_names` filter in `image_dataset_from_directory` | Dataset hygiene: exclude noisy folder deterministically | **Already done** — M1/M2 exclude `Caries_Gingivitus_..._yolo`; `split_metadata.json` holds the 6-class map | None (verify in M3 assert block) |
| C2 | Two-phase transfer learning (frozen → fine-tune @ 1e-5) | Stage-1 feature extraction, stage-2 adaptation with 10× lower LR | `scripts/build_train_notebook.py` → stage 1 (`C_TRAIN_STAGE1`) + stage 2 (`C_TRAIN_STAGE2`) | Already implemented; refactor `fit_stage` into `src/training.py` |
| C3 | Partial unfreezing ("top 50" / "last 40") | Freeze generic early layers, adapt task-specific top blocks | M3 stage 2 unfreezes from `block6a` (equivalent, block-based — more precise than layer-count) | Document equivalence; keep block-based logic |
| C4 | `ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-6)` | LR annealing evidence | `artifacts/training_config.json` (already configured) | **Adopt reference factor 0.3 / patience 3 for stage 2** (we used 0.5 / 2) |
| C5 | Per-architecture preprocessing | Input contract per backbone | `02_Preprocessing.ipynb` — verified [0,255] contract for EffNetB0 (M2 verification, correlation test) | None — already solved correctly; reference is inconsistent across models |
| C6 | Baseline-first methodology | Quantify transfer-learning gain | M3 intro section + M4 report | Add a **head-only baseline row** (stage 1) as "our baseline" — stage 1 *is* the baseline; report it that way in M4 |
| C7 | Champion comparison bar chart | Comparative reporting | M4 (`04_Model_Evaluation.ipynb`, to build) | `src/visualization.py: plot_model_comparison()` |
| C8 | History/curves plotting | Acc/loss learning curves | `C_TRAIN_STAGE1/2` + `C_ANALYSIS` (already present) | Extract to `src/visualization.py: plot_history()` |
| C9 | `SparsePrecision` / `SparseRecall` adapters | Per-class-aware metrics with sparse labels | Inline classes in `C_TRAIN_STAGE1` | **Extract to `src/metrics.py`** (reused by M4) |
| C10 | Callback stack (ES + MC + RLROP + CSV) | Robust training control + logging | M3 config cell (already) | Extract to `src/training.py: build_callbacks()`; add epoch wall-clock timing |
| C11 | Gradio deployment idea | "Model as a web second opinion" | Streamlit app (M5+) | We keep **Streamlit** (course requirement); reference's Gradio is ignored |
| C12 | Model naming `best_<name>.keras` | Versioned, comparable artifacts | `models/best_model_stage1.keras`, `models/best_model.keras` (already) | None |

### Explicitly rejected (do not extract)

| Reference item | Why rejected |
| --- | --- |
| Hardcoded fallback accuracies (0.8397 / 0.8957 / 0.9302 / 0.9257) | Fabricates results on partial execution; our verification script reports real numbers only |
| Random 80/20 split, seed 123 | Not stratified, not persisted; our 70/15/15 verified split is strictly better |
| Custom CNN with `Flatten → Dense(256)` | 12.8 M-param head; GAP is the correct pattern |
| Full-unfreeze MobileNetV2 at 1e-5 | Less safe than partial unfreezing; unnecessary for our single-model plan |
| EfficientNetB3 / DenseNet121 / MobileNetV2 models | Not feasible on our 1-core CPU; EffNetB0 already justified in M3 (16 MB, 5.3 M params, ~50 ms/img measured) |
| `ImageDataGenerator` import | Dead code |
| Gradio | Project requirement is Streamlit |

---

## Part 3 — Refactoring Plan

Every extracted component is rewritten for our architecture: PEP 8, docstrings, type
hints, `src/` modules, `project_style` theming, no deprecated APIs.

| New module | Contents (function signatures) | Refactored from |
| --- | --- | --- |
| `src/metrics.py` | `SparsePrecision`, `SparseRecall` (Keras metric subclasses), `classwise_report(y_true, y_pred, class_names) -> pd.DataFrame` | C9 + M4 needs |
| `src/training.py` | `build_callbacks(cfg, log_path, best_path) -> list`, `compile_stage(model, lr, metrics) -> None`, `fit_stage(model, train_ds, val_ds, cfg, log_path, best_path) -> (History, StageTiming)`, `unfreeze_from(model, layer_prefix) -> int` | C2, C4, C10 + existing `fit_stage` |
| `src/model_builder.py` | `build_transfer_model(backbone, input_shape, num_classes, dropout) -> tf.keras.Model` (factory with `weights="imagenet"`, `include_top=False`) | C2/C3 pattern + M3 model cell |
| `src/visualization.py` | `plot_history(history, title, save_path)`, `plot_comparison(results: dict[str, float], save_path)`, `plot_lr_history(log_df, save_path)` | C7, C8 + existing chart cells |
| `src/evaluate.py` | `evaluate_on_test(model, test_ds, class_names) -> dict`, `confusion_matrix_plot(...)` | M4 (new; extends C6/C7) |
| `src/gradcam.py` | `gradcam_map(model, img, last_conv_layer, class_idx) -> np.ndarray`, `overlay_heatmap(...)` | M5 (new; reference has none) |
| `src/data.py` | `load_m2_artifacts() -> SplitBundle`, `make_dataset(paths, labels, training, batch_size)` | Extract duplicated code from M3 builder; single source of truth for M3/M4 |
| `src/inference.py` | `load_classifier(path, class_names)`, `predict_proba(img_path) -> (label, probs)` | Streamlit app (M5+) |

**Refactoring rules applied to every component**

- No `input_shape=` (use `tf.keras.Input(shape=...)`); no `verbose=` in callbacks.
- All paths resolved via `Path(ROOT)`; no `os.path.join` strings.
- Type hints on every public function; module-level docstrings.
- No duplicated pipeline code between notebooks — `src/data.py` is imported by M3, M4, M5.
- Numbers never hardcoded into reporting code — always read from `artifacts/*.json`.
- Charts always themed through `apply_style()` + palette constants.

---

## Part 4 — Integration Roadmap

### 4.1 `03_Model_Training.ipynb` (Milestone 3 — next)

| Change | Source component | Effect |
| --- | --- | --- |
| Import `fit_stage`, `build_callbacks`, `unfreeze_from` from `src/training.py`; `SparsePrecision/Recall` from `src/metrics.py`; plots from `src/visualization.py` | C2, C4, C9, C10, C8 | Notebook becomes orchestration; logic lives in testable modules |
| Stage-2 `ReduceLROnPlateau(factor=0.3, patience=3)` | C4 | Aligns with reference's proven annealing |
| Epoch wall-clock timing in `CSVLogger` complement (`on_epoch_end` with `time.perf_counter`) | C10 | Replaces the equal-split approximation |
| Assert block re-validates 6-class map + folder exclusion | C1 | Guarantees the M2 contract holds before 1.2 h of training |
| Artifact inventory cell (unchanged) | — | Already good |

### 4.2 `04_Model_Evaluation.ipynb` (Milestone 4)

| Component | Source | Implementation |
| --- | --- | --- |
| Held-out test evaluation (1,848 images, never seen) | closes reference's biggest gap | `src/evaluate.py: evaluate_on_test` — accuracy, macro/micro precision-recall, per-class confusion |
| Class-wise report | C9 extension | `classwise_report()` on test set |
| Champion/comparison chart | C7 | `plot_comparison()`: stage-1 vs stage-2 vs (documented) reference-reported numbers, clearly labeled "reference, not reproduced" |
| Learning curves + LR history | C8 | Re-plot from `artifacts/history.json` + `training_log_stage*.csv` |
| Threshold analysis | new | Confidence histogram; suggest operating threshold |

### 4.3 `05_GradCAM.ipynb` (Milestone 5 — explainability)

| Component | Source | Implementation |
| --- | --- | --- |
| Grad-CAM heatmaps | new (reference has none) | `src/gradcam.py`; last conv layer = `top_conv` of EffNetB0 |
| Sample grid: true/pred/prob/heatmap | new | Themed with `project_style` |
| Edge cases (misclassifications, low-confidence) | new | Error analysis bridges M4 → M5 |

### 4.4 Streamlit application

| Component | Source | Implementation |
| --- | --- | --- |
| Upload → predict → top-3 probabilities | C11 | `src/inference.py`; `models/best_model.keras` |
| Grad-CAM overlay on the uploaded image | `src/gradcam.py` | Side-by-side raw vs heatmap |
| Class descriptions + **medical disclaimer** | new | University/course requirement; reference lacks it |
| Dark design system | `src/project_style.py` | CSS tokens from the palette |

---

## Part 5 — Comparison: Reference Notebook vs. Our Project

### 5.1 What we already do better

| Dimension | Reference | Us |
| --- | --- | --- |
| Test methodology | 80/20 random; val double-role | 70/15/15 stratified, persisted, verified (M2 report §2) |
| Imbalance handling | none | Inverse-frequency class weights (0.73–1.64) |
| Reproducibility | no seeds | seed 42 + partition file + builder scripts |
| Metrics | accuracy only | accuracy + sparse precision/recall (M3 config) |
| Artifacts | in-memory histories | `history.json`, `training_config.json`, `model_statistics.json`, CSV logs |
| Reporting integrity | hardcoded fallback numbers | All numbers from executed runs; `verify_*.py` audits |
| Preprocessing correctness | correct per model, but undisclosed in M1/M2 | EffNetB0 [0,255] contract verified with a correlation test (M2 §4) |
| Explainability | none | Grad-CAM planned (M5) |
| Code quality | inline, no seeds, dead imports | PEP 8, type hints, docstrings, design system |

### 5.2 What the reference does better

| Dimension | Reference | Us |
| --- | --- | --- |
| Multi-backbone benchmarking | 4 models compared scientifically | Single EfficientNetB0 (justified by hardware, but no internal ablation) |
| LR annealing evidence | factor 0.3 / patience 3 proved useful | Configured but untuned (0.5 / 2) |
| Baseline quantification | explicit custom-CNN floor | Our stage-1 is the floor; we should *label it as baseline* in reporting |
| Deployment narrative | end-to-end story (model → web app) | Partially planned; app is M5+ |
| Augmentation aggressiveness | escalates with model capacity | Fixed mild set from M2 |

### 5.3 What should be adopted

1. `ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-6)` for stage 2 (M3 config).
2. Explicit "baseline vs transfer" framing in M3/M4 reporting (stage 1 = baseline).
3. Comparison/champion chart in M4 (with honest labeling of reference numbers).
4. Per-epoch wall-clock timing.
5. Model-naming convention already matches (`best_*.keras`).

### 5.4 What should be ignored

1. Hardcoded fallback accuracies — fabrication risk.
2. Random non-stratified split.
3. `Flatten → Dense(256)` head (12.8 M params).
4. Full-network fine-tune of MobileNetV2.
5. EfficientNetB3/DenseNet121/MobileNetV2 backbones (CPU constraints; B0 justified).
6. Gradio (Streamlit is the course requirement).
7. Dead imports and `InputLayer(input_shape=...)`.

---

## Part 6 — Recommendations (Prioritized)

### 🔴 High priority — do before/at M3 execution

| # | Action | Why |
| --- | --- | --- |
| H1 | Adopt `ReduceLROnPlateau(factor=0.3, patience=3)` for stage 2 | Direct, evidence-backed transfer from the reference; cheap, low risk |
| H2 | Extract `src/metrics.py`, `src/training.py`, `src/visualization.py`, `src/data.py` | Modularization before training means M4/M5 reuse clean code; GitHub-ready requirement |
| H3 | Add per-epoch wall-clock timing | The current equal-split timing estimate is inaccurate; timing evidence matters for the course report |
| H4 | Execute M3 via `execute_notebook.py` (GPU if available; CPU fallback with documented epoch budget) | M2 already approved "READY FOR TRAINING"; the run must produce the artifact inventory |
| H5 | Never touch the test split until M4 | Reference's biggest methodological sin; our 1,848-image test set must stay pristine |

### 🟡 Medium priority — M4/M5 planning

| # | Action | Why |
| --- | --- | --- |
| M1 | Held-out test evaluation with per-class precision/recall + confusion matrix | The reference's missing half; makes the system credible for clinicians |
| M2 | Comparison chart with honest labeling (ours vs reference-reported) | Demonstrates our engineering superiority for viva/demo without claiming reproduced numbers |
| M3 | Label stage-1 as "baseline" in all reporting | Adopts the reference's scientific framing at zero cost |
| M4 | Confidence threshold + low-confidence handling in M4 report | Medical tool needs an "uncertain" path, not just top-1 |
| M5 | Grad-CAM error analysis (misclassifications) | Bridges explainability to clinical credibility |

### 🟢 Low priority — polish

| # | Action | Why |
| --- | --- | --- |
| L1 | File-backed `cache('')` if RAM pressure appears on GPU runs | Cheap safety valve (already documented in M2) |
| L2 | Aggressive-augmentation variant during fine-tuning (reference pattern) | Optional experiment, not required for passing |
| L3 | Mixed precision (float16) if a GPU becomes available | ~2× step speedup; zero on CPU |
| L4 | TFLite/ONNX export note for deployment size | Nice future work slide for the presentation |

---

## Part 7 — Recommended Implementation Order

| Step | Action | Output | Est. effort |
| --- | --- | --- | --- |
| 1 | **Approve this analysis** (owner decision) | — | — |
| 2 | Create `src/metrics.py`, `src/training.py`, `src/model_builder.py`, `src/visualization.py`, `src/data.py` | PEP 8 modules with docstrings/type hints | 1 session |
| 3 | Update `scripts/build_train_notebook.py` to import the modules + H1 config change; rebuild `03_Model_Training.ipynb` | Notebook v2 | 1 session |
| 4 | Execute M3 (`python scripts/execute_notebook.py notebooks/03_Model_Training.ipynb`) | `best_model.keras`, artifacts, curves | 1–2 h GPU / ~3 h CPU |
| 5 | `verify_consistency.py` + artifact inventory check | M3 verification | 30 min |
| 6 | Build + execute `04_Model_Evaluation.ipynb` (H2/M1–M4) | Evaluation report + charts | 1 session |
| 7 | Build `05_GradCAM.ipynb` (`src/gradcam.py`) + Streamlit app | Explainability + demo | 2 sessions |
| 8 | Update README, deliverables, PowerPoint for M3–M5 | Course submission | 1 session |

**Critical path:** Step 3 → 4 (training) — everything else waits on the trained model.

---

## Summary for the owner

- **The reference's core value for us is methodological, not code:** two-phase transfer
  learning, partial unfreezing, per-architecture preprocessing, and LR annealing — all of
  which our M3 plan already embodies or trivially adopts (H1).
- **Our project is already ahead** in every area that matters for a university AI Tools
  course: held-out evaluation, stratification, class weights, reproducibility, artifact
  persistence, explainability, and reporting integrity.
- **Nothing from the reference needs to be copied.** The single actionable change before
  training is the `ReduceLROnPlateau` tuning (H1); the rest is refactoring for reuse.

*No project source files were modified during this analysis. Awaiting approval to proceed with the Refactoring Plan (Step 2–3) and Milestone 3 training.*
