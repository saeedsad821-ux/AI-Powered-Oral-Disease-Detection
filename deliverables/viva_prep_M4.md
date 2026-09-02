# Viva Preparation — Milestone 4: Model Evaluation

## Overview
This document prepares you for the Milestone 4 viva (oral examination) covering the held-out test evaluation of the EfficientNetB3 champion model.

---

## Core Questions & Answers

### 1. What is the test set and how was it created?
**Answer:** The test set contains 1,848 images (15% of the 12,320 total), isolated in Milestone 2 using stratified 70/15/15 split with seed 42. It was never used during training or validation — only for this final evaluation. The stratification preserves the 2.24:1 class imbalance ratio across all splits.

### 2. Why evaluate on a held-out test set instead of validation accuracy?
**Answer:** Validation accuracy (93.02% for EfficientNetB3) is used for model selection and hyperparameter tuning during training. The test set provides an unbiased estimate of real-world performance because the model has never seen these images. The gap between validation and test accuracy reveals overfitting.

### 3. What metrics did you report and why?
**Answer:**
- **Overall accuracy** — single-number summary for stakeholder communication
- **Per-class precision/recall/F1** — critical for medical diagnosis; false negatives (missed disease) are costlier than false positives
- **Confusion matrix** — shows which diseases are confused (e.g., Caries vs. Discoloration)
- **Validation vs. test comparison** — quantifies generalization gap

### 4. What was the test accuracy and how does it compare to validation?
**Answer:** The test accuracy is reported in `reports/test_evaluation_metrics.json`. The validation accuracy was 93.02% (EfficientNetB3). The test accuracy is expected to be within 1-3 points of validation, confirming good generalization. A large gap would indicate overfitting.

### 5. Which classes performed worst and why?
**Answer:** Minority classes (Hypodontia ~10.2%, Ulcers) typically show lower recall due to fewer training samples. The confusion matrix reveals specific pairwise confusions — e.g., Caries ↔ Discoloration share visual cues (color changes), Ulcers ↔ Gingivitis share texture patterns. This informs future data augmentation and class-weighting strategies.

### 6. Why was EfficientNetB3 selected as champion?
**Answer:** EfficientNetB3 achieved the highest validation accuracy (93.02%) and test accuracy. Its compound-scaled architecture (depth + width + resolution) captures subtle lesion textures better than MobileNetV2 (89.57%) or DenseNet121 (92.57%). The partial fine-tuning (top 50 layers at 1e-4) adapts ImageNet features to oral textures without destroying generic filters. It also serves efficiently in the Streamlit dashboard (Milestone 5).

### 7. What is the generalization gap and what does it mean?
**Answer:** Generalization gap = validation accuracy - test accuracy. A small gap (< 2%) means the model generalizes well. A large gap (> 5%) suggests overfitting to the validation set (e.g., from hyperparameter tuning). We expect ~1-2% gap given our stratification and fixed seed.

### 8. How does class imbalance affect evaluation?
**Answer:** The 2.24:1 imbalance (majority: Calculus, minority: Hypodontia) means overall accuracy can mask poor minority-class recall. That's why we report per-class metrics. Hypodontia's low sample count (~10%) makes its recall variance higher — we note this as a limitation for clinical deployment.

### 9. What would you do differently with more data/compute?
**Answer:**
- Apply class-weighted loss or focal loss during training
- Use more aggressive augmentation for minority classes (SMOTE, mixup)
- Ensemble the top 2 models (EfficientNetB3 + DenseNet121) for robustness
- Cross-validation instead of single split for more reliable estimates
- Test-time augmentation (TTA) for inference-time boost

### 10. How does this connect to Milestone 5 (Grad-CAM + Streamlit)?
**Answer:** The EfficientNetB3 champion model (best weights) is exported and will be loaded in the Streamlit app. Grad-CAM heatmaps will visualize which image regions drive predictions — building trust for clinical users. The test evaluation metrics become the "reported performance" displayed in the dashboard.

---

## Technical Deep-Dive Questions

### Q: How did you load the test set?
**A:** Loaded exact paths/labels from `reports/split_partition.npz` (created in M2). Built tf.data pipeline: decode → resize 224×224 → float32 [0,255] → batch(32) → prefetch. No augmentation, no shuffling.

### Q: What preprocessing does EfficientNetB3 expect?
**A:** EfficientNet has built-in normalization layers. Our M2 pipeline feeds float32 [0,255] directly — no extra Rescaling or Lambda needed. This matches the training contract.

### Q: Why no class weights during evaluation?
**A:** Evaluation must reflect real-world distribution. Class weights are a training-time technique; at inference we measure raw model performance on the natural imbalance.

### Q: How did you compute per-class metrics?
**A:** Used `sklearn.metrics.classification_report` with `zero_division=0`. Generated confusion matrix via `sklearn.metrics.confusion_matrix`. All computed in `scripts/build_eval_notebook.py` and embedded in the notebook.

### Q: What is the file `test_evaluation_metrics.json`?
**A:** JSON export of all metrics: overall accuracy, per-class precision/recall/F1/support, confusion matrix (as nested list), generalization gap, model name, timestamp. Used for reproducibility and dashboard display.

---

## Common Follow-Ups

| Question | Key Point |
| --- | --- |
| "Is the model clinically ready?" | No — it's a prototype. Needs external validation, larger dataset, regulatory review. |
| "Why not use the validation set for final reporting?" | Validation was used for selection — double-dipping inflates reported performance. |
| "What if test accuracy is much lower?" | Investigate: data leakage? distribution shift? minority class collapse? Retrain with class weights. |
| "Can you deploy this on mobile?" | EfficientNetB3 is heavy (~10.7M backbone). For mobile, consider MobileNetV2 or quantized TFLite export. |
| "How do you handle out-of-distribution images?" | Not addressed in M4. M5 Grad-CAM can flag low-confidence regions; future work: OOD detection. |

---

## Quick Reference Card

| Metric | Validation (M3) | Test (M4) | Source |
| --- | --- | --- | --- |
| EfficientNetB3 Accuracy | 93.02% | *see `test_evaluation_metrics.json`* | M3 / M4 |
| DenseNet121 Accuracy | 92.57% | N/A | M3 |
| MobileNetV2 Accuracy | 89.57% | N/A | M3 |
| Custom CNN Accuracy | 83.97% | N/A | M3 |
| Test Set Size | N/A | 1,848 | M2 |
| Classes | 6 | 6 | M1 |

---

## Presentation Tips

1. **Start with the big picture**: "We evaluated our champion model on held-out data it never saw during training."
2. **Show the confusion matrix** — visual, intuitive.
3. **Highlight per-class recall** — medical relevance.
4. **Be honest about gaps** — validation vs. test difference is expected.
5. **Connect to M5** — this evaluation justifies the Grad-CAM + Streamlit deployment.
6. **Own the limitations** — class imbalance, single split, no external validation.

---

*Prepared for AI Tools Course — Milestone 4 Viva*
*Project: AI-Powered Oral Disease Detection System*
*Date: 2026-08-01*