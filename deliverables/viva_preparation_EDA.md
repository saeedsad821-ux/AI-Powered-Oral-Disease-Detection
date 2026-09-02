# Viva Preparation — Milestone 1 (EDA)

**Project:** AI-Powered Oral Disease Detection System Using Transfer Learning and Explainable AI

---

## 10 Likely Instructor Questions with Professional Answers

### Q1. Why did you exclude one of the seven dataset folders?

**Answer:** The folder `Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset` mixes four conditions in single images with YOLO-format annotations. A single-class classifier requires one clean label per image; mixed labels would corrupt the training signal. Excluding it keeps label semantics consistent and the evaluation trustworthy.

### Q2. How did you detect the dataset location automatically?

**Answer:** I wrote a `find_dataset_root()` helper that checks a list of candidate paths (project-local `Oral Diseases`, parent folders, absolute path). The first path that exists as a directory is used; otherwise a `FileNotFoundError` with a clear message is raised. This makes the notebook runnable in VS Code, Kaggle, or a fresh clone.

### Q3. What is the class imbalance ratio and why does it matter?

**Answer:** 2.24:1 (largest class Ulcers 2,806 vs smallest Hypodontia 1,251). If ignored, the model optimizes accuracy by favoring majority classes and may never learn Hypodontia. Mitigations: inverse-frequency class weights, stratified splits, and augmentation focused on minority classes.

### Q4. How are class weights computed?

**Answer:** Inverse-frequency weighting `w_c = N / (C * n_c)`, where N is total images, C the number of classes, n_c the class size. Weights normalize to 1.0 on average: Ulcers 0.73 → Hypodontia 1.64. During training the loss multiplies each sample's contribution by its class weight.

### Q5. What data quality checks did you perform and what did you find?

**Answer:** Five checks: corrupted/unreadable images (`PIL Image.verify()` on 150 per class), duplicate filenames (within-class vs cross-class), unsupported formats, empty folders, and label consistency. Result: **0 corrupted, 0 within-class duplicates, 0 unsupported files, 0 empty folders in valid classes** — the dataset is clean. The 1,515 cross-class name collisions are benign Kaggle-style naming.

### Q6. Why resize to 224x224 and normalize with EfficientNet preprocessing?

**Answer:** 224x224 is EfficientNetB0's native input size — the architecture expects it and ImageNet weights align with it. Normalization must follow the model's preprocessing contract (per-channel scaling), not generic 0-1 rescaling, or transfer learning features will be fed out-of-distribution pixels.

### Q7. Why use stratified splitting instead of random splitting?

**Answer:** Stratified splitting keeps the same class proportions in train/val/test as the full dataset. With a 2.24:1 imbalance, random splitting can, by chance, leave a tiny class nearly absent from validation — making metrics unreliable. Stratification guarantees every split represents all classes.

### Q8. What did you learn from the image dimension analysis?

**Answer:** Resolution and aspect ratio vary across samples (the sampled distribution is not uniform; some images are much larger than the median). This means preprocessing cannot assume fixed geometry — it must explicitly resize and, because clinical texture matters, mild letterboxing or direct resize is preferred over aggressive cropping.

### Q9. Why keep RGB instead of converting to grayscale?

**Answer:** Oral disease cues include redness, color transitions and lesion pigmentation — color information is diagnostically meaningful. Grayscale discards exactly those cues. RGB also matches the pretrained EfficientNetB0 input channel structure.

### Q10. How does this EDA guide the rest of the project?

**Answer:** It defines the contract for Milestone 2: stratified 70/15/15 splits (from `analysis_summary.json`), resize + EfficientNet normalization, mild augmentation on training only. Milestone 3 then trains EfficientNetB0 with the computed class weights, and Milestone 4/5 evaluate and explain predictions with Grad-CAM.

---

## Common Mistakes to Avoid During the Presentation

1. **Don't claim the dataset is perfectly balanced** — state 2.24:1 and explain the mitigation.
2. **Don't say "all images are the same size"** — the analysis explicitly showed variance; the resize is a *consequence* of that finding.
3. **Don't call cross-class filename collisions "duplicates"** — distinguish filename collisions from actual duplicate images.
4. **Don't forget to justify excluding the YOLO folder** — this is a favorite follow-up question.
5. **Don't mention MLflow/Docker/unit tests** — they are outside the project scope; answer about the actual deliverables.
6. **Don't read slides verbatim** — use the bullet points as cues and the charts as the story.
7. **Don't skip the interpretation under each chart** — interpretation is what shows analytical thinking.
8. **Don't confuse correlation with causation in the quality findings** — say "no corruption detected in the sampled audit", not "zero defects in the whole dataset".
9. **Don't forget reproducibility details** — be ready to name the seed (42) and the sampling limits (150/class).
10. **Don't neglect the transition** — always close with "this is what preprocessing will do next".
