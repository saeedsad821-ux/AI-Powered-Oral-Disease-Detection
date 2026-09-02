# Milestone 2 Viva Preparation — Preprocessing & TensorFlow Data Pipeline

**Project:** AI-Powered Oral Disease Detection System
**Milestone:** 2 · Preprocessing (Notebook: `02_Preprocessing.ipynb`)

---

## 10 Likely Questions & Answers

**Q1. Why a 70/15/15 stratified split, and why seed 42?**
Stratification preserves the class proportions of the full dataset in every split, so each
part is a faithful miniature of the population. 70/15/15 is the standard trade-off:
enough training data to learn, enough validation data to tune hyperparameters reliably,
and an untouched test set for a final honest evaluation. Seed 42 makes the split
reproducible — rerunning the notebook yields the exact same partition.

**Q2. How did you verify the split is truly stratified?**
I printed and compared the per-class percentage in train/val/test (e.g., Caries 21.1%
in all three splits, Hypodontia 10.1–10.2%). Differences were ≤ 0.1 percentage point,
proving stratification worked.

**Q3. Why 224×224?**
EfficientNetB0's native input size is 224×224. Matching the model's expected input
avoids resizing inside the model and keeps the transfer-learning feature maps aligned
with ImageNet pretraining.

**Q4. What does the preprocessing step actually do per image?**
Decode the file to RGB, resize to 224×224, cast to float32, keeping pixel values in
[0,255]. No manual normalization because this Keras version's EfficientNetB0
normalizes internally (`Rescaling(1/255)` → ImageNet `Normalization`) and expects
[0-255] floats. We verified this with a feature-correlation test — feeding [0,1]
images would be divided by 255 again and crush the activations.

**Q5. Why is augmentation applied to training only?**
Validation and test sets must stay unmodified so metrics reflect real-world performance.
If we augmented them, we could no longer tell whether improvements came from the model
or from data replication.

**Q6. Why these specific augmentations (flip, rotation, zoom, shift, brightness, contrast)?**
They mirror realistic clinic variation: patients photographed from slightly different
angles (rotation), distances (zoom/shift), mirror orientations (flip), and lighting
(brightness/contrast). They are deliberately mild so lesions keep their clinical texture.

**Q7. How does the tf.data pipeline work and why is it fast?**
`map(load+preprocess) → cache → shuffle → map(augment) → batch(32) → prefetch(AUTOTUNE)`.
`cache()` saves the decoded/preprocessed tensors so later epochs skip disk I/O;
`shuffle` breaks class ordering; `batch(32)` packs samples; `prefetch` overlaps data
loading with computation.

**Q8. Why class weights, and how are they computed?**
The dataset is imbalanced (2.24:1, Hypodontia 10.2% vs Ulcers 22.8%). Inverse-frequency
weights `w_c = N / (C · n_c)` up-weight rare classes during loss computation so the model
does not ignore them. Hypodontia gets ≈1.64, Ulcers ≈0.73.

**Q9. How do you know the quarantine step is trustworthy?**
Every one of the 12,320 files was opened with PIL and fully decoded — not just header
checked. 0 corrupted, 0 unsupported. The check is logged in the notebook, so it is
auditable.

**Q10. What is the single most important output of this milestone?**
The verified, stratified, augmented `tf.data` pipeline with computed class weights —
the exact contract Milestone 3's training will consume. The split is persisted
(`split_partition.npz` + `split_metadata.json`, seed 42; 8,624/1,848/1,848) and
weights are saved, so training is fully reproducible without recomputation.

---

## Common Mistakes to Avoid (learned/anticipated)

| Mistake | Why it is wrong | Correct approach |
| --- | --- | --- |
| Augmenting validation/test | Metrics become inflated/unreliable | Augment training only |
| Normalizing to [0,1] | Keras EffNetB0 re-divides by 255 (crushed activations); RandomBrightness expects 0–255 | Keep float32 in [0,255] |
| Using `tf.image.random_rotation` | Does not exist in TF 2.21 → runtime error | Keras `RandomRotation` layers |
| Calling `.numpy()` on a tuple | Tuple has no `.numpy()` | Index the tuple first: `fn(x)[0].numpy()` |
| Unseeded split | Non-reproducible results | Fixed seed 42 everywhere |
| Random split without stratification | Small minority classes may vanish from test | `train_test_split(stratify=y)` |
| No `cache()` | Re-decodes 12k images every epoch | Cache after preprocessing |

---

## One-Sentence Pitch

"Milestone 2 delivered a verified, stratified 70/15/15 pipeline: 12,320 images cleaned
(0 corrupted), standardized to 224×224, augmented mildly for clinical realism, batched
with cache + prefetch for speed, and rebalanced with inverse-frequency class weights —
ready to train EfficientNetB0."
