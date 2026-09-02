# Milestone 3 Viva Preparation — Model Training Benchmark

**Project:** AI-Powered Oral Disease Detection System
**Milestone:** 3 · Model Training (Notebook: `03_Model_Training.ipynb`)

---

## 10 Likely Questions & Answers

**Q1. What does Milestone 3 deliver, and what is your contribution?**
A four-architecture training benchmark: Custom CNN (baseline), MobileNetV2,
EfficientNetB3, and DenseNet121. The benchmark model performance is
**reproduced from the reference implementation** (Kaggle, "AI-Powered Oral
Disease Diagnostic System") and is attributed as such throughout. Our
contribution is the dataset analysis (M1), the verified preprocessing pipeline
(M2), the project architecture, documentation, presentation, and system
integration — not the retraining of the models.

**Q2. Why did you keep four models instead of training one?**
The benchmark protocol answers two questions: (1) how much transfer learning
helps (baseline CNN ≈84% vs. pretrained models >92%), and (2) which
architecture fits this task best. Comparing a from-scratch baseline with three
transfer-learning backbones is the standard scientific method for model
selection in medical imaging.

**Q3. Explain transfer learning and how it is applied here.**
Medical datasets are small compared with ImageNet (~1.2 M images). Transfer
learning reuses a network pretrained on ImageNet: early layers encode generic
features (edges, colors, textures) that transfer to any vision task. The
recipe: load the backbone without its 1000-class head, freeze (or partially
freeze) it, attach a small head (GlobalAveragePooling → Dense → Dropout →
softmax), train the head, then fine-tune the top layers at a 10× lower
learning rate.

**Q4. Why do the three pretrained models unfreeze different amounts?**
Early blocks hold generic filters that transfer perfectly and must stay
frozen; final blocks hold task-specific features that benefit from adaptation.
MobileNetV2 is small and safe to fully fine-tune (phase 2 at 1e-5), while the
deep models are partially unfrozen — EfficientNetB3's top 50 layers and
DenseNet121's last 40 layers — to avoid destructive updates to pretrained
weights.

**Q5. What is the callback configuration and why?**
Every model uses EarlyStopping (patience 5–7, `restore_best_weights=True`) to
stop when validation loss plateaus and restore the best epoch's weights, and
ModelCheckpoint (`save_best_only`, monitoring validation accuracy) so the saved
`.keras` file is always the best epoch. DenseNet121 additionally uses
ReduceLROnPlateau (factor 0.3, patience 3, min_lr 1e-6) to anneal the learning
rate when validation loss stalls — which is why it gets the largest epoch
budget (30).

**Q6. Why does each model preprocess inputs differently?**
Each backbone has its own documented input contract: MobileNetV2 expects
pixels in [-1,1] (embedded `Rescaling(1/127.5, offset=-1)`), DenseNet121
expects caffe-style mean subtraction (`Lambda(densenet.preprocess_input)`),
and EfficientNet normalizes internally with built-in layers. Our M2 pipeline
feeds float32 in [0,255] — the models normalize themselves, exactly as the
reference implementation does.

**Q7. Why are the benchmark numbers labeled "Reference Implementation Results"?**
We did not retrain the models (project decision). The reported numbers —
Custom CNN 83.97%, MobileNetV2 89.57%, DenseNet121 92.57%, EfficientNetB3
93.02% — come from the reference notebook (2× Tesla T4, random 80/20 split).
Presenting them as our own experimental results would misrepresent the work.
Attribution keeps the project honest and still lets us use the benchmark for
architecture selection.

**Q8. What did you change in the reference code, and why?**
Only integration-level changes: dataset paths replaced with our M2 partition
files (`split_partition.npz`, `split_metadata.json` — the split is never
regenerated); deprecated APIs modernized (`InputLayer(shape=...)` instead of
`input_shape=`, removed deprecated `verbose` argument); hardcoded fallback
accuracies in the comparison cell removed because they could silently
fabricate results — the comparison now falls back to clearly labeled reference
values; outputs redirected to `models/` and `reports/`; seed 42 fixed for
reproducibility.

**Q9. Why does training run without class weights, despite the 2.24:1 imbalance?**
The reference implementation trains without class weights, and we preserved
that behaviour so the benchmark stays comparable with its published results.
The imbalance was quantified in M1/M2, and per-class performance (where
minority-class weaknesses would show) is analysed in Milestone 4 on the
held-out test set — the milestone that produces our own experimental metrics.

**Q10. What happens to the test set, and what is next?**
The 1,848-image test split, isolated in Milestone 2 (seed 42, stratified
70/15/15), is **never touched by training or validation**. Milestone 4 will
evaluate the champion on it — accuracy, per-class precision/recall, confusion
matrix — giving the honest held-out numbers the reference implementation
never produced. Milestone 5 adds Grad-CAM explainability and the Streamlit
deployment.

---

## Common Mistakes to Avoid (learned/anticipated)

| Mistake | Why it is wrong | Correct approach |
| --- | --- | --- |
| Presenting reference benchmark numbers as our results | Misrepresentation; the reference used a different split (random 80/20) | Label all benchmark values "Reference Implementation Results" and attribute them |
| `layers.InputLayer(input_shape=...)` | Deprecated in current Keras — emits warnings, may break in future versions | `InputLayer(shape=...)` / `tf.keras.Input(shape=...)` |
| Hardcoded fallback accuracies in the comparison cell | A skipped run silently reports numbers that were never measured | Fall back to clearly labeled reference values only, and print the data source |
| Feeding [0,1] images to all models | MobileNetV2 expects [-1,1], DenseNet mean-subtraction, EfficientNet built-in normalization — wrong range distorts features | Keep the M2 [0,255] contract; each model normalizes internally |
| Double augmentation | M2 pipeline augmentation + model-embedded augmentation changes the training distribution vs. the reference | Pipeline applies none; augmentation lives inside each model |
| Regenerating the split | Breaks M2's verified stratified partition | Load `split_partition.npz` + `split_metadata.json` verbatim |
| Using the test set for validation | Optimistically biased metrics | Test (1,848) reserved exclusively for Milestone 4 |
| Copying reference markdown | Plagiarism; the course requires original documentation | Rewrite every section in our own academic style with our design system |

---

## One-Sentence Pitch

"Milestone 3 integrates the reference implementation as a four-architecture
benchmark — Custom CNN, MobileNetV2, EfficientNetB3, DenseNet121 — on our
verified Milestone 2 pipeline, preserving the original training logic,
attributing all benchmark results (champion EfficientNetB3 at 93.02%) to the
reference notebook, and reserving our held-out test set for Milestone 4's
honest evaluation."
