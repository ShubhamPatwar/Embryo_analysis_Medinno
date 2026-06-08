# 🧬 DINO-Based Embryo Viability Classification Report

---

## 1. Project Overview

This project explores the use of **Self-Supervised Learning (DINO — Self-Distillation with No Labels)** to extract meaningful visual representations from embryo images for the task of **viability classification (viable vs non-viable)**.

The core motivation: given the scarcity of labeled biomedical data, can large-scale unlabeled pretraining produce representations strong enough to support accurate viability prediction?

---

## 2. Dataset Description

| Category | Count |
|---|---|
| Total images | 2,344 |
| Unlabeled (SSL pretraining) | 1,590 |
| Labeled (supervised training) | 754 |

### Label Distribution (Labeled Set)

| Class | Proportion |
|---|---|
| Viable (Class 1) | ~30% |
| Non-viable (Class 0) | ~70% |

The dataset is notably **imbalanced**, with non-viable embryos comprising roughly 70% of labeled samples. This imbalance heavily influenced evaluation choices — balanced accuracy was used as the primary metric rather than raw accuracy.

---

## 3. Methodology

### Stage 1: Data Preparation

All 2,344 images were pooled for self-supervised pretraining. Standard augmentations were applied:

- Random resized crop
- Horizontal flip
- Color jitter
- Gaussian blur
- Normalization: ImageNet mean/std
- Input resolution: 224 × 224

---

### Stage 2: DINO Pretraining

A Vision Transformer (ViT-Small, patch size 16) was used as the backbone.

| Parameter | Value |
|---|---|
| Backbone | ViT-Small/16 |
| Initialization | ImageNet DINO checkpoint |
| Objective | Self-distillation (student-teacher) |
| Optimizer | AdamW |
| Training epochs | 50 |
| Teacher update | EMA momentum |
| Batch size | 32 |

**Outcome:**

- Stable convergence throughout training
- Loss decreased from **~11.1 → ~2.3**
- No collapse or instability detected

The pretraining phase was successful — the model learned to produce consistent patch-level features from embryo images.

---

### Stage 3: Representation Evaluation (k-NN)

Before any supervised training, the quality of the frozen DINO embeddings was probed using **k-Nearest Neighbours (k-NN) classification** — a zero-parameter test of raw embedding separability.

---

### Stage 4A: Linear Probe

A **single linear layer** was trained on top of the frozen DINO backbone using the labeled set. Class weights were applied to the loss function to counter the imbalance.

---

### Stage 4B: Full Fine-Tuning

The **entire ViT backbone + classifier** were trained end-to-end on the labeled data. Same class weighting applied.

---

### Stage 4C: MLP Head (Strategy Pivot)

After observing that fine-tuning degraded performance, the strategy shifted to keeping the **backbone frozen** and replacing the linear classifier with a stronger **MLP head** featuring:

- Multiple fully connected layers
- Batch Normalization
- Dropout for regularization
- Class-weighted cross-entropy loss

---

## 4. Results

### 4.1 k-NN Probe (Frozen Embeddings)

| Metric | Value |
|---|---|
| Accuracy | 64.9% |
| Balanced Accuracy | 52.9% |

**Class-wise recall:**

| Class | Recall |
|---|---|
| Non-viable (Class 0) | 85% |
| Viable (Class 1) | 21% |

The high non-viable recall vs very low viable recall shows the embedding space is biased — features cluster around the majority class. DINO captures general embryo morphology, but viability boundaries are not clearly encoded.

---

### 4.2 Linear Probe (Frozen Backbone)

| Metric | Value |
|---|---|
| Accuracy | 62.3% |
| Balanced Accuracy | **58.0%** ✅ Best |

**Class-wise recall:**

| Class | Recall |
|---|---|
| Non-viable (Class 0) | ~69% |
| Viable (Class 1) | **47%** (up from 21%) |

A significant jump in viable recall compared to k-NN. The linear classifier was able to find a better decision boundary in the embedding space, confirming that the DINO features contain *some* viability signal even if it is not geometrically obvious.

---

### 4.3 Full Fine-Tuning

| Metric | Value |
|---|---|
| Accuracy | 60.0% |
| Balanced Accuracy | 53.0% |

**Class-wise recall:**

| Class | Recall |
|---|---|
| Non-viable (Class 0) | 71% |
| Viable (Class 1) | 34% |

Performance dropped relative to the linear probe. Fine-tuning the backbone on only 754 samples led to partial overfitting and erosion of the pretrained representations.

---

### 4.4 Summary Comparison

| Method | Accuracy | Balanced Accuracy |
|---|---|---|
| k-NN (frozen) | 64.9% | 52.9% |
| Linear Probe (frozen) | 62.3% | **58.0%** |
| Full Fine-Tuning | 60.0% | 53.0% |

---

## 5. Key Observations

### ✅ DINO pretraining was stable and beneficial
The SSL phase converged cleanly and produced embeddings that outperform a random baseline. The linear probe's 58% balanced accuracy confirms the representations carry real viability signal.

### ⚠️ Viability is weakly encoded in visual morphology
Viable and non-viable embryos often look morphologically similar. Static image features alone may be insufficient to reliably distinguish viability — this is a fundamental biological constraint, not a modelling failure.

### ❌ Fine-tuning degraded performance
With only ~226 training samples (30% of 754), the ViT-Small backbone does not have enough supervised signal to fine-tune safely. The model began overwriting useful pretrained structure, leading to worse generalisation.

### ⚠️ Class imbalance dominates behaviour
Across all methods, non-viable recall is consistently high while viable recall stays low. The model defaults to the majority class under uncertainty. Class weighting helped but did not fully resolve this.

---

## 6. Problems Encountered

| Problem | Impact | Root Cause |
|---|---|---|
| Weak visual-viability correlation | Low ceiling on performance | Viability not fully encoded in morphology |
| Fine-tuning hurt performance | Balanced accuracy fell 5% | Insufficient labeled data for ViT fine-tuning |
| Class imbalance | Low viable recall across all stages | ~70/30 non-viable/viable split |
| Very small supervised dataset | Overfitting risk | Only 754 labeled samples |
| Aggressive train/val split (30:70) | ~226 training samples | Chosen to ensure robust validation |

---

## 7. Conclusions

DINO provides a **useful but not sufficient** visual representation for embryo viability prediction on this dataset.

> The pretrained embeddings contain signal, but supervised fine-tuning on small data is not stable enough to improve beyond a linear probe.

The best strategy going forward is to maximise what can be extracted from frozen representations rather than attempting to modify the backbone.

---

## 8. Recommendations

### Immediate (High Priority)

- **MLP head on frozen backbone** — replace the linear probe with a deeper head (batch norm + dropout) for better non-linear separation
- **Threshold tuning** — rather than hard argmax, tune the decision threshold to optimise for balanced accuracy or F1
- **Stronger augmentation** during supervised training (flips, colour jitter, mixup) to regularise the small labeled set

### Model Improvements (Medium Priority)

- **UMAP / t-SNE visualisation** of DINO embeddings to understand what the feature space actually captures
- **Layer-wise learning rates or gradual unfreezing** if fine-tuning is revisited — rather than training all layers at once
- **ViT + CNN hybrid** — combine ViT global features with CNN local texture features

### Data Improvements (Longer Term)

- **Increase labeled dataset size** — even 200–300 additional labeled samples could significantly change the fine-tuning picture
- **Temporal / developmental sequence data** — viability may be better predicted from time-lapse sequences than static images
- **Non-visual metadata** — patient age, cycle day, clinical annotations could complement visual features

---

*Report compiled from experimental results — DINO Embryo Viability Classification Project*
