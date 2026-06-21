# Understanding MiDaS — Model Notes

This document explains how MiDaS (the depth estimation model used in this project) actually works — the problem it solves, its architecture, how it was trained, and its known limitations. It's written for anyone reviewing this repo who wants to understand the model behind the application, not just the application code.

## Table of Contents
- [Why Monocular Depth Estimation Is Hard](#why-monocular-depth-estimation-is-hard)
- [Architecture: Encoder-Decoder & DPT](#architecture-encoder-decoder--dpt)
- [Training Data: The Mixed-Dataset Approach](#training-data-the-mixed-dataset-approach)
- [The Loss Function: Scale & Shift Invariance](#the-loss-function-scale--shift-invariance)
- [Evaluating Accuracy](#evaluating-accuracy)
- [Limitations](#limitations)
- [Relevance to This Project](#relevance-to-this-project)

---

## Why Monocular Depth Estimation Is Hard

With stereo vision (two eyes or two cameras), the brain compares the slightly different angle each eye sees. The difference between the two views — disparity — directly gives distance through geometry. With a single camera, there's only one viewpoint, so there's no second angle to triangulate against.

A single 2D image is a projection of the 3D world, and that projection is **not reversible** — infinite different 3D scenes can produce the exact same 2D image. This is called **scale ambiguity**. For example, from a single photo alone, it's mathematically impossible to distinguish between:

- A real basketball, 2 meters away, normal size
- A giant basketball, 20 meters away, photographed to look the same size
- A printed photo of a basketball taped to a wall, 1 meter away

All three produce identical pixels. Geometry alone cannot resolve this.

**How MiDaS gets around this:** it doesn't use geometry — it uses learned visual priors. By training on millions of images, it learns patterns like typical object sizes, perspective convergence in floors/walls, and occlusion cues (closer objects blocking farther ones). This is the same trick a human uses when judging distance with one eye closed.

---

## Architecture: Encoder-Decoder & DPT

MiDaS follows an **encoder-decoder** pattern:

```
Input Image → ENCODER → feature representation → DECODER → Depth Map
```

The encoder compresses the image into features (what objects/edges/textures exist where). The decoder upsamples those features back into a full-resolution, single-channel depth map.

**The encoder** in the higher-accuracy variants (`DPT_Large`, `DPT_Hybrid`) is a **Vision Transformer (ViT)** — DPT stands for *Dense Prediction Transformer*. It works by:

1. Splitting the image into a grid of small patches (e.g. 16×16 pixels)
2. Flattening each patch into a vector, treated like a "word" in a sentence
3. Applying **self-attention** — every patch attends to every other patch to determine relevance

This matters because depth is contextual: a sky patch needs context from other sky patches to be classified "far"; a road patch needs the whole road's perspective to judge its gradient correctly. Self-attention captures this global context far better than a CNN, which only sees local neighborhoods.

`MiDaS_small` instead uses a lighter CNN-based encoder (EfficientNet) — faster, less accurate, suited for weaker hardware (this is the variant used in this project for CPU-friendly inference).

---

## Training Data: The Mixed-Dataset Approach

MiDaS's core research contribution wasn't a new architecture — it was solving a **practical data problem**. Depth datasets come from very different sources:

| Dataset | Domain | Sensor | Depth Range |
|---|---|---|---|
| KITTI | Outdoor driving | LIDAR | up to ~80m |
| NYU Depth V2 | Indoor rooms | Kinect | up to ~10m |
| MegaDepth | Internet photos | Structure-from-motion | varied |
| *(+ several others)* | | | |

Each dataset uses different units and ranges, so directly mixing them into a standard regression loss would confuse the model about what a "depth value" even means.

**The fix:** MiDaS predicts **relative, scale-invariant depth** instead of absolute metric depth — it only learns "this pixel is closer or farther than that one," never an exact real-world distance. This is *why MiDaS output has no fixed unit* — it's a direct consequence of training across mismatched datasets, not a missing feature.

Training across ~10–12 diverse datasets is also what gives MiDaS its generalization — it performs reasonably on almost any uploaded photo (indoor, outdoor, day, night) without retraining, because it has seen huge variety during training.

---

## The Loss Function: Scale & Shift Invariance

For a given image, let `d` be true (ground-truth) depth and `d̂` ("d-hat") be the predicted depth. A naive loss like `(d − d̂)²` breaks across datasets with different units/ranges.

**MiDaS's fix:** for each image, compute the optimal scale `s` and shift `t` (via least-squares) that best align the prediction to ground truth:

```
d_aligned = s · d̂ + t
```

Then compute the loss *after* this alignment:

```
L = (1/n) · Σ |d_aligned − d|²
```

Because `s` and `t` are recalculated per image, the model is never penalized for predicting in an arbitrary internal scale — only for getting the *relative* ordering and proportions wrong. This is what lets KITTI's 80m range and NYU's 10m range coexist in the same training run.

A more outlier-robust variant of this normalization uses the **median** and **mean absolute deviation (MAD)** instead of least-squares:

```
d_normalized = (d − median(d)) / MAD(d)
```

Median/MAD are far less sensitive to outliers (e.g. a sliver of "infinite depth" sky, or a sensor glitch) than mean/standard-deviation would be, which keeps training stable.

---

## Evaluating Accuracy

Depth estimation isn't measured with a single accuracy percentage like classification. The standard metrics are:

- **AbsRel (Absolute Relative Error):** `(1/n) Σ |d_pred − d_gt| / d_gt` — average error as a percentage of true depth. Lower is better.
- **δ accuracy:** the percentage of pixels where `max(d_pred/d_gt, d_gt/d_pred) < 1.25` (and looser thresholds 1.25², 1.25³). This is the closest equivalent to a "percentage accuracy" metric.
- **Zero-shot cross-dataset evaluation:** MiDaS's headline result — tested on datasets it never trained on, performing competitively against models trained specifically for those datasets. Generalization, not a raw accuracy number, is the actual achievement.

No formal ground-truth benchmarking (e.g. against LIDAR) was performed for this project — there's no sensor-labeled data available for arbitrary uploaded photos. Evaluation here is qualitative: does the model correctly order near vs. far objects, and does the zone-based obstacle logic built on top of it behave sensibly across test images.

---

## Limitations

- **No absolute scale** — output is relative depth only, never real-world distance/units.
- **Thin or reflective objects** (glass, thin railings) are harder for the model to capture accurately, since it relies on learned visual context rather than true geometric measurement.
- **Unusual perspectives** (e.g. extreme angles, atypical scenes) can reduce reliability, since the model's priors come from "typical" photo compositions.
- For a real-world assistive application, MiDaS alone shouldn't be the sole safety mechanism — pairing it with direct-measurement sensors (e.g. ultrasonic, LIDAR) for close-range certainty is the more robust design.

---

## Relevance to This Project

This project's `normalize_depth()` function:

```python
def normalize_depth(depth_map):
    d_min, d_max = depth_map.min(), depth_map.max()
    return (depth_map - d_min) / (d_max - d_min + 1e-8)
```

mirrors the same idea as MiDaS's scale-invariant training — rescaling an arbitrary-range output to a fixed, comparable range (0–1) before any further processing. This step is necessary precisely *because* MiDaS's raw output carries no fixed unit, only relative ordering — a direct consequence of how the model was trained, as explained above.

The zone-based obstacle detection built on top of the depth map (see `zone_analyzer.py`) works entirely on this relative ordering — it never assumes or requires absolute distances, which is consistent with what MiDaS can actually guarantee.
