## Overview

This project investigates the impact of quantization on CNN performance.

evaluate how **FP16 (half precision)** and **INT8 (post-training quantization)** affect:

* Classification accuracy
* Model calibration (Expected Calibration Error, ECE)
* Robustness under additive Gaussian noise

---

## Experimental Setup

**Dataset:** CIFAR-10
**Architecture:** ResNet-18 (CIFAR-adapted)
**Training Precision:** FP32 (single baseline checkpoint)
**Inference Variants:**

* FP32 (baseline)
* FP16 (half precision inference)
* INT8 (FX-based post-training static quantization)

**Metrics:**

* Top-1 Accuracy (%)
* Expected Calibration Error (ECE)
* Accuracy under Gaussian noise (ε = 0.01, 0.02)

---

## Key Results

| Precision | Accuracy (%) | ECE    | Acc (ε=0.01) | Acc (ε=0.02) |
| --------- | ------------ | ------ | ------------ | ------------ |
| FP32      | 93.75        | 0.0659 | 53.02        | 53.10        |
| FP16      | 93.75        | 0.0662 | 53.06        | 53.17        |
| INT8      | 93.75        | 0.0651 | 53.45        | 53.20        |

### Observations

* **No accuracy degradation** from FP32 → INT8.
* Calibration remains stable across precision levels.
* Noise robustness degradation is dominated by input perturbation, not numeric precision.
* INT8 quantization does not meaningfully harm predictive confidence quality.

---

## Quantization Method

INT8 evaluation uses **FX Graph Mode Post-Training Static Quantization** (PyTorch 2.x).

Pipeline:

1. Load trained FP32 checkpoint
2. Apply FX graph tracing
3. Insert observers
4. Calibrate on 512-sample subset
5. Convert to INT8
6. Evaluate on CPU

---

## Reproducibility

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2. Train Baseline (if checkpoint not present)

Run notebook or script to train FP32 model.

### 3. Evaluate Precision Variants

Notebook automatically:

* Loads checkpoint
* Evaluates FP32
* Casts to FP16
* Performs FX INT8 quantization
* Generates comparison plots

---

## Generated Outputs

* `precision_comparison.png`

  * Accuracy vs Precision
  * ECE vs Precision
  * Accuracy vs Noise Level

* `reliability_diagrams.png`

  * Per-bin calibration analysis for each precision

---

## Takeaways

* Post-training static quantization (INT8) preserves performance on CIFAR-10.
* Calibration stability suggests limited confidence distortion under reduced precision.
* Robustness to Gaussian noise is primarily governed by model sensitivity rather than arithmetic precision.
* FX graph mode quantization is the recommended approach in modern PyTorch for CNN architectures with residual connections.

---

## Potential Extensions

* Measure inference latency (FP32 vs FP16 vs INT8)
* Model size comparison
* Per-class accuracy shifts
* Quantization-aware training (QAT)
* Evaluate on larger datasets (CIFAR-100, TinyImageNet)

---