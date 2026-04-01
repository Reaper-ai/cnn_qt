## Overview

This project evaluates quantization effects on CNNs across datasets, model families, and precision modes.

It measures:

* Top-1 accuracy
* Expected Calibration Error (ECE)
* Inference latency (ms/sample)

The code now supports multiple datasets, architectures, and both PTQ and QAT workflows.

---

## What Changed In The Expanded Project

* **Datasets:** CIFAR-10 and CIFAR-100
* **Models:** ResNet-18 (CIFAR-adapted) and MobileNetV2 (CIFAR-adapted)
* **Precisions:** FP32, FP16, INT8 PTQ, INT8 QAT
* **Metrics:** Accuracy, ECE, and latency
* **Automation:** Full experiment grid runner with resume support
* **Plots:** Aggregated plots and per-run reliability diagrams

---

## Project Structure

* `main.py` runs a single configurable experiment via Hydra.
* `automaton.py` runs the full grid of datasets, models, seeds, and precisions.
* `generate.py` builds clean comparison plots from `metrics.json`.
* `config.yaml` is the Hydra configuration used by `main.py`.
* `src/` contains modular training, evaluation, quantization, and plotting code.
* `model_wts/` stores checkpoints.
* `fig/` stores generated plots.

---

## Quickstart

### 1. Install Dependencies

```
pip install -e .
```

### 2. Run A Single Experiment

```
python main.py
```

Override any Hydra config value from the CLI, for example:

```
python main.py dataset=cifar100 model=mobilenetv2 precision=int8_ptq
```

### 3. Run The Full Experiment Grid

```
python automaton.py
```

This will train or reuse checkpoints, evaluate all precisions, and append results to `metrics.json`.

### 4. Generate Aggregated Plots

```
python generate.py
```

---

## Configuration

The default settings live in `config.yaml`. Key options:

* `dataset`: `cifar10` or `cifar100`
* `model`: `resnet18` or `mobilenetv2`
* `precision`: `fp32`, `fp16`, `int8_ptq`, `int8_qat`, or `all`
* `calib_size`: size of the PTQ calibration subset
* `qat_epochs`: fine-tuning epochs for QAT

---

## Quantization Methods

* **INT8 PTQ:** FX Graph Mode Post-Training Static Quantization
* **INT8 QAT:** FX Graph Mode Quantization-Aware Training

Pipeline (PTQ):

1. Load trained FP32 checkpoint
2. FX trace the model
3. Insert observers
4. Calibrate on a subset
5. Convert to INT8
6. Evaluate on CPU

---

## Outputs

* `model_wts/` checkpoints per model and dataset
* `metrics.json` with per-run metrics (used by `generate.py`)
* `fig/` with:
  * Per-precision reliability diagrams from `main.py`
  * Aggregated accuracy, latency, and ECE plots from `automaton.py` or `generate.py`

---

## Notes

* INT8 models evaluate on CPU, while FP32/FP16 evaluate on the configured device.
* `automaton.py` resumes from existing `metrics.json` and skips completed runs.

---

## Potential Extensions

* Model size comparison
* Per-class accuracy analysis
* Larger datasets (TinyImageNet)
* QAT hyperparameter sweeps
* End-to-end throughput benchmarks

---