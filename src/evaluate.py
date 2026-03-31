import torch
import numpy as np
import time

@torch.no_grad()
def evaluate(model, loader, device=torch.device("cuda"), half=False, noise_eps=0.0, cpu_model=False):
    """
    Returns accuracy percentage, all confidences, and all correct flags.
    """
    model.eval()
    correct, total = 0, 0
    all_conf, all_correct = [], []

    eval_device = torch.device("cpu") if cpu_model else device

    for inputs, targets in loader:
        inputs  = inputs.to(eval_device)
        targets = targets.to(eval_device)

        if noise_eps > 0.0:
            noise   = torch.randn_like(inputs) * noise_eps
            inputs  = (inputs + noise).clamp(0.0, 1.0)

        if half:
            inputs = inputs.half()

        outputs = model(inputs)
        probs   = torch.softmax(outputs.float(), dim=1)
        conf, preds = probs.max(1)

        correct += preds.eq(targets).sum().item()
        total   += targets.size(0)

        all_conf.extend(conf.cpu().tolist())
        all_correct.extend(preds.eq(targets).cpu().tolist())

    acc = 100.0 * correct / total
    return acc, np.array(all_conf), np.array(all_correct, dtype=bool)

def compute_ece(confidences, corrects, n_bins=10):
    """Calculates Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc  = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_frac = np.zeros(n_bins)
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0: continue
        bin_acc[i]  = corrects[mask].mean()
        bin_conf[i] = confidences[mask].mean()
        bin_frac[i] = mask.sum() / n

    ece = float(np.sum(bin_frac * np.abs(bin_acc - bin_conf)))
    return ece, bin_acc, bin_conf, bin_frac

@torch.no_grad()
def measure_latency(model, loader, device=torch.device("cuda"), half=False, cpu_model=False, num_batches=20):
    """
    Measures inference latency in milliseconds per sample.
    Includes a warmup phase to avoid cold-start penalties.
    """
    model.eval()
    eval_device = torch.device("cpu") if cpu_model else device
    
    # Grab a single batch for measurement to isolate inference time from data loading
    inputs, _ = next(iter(loader))
    inputs = inputs.to(eval_device)
    if half:
        inputs = inputs.half()

    batch_size = inputs.size(0)

    # Warmup
    for _ in range(5):
        _ = model(inputs)

    # Synchronize if using CUDA
    if not cpu_model and device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    
    for _ in range(num_batches):
        _ = model(inputs)
        
    if not cpu_model and device.type == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    
    total_time_ms = (end_time - start_time) * 1000
    ms_per_sample = total_time_ms / (num_batches * batch_size)
    
    return ms_per_sample