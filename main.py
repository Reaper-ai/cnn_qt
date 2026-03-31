import os
import copy
import random
import warnings
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import hydra
from omegaconf import DictConfig

from src.data import get_dataloaders
from src.model import build_resnet18, build_mobilenetv2
from src.train import train_one_epoch
from src.evaluate import evaluate, compute_ece, measure_latency
from src.quantize import cast_to_fp16, apply_int8_ptq, prepare_int8_qat, convert_qat_to_int8
from src.visualize import plot_metrics_comparison, plot_reliability_diagram

warnings.filterwarnings("ignore")

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Setup Reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.model_dir, exist_ok=True)
    fig_dir = os.path.join(os.getcwd(), "fig")
    os.makedirs(fig_dir, exist_ok=True)
    
    num_classes = 100 if cfg.dataset.lower() == "cifar100" else 10

    print(f"Loading {cfg.dataset.upper()} dataset...")
    train_loader, test_loader, calib_loader = get_dataloaders(
        dataset_name=cfg.dataset,
        batch_size=cfg.batch_size, 
        calib_batch_size=cfg.calib_batch_size,
        calib_size=cfg.calib_size,
        calib_seed=cfg.calib_seed
    )

    if cfg.model.lower() == "mobilenetv2":
        fp32_model = build_mobilenetv2(num_classes).to(device)
    else:
        fp32_model = build_resnet18(num_classes).to(device)

    model_filename = f"{cfg.model}_{cfg.dataset}_fp32.pth"
    fp32_path = os.path.join(cfg.model_dir, model_filename)

    # --- 1. Training ---
    if os.path.exists(fp32_path):
        print(f"Checkpoint found: {fp32_path} — skipping base training.")
        fp32_model.load_state_dict(torch.load(fp32_path, map_location=device))
    else:
        print(f"Training {cfg.model} on {cfg.dataset} (FP32) ...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(fp32_model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

        for epoch in range(1, cfg.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(fp32_model, train_loader, criterion, optimizer, device)
            scheduler.step()
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{cfg.epochs} | loss {tr_loss:.4f} | acc {tr_acc:.2f}%")

        torch.save(fp32_model.state_dict(), fp32_path)
    
    fp32_model.eval()

    # Dictionary to hold all metrics for JSON logging and plotting
    results_metrics = {}

    # --- 2. FP32 Evaluation ---
    if cfg.precision in ["fp32", "all"]:
        print("\n--- FP32 Baseline ---")
        acc, conf, correct = evaluate(fp32_model, test_loader, device=device)
        ece, bin_acc, bin_conf, _ = compute_ece(conf, correct)
        latency = measure_latency(fp32_model, test_loader, device=device)
        print(f"Accuracy: {acc:.2f}% | ECE: {ece:.4f} | Latency: {latency:.4f} ms/sample")
        
        results_metrics["FP32"] = {"accuracy": acc, "ece": ece, "latency_ms": latency}
        plot_reliability_diagram(bin_acc, bin_conf, ece, "FP32 Baseline", os.path.join(fig_dir, "rel_diagram_fp32.png"))

    # --- 3. FP16 Evaluation ---
    if cfg.precision in ["fp16", "all"]:
        print("\n--- FP16 Evaluation ---")
        model_fp16 = cast_to_fp16(copy.deepcopy(fp32_model))
        acc, conf, correct = evaluate(model_fp16, test_loader, device=device, half=True)
        ece, bin_acc, bin_conf, _ = compute_ece(conf, correct)
        latency = measure_latency(model_fp16, test_loader, device=device, half=True)
        print(f"Accuracy: {acc:.2f}% | ECE: {ece:.4f} | Latency: {latency:.4f} ms/sample")
        
        results_metrics["FP16"] = {"accuracy": acc, "ece": ece, "latency_ms": latency}
        plot_reliability_diagram(bin_acc, bin_conf, ece, "FP16", os.path.join(fig_dir, "rel_diagram_fp16.png"))

    # --- 4. INT8 PTQ Evaluation ---
    if cfg.precision in ["int8_ptq", "all"]:
        print(f"\n--- INT8 PTQ (Calib Size: {cfg.calib_size}) ---")
        model_int8_ptq = apply_int8_ptq(copy.deepcopy(fp32_model), calib_loader)
        acc, conf, correct = evaluate(model_int8_ptq, test_loader, cpu_model=True)
        ece, bin_acc, bin_conf, _ = compute_ece(conf, correct)
        latency = measure_latency(model_int8_ptq, test_loader, cpu_model=True)
        print(f"Accuracy: {acc:.2f}% | ECE: {ece:.4f} | Latency: {latency:.4f} ms/sample")
        
        results_metrics["INT8_PTQ"] = {"accuracy": acc, "ece": ece, "latency_ms": latency}
        plot_reliability_diagram(bin_acc, bin_conf, ece, "INT8 PTQ", os.path.join(fig_dir, "rel_diagram_int8_ptq.png"))

    # --- 5. INT8 QAT Evaluation ---
    if cfg.precision in ["int8_qat", "all"]:
        print(f"\n--- INT8 QAT ({cfg.qat_epochs} Epochs) ---")
        model_qat = prepare_int8_qat(copy.deepcopy(fp32_model), device=device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_qat.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)

        for epoch in range(1, cfg.qat_epochs + 1):
            train_one_epoch(model_qat, train_loader, criterion, optimizer, device)

        model_int8_qat = convert_qat_to_int8(model_qat)
        acc, conf, correct = evaluate(model_int8_qat, test_loader, cpu_model=True)
        ece, bin_acc, bin_conf, _ = compute_ece(conf, correct)
        latency = measure_latency(model_int8_qat, test_loader, cpu_model=True)
        print(f"Accuracy: {acc:.2f}% | ECE: {ece:.4f} | Latency: {latency:.4f} ms/sample")
        
        results_metrics["INT8_QAT"] = {"accuracy": acc, "ece": ece, "latency_ms": latency}
        plot_reliability_diagram(bin_acc, bin_conf, ece, "INT8 QAT", os.path.join(fig_dir, "rel_diagram_int8_qat.png"))

    # --- 6. Final Logging and Visualizations ---
    print("\n--- Final Results ---")
    metrics_path = os.path.join(cfg.model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results_metrics, f, indent=4)
    print(f"Saved JSON metrics to {metrics_path}")

    if len(results_metrics) > 1:
        plot_metrics_comparison(results_metrics, fig_dir)

if __name__ == "__main__":
    main()