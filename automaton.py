import os
import copy
import json
import random
import warnings
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import all our modular functions
from src.data import get_dataloaders
from src.model import build_resnet18, build_mobilenetv2
from src.train import train_one_epoch
from src.evaluate import evaluate, compute_ece, measure_latency

# Use the appropriate quantize imports based on whether you are using FX or PT2E 
# (Assuming the functions are named as defined in the previous steps)
from src.quantize import (
    cast_to_fp16, 
    apply_int8_ptq, 
    prepare_int8_qat,   # or prepare_int8_qat_pt2e
    convert_qat_to_int8 # or convert_qat_to_int8_pt2e
)
from src.visualize import plot_aggregated_results

warnings.filterwarnings("ignore")

# ==========================================
# EXPERIMENT GRID
# ==========================================
MODELS      = ["resnet18", "mobilenetv2"]
DATASETS    = ["cifar10", "cifar100"]
SEEDS       = [42, 123, 999]
PRECISIONS  = ["FP32", "FP16", "INT8_PTQ", "INT8_QAT"]

# Hyperparameters
EPOCHS       = 30
QAT_EPOCHS   = 3
BATCH_SIZE   = 128
CALIB_SIZE   = 512
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_FILE = "metrics.json"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    os.makedirs("model_wts", exist_ok=True)
    os.makedirs("fig", exist_ok=True)
    
    # Load existing results to allow resuming if Colab crashes
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    def result_exists(m, d, s, p):
        return any(r["model"] == m and r["dataset"] == d and r["seed"] == s and r["precision"] == p for r in all_results)

    # ------------------------------------------
    # THE AUTOMATON LOOP
    # ------------------------------------------
    for dataset, model_name, seed in itertools.product(DATASETS, MODELS, SEEDS):
        print(f"\n{'='*50}")
        print(f"EXPERIMENT: {model_name.upper()} | {dataset.upper()} | SEED: {seed}")
        print(f"{'='*50}")
        
        set_seed(seed)
        num_classes = 100 if dataset == "cifar100" else 10
        
        train_loader, test_loader, calib_loader = get_dataloaders(
            dataset_name=dataset, batch_size=BATCH_SIZE, calib_size=CALIB_SIZE, calib_seed=seed
        )

        # 1. Initialize Model
        if model_name == "mobilenetv2":
            fp32_model = build_mobilenetv2(num_classes).to(DEVICE)
        else:
            fp32_model = build_resnet18(num_classes).to(DEVICE)

        model_path = f"model_wts/{model_name}_{dataset}_seed{seed}_fp32.pth"

        # 2. Train or Load Base FP32 Model
        if os.path.exists(model_path):
            print(f"[*] Found existing FP32 checkpoint -> {model_path}")
            fp32_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        else:
            print(f"[*] Training Base FP32 Model...")
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(fp32_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

            for epoch in range(1, EPOCHS + 1):
                tr_loss, tr_acc = train_one_epoch(fp32_model, train_loader, criterion, optimizer, DEVICE)
                scheduler.step()
                if epoch % 10 == 0:
                    print(f"    Epoch {epoch}/{EPOCHS} | Loss: {tr_loss:.4f} | Acc: {tr_acc:.2f}%")
            
            torch.save(fp32_model.state_dict(), model_path)

        fp32_model.eval()

        # ------------------------------------------
        # EVALUATE ALL PRECISIONS
        # ------------------------------------------
        for precision in PRECISIONS:
            if result_exists(model_name, dataset, seed, precision):
                print(f"[-] Skipping {precision} (Already computed)")
                continue
                
            print(f"\n[>] Running {precision} Evaluation...")
            
            # Helper to log and save
            def log_metrics(acc, ece, lat):
                record = {
                    "model": model_name, "dataset": dataset, "seed": seed,
                    "precision": precision, "accuracy": acc, "ece": ece, "latency_ms": lat
                }
                all_results.append(record)
                with open(RESULTS_FILE, "w") as f:
                    json.dump(all_results, f, indent=4)
                print(f"    -> Acc: {acc:.2f}% | ECE: {ece:.4f} | Latency: {lat:.3f} ms")

            # Execute based on precision type
            if precision == "FP32":
                acc, conf, correct = evaluate(fp32_model, test_loader, device=DEVICE)
                ece, *_ = compute_ece(conf, correct)
                lat = measure_latency(fp32_model, test_loader, device=DEVICE)
                log_metrics(acc, ece, lat)

            elif precision == "FP16":
                model_fp16 = cast_to_fp16(copy.deepcopy(fp32_model))
                acc, conf, correct = evaluate(model_fp16, test_loader, device=DEVICE, half=True)
                ece, *_ = compute_ece(conf, correct)
                lat = measure_latency(model_fp16, test_loader, device=DEVICE, half=True)
                log_metrics(acc, ece, lat)

            elif precision == "INT8_PTQ":
                model_ptq = apply_int8_ptq(copy.deepcopy(fp32_model), calib_loader)
                acc, conf, correct = evaluate(model_ptq, test_loader, cpu_model=True)
                ece, *_ = compute_ece(conf, correct)
                lat = measure_latency(model_ptq, test_loader, cpu_model=True)
                log_metrics(acc, ece, lat)

            elif precision == "INT8_QAT":
                model_qat = prepare_int8_qat(copy.deepcopy(fp32_model), device=DEVICE)
                criterion = nn.CrossEntropyLoss()
                # Low learning rate for fine-tuning
                optimizer = optim.SGD(model_qat.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)

                for _ in range(QAT_EPOCHS):
                    train_one_epoch(model_qat, train_loader, criterion, optimizer, DEVICE)

                model_qat_int8 = convert_qat_to_int8(model_qat)
                acc, conf, correct = evaluate(model_qat_int8, test_loader, cpu_model=True)
                ece, *_ = compute_ece(conf, correct)
                lat = measure_latency(model_qat_int8, test_loader, cpu_model=True)
                log_metrics(acc, ece, lat)

    # 3. Generate Aggregated Plots automatically at the end
    print("\nAll experiments complete. Generating aggregated plots...")
    plot_aggregated_results(all_results, "fig")

if __name__ == "__main__":
    main()