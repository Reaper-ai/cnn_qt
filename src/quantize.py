import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig, get_default_qat_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx

def cast_to_fp16(model):
    fp16_model = model.half()
    for m in fp16_model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.float()
    return fp16_model

def apply_int8_ptq(model_fp32, calib_loader, example_input_shape=(1, 3, 32, 32), device="cpu"):
    """Applies Post-Training Static Quantization."""
    torch.backends.quantized.engine = "fbgemm"
    model_fp32.eval()
    model_fp32.cpu()

    qconfig_dict = {"": get_default_qconfig("fbgemm")}
    example_input = torch.randn(*example_input_shape)

    model_prepared = prepare_fx(model_fp32, qconfig_dict, example_input)

    print("Calibrating INT8 model...")
    with torch.no_grad():
        for inputs, _ in calib_loader:
            model_prepared(inputs.cpu())

    model_int8 = convert_fx(model_prepared)
    return model_int8.eval()

def prepare_int8_qat(model_fp32, example_input_shape=(1, 3, 32, 32), device="cuda"):
    """Prepares the model with fake-quantization nodes for QAT."""
    torch.backends.quantized.engine = "fbgemm"
    model_fp32.train()
    
    # QAT preparation usually happens on CPU, then moved to GPU for training
    model_fp32.cpu()
    qconfig_dict = {"": get_default_qat_qconfig("fbgemm")}
    example_input = torch.randn(*example_input_shape)

    model_prepared = prepare_qat_fx(model_fp32, qconfig_dict, example_input)
    return model_prepared.to(device)

def convert_qat_to_int8(model_prepared):
    """Converts the QAT fine-tuned model to actual INT8."""
    model_prepared.eval()
    model_prepared.cpu()
    model_int8 = convert_fx(model_prepared)
    return model_int8.eval()