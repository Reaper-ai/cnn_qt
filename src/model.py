import torch.nn as nn
import torchvision.models as tvm

def build_resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for CIFAR (32x32 inputs)."""
    model = tvm.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_mobilenetv2(num_classes: int = 10) -> nn.Module:
    """MobileNetV2 adapted for CIFAR (32x32 inputs)."""
    model = tvm.mobilenet_v2(weights=None)
    # Adjust the first convolutional layer stride for 32x32 input to prevent aggressive downsampling
    model.features[0][0].stride = (1, 1)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model