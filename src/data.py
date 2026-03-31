import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD  = (0.2673, 0.2564, 0.2761)

def get_dataloaders(dataset_name="cifar10", batch_size=128, calib_batch_size=64, calib_size=512, calib_seed=42, num_workers=2):
    if dataset_name.lower() == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        dataset_class = torchvision.datasets.CIFAR100
    else:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        dataset_class = torchvision.datasets.CIFAR10

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = dataset_class(root="./data", train=True, download=True, transform=train_transform)
    test_dataset  = dataset_class(root="./data", train=False, download=True, transform=test_transform)

    # Seeded random subset for calibration
    rng = np.random.default_rng(calib_seed)
    calib_indices = rng.choice(len(train_dataset), calib_size, replace=False).tolist()
    calib_dataset = Subset(train_dataset, calib_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    calib_loader = DataLoader(calib_dataset, batch_size=calib_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, calib_loader