import torch
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader


def load_data(batch_size=128):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    transform_tr = v2.Compose([v2.ToImage(),
                               v2.RandomCrop(
                                   32, padding=4), v2.RandomHorizontalFlip(),
                               v2.ToDtype(torch.float32, scale=True),
                               v2.Normalize(mean, std)])
    transform_ts = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])
    data_tr = datasets.CIFAR10(
        "data", True, transform=transform_tr, download=True)
    data_ts = datasets.CIFAR10(
        "data", False, transform=transform_ts, download=True)
    tr_common = {
        "shuffle": True,
        "batch_size": batch_size,
        "pin_memory": True if torch.cuda.is_available() else False,
        "num_workers": 4,
    }
    ts_common = {
        "shuffle": False,
        "batch_size": batch_size,
        "pin_memory": True if torch.cuda.is_available() else False,
        "num_workers": 4,
    }
    dataloader_tr = DataLoader(data_tr, **tr_common)
    dataloader_ts = DataLoader(data_ts, **ts_common)
    return dataloader_tr, dataloader_ts
