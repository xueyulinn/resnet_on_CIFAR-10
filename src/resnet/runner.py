from resnet import data, model, engine
import torch.nn.functional as F
import torch
from pathlib import Path
import json


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(torch.cuda.get_device_name(0))
        print(torch.version.cuda)
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"


def run(results_path, seed, ):
    torch.manual_seed(seed)

    dataloader_tr, dataloader_ts = data.load_data()
    epochs = 1
    device = get_device()

    resnet = model.ResNet()
    resnet.to(device)

    loss_fn = F.cross_entropy
    optimizer = torch.optim.SGD(
        params=resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    history = {"tr_loss": [], "tr_error": [], "ts_loss": [], "ts_error": []}
    for _ in range(epochs):
        tr_loss, tr_error = engine.train(
            resnet, dataloader_tr, device, loss_fn, optimizer)
        ts_loss, ts_error = engine.test(resnet, dataloader_ts, device, loss_fn)
        history["tr_loss"].append(tr_loss)
        history["tr_error"].append(tr_error)
        history["ts_loss"].append(ts_loss)
        history["ts_error"].append(ts_error)

    with open(results_path / f"{seed}.json", "w") as f:
        json.dump(history, f)
