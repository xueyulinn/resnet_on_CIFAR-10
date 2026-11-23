import torch


def train(model, data, device, loss_fn, optimizer):
    model.train()
    lossi = 0.0
    errori = 0.0
    size = len(data.dataset)
    for X, y in data:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(X)

        errori += (pred.argmax(1) == y).sum().item()
        loss = loss_fn(pred, y)
        lossi += loss.item() * X.shape[0]

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return lossi / size, (1 - errori / size) * 100


def test(model, data, device, loss_fn):
    model.eval()
    lossi = 0.0
    errori = 0.0
    size = len(data.dataset)
    for X, y in data:
        with torch.no_grad():
            X, y = X.to(device, non_blocking=True), y.to(
                device, non_blocking=True)
            pred = model(X)

            errori += (pred.argmax(1) == y).sum().item()
            loss = loss_fn(pred, y)
            lossi += loss.item() * X.shape[0]

    return lossi / size, (1 - errori / size) * 100
