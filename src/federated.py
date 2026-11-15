import copy
import torch
from torch import nn
from torch.utils.data import DataLoader

def train_local(model, loader, device, epochs, lr, global_state=None, mu=0.0):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.SmoothL1Loss()
    ce = nn.CrossEntropyLoss()
    reference = None
    if global_state is not None and mu > 0:
        reference = {k: v.detach().clone().to(device) for k, v in global_state.items()}
    for _ in range(epochs):
        for xb, eb, qb in loader:
            xb, eb, qb = xb.to(device), eb.to(device), qb.to(device)
            optimizer.zero_grad()
            pred_error, logits = model(xb)
            loss = mse(pred_error, eb) + ce(logits, qb)
            # FedProx: discourage local model from drifting too far from global model.
            if reference is not None:
                prox = sum(torch.sum((p - reference[n]) ** 2)
                           for n, p in model.named_parameters())
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            optimizer.step()
    return copy.deepcopy(model.state_dict())

def fedavg(state_dicts, weights):
    total = float(sum(weights))
    result = {}
    for key in state_dicts[0]:
        if torch.is_floating_point(state_dicts[0][key]):
            result[key] = sum(sd[key] * (w / total)
                              for sd, w in zip(state_dicts, weights))
        else:
            result[key] = state_dicts[0][key].clone()
    return result

def federated_round(model, client_loaders, device, epochs, lr, algorithm="fedavg", mu=0.01):
    global_state = copy.deepcopy(model.state_dict())
    client_states, weights = [], []
    for loader in client_loaders:
        local_model = copy.deepcopy(model).to(device)
        local_model.load_state_dict(global_state)
        state = train_local(local_model, loader, device, epochs, lr,
                            global_state=global_state,
                            mu=(mu if algorithm.lower() == "fedprox" else 0.0))
        client_states.append(state)
        weights.append(len(loader.dataset))
    model.load_state_dict(fedavg(client_states, weights))
    return model
