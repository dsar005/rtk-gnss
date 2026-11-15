import argparse, json, random
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import load_client_csv, chronological_split, SequenceDataset, fit_normalizer, normalize
from model import NavigationQualityNet
from federated import federated_round
from evaluate import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--algorithm", choices=["fedavg","fedprox"], default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alg = args.algorithm or cfg["algorithm"]
    clients = {}
    train_raw = []
    for name, path in cfg["clients"].items():
        x, e, q = load_client_csv(path, cfg["feature_columns"],
            cfg["gnss_position_columns"], cfg["ground_truth_columns"],
            cfg["high_threshold_m"], cfg["medium_threshold_m"])
        tr, va, te = chronological_split(x,e,q,cfg["train_fraction"],cfg["validation_fraction"])
        clients[name] = {"tr":tr, "va":va, "te":te}
        train_raw.append(tr[0])
    # Shared global feature scaling; only aggregate summary statistics in a real deployment.
    mean, std = fit_normalizer(train_raw)
    train_loaders, test_loaders = [], {}
    for name, parts in clients.items():
        norm_parts = {}
        for split, data in parts.items():
            norm_parts[split] = (normalize(data[0],mean,std),data[1],data[2])
        tr, va, te = [SequenceDataset(*norm_parts[k],cfg["sequence_length"]) for k in ("tr","va","te")]
        train_loaders.append(DataLoader(tr,batch_size=cfg["batch_size"],shuffle=True))
        test_loaders[name] = DataLoader(te,batch_size=cfg["batch_size"],shuffle=False)
    model = NavigationQualityNet(len(cfg["feature_columns"])).to(device)
    history = []
    for rnd in range(1,cfg["rounds"]+1):
        federated_round(model,train_loaders,device,cfg["local_epochs"],cfg["learning_rate"],
                        alg,cfg["fedprox_mu"])
        # Client-wise held-out test reporting; test data is not used for training.
        metrics = {name:evaluate(model,loader,device) for name,loader in test_loaders.items()}
        history.append({"round":rnd,"clients":metrics})
        print(f"Round {rnd}/{cfg['rounds']} complete")
    Path("outputs").mkdir(exist_ok=True)
    torch.save({"model_state":model.state_dict(),"feature_columns":cfg["feature_columns"],
                "mean":mean,"std":std,"sequence_length":cfg["sequence_length"],
                "algorithm":alg}, "outputs/navigation_quality_model.pt")
    with open("outputs/metrics.json","w",encoding="utf-8") as f: json.dump(history,f,indent=2)
    print(json.dumps(history[-1],indent=2))
    print("Saved model: outputs/navigation_quality_model.pt")
if __name__ == "__main__":
    main()
