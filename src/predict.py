import argparse, yaml, numpy as np, pandas as pd, torch
from model import NavigationQualityNet

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,help="CSV containing configured sensor feature columns")
    p.add_argument("--model",default="outputs/navigation_quality_model.pt")
    p.add_argument("--config",default="config.yaml")
    p.add_argument("--output",default="outputs/predictions.csv")
    a=p.parse_args()
    cfg=yaml.safe_load(open(a.config,encoding="utf-8"))
    ckpt=torch.load(a.model,map_location="cpu",weights_only=False)
    df=pd.read_csv(a.input)
    cols=ckpt["feature_columns"]
    missing=[c for c in cols if c not in df]
    if missing: raise ValueError(f"Missing input columns: {missing}")
    x=df[cols].replace([np.inf,-np.inf],np.nan).interpolate(limit_direction="both").fillna(0).to_numpy(np.float32)
    x=(x-ckpt["mean"])/ckpt["std"]
    seq=ckpt["sequence_length"]
    model=NavigationQualityNet(len(cols)); model.load_state_dict(ckpt["model_state"]); model.eval()
    rows=[]
    with torch.no_grad():
        for i in range(max(0,len(x)-seq+1)):
            xb=torch.tensor(x[i:i+seq]).unsqueeze(0)
            error,logits=model(xb); cls=int(logits.argmax(1).item())
            rows.append({"window_end_row":i+seq-1,"predicted_position_error_m":float(error.item()),
                         "predicted_quality":["High","Medium","Low"][cls]})
    pd.DataFrame(rows).to_csv(a.output,index=False)
    print(f"Wrote {len(rows)} predictions to {a.output}")
if __name__=="__main__": main()
