import numpy as np
import torch
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    yp, yt, cp, ct = [], [], [], []
    for xb, eb, qb in loader:
        pred, logits = model(xb.to(device))
        yp.extend(pred.cpu().numpy()); yt.extend(eb.numpy())
        cp.extend(logits.argmax(1).cpu().numpy()); ct.extend(qb.numpy())
    yp, yt, cp, ct = map(np.asarray, (yp, yt, cp, ct))
    if len(yt) == 0:
        return {"error": "No evaluation sequences. Reduce sequence_length or provide more rows."}
    p, r, f, _ = precision_recall_fscore_support(ct, cp, labels=[0,1,2],
                                                  average="macro", zero_division=0)
    return {
        "position_error_MAE_m": float(mean_absolute_error(yt, yp)),
        "position_error_RMSE_m": float(np.sqrt(mean_squared_error(yt, yp))),
        "position_error_R2": float(r2_score(yt, yp)) if len(yt) > 1 else None,
        "quality_accuracy": float(accuracy_score(ct, cp)),
        "quality_macro_precision": float(p),
        "quality_macro_recall": float(r),
        "quality_macro_f1": float(f),
        "quality_confusion_matrix_labels_High_Medium_Low": confusion_matrix(ct, cp, labels=[0,1,2]).tolist()
    }
