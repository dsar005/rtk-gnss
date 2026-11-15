# Federated GNSS–IMU Navigation Quality Prediction

A research starter implementation for training one temporal navigation-quality model across four independent environment clients. It uses a 1D CNN + GRU with two output heads: position-error regression (metres) and navigation-quality classification (High/Medium/Low).

## Important before interpreting results
The CSVs in `data/sample/` are **synthetic demonstration data**, not your actual GNSS/IMU recordings. They let you test the complete pipeline end-to-end. Replace them with properly aligned real sensor records before reporting scientific results. A model trained on synthetic data is not evidence of real-world performance.

## Architecture and data flow
1. Each environment is a client: Medium Urban Canyon, Deep Urban Canyon, Harsh Urban Canyon, Tunnel.
2. Client CSV rows contain GNSS/IMU measurements and synchronized ground-truth coordinates.
3. The label is 3D Euclidean GNSS position error: `sqrt((gnss_x-gt_x)^2 + (gnss_y-gt_y)^2 + (gnss_z-gt_z)^2)`.
4. Error is converted into a quality class: High <= 2 m, Medium <= 5 m, Low > 5 m. These are configurable starting thresholds, not universal standards.
5. Consecutive sensor rows are grouped into 30-sample windows. CNN extracts short-term patterns; GRU learns temporal behaviour.
6. The model predicts error and quality together.
7. Each client trains a local copy. FedAvg combines client model weights; FedProx adds a penalty that limits local model drift for non-identically distributed (non-IID) environments.
8. Held-out later portions of each client dataset are evaluated using MAE, RMSE, R², accuracy, macro precision/recall/F1 and confusion matrix.

## Install and run (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_sample_data.py
python src/train.py --algorithm fedavg
# Or:
python src/train.py --algorithm fedprox
```
Outputs are written to `outputs/`: model checkpoint and per-round metrics.

## Predict on a new sensor CSV
The prediction CSV must contain every configured feature column. Ground truth is not required for inference.
```powershell
python src/predict.py --input path\to\new_sensor_data.csv
```
Predictions are written to `outputs/predictions.csv`.

## Real-data CSV format
One row per synchronized timestamp. Default feature columns are configured in `config.yaml`:
`gnss_x, gnss_y, gnss_z, gnss_speed, gnss_cn0, gnss_satellites, gnss_hdop, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z`.
Also required for training/label creation: `gt_x, gt_y, gt_z`.
Optional columns such as `timestamp` and `trajectory_id` can be retained. Ensure GNSS and ground-truth coordinates use the **same coordinate frame and units (metres)** and timestamps are synchronized. Convert latitude/longitude to a suitable local metric coordinate system first; do not subtract degrees as if they were metres. Map your ROS topics/bag data into this tabular schema with a separate extraction step.

Edit `clients` paths in `config.yaml` to point to your real client CSV files. Change `feature_columns` to match columns actually available in every client. If a sensor channel is unavailable in one environment, use a consistent feature schema across clients and document any imputation.

## Research and deployment caveats
- The current implementation is a **simulated federated-learning experiment in one Python process**. Client datasets are loaded locally by the simulation runner, and only model weights are aggregated. It demonstrates the FL algorithm but is not yet a secure distributed deployment.
- No raw records are sent to the aggregation function, but this code does not implement encryption, secure aggregation, differential privacy, authentication, or network transport.
- For a true multi-device deployment, run each client as a separate process/device and add secure communication and secure aggregation.
- The scaler in this baseline is fitted from training partitions across clients. For strict privacy, replace this with per-client normalization or a privacy-preserving statistics protocol and state the choice in the report.
- The split is chronological within each CSV. For rigorous evaluation, split by complete trajectory/session, not random rows, and avoid overlapping windows across split boundaries. Keep all sessions from a test trajectory out of training.
- Skymask and 3D maps are not consumed by this baseline. They can be added as engineered visibility/occlusion features or a separate spatial encoder once their formats and alignment are known.
- Compare local-only, FedAvg and FedProx; optionally add a centralized upper-bound experiment. Report per-environment metrics and mean/std across repeated seeds.
