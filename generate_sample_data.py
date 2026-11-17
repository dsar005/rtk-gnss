"""Generate illustrative client datasets based on UrbanNav Hong Kong environments and ENU coordinate frames."""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
out = Path("data/sample")
out.mkdir(parents=True, exist_ok=True)

# UrbanNav scenarios modeled after Hong Kong urban canyon conditions:
# - Medium Urban Canyon (e.g., Tsim Sha Tsui / TST): Moderate multipath, ~2-4m GNSS error
# - Deep Urban Canyon (e.g., Mong Kok / Whampoa): Heavy skyscraper obstruction, high NLOS, ~5-12m GNSS error
# - Harsh Urban Canyon: Severe satellite obscuration, low satellite visibility, high HDOP, ~8-20m GNSS error
# - Tunnel / Underpass (e.g., Cross-Harbour Tunnel approaches): GNSS outage/severe degradation, ~15-30m drift
settings = {
    "Medium_Urban_Canyon": {"noise": 2.2, "nlos_bias": 1.5, "cn0_base": 38.0, "sats": 14, "hdop": 1.2},
    "Deep_Urban_Canyon":   {"noise": 4.5, "nlos_bias": 6.0, "cn0_base": 30.0, "sats": 9,  "hdop": 2.4},
    "Harsh_Urban_Canyon":  {"noise": 7.5, "nlos_bias": 12.0, "cn0_base": 24.0, "sats": 6,  "hdop": 4.2},
    "Tunnel":              {"noise": 14.0, "nlos_bias": 20.0, "cn0_base": 15.0, "sats": 4,  "hdop": 6.5}
}

for name, s in settings.items():
    n = 2500
    # 10 Hz sampling rate (0.1s delta) matching UrbanNav benchmark logging
    dt = 0.1
    t = np.arange(n) * dt

    # Vehicle simulated kinematics in local East-North-Up (ENU) frame in meters
    speed = 8.0 + 3.0 * np.sin(t / 15.0)  # vehicle speed ~5-11 m/s (18-40 km/h urban driving)
    yaw_rate = 0.05 * np.cos(t / 10.0)    # turning yaw rate
    heading = np.cumsum(yaw_rate) * dt

    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)

    gt_e = np.cumsum(vx) * dt
    gt_n = np.cumsum(vy) * dt
    gt_u = 5.0 + 0.5 * np.sin(t / 20.0)

    # IMU angular velocities and accelerations (UrbanNav format: rad/s, m/s^2)
    accel_forward = np.gradient(speed, dt)
    accel_lateral = speed * yaw_rate
    accel_x = accel_forward * np.cos(heading) - accel_lateral * np.sin(heading) + rng.normal(0, 0.05, n)
    accel_y = accel_forward * np.sin(heading) + accel_lateral * np.cos(heading) + rng.normal(0, 0.05, n)
    accel_z = 9.81 + rng.normal(0, 0.05, n)  # gravity component on body-up axis

    gyro_x = rng.normal(0, 0.01, n)
    gyro_y = rng.normal(0, 0.01, n)
    gyro_z = yaw_rate + rng.normal(0, 0.01, n)

    # GNSS position measurements degraded by urban multipath and non-line-of-sight (NLOS)
    noise = s["noise"]
    nlos_bias = s["nlos_bias"] * np.clip(np.sin(t / 8.0), 0, 1)  # episodic multipath reflection bursts
    err_e = rng.normal(0, noise, n) + nlos_bias * 0.8
    err_n = rng.normal(0, noise, n) + nlos_bias * 0.6
    err_u = rng.normal(0, noise * 1.5, n) + nlos_bias * 0.5

    gnss_e = gt_e + err_e
    gnss_n = gt_n + err_n
    gnss_u = gt_u + err_u

    # Receiver GNSS quality indicators
    gnss_speed = speed + rng.normal(0, 0.25, n)
    gnss_cn0 = np.clip(s["cn0_base"] + rng.normal(0, 2.5, n) - (nlos_bias * 0.5), 10.0, 50.0)
    gnss_satellites = np.clip(np.round(s["sats"] + rng.normal(0, 1.5, n)), 3, 24)
    gnss_hdop = np.clip(s["hdop"] + rng.normal(0, 0.2, n) + (nlos_bias * 0.1), 0.5, 9.9)

    df = pd.DataFrame({
        "timestamp": t,
        "gnss_e": gnss_e,
        "gnss_n": gnss_n,
        "gnss_u": gnss_u,
        "gnss_speed": gnss_speed,
        "gnss_cn0": gnss_cn0,
        "gnss_satellites": gnss_satellites,
        "gnss_hdop": gnss_hdop,
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,
        "gt_e": gt_e,
        "gt_n": gt_n,
        "gt_u": gt_u
    })

    file_path = out / f"{name}.csv"
    df.to_csv(file_path, index=False)

print("Created updated UrbanNav-aligned client CSV files in data/sample/ (using ENU coordinate frame).")
