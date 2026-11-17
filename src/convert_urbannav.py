"""
Convert UrbanNav dataset raw files into synchronized client CSVs ready for federated training.

Supports:
- IMU CSV (GPS Time, angular velocity x/y/z, linear acceleration x/y/z)
- Ground Truth Reference (NovAtel SPAN-CPT or Applanix POS LV: lat, lon, height or ENU)
- GNSS receiver solution (u-blox / SPAN / standard NMEA/PVT lat, lon, height, satellites, HDOP, C/N0)

Outputs synchronized tabular CSV with East-North-Up (ENU) coordinates and kinematic features.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """
    Convert geodetic coordinates (lat, lon, alt in degrees & meters WGS84)
    to local East, North, Up (meters) relative to a reference anchor (lat0, lon0, alt0).
    Uses WGS84 ellipsoid parameters without requiring external external heavy C-libraries.
    """
    # WGS-84 constants
    a = 6378137.0          # semi-major axis
    f = 1.0 / 298.257223563  # flattening
    e2 = 2 * f - f ** 2    # first eccentricity squared

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    # Prime vertical radius of curvature
    def get_xyz(phi, lam, h):
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        n = a / np.sqrt(1.0 - e2 * sin_phi ** 2)
        x = (n + h) * cos_phi * np.cos(lam)
        y = (n + h) * cos_phi * np.sin(lam)
        z = (n * (1.0 - e2) + h) * sin_phi
        return x, y, z

    x, y, z = get_xyz(lat_rad, lon_rad, alt)
    x0, y0, z0 = get_xyz(lat0_rad, lon0_rad, alt0)

    dx, dy, dz = x - x0, y - y0, z - z0

    # Rotation to ENU
    sin_lat0 = np.sin(lat0_rad)
    cos_lat0 = np.cos(lat0_rad)
    sin_lon0 = np.sin(lon0_rad)
    cos_lon0 = np.cos(lon0_rad)

    e = -sin_lon0 * dx + cos_lon0 * dy
    n = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
    u = cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz

    return e, n, u


def load_raw_imu(imu_path):
    """Load UrbanNav imu.csv."""
    df = pd.read_csv(imu_path)
    # UrbanNav IMU files typically have columns:
    # timestamp / %time / gps_time, ang_vel_x, ang_vel_y, ang_vel_z, linear_acc_x, linear_acc_y, linear_acc_z
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip().replace("%", "")
        if "time" in cl:
            col_map[c] = "timestamp"
        elif "ang" in cl or "gyro" in cl:
            if "x" in cl: col_map[c] = "gyro_x"
            elif "y" in cl: col_map[c] = "gyro_y"
            elif "z" in cl: col_map[c] = "gyro_z"
        elif "acc" in cl:
            if "x" in cl: col_map[c] = "accel_x"
            elif "y" in cl: col_map[c] = "accel_y"
            elif "z" in cl: col_map[c] = "accel_z"

    if len(col_map) >= 7:
        df = df.rename(columns=col_map)
    else:
        # Fallback to positional mapping for headerless exports
        default_names = ["timestamp", "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]
        df.columns = default_names[:len(df.columns)]

    return df[["timestamp", "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]].sort_values("timestamp")


def load_raw_trajectory(csv_path, is_ground_truth=False):
    """Load GNSS or Ground Truth reference trajectory."""
    df = pd.read_csv(csv_path)
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip().replace("%", "")
        if "time" in cl: col_map[c] = "timestamp"
        elif "lat" in cl: col_map[c] = "lat"
        elif "lon" in cl: col_map[c] = "lon"
        elif "alt" in cl or "height" in cl: col_map[c] = "alt"
        elif "speed" in cl or "vel" in cl: col_map[c] = "speed"
        elif "hdop" in cl: col_map[c] = "hdop"
        elif "sat" in cl or "num_sv" in cl: col_map[c] = "satellites"
        elif "cn0" in cl or "snr" in cl: col_map[c] = "cn0"

    df = df.rename(columns=col_map)
    return df.sort_values("timestamp")


def synchronize_and_convert(imu_df, gnss_df, gt_df, output_path, resample_hz=10.0):
    """
    Synchronizes IMU, GNSS, and GT streams to a target frequency (e.g. 10 Hz)
    and converts global coordinates to local ENU coordinates.
    """
    # Define common time span
    t_start = max(imu_df["timestamp"].min(), gnss_df["timestamp"].min(), gt_df["timestamp"].min())
    t_end = min(imu_df["timestamp"].max(), gnss_df["timestamp"].max(), gt_df["timestamp"].max())

    if t_start >= t_end:
        raise ValueError("No overlapping timestamp range between IMU, GNSS, and Ground Truth data.")

    step = 1.0 / resample_hz
    target_time = np.arange(t_start, t_end, step)

    # Reference origin for ENU projection is the initial Ground Truth point
    lat0 = gt_df["lat"].iloc[0]
    lon0 = gt_df["lon"].iloc[0]
    alt0 = gt_df["alt"].iloc[0]

    # Convert GNSS lat/lon/alt -> ENU
    gnss_e, gnss_n, gnss_u = geodetic_to_enu(gnss_df["lat"], gnss_df["lon"], gnss_df["alt"], lat0, lon0, alt0)
    gnss_df["gnss_e"] = gnss_e
    gnss_df["gnss_n"] = gnss_n
    gnss_df["gnss_u"] = gnss_u

    # Convert GT lat/lon/alt -> ENU
    gt_e, gt_n, gt_u = geodetic_to_enu(gt_df["lat"], gt_df["lon"], gt_df["alt"], lat0, lon0, alt0)
    gt_df["gt_e"] = gt_e
    gt_df["gt_n"] = gt_n
    gt_df["gt_u"] = gt_u

    # Interpolate onto synchronized target grid
    out = pd.DataFrame({"timestamp": target_time})

    for col in ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"]:
        out[col] = np.interp(target_time, imu_df["timestamp"], imu_df[col])

    for col in ["gnss_e", "gnss_n", "gnss_u"]:
        out[col] = np.interp(target_time, gnss_df["timestamp"], gnss_df[col])

    for col in ["speed", "hdop", "satellites", "cn0"]:
        if col in gnss_df.columns:
            out[f"gnss_{col}"] = np.interp(target_time, gnss_df["timestamp"], gnss_df[col])
        else:
            # Reasonable fallback defaults if receiver doesn't export them
            if col == "speed": out["gnss_speed"] = 0.0
            elif col == "hdop": out["gnss_hdop"] = 1.5
            elif col == "satellites": out["gnss_satellites"] = 12.0
            elif col == "cn0": out["gnss_cn0"] = 35.0

    for col in ["gt_e", "gt_n", "gt_u"]:
        out[col] = np.interp(target_time, gt_df["timestamp"], gt_df[col])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Successfully processed {len(out)} synchronized records -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess UrbanNav trajectory sequences into federated client CSVs.")
    parser.add_argument("--imu", required=True, help="Path to raw imu.csv")
    parser.add_argument("--gnss", required=True, help="Path to GNSS receiver trajectory CSV (with lat, lon, alt)")
    parser.add_argument("--gt", required=True, help="Path to Ground Truth CSV (with lat, lon, alt)")
    parser.add_argument("--output", required=True, help="Output destination CSV path")
    parser.add_argument("--hz", type=float, default=10.0, help="Resampling frequency (Hz)")
    args = parser.parse_args()

    imu = load_raw_imu(args.imu)
    gnss = load_raw_trajectory(args.gnss)
    gt = load_raw_trajectory(args.gt, is_ground_truth=True)
    synchronize_and_convert(imu, gnss, gt, args.output, args.hz)


if __name__ == "__main__":
    main()
