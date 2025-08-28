"""
Generate synthetic but realistic vessel routing dataset for one route (many vessels / voyages).
Output: CSV or Excel with full feature set suitable for Directed-Info GAIL + VAE pretrain.

Usage example:
python generate_vessel_dataset.py --n_trajectories 520 --samples 1000 --timestep 5 \
  --origin_lat 37.94 --origin_lon 23.64 --dest_lat 1.29 --dest_lon 103.85 \
  --out vessel_dataset_full_geo.csv --format csv
"""

import argparse
import os
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
from pyproj import Geod

# -------------------------
# Utilities
# -------------------------
geod = Geod(ellps="WGS84")

def haversine_nm(lat1, lon1, lat2, lon2):
    # returns nautical miles
    # use geod to get distance in meters
    az12, az21, dist_m = geod.inv(lon1, lat1, lon2, lat2)
    return dist_m / 1852.0

def interpolate_geodesic(lat1, lon1, lat2, lon2, n):
    """
    Return n points including endpoints along geodesic from (lat1,lon1) to (lat2,lon2)
    """
    # use geod.npts to get n-2 points between endpoints
    if n == 1:
        return [lat1], [lon1]
    if n == 2:
        return [lat1, lat2], [lon1, lon2]
    npts = geod.npts(lon1, lat1, lon2, lat2, n - 2)
    lons = [lon1] + [p[0] for p in npts] + [lon2]
    lats = [lat1] + [p[1] for p in npts] + [lat2]
    return lats, lons

# -------------------------
# Generator
# -------------------------
def generate_dataset(
    n_trajectories=520,
    samples_per_traj=1000,
    time_step_min=5,
    origin=(37.94, 23.64),
    dest=(1.29, 103.85),
    out_path="vessel_dataset_full_geo.csv",
    out_format="csv",
    seed=123
):
    np.random.seed(seed)

    # column order: timestamp after trajectory_id as requested
    columns = [
        "vessel_id", "trajectory_id", "timestamp",
        # position & motion
        "lat", "lon", "dlat", "dlon", "sog_kn",
        # kinematics & headings (derived)
        "ax_kn_per_h", "yaw_rate_deg_per_min", "cross_track_nm", "along_track_nm",
        # metocean
        "wind_u10_ms", "wind_v10_ms", "wind_speed_ms", "wind_dir_deg",
        "wave_hs_m", "wave_tp_s", "wave_dir_deg",
        "current_u_ms", "current_v_ms",
        # modes / bins
        "sea_state_bin", "proximity_bin", "traffic_bin",
        # ship state / loading
        "draft_fwd_m", "draft_aft_m", "trim_m", "displacement_t", "load_bin",
        "eca_flag", "tss_lane_flag",
        # traffic / safety
        "targets_in_5nm", "min_cpa_nm", "min_tcpa_min", "depth_under_keel_m", "speed_limit_kn",
        # energy / fuel
        "est_power_kw", "sfoc_gpkwh", "fuel_tph", "fuel_per_nm_t",
        "fuel_price_usd_per_ton", "fuel_consumed_t", "fuel_cost_usd",
        # actions (expert / demo)
        "delta_heading_cmd_deg", "speed_setpoint_kn",
        # waypoint / helpers
        "remaining_distance_nm", "eta_error_min",
        # mode labels for weak supervision
        "mode_sea_state", "mode_load", "mode_proximity", "mode_traffic"
    ]

    rows = []
    total_distance_nm = haversine_nm(origin[0], origin[1], dest[0], dest[1])
    # approximate total required hours if steady speed ~ 12-16 knots -> choose baseline duration per traj
    # We'll compute SOG per step from distance/time with jitter below.

    # For each trajectory (vessel + voyage day)
    for traj_idx in range(1, n_trajectories + 1):
        vessel_id = f"V{traj_idx:04d}"
        traj_id = f"TR{traj_idx:04d}"

        # start time (spread voyages across days)
        start_time = datetime(2025, 1, 1) + timedelta(hours=traj_idx)

        # generate geodesic path (lat/lon samples)
        lats, lons = interpolate_geodesic(origin[0], origin[1], dest[0], dest[1], samples_per_traj)

        # add moderate low-frequency deviation to path to simulate route variation (weather/avoidance)
        # create smooth noise by cumulative sum of small gaussian increments filtered
        smooth_scale = 0.02  # degrees
        noise_lat = np.cumsum(np.random.normal(scale=smooth_scale, size=samples_per_traj))
        noise_lon = np.cumsum(np.random.normal(scale=smooth_scale, size=samples_per_traj))
        # center noise to zero mean to keep endpoints approximately same
        noise_lat = noise_lat - np.mean(noise_lat) * (np.linspace(0,1,samples_per_traj))
        noise_lon = noise_lon - np.mean(noise_lon) * (np.linspace(0,1,samples_per_traj))
        lats = np.array(lats) + noise_lat * 0.2  # damp noise
        lons = np.array(lons) + noise_lon * 0.2

        # vessel / voyage specific params
        displacement = float(np.random.choice([60000, 80000, 100000, 120000, 140000]))  # tonnes
        draft_fwd = float(np.clip(np.random.normal(11.0, 0.8), 8.5, 15.0))
        draft_aft = float(np.clip(draft_fwd + np.random.normal(0.2, 0.3), 8.5, 15.5))
        trim = draft_aft - draft_fwd
        load_bin = int(np.random.choice([0,1,2], p=[0.15,0.35,0.5]))  # 0 ballast, 2 full
        # fuel price varies by voyage but consistent along it
        fuel_price = float(np.random.uniform(450, 750))  # USD/ton
        # SFOC baseline
        sfoc = float(np.random.uniform(165, 190))  # g/kWh
        # engine scaling constant k (approx) from displacement to power
        # choose base_engine_power_kw proportional to displacement
        base_engine_power_kw = float(np.random.uniform(0.09, 0.18) * displacement)  # rough MCR-like
        # set sea-state baseline
        base_sea_state = int(np.clip(np.random.choice([0,1,2,3]), 0, 5))

        # compute distances between successive lat/lon points (NM)
        dists_nm = []
        for i in range(samples_per_traj-1):
            nm = haversine_nm(lats[i], lons[i], lats[i+1], lons[i+1])
            dists_nm.append(nm)
        dists_nm.append(0.0)
        dists_nm = np.array(dists_nm)

        # estimate expected mean SOG such that completion time reasonable:
        # choose baseline mean_speed between 12-18 knots
        mean_speed = np.random.uniform(12, 17)
        # compute SOG per step as distance/(time_hours) + noise
        dt_hours = time_step_min / 60.0

        # generate per-step random perturbations of speed but keep average ~ mean_speed
        speed_noise = np.random.normal(scale=0.6, size=samples_per_traj)
        # compute SOG: if geodesic points spacing small, derive from dists; if zero due to rounding, fallback to mean_speed
        sog = np.where(dists_nm > 1e-6, dists_nm / dt_hours, mean_speed)  # knots
        # Blend with mean_speed + noise to avoid exact dists when user wants standard speed pattern
        sog = 0.6 * (mean_speed + speed_noise) + 0.4 * sog
        sog = np.clip(sog, 1.0, 24.0)

        # compute along_track cumulative and remaining distance
        along_track = np.cumsum(dists_nm)
        remaining = total_distance_nm - along_track

        # generate environmental fields as slowly-varying time series
        wind_u = np.random.normal(loc=0.0, scale=4.0, size=samples_per_traj)
        wind_v = np.random.normal(loc=0.0, scale=3.5, size=samples_per_traj)
        # scale baseline
        wind_u += np.random.uniform(-8,8)
        wind_v += np.random.uniform(-8,8)
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        wind_dir = (np.degrees(np.arctan2(wind_v, wind_u)) + 360) % 360

        wave_hs = np.abs(np.random.normal(loc=1.6 + base_sea_state*0.5, scale=0.6, size=samples_per_traj))
        wave_tp = np.clip(np.random.normal(loc=7.0, scale=1.2, size=samples_per_traj), 3.0, 20.0)
        wave_dir = (wind_dir + np.random.normal(scale=30.0, size=samples_per_traj)) % 360

        current_u = np.random.normal(loc=0.0, scale=0.3, size=samples_per_traj)
        current_v = np.random.normal(loc=0.0, scale=0.25, size=samples_per_traj)

        # generate traffic density time-series (Poisson-like variations)
        # average around 2 targets for open sea, larger near approach
        traffic_base = np.random.poisson(lam=1.5, size=samples_per_traj) + np.random.choice([0,1,2], p=[0.7,0.2,0.1])
        # set proximity bin: near coastal in last 10% of samples
        proximity_bin_series = np.zeros(samples_per_traj, dtype=int)
        last_approach_idx = int(samples_per_traj * 0.12)
        proximity_bin_series[-last_approach_idx:] = 2  # harbor approach
        # mid-coast region earlier
        mid_idx = int(samples_per_traj * 0.35)
        proximity_bin_series[mid_idx:mid_idx+int(samples_per_traj*0.2)] = 1

        # fuel/power model per sample
        # est_power_kw = k * speed^3, choose k such that base_engine_power_kw ~ mean_speed^3 * k
        k = base_engine_power_kw / max((mean_speed**3), 1e-6)
        est_power_kw_arr = k * (sog ** 3)
        fuel_tph = (est_power_kw_arr * sfoc) / 1e6  # tonnes per hour
        fuel_per_nm_t = fuel_tph / np.maximum(sog, 1e-3)  # t per nm
        fuel_consumed_t = fuel_tph * dt_hours  # per sample (5 minutes -> dt_hours = 5/60)
        fuel_cost_usd = fuel_consumed_t * fuel_price

        # other safety/proximity proxies per sample
        # min_cpa random small values higher in open sea, smaller near port
        min_cpa = np.random.uniform(0.5, 5.0, size=samples_per_traj)
        min_tcpa = np.random.uniform(5, 120, size=samples_per_traj)
        depth_under_keel = np.random.uniform(10, 60, size=samples_per_traj)
        speed_limit = np.where(proximity_bin_series == 2, np.random.uniform(8,14), np.random.uniform(16,22))

        # actions: expert behaviour as small adjustments around heading + speed
        # heading estimate from successive lat/lon
        headings = []
        for i in range(samples_per_traj):
            if i < samples_per_traj-1:
                az12, az21, _ = geod.inv(lons[i], lats[i], lons[i+1], lats[i+1])
                hdg = (az12 + 360) % 360
            else:
                hdg = headings[-1] if headings else 0.0
            headings.append(hdg)
        headings = np.array(headings)
        # delta heading commands = small gaussian adjustments
        delta_heading = np.random.normal(loc=0.0, scale=2.0, size=samples_per_traj)
        # speed setpoint near sog with corrections
        speed_setpoint = np.clip(sog + np.random.normal(scale=0.4, size=samples_per_traj), 1.0, 24.0)

        # mode labels for weak supervision (consistent with continuous features)
        mode_sea_state = np.clip((wave_hs // 1.5).astype(int), 0, 5)
        mode_load = np.full(samples_per_traj, load_bin, dtype=int)
        mode_proximity = proximity_bin_series
        # traffic mode from traffic_base
        mode_traffic = np.clip((traffic_base > 3).astype(int) + (traffic_base > 7).astype(int), 0, 2)

        # cross track error: small deviations from ideal geodesic (we used noise earlier)
        # compute great circle interpolation ideal lat/lon without noise for error calculation
        ideal_lats, ideal_lons = interpolate_geodesic(origin[0], origin[1], dest[0], dest[1], samples_per_traj)
        cross_track = np.array([haversine_nm(lat, lon, ilat, ilon) for lat, lon, ilat, ilon in zip(lats, lons, ideal_lats, ideal_lons)])

        # assemble rows for the trajectory
        for i in range(samples_per_traj):
            ts = (start_time + timedelta(minutes=i * time_step_min)).isoformat()
            row = {
                "vessel_id": vessel_id,
                "trajectory_id": traj_id,
                "timestamp": ts,
                "lat": float(lats[i]),
                "lon": float(lons[i]),
                "dlat": float(lats[i] - lats[i-1]) if i>0 else 0.0,
                "dlon": float(lons[i] - lons[i-1]) if i>0 else 0.0,
                "sog_kn": float(sog[i]),
                "ax_kn_per_h": float((sog[i] - sog[i-1]) / dt_hours) if i>0 else 0.0,
                "yaw_rate_deg_per_min": float((headings[i] - headings[i-1]) / time_step_min) if i>0 else 0.0,
                "cross_track_nm": float(cross_track[i]),
                "along_track_nm": float(along_track[i]),
                "wind_u10_ms": float(wind_u[i]),
                "wind_v10_ms": float(wind_v[i]),
                "wind_speed_ms": float(wind_speed[i]),
                "wind_dir_deg": float(wind_dir[i]),
                "wave_hs_m": float(wave_hs[i]),
                "wave_tp_s": float(wave_tp[i]),
                "wave_dir_deg": float(wave_dir[i]),
                "current_u_ms": float(current_u[i]),
                "current_v_ms": float(current_v[i]),
                "sea_state_bin": int(mode_sea_state[i]),
                "proximity_bin": int(mode_proximity[i]),
                "traffic_bin": int(mode_traffic[i]),
                "draft_fwd_m": float(draft_fwd),
                "draft_aft_m": float(draft_aft),
                "trim_m": float(trim),
                "displacement_t": float(displacement),
                "load_bin": int(load_bin),
                "eca_flag": int(0),  # optionally set to 1 if route passes ECA (could be time/segment based)
                "tss_lane_flag": int(1 if i % 300 < 60 else 0),  # simple proxy for being inside TSS segments
                "targets_in_5nm": int(max(0, int(traffic_base[i]))),
                "min_cpa_nm": float(min_cpa[i]),
                "min_tcpa_min": float(min_tcpa[i]),
                "depth_under_keel_m": float(depth_under_keel[i]),
                "speed_limit_kn": float(speed_limit[i]),
                "est_power_kw": float(est_power_kw_arr[i]),
                "sfoc_gpkwh": float(sfoc),
                "fuel_tph": float(fuel_tph[i]),
                "fuel_per_nm_t": float(fuel_per_nm_t[i]),
                "fuel_price_usd_per_ton": float(fuel_price),
                "fuel_consumed_t": float(fuel_consumed_t[i]),
                "fuel_cost_usd": float(fuel_cost_usd[i]),
                "delta_heading_cmd_deg": float(delta_heading[i]),
                "speed_setpoint_kn": float(speed_setpoint[i]),
                "remaining_distance_nm": float(max(0.0, remaining[i])),
                "eta_error_min": float(np.random.normal(loc=0.0, scale=15.0)),  # placeholder
                "mode_sea_state": int(mode_sea_state[i]),
                "mode_load": int(mode_load[i]),
                "mode_proximity": int(mode_proximity[i]),
                "mode_traffic": int(mode_traffic[i])
            }
            rows.append(row)

    # done all trajectories
    # convert to DataFrame and write output
    print(f"Generated {len(rows)} rows across {n_trajectories} trajectories.")

    df = pd.DataFrame(rows, columns=columns)

    # Save CSV or Excel depending on out_format
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if out_format.lower() == "csv":
        df.to_csv(out_path, index=False)
    elif out_format.lower() in ("xlsx", "excel"):
        # Excel supports up to ~1,048,576 rows — 520k fits, but could be slower
        df.to_excel(out_path, index=False, engine="openpyxl")
    else:
        # default to parquet for speed if user prefers
        if not out_path.endswith(".parquet"):
            out_path = out_path + ".parquet"
        df.to_parquet(out_path, index=False)

    print(f"Saved dataset to {out_path} (rows: {len(df)})")
    return out_path

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate synthetic vessel routing dataset (one route, many vessels).")
    p.add_argument("--n_trajectories", type=int, default=520)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--timestep", type=int, default=5, help="minutes per sample")
    p.add_argument("--origin_lat", type=float, default=37.94)
    p.add_argument("--origin_lon", type=float, default=23.64)
    p.add_argument("--dest_lat", type=float, default=1.29)
    p.add_argument("--dest_lon", type=float, default=103.85)
    p.add_argument("--out", type=str, default="vessel_dataset_full_geo.csv")
    p.add_argument("--format", type=str, default="csv", choices=["csv","xlsx","parquet"])
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    generate_dataset(
        n_trajectories=args.n_trajectories,
        samples_per_traj=args.samples,
        time_step_min=args.timestep,
        origin=(args.origin_lat, args.origin_lon),
        dest=(args.dest_lat, args.dest_lon),
        out_path=args.out,
        out_format=args.format,
        seed=args.seed
    )