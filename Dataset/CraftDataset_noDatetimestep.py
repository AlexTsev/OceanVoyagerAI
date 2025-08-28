import sys
import os
import numpy as np
import pandas as pd

# ---------------------------
# Include environment folder
# ---------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), 'Environment'))
from Environment.environment_ATH_SG import VesselEnvironment


# =========================
# Dataset Generator
# =========================
def generate_dataset(
        num_trajectories=500,
        samples_per_traj=1000,
        time_step_min=5,
        output_csv="../Dataset/vesseldataset.csv"
):
    """
    Generate synthetic vessel trajectories directly from VesselEnvironment.

    Args:
        num_trajectories (int): number of trajectories
        samples_per_traj (int): number of samples per trajectory
        time_step_min (int): minutes between samples
        output_csv (str): file path for saving dataset
    """
    rows = []

    for traj_id in range(num_trajectories):
        env = VesselEnvironment()
        env.reset()
        vessel_id = traj_id

        # Extract arrays directly from environment
        lats = env.lat
        lons = env.lon
        sog = env.sog_kn
        headings = np.zeros(samples_per_traj)  # environment has no heading, set 0
        cross_track = np.zeros(samples_per_traj)
        along_track = np.zeros(samples_per_traj)

        wind_u = env.wind_u10_ms
        wind_v = env.wind_v10_ms
        wind_speed = env.wind_speed_ms
        wind_dir = env.wind_dir_deg

        wave_hs = env.wave_hs_m
        wave_tp = env.wave_tp_s
        wave_dir = env.wave_dir_deg

        current_u = env.current_u_ms
        current_v = env.current_v_ms

        draft_fwd = np.full(samples_per_traj, env.draft_fwd_m)
        draft_aft = np.full(samples_per_traj, env.draft_aft_m)
        trim = np.full(samples_per_traj, env.trim_m)
        displacement = np.full(samples_per_traj, env.displacement_t)
        load_bin = np.full(samples_per_traj, env.load_bin)

        mode_sea_state = env.sea_state_bin
        mode_load = np.full(samples_per_traj, env.load_bin)
        mode_proximity = env.proximity_bin
        mode_traffic = env.traffic_bin

        sfoc = np.full(samples_per_traj, env.sfoc_gpkwh)
        est_power_kw_arr = env.est_power_kw
        fuel_tph = env.fuel_tph
        fuel_per_nm_t = env.fuel_per_nm_t
        fuel_price = np.full(samples_per_traj, env.fuel_price_usd_per_ton)
        fuel_consumed_t = env.fuel_consumed_t
        fuel_cost_usd = env.fuel_cost_usd

        remaining = np.zeros(samples_per_traj)
        for i in range(samples_per_traj):
            # compute remaining distance to destination
            remaining[i] = env._get_state(i)['remaining_distance_nm']

        dt_hours = time_step_min / 60.0

        # Build trajectory rows
        for i in range(samples_per_traj):
            minutes = i * time_step_min

            row = {
                "vessel_id": vessel_id,
                "trajectory_id": traj_id,
                "minutes": minutes,
                "lat": float(lats[i]),
                "lon": float(lons[i]),
                "dlat": float(lats[i] - lats[i - 1]) if i > 0 else 0.0,
                "dlon": float(lons[i] - lons[i - 1]) if i > 0 else 0.0,
                "sog_kn": float(sog[i]),
                "ax_kn_per_h": float((sog[i] - sog[i - 1]) / dt_hours) if i > 0 else 0.0,
                "yaw_rate_deg_per_min": float((headings[i] - headings[i - 1]) / time_step_min) if i > 0 else 0.0,
                "heading_deg": float(headings[i]),
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

                "draft_fwd_m": float(draft_fwd[i]),
                "draft_aft_m": float(draft_aft[i]),
                "trim_m": float(trim[i]),
                "displacement_t": float(displacement[i]),
                "load_bin": int(load_bin[i]),

                "sea_state_bin": int(mode_sea_state[i]),
                "proximity_bin": int(mode_proximity[i]),
                "traffic_bin": int(mode_traffic[i]),

                "eca_flag": int(env.eca_flag),
                "tss_lane_flag": int(env.tss_lane_flag),
                "targets_in_5nm": 0,
                "min_cpa_nm": 0,
                "min_tcpa_min": 0,

                "depth_under_keel_m": 0,
                "speed_limit_kn": float(env.sog_kn[i]),
                "est_power_kw": float(est_power_kw_arr[i]),

                "sfoc_gpkwh": float(sfoc[i]),
                "fuel_tph": float(fuel_tph[i]),
                "fuel_per_nm_t": float(fuel_per_nm_t[i]),
                "fuel_price_usd_per_ton": float(fuel_price[i]),
                "fuel_consumed_t": float(fuel_consumed_t[i]),
                "fuel_cost_usd": float(fuel_cost_usd[i]),

                "delta_heading_cmd_deg": 0.0,
                "speed_setpoint_kn": float(sog[i]),

                "remaining_distance_nm": float(max(0.0, remaining[i])),
                "eta_error_min": float(np.random.normal(loc=0.0, scale=15.0)),

                "mode_sea_state": int(mode_sea_state[i]),
                "mode_load": int(mode_load[i]),
                "mode_proximity": int(mode_proximity[i]),
                "mode_traffic": int(mode_traffic[i]),
            }

            rows.append(row)

    # Save dataset
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved to {output_csv} with shape {df.shape}")
    return df


# =========================
# Main Run
# =========================
if __name__ == "__main__":
    dataset = generate_dataset(
        num_trajectories=500,
        samples_per_traj=1000,
        time_step_min=5,
        output_csv="../Dataset/vesseldataset.csv"
    )
