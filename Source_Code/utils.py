import numpy as np
import scipy.signal
import random
from Source_Code.myconfig import myconfig
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from Source_Code.myconfig import myconfig

# =========================
# Conjugate Gradient
# =========================
def conjugate_gradient(f_Ax, b, feed=None, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p, feed)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        new_rdotr = r.dot(r)
        mu = new_rdotr / rdotr
        p = r + mu * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

# =========================
# Discount rewards
# =========================
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self):
        self.size = 0
        self.observations = []
        self.actions = []

    def seed_buffer(self, observations, actions):
        assert len(observations) == len(actions)
        self.observations = observations
        self.actions = actions
        self.size = len(observations)

    def get_batch(self, batch_size):
        idxs = random.choices(range(self.size), k=batch_size)
        batch_obs = [self.observations[i] for i in idxs]
        batch_act = [self.actions[i] for i in idxs]
        return batch_obs, batch_act

def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    start_time = df['timestamp'].min()
    df['timestamp'] = (df['timestamp'] - start_time).dt.total_seconds() / 60.0

    drop_cols = ['vessel_id', 'trajectory_id', 'delta_heading_cmd_deg', 'speed_setpoint_kn', 'dlon', 'dlat']
    X_columns = [c for c in df.columns if c not in drop_cols]
    pd.set_option("display.max_columns", None)
    #X2=df[X_columns]
    #print('X STATE:', X2)
    X = df[X_columns].values.astype(np.float32)
    action_cols = ['delta_heading_cmd_deg', 'speed_setpoint_kn', 'dlon', 'dlat']
    y = df[action_cols].values.astype(np.float32)

    print(f"Loaded dataset {csv_file}: X shape {X.shape}, y shape {y.shape}")
    return X, y, X_columns

# =========================
# Normalize Observation
# =========================
def normalize_observation(observation):
    timestamp = (observation[0] - myconfig['timestamp_avg']) / myconfig['timestamp_std']
    lon = (observation[1] - myconfig['lon_avg']) / myconfig['lon_std']
    lat = (observation[2] - myconfig['lat_avg']) / myconfig['lat_std']
    #dlon = (observation[3] - myconfig['dlon_avg']) / myconfig['dlon_std']
    #dlat = (observation[4] - myconfig['dlat_avg']) / myconfig['dlat_std']
    sog_kn = (observation[3] - myconfig['sog_kn_avg']) / myconfig['sog_kn_std']
    ax_kn_per_h = (observation[4] - myconfig['ax_kn_per_h_avg']) / myconfig['ax_kn_per_h_std']
    yaw_rate_deg_per_min = (observation[5] - myconfig['yaw_rate_deg_per_min_avg']) / myconfig['yaw_rate_deg_per_min_std']
    cross_track_nm = (observation[6] - myconfig['cross_track_nm_avg']) / myconfig['cross_track_nm_std']
    along_track_nm = (observation[7] - myconfig['along_track_nm_avg']) / myconfig['along_track_nm_std']
    wind_u10_ms = (observation[8] - myconfig['wind_u10_ms_avg']) / myconfig['wind_u10_ms_std']
    wind_v10_ms = (observation[9] - myconfig['wind_v10_ms_avg']) / myconfig['wind_v10_ms_std']
    wind_speed_ms = (observation[10] - myconfig['wind_speed_ms_avg']) / myconfig['wind_speed_ms_std']
    wind_dir_deg = (observation[11] - myconfig['wind_dir_deg_avg']) / myconfig['wind_dir_deg_std']
    wave_hs_m = (observation[12] - myconfig['wave_hs_m_avg']) / myconfig['wave_hs_m_std']
    wave_tp_s = (observation[13] - myconfig['wave_tp_s_avg']) / myconfig['wave_tp_s_std']
    wave_dir_deg = (observation[14] - myconfig['wave_dir_deg_avg']) / myconfig['wave_dir_deg_std']
    current_u_ms = (observation[15] - myconfig['current_u_ms_avg']) / myconfig['current_u_ms_std']
    current_v_ms = (observation[16] - myconfig['current_v_ms_avg']) / myconfig['current_v_ms_std']
    sea_state_bin = (observation[17] - myconfig['sea_state_bin_avg']) / myconfig['sea_state_bin_std']
    proximity_bin = (observation[18] - myconfig['proximity_bin_avg']) / myconfig['proximity_bin_std']
    traffic_bin = (observation[19] - myconfig['traffic_bin_avg']) / myconfig['traffic_bin_std']
    draft_fwd_m = (observation[20] - myconfig['draft_fwd_m_avg']) / myconfig['draft_fwd_m_std']
    draft_aft_m = (observation[21] - myconfig['draft_aft_m_avg']) / myconfig['draft_aft_m_std']
    trim_m = (observation[22] - myconfig['trim_m_avg']) / myconfig['trim_m_std']
    displacement_t = (observation[23] - myconfig['displacement_t_avg']) / myconfig['displacement_t_std']
    load_bin = (observation[24] - myconfig['load_bin_avg']) / myconfig['load_bin_std']
    eca_flag = (observation[25] - myconfig['eca_flag_avg']) / (myconfig['eca_flag_std'] + 1e-8)
    tss_lane_flag = (observation[26] - myconfig['tss_lane_flag_avg']) / myconfig['tss_lane_flag_std']
    targets_in_5nm = (observation[27] - myconfig['targets_in_5nm_avg']) / myconfig['targets_in_5nm_std']
    min_cpa_nm = (observation[28] - myconfig['min_cpa_nm_avg']) / myconfig['min_cpa_nm_std']
    min_tcpa_min = (observation[29] - myconfig['min_tcpa_min_avg']) / myconfig['min_tcpa_min_std']
    depth_under_keel_m = (observation[30] - myconfig['depth_under_keel_m_avg']) / myconfig['depth_under_keel_m_std']
    speed_limit_kn = (observation[31] - myconfig['speed_limit_kn_avg']) / myconfig['speed_limit_kn_std']
    est_power_kw = (observation[32] - myconfig['est_power_kw_avg']) / myconfig['est_power_kw_std']
    sfoc_gpkwh = (observation[33] - myconfig['sfoc_gpkwh_avg']) / myconfig['sfoc_gpkwh_std']
    fuel_tph = (observation[34] - myconfig['fuel_tph_avg']) / myconfig['fuel_tph_std']
    fuel_per_nm_t = (observation[35] - myconfig['fuel_per_nm_t_avg']) / myconfig['fuel_per_nm_t_std']
    fuel_price_usd_per_ton = (observation[36] - myconfig['fuel_price_usd_per_ton_avg']) / myconfig['fuel_price_usd_per_ton_std']
    fuel_consumed_t = (observation[37] - myconfig['fuel_consumed_t_avg']) / myconfig['fuel_consumed_t_std']
    fuel_cost_usd = (observation[38] - myconfig['fuel_cost_usd_avg']) / myconfig['fuel_cost_usd_std']
    #delta_heading_cmd_deg = (observation[39] - myconfig['delta_heading_cmd_deg_avg']) / myconfig['delta_heading_cmd_deg_std']
    #speed_setpoint_kn = (observation[40] - myconfig['speed_setpoint_kn_avg']) / myconfig['speed_setpoint_kn_std']
    remaining_distance_nm = (observation[39] - myconfig['remaining_distance_nm_avg']) / myconfig['remaining_distance_nm_std']
    eta_error_min = (observation[40] - myconfig['eta_error_min_avg']) / myconfig['eta_error_min_std']
    mode_sea_state = (observation[41] - myconfig['mode_sea_state_avg']) / myconfig['mode_sea_state_std']
    mode_load = (observation[42] - myconfig['mode_load_avg']) / myconfig['mode_load_std']
    mode_proximity = (observation[43] - myconfig['mode_proximity_avg']) / myconfig['mode_proximity_std']
    mode_traffic = (observation[44] - myconfig['mode_traffic_avg']) / myconfig['mode_traffic_std']

    return [timestamp, lon, lat, sog_kn, ax_kn_per_h, yaw_rate_deg_per_min, cross_track_nm, along_track_nm,
            wind_u10_ms, wind_v10_ms, wind_speed_ms, wind_dir_deg, wave_hs_m, wave_tp_s, wave_dir_deg, current_u_ms,
            current_v_ms, sea_state_bin, proximity_bin, traffic_bin, draft_fwd_m, draft_aft_m, trim_m, displacement_t,
            load_bin, eca_flag, tss_lane_flag, targets_in_5nm, min_cpa_nm, min_tcpa_min, depth_under_keel_m,
            speed_limit_kn, est_power_kw, sfoc_gpkwh, fuel_tph, fuel_per_nm_t, fuel_price_usd_per_ton, fuel_consumed_t,
            fuel_cost_usd, remaining_distance_nm, eta_error_min, mode_sea_state,
            mode_load, mode_proximity, mode_traffic]#, dlat, dlon, delta_heading_cmd_deg, speed_setpoint_kn


# =========================
# Normalize Action
# =========================
def normalize_action(action):
    return [
        (action[0] - myconfig['delta_heading_cmd_deg_avg']) / myconfig['delta_heading_cmd_deg_std'],
        (action[1] - myconfig['speed_setpoint_kn_avg']) / myconfig['speed_setpoint_kn_std'],
        (action[2] - myconfig['dlon_avg']) / myconfig['dlon_std'],
        (action[3] - myconfig['dlat_avg']) / myconfig['dlat_std']
    ]


def normalize_observations_MinMaxScaler(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled.astype(np.float32), scaler

def normalize_actions_MinMaxScaler(y):
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y)
    return y_scaled.astype(np.float32), scaler

def unnormalize_observations_MinMaxScaler(X_scaled, scaler):
    """
    Inverse transform normalized actions back to original scale.
    y_scaled: np.ndarray (normalized actions)
    scaler: MinMaxScaler fitted on original y
    """
    return scaler.inverse_transform(X_scaled)

def unnormalize_actions_MinMaxScaler(y_scaled, scaler):
    """
    Inverse transform normalized actions back to original scale.
    y_scaled: np.ndarray (normalized actions)
    scaler: MinMaxScaler fitted on original y
    """
    return scaler.inverse_transform(y_scaled)


# =========================
# Unnormalize Observations
# =========================
def unnormalize_observation(observation):
    timestamp = observation[0] * myconfig['timestamp_std'] + myconfig['timestamp_avg']
    lon = observation[1] * myconfig['lon_std'] + myconfig['lon_avg']
    lat = observation[2] * myconfig['lat_std'] + myconfig['lat_avg']
    #dlon = observation[3] * myconfig['dlon_std'] + myconfig['dlon_avg']
    #dlat = observation[4] * myconfig['dlat_std'] + myconfig['dlat_avg']
    sog_kn = observation[3] * myconfig['sog_kn_std'] + myconfig['sog_kn_avg']
    ax_kn_per_h = observation[4] * myconfig['ax_kn_per_h_std'] + myconfig['ax_kn_per_h_avg']
    yaw_rate_deg_per_min = observation[5] * myconfig['yaw_rate_deg_per_min_std'] + myconfig['yaw_rate_deg_per_min_avg']
    cross_track_nm = observation[6] * myconfig['cross_track_nm_std'] + myconfig['cross_track_nm_avg']
    along_track_nm = observation[7] * myconfig['along_track_nm_std'] + myconfig['along_track_nm_avg']
    wind_u10_ms = observation[8] * myconfig['wind_u10_ms_std'] + myconfig['wind_u10_ms_avg']
    wind_v10_ms = observation[9] * myconfig['wind_v10_ms_std'] + myconfig['wind_v10_ms_avg']
    wind_speed_ms = observation[10] * myconfig['wind_speed_ms_std'] + myconfig['wind_speed_ms_avg']
    wind_dir_deg = observation[11] * myconfig['wind_dir_deg_std'] + myconfig['wind_dir_deg_avg']
    wave_hs_m = observation[12] * myconfig['wave_hs_m_std'] + myconfig['wave_hs_m_avg']
    wave_tp_s = observation[13] * myconfig['wave_tp_s_std'] + myconfig['wave_tp_s_avg']
    wave_dir_deg = observation[14] * myconfig['wave_dir_deg_std'] + myconfig['wave_dir_deg_avg']
    current_u_ms = observation[15] * myconfig['current_u_ms_std'] + myconfig['current_u_ms_avg']
    current_v_ms = observation[16] * myconfig['current_v_ms_std'] + myconfig['current_v_ms_avg']
    sea_state_bin = observation[17] * myconfig['sea_state_bin_std'] + myconfig['sea_state_bin_avg']
    proximity_bin = observation[18] * myconfig['proximity_bin_std'] + myconfig['proximity_bin_avg']
    traffic_bin = observation[19] * myconfig['traffic_bin_std'] + myconfig['traffic_bin_avg']
    draft_fwd_m = observation[20] * myconfig['draft_fwd_m_std'] + myconfig['draft_fwd_m_avg']
    draft_aft_m = observation[21] * myconfig['draft_aft_m_std'] + myconfig['draft_aft_m_avg']
    trim_m = observation[22] * myconfig['trim_m_std'] + myconfig['trim_m_avg']
    displacement_t = observation[23] * myconfig['displacement_t_std'] + myconfig['displacement_t_avg']
    load_bin = observation[24] * myconfig['load_bin_std'] + myconfig['load_bin_avg']
    eca_flag = observation[25] * (myconfig['eca_flag_std']+ 1e-8) + myconfig['eca_flag_avg']
    tss_lane_flag = observation[26] * myconfig['tss_lane_flag_std'] + myconfig['tss_lane_flag_avg']
    targets_in_5nm = observation[27] * myconfig['targets_in_5nm_std'] + myconfig['targets_in_5nm_avg']
    min_cpa_nm = observation[28] * myconfig['min_cpa_nm_std'] + myconfig['min_cpa_nm_avg']
    min_tcpa_min = observation[29] * myconfig['min_tcpa_min_std'] + myconfig['min_tcpa_min_avg']
    depth_under_keel_m = observation[30] * myconfig['depth_under_keel_m_std'] + myconfig['depth_under_keel_m_avg']
    speed_limit_kn = observation[31] * myconfig['speed_limit_kn_std'] + myconfig['speed_limit_kn_avg']
    est_power_kw = observation[32] * myconfig['est_power_kw_std'] + myconfig['est_power_kw_avg']
    sfoc_gpkwh = observation[33] * myconfig['sfoc_gpkwh_std'] + myconfig['sfoc_gpkwh_avg']
    fuel_tph = observation[34] * myconfig['fuel_tph_std'] + myconfig['fuel_tph_avg']
    fuel_per_nm_t = observation[35] * myconfig['fuel_per_nm_t_std'] + myconfig['fuel_per_nm_t_avg']
    fuel_price_usd_per_ton = observation[36] * myconfig['fuel_price_usd_per_ton_std'] + myconfig['fuel_price_usd_per_ton_avg']
    fuel_consumed_t = observation[37] * myconfig['fuel_consumed_t_std'] + myconfig['fuel_consumed_t_avg']
    fuel_cost_usd = observation[38] * myconfig['fuel_cost_usd_std'] + myconfig['fuel_cost_usd_avg']
    #delta_heading_cmd_deg = observation[41] * myconfig['delta_heading_cmd_deg_std'] + myconfig['delta_heading_cmd_deg_avg']
    #speed_setpoint_kn = observation[42] * myconfig['speed_setpoint_kn_std'] + myconfig['speed_setpoint_kn_avg']
    remaining_distance_nm = observation[39] * myconfig['remaining_distance_nm_std'] + myconfig['remaining_distance_nm_avg']
    eta_error_min = observation[40] * myconfig['eta_error_min_std'] + myconfig['eta_error_min_avg']
    mode_sea_state = observation[41] * myconfig['mode_sea_state_std'] + myconfig['mode_sea_state_avg']
    mode_load = observation[42] * myconfig['mode_load_std'] + myconfig['mode_load_avg']
    mode_proximity = observation[43] * myconfig['mode_proximity_std'] + myconfig['mode_proximity_avg']
    mode_traffic = observation[44] * myconfig['mode_traffic_std'] + myconfig['mode_traffic_avg']

    return [timestamp, lon,lat, sog_kn, ax_kn_per_h, yaw_rate_deg_per_min, cross_track_nm, along_track_nm, wind_u10_ms, wind_v10_ms,
            wind_speed_ms, wind_dir_deg, wave_hs_m, wave_tp_s, wave_dir_deg, current_u_ms, current_v_ms, sea_state_bin, proximity_bin,
            traffic_bin, draft_fwd_m, draft_aft_m, trim_m, displacement_t, load_bin, eca_flag, tss_lane_flag, targets_in_5nm,
            min_cpa_nm, min_tcpa_min, depth_under_keel_m, speed_limit_kn, est_power_kw, sfoc_gpkwh, fuel_tph, fuel_per_nm_t,
            fuel_price_usd_per_ton, fuel_consumed_t, fuel_cost_usd, remaining_distance_nm, eta_error_min, mode_sea_state,
            mode_load, mode_proximity, mode_traffic]
#,	dlat, dlon, delta_heading_cmd_deg, speed_setpoint_kn

# =========================
# Unnormalize Action
# =========================
def unnormalize_action(action):
    return [
        action[0]*myconfig['delta_heading_cmd_deg_std'] + myconfig['delta_heading_cmd_deg_avg'],
        action[1]*myconfig['speed_setpoint_std'] + myconfig['speed_setpoint_kn_avg'],
        action[2] * myconfig['dlon_std'] + myconfig['dlon_avg'],
        action[3] * myconfig['dlat_std'] + myconfig['dlat_avg']

    ]