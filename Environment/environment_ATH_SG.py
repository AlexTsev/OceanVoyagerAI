# vessel_environment.py
import numpy as np
import random
from geo.geo import distance
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'Source_Code'))
from Source_Code.utils import normalize_observation, normalize_action, unnormalize_observation, unnormalize_action, load_dataset
from Source_Code.myconfig import myconfig as norm_config

# -----------------------
# Configuration
# -----------------------
conf = {
    'origin': (37.94, 23.64),        # Piraeus (lat, lon)
    'destination': (1.29, 103.85),   # Singapore (lat, lon)
    'samples_per_vessel': 1000,
    'time_step_minutes': 5,
    'max_speed_kn': 22,
    'min_speed_kn': 10,
    'mbr_buffer_deg': 2.5,          # bounding box buffer (deg) around origin-destination
    'destination_tolerance_nm': 1.0, # when to consider "arrived"
}

# Compute a simple bounding box (min bounding rectangle) with buffer
lat_min = min(conf['origin'][0], conf['destination'][0]) - conf['mbr_buffer_deg']
lat_max = max(conf['origin'][0], conf['destination'][0]) + conf['mbr_buffer_deg']
lon_min = min(conf['origin'][1], conf['destination'][1]) - conf['mbr_buffer_deg']
lon_max = max(conf['origin'][1], conf['destination'][1]) + conf['mbr_buffer_deg']

conf['mbr_top_left'] = (lat_max, lon_min)    # (lat, lon)
conf['mbr_bot_right'] = (lat_min, lon_max)   # (lat, lon)


class VesselEnvironment:
    """
    Simple vessel environment used for training/testing the VAE imitation agent.

    State dictionary keys match the fields used throughout the project.
    Observations returned by get_observation() are normalized using myconfig (mean/std).
    """

    def __init__(self, randomize=True):
        self.origin = conf['origin']
        self.destination = conf['destination']
        self.randomize = randomize
        self.time_step = 0
        self.done = False
        self.reset()

    def reset(self):
        """
        Initialize a new trajectory (episode). Creates time-series arrays for
        environmental variables and ship parameters used by _get_state().
        Returns: (obs_norm, state_dict) same as Gym-style reset: normalized observation and raw state.
        """
        N = conf['samples_per_vessel']

        # Create linear route between origin and destination (expert track)
        # lon/lat arrays are expert reference points (we keep them for reward shaping / comparison)
        self.lon = np.linspace(self.origin[1], self.destination[1], N)
        self.lat = np.linspace(self.origin[0], self.destination[0], N)

        # Kinematic/environmental arrays (per timestep)
        self.sog_kn = np.clip(
            np.random.normal((conf['min_speed_kn'] + conf['max_speed_kn']) / 2, 1.5, N),
            conf['min_speed_kn'], conf['max_speed_kn']
        )
        self.wind_u10_ms = np.random.uniform(-15, 15, N)
        self.wind_v10_ms = np.random.uniform(-15, 15, N)
        self.wind_speed_ms = np.sqrt(self.wind_u10_ms ** 2 + self.wind_v10_ms ** 2)
        self.wind_dir_deg = (np.degrees(np.arctan2(self.wind_v10_ms, self.wind_u10_ms)) + 360) % 360
        self.wave_hs_m = np.random.uniform(0, 6, N)
        self.wave_tp_s = np.random.uniform(4, 12, N)
        self.wave_dir_deg = np.random.uniform(0, 360, N)
        self.current_u_ms = np.random.uniform(-1.5, 1.5, N)
        self.current_v_ms = np.random.uniform(-1.5, 1.5, N)
        self.sea_state_bin = np.clip((self.wave_hs_m // 1.5).astype(int), 0, 5)
        self.proximity_bin = np.random.choice([0, 1, 2], N, p=[0.7, 0.2, 0.1])
        self.traffic_bin = np.random.choice([0, 1, 2], N, p=[0.6, 0.3, 0.1])

        # Ship fixed characteristics (single values)
        self.displacement_t = random.uniform(50000, 150000)
        self.draft_fwd_m = random.uniform(8, 12)
        self.draft_aft_m = self.draft_fwd_m + random.uniform(-0.5, 0.5)
        self.trim_m = self.draft_aft_m - self.draft_fwd_m
        self.load_bin = random.choice([0, 1, 2])
        self.eca_flag = random.choice([0, 1])
        self.tss_lane_flag = random.choice([0, 1])

        # Power & fuel (arrays driven by SOG)
        # Use element-wise ops; ensure shapes align
        self.sfoc_gpkwh = np.random.uniform(160, 190)
        # est_power_kw and fuel arrays will be computed per-timestep
        self.est_power_kw = (self.displacement_t / 1000.0) * (self.sog_kn ** 3) * 0.05
        self.fuel_tph = self.est_power_kw * self.sfoc_gpkwh / 1e6
        self.fuel_per_nm_t = self.fuel_tph / np.maximum(self.sog_kn, 1e-3)
        self.fuel_price_usd_per_ton = np.random.uniform(500, 800)
        self.fuel_consumed_t = self.fuel_tph * (conf['time_step_minutes'] / 60.0)
        self.fuel_cost_usd = self.fuel_consumed_t * self.fuel_price_usd_per_ton

        # Initial dynamic variables (actions / internal)
        # Heading is tracked and updated by delta_heading actions (degrees)
        # Start heading as the bearing from origin to next expert point
        dy = (self.lat[1] - self.lat[0])
        dx = (self.lon[1] - self.lon[0])
        initial_bearing = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
        self.heading_deg = float(initial_bearing)  # keep floating

        # Initialize last commanded actions (for logging / state)
        self.delta_heading_cmd_deg = 0.0
        self.speed_setpoint_kn = float(self.sog_kn[0])

        # Modes (categorical) saved for state
        self.mode_sea_state = int(self.sea_state_bin[0])
        self.mode_load = int(self.load_bin)
        self.mode_proximity = int(self.proximity_bin[0])
        self.mode_traffic = int(self.traffic_bin[0])

        # Time tracking
        self.time_step = 0
        self.done = False

        # Current state (dictionary)
        self.state = self._get_state(self.time_step)

        # Return normalized observation and raw state
        return self.get_observation(), self.state

    def _get_state(self, idx):
        """Return raw/un-normalized state dictionary for timestep idx."""
        # Compute remaining distance (nm)
        rem_distance = distance((self.lat[idx], self.lon[idx]), self.destination) / 1852.0

        return {
            'timestamp': idx * conf['time_step_minutes'],
            'lon': float(self.lon[idx]),
            'lat': float(self.lat[idx]),
            'sog_kn': float(self.sog_kn[idx]),
            'ax_kn_per_h': 0.0,
            'yaw_rate_deg_per_min': 0.0,
            'cross_track_nm': 0.0,
            'along_track_nm': 0.0,
            'wind_u10_ms': float(self.wind_u10_ms[idx]),
            'wind_v10_ms': float(self.wind_v10_ms[idx]),
            'wind_speed_ms': float(self.wind_speed_ms[idx]),
            'wind_dir_deg': float(self.wind_dir_deg[idx]),
            'wave_hs_m': float(self.wave_hs_m[idx]),
            'wave_tp_s': float(self.wave_tp_s[idx]),
            'wave_dir_deg': float(self.wave_dir_deg[idx]),
            'current_u_ms': float(self.current_u_ms[idx]),
            'current_v_ms': float(self.current_v_ms[idx]),
            'sea_state_bin': int(self.sea_state_bin[idx]),
            'proximity_bin': int(self.proximity_bin[idx]),
            'traffic_bin': int(self.traffic_bin[idx]),
            'draft_fwd_m': float(self.draft_fwd_m),
            'draft_aft_m': float(self.draft_aft_m),
            'trim_m': float(self.trim_m),
            'displacement_t': float(self.displacement_t),
            'load_bin': int(self.load_bin),
            'eca_flag': int(self.eca_flag),
            'tss_lane_flag': int(self.tss_lane_flag),
            'targets_in_5nm': 0,
            'min_cpa_nm': 0,
            'min_tcpa_min': 0,
            'depth_under_keel_m': 0,
            'speed_limit_kn': conf['max_speed_kn'],
            'est_power_kw': float(self.est_power_kw[idx]),
            'sfoc_gpkwh': float(self.sfoc_gpkwh),
            'fuel_tph': float(self.fuel_tph[idx]),
            'fuel_per_nm_t': float(self.fuel_per_nm_t[idx]),
            'fuel_price_usd_per_ton': float(self.fuel_price_usd_per_ton),
            'fuel_consumed_t': float(self.fuel_consumed_t[idx]),
            'fuel_cost_usd': float(self.fuel_cost_usd[idx]),
            # last commanded actions (kept in state for logging)
            'delta_heading_cmd_deg': float(self.delta_heading_cmd_deg),
            'speed_setpoint_kn': float(self.speed_setpoint_kn),
            'remaining_distance_nm': float(rem_distance),
            'eta_error_min': 0,
            'mode_sea_state': int(self.mode_sea_state),
            'mode_load': int(self.mode_load),
            'mode_proximity': int(self.mode_proximity),
            'mode_traffic': int(self.mode_traffic),
        }

    def get_observation(self):
        """
        Produce normalized observation dictionary (same keys as state, but normalized).
        Uses norm_config (myconfig) mean/std values.
        """
        obs = self.state.copy()
        # list of keys to normalize using mean/std
        normalize_keys = [
            'timestamp', 'lon', 'lat', 'sog_kn', 'wind_u10_ms', 'wind_v10_ms', 'wind_speed_ms',
            'wind_dir_deg', 'wave_hs_m', 'wave_tp_s', 'wave_dir_deg', 'current_u_ms', 'current_v_ms'
        ]

        for key in normalize_keys:
            mean_key = f'{key}_avg'
            std_key = f'{key}_std'
            if mean_key in norm_config and std_key in norm_config and norm_config[std_key] != 0:
                obs[key] = (obs[key] - norm_config[mean_key]) / norm_config[std_key]
            else:
                # if missing config or zero std — keep raw (but warn might be useful)
                obs[key] = obs[key]

        # categorical / binary features can be left as-is (or one-hot if needed by model)
        return obs

    def step(self, action):
        """
        Take an action (normalized) and update the vessel state.

        Expected action layout (consistent with your VAE):
            action = [delta_heading_norm, speed_setpoint_norm, dlon_norm, dlat_norm]

        Action decoding uses norm_config (mean/std).
        Returns: (obs_norm, state_raw, reward:float, done:bool)
        """
        if self.done:
            # Episode already finished; return terminal observation
            return self.get_observation(), self.state, 0.0, True

        # --- 1) Decode action using mean/std ---
        try:
            delta_heading_norm, speed_setpoint_norm, dlon_norm, dlat_norm = action
        except Exception:
            # support also numpy arrays with shape (n,) or (1,n)
            arr = np.asarray(action).ravel()
            delta_heading_norm, speed_setpoint_norm, dlon_norm, dlat_norm = arr[:4]

        # Un-normalize
        delta_heading = delta_heading_norm * norm_config.get('delta_heading_cmd_deg_std', 1.0) + \
                        norm_config.get('delta_heading_cmd_deg_avg', 0.0)
        speed_setpoint = speed_setpoint_norm * norm_config.get('speed_setpoint_kn_std', 1.0) + \
                         norm_config.get('speed_setpoint_kn_avg', 0.0)
        dlon_action = dlon_norm * norm_config.get('dlon_std', 1.0) + norm_config.get('dlon_avg', 0.0)
        dlat_action = dlat_norm * norm_config.get('dlat_std', 1.0) + norm_config.get('dlat_avg', 0.0)

        # Clip speed to reasonable bounds
        speed_setpoint = float(np.clip(speed_setpoint, conf['min_speed_kn'], max(conf['max_speed_kn'], conf['min_speed_kn'])))

        # --- 2) Update internal commanded action state (log / for state) ---
        self.delta_heading_cmd_deg = float(delta_heading)
        self.speed_setpoint_kn = float(speed_setpoint)

        # Update heading (apply commanded delta)
        # Keep heading in [0, 360)
        self.heading_deg = (self.heading_deg + float(delta_heading)) % 360.0

        # --- 3) Movement update ---
        # Two compatible ways to move the vessel:
        #  - Use dlon/dlat exactly (dataset-style: dlon/dlat are delta lon/lat)
        #  - Or compute movement from heading+speed (distance traveled during timestep)
        #
        # We will apply dlon/dlat if they are non-zero (this keeps compatibility with your dataset
        # which used dlon/dlat), and also accept heading+speed motion if desired. To be safe
        # we use both: primary movement follows dlon/dlat, and we ensure the speed/heading
        # are updated for realism.
        #
        # NOTE: dlon_action/dlat_action are assumed to be **degrees** (as in original dataset).
        # If they are in other units, adjust conversion accordingly.
        prev_lon = float(self.state['lon'])
        prev_lat = float(self.state['lat'])

        # Apply dlon/dlat movement (this matches how dataset appears to be structured)
        new_lon = prev_lon + float(dlon_action)
        new_lat = prev_lat + float(dlat_action)

        # As a safeguard: if dlon/dlat are extremely small (close to 0), fallback to heading+speed movement
        if abs(dlon_action) < 1e-6 and abs(dlat_action) < 1e-6:
            # Move according to heading + speed_setpoint
            distance_nm = speed_setpoint * (conf['time_step_minutes'] / 60.0)  # nautical miles traveled in this step
            # Convert nm to degrees latitude: 1 deg latitude ≈ 60 nm
            dlat_from_speed = (distance_nm / 60.0) * np.cos(np.radians(self.heading_deg))
            # degrees longitude depends on latitude
            denom = max(1e-6, np.cos(np.radians(prev_lat)))
            dlon_from_speed = (distance_nm / 60.0) * np.sin(np.radians(self.heading_deg)) / denom
            new_lon = prev_lon + dlon_from_speed
            new_lat = prev_lat + dlat_from_speed

        # Update timestamp and state geometry
        self.state['lon'] = float(new_lon)
        self.state['lat'] = float(new_lat)
        self.state['timestamp'] = int(self.state['timestamp'] + conf['time_step_minutes'])

        # --- 4) Safety/bounding-box check (MBR) ---
        # If vessel leaves the bounding box, terminate episode with penalty
        if (self.state['lat'] > conf['mbr_top_left'][0] or
            self.state['lat'] < conf['mbr_bot_right'][0] or
            self.state['lon'] < conf['mbr_top_left'][1] or
            self.state['lon'] > conf['mbr_bot_right'][1]):
            self.done = True
            # update remaining distance for completeness
            self.state['remaining_distance_nm'] = distance((self.state['lat'], self.state['lon']), self.destination) / 1852.0
            return self.get_observation(), self.state, -10.0, True

        # --- 5) Update fuel / power entries for current time_step if arrays available ---
        t = min(self.time_step, conf['samples_per_vessel'] - 1)
        # If fuel arrays present, use them (otherwise leave as-is)
        if hasattr(self, 'fuel_tph'):
            self.state['fuel_tph'] = float(self.fuel_tph[t])
            self.state['fuel_per_nm_t'] = float(self.fuel_per_nm_t[t])
            self.state['est_power_kw'] = float(self.est_power_kw[t])
            self.state['fuel_consumed_t'] = float(self.fuel_consumed_t[t])
            self.state['fuel_cost_usd'] = float(self.fuel_cost_usd[t])

        # --- 6) Remaining distance (nm) and reward shaping ---
        self.state['remaining_distance_nm'] = distance((self.state['lat'], self.state['lon']), self.destination) / 1852.0

        # Basic reward: negative distance to the next expert point (imitation)
        reward = 0.0
        if hasattr(self, "lat") and len(self.lat) > self.time_step + 1:
            idx_next = min(self.time_step + 1, conf['samples_per_vessel'] - 1)
            expert_lat, expert_lon = self.lat[idx_next], self.lon[idx_next]
            distance_error = distance((self.state['lat'], self.state['lon']), (expert_lat, expert_lon)) / 1852.0
            reward -= float(distance_error)

        # Penalize overspeed moderately
        if self.state['speed_setpoint_kn'] > conf['max_speed_kn']:
            reward -= 0.01 * (self.state['speed_setpoint_kn'] - conf['max_speed_kn'])

        # Small bonus if moving closer to the destination than the expert's current point
        prev_rem_dist = distance((self.lat[self.time_step], self.lon[self.time_step]), self.destination) / 1852.0 \
            if hasattr(self, 'lat') and len(self.lat) > self.time_step else self.state['remaining_distance_nm']
        if self.state['remaining_distance_nm'] < prev_rem_dist:
            reward += 0.1

        # --- 7) Advance time and termination checks ---
        self.time_step += 1

        # Terminal success if within tolerance_nm
        if self.state['remaining_distance_nm'] <= conf['destination_tolerance_nm']:
            self.done = True
            reward += 10.0  # large terminal reward for reaching destination
            return self.get_observation(), self.state, float(reward), True

        # Terminal failure if out of steps
        if self.time_step >= conf['samples_per_vessel']:
            self.done = True
            reward -= 5.0
            return self.get_observation(), self.state, float(reward), True

        # Not done yet: return new observation
        return self.get_observation(), self.state, float(reward), False
