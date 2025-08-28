import tensorflow as tf
from tensorflow.keras import layers, optimizers, Model
import numpy as np
from critic import Critic


# ----------------------------
# Helper: Convert dict observation to numeric array
# ----------------------------
def obs_to_array(obs_dict):
    keys = ['lat', 'lon', 'dlat', 'dlon', 'sog_kn', 'yaw_rate_deg_per_min',
            'cross_track_nm', 'along_track_nm', 'wind_u10_ms', 'wind_v10_ms',
            'wind_speed_ms', 'wind_dir_deg', 'wave_hs_m', 'wave_tp_s', 'wave_dir_deg',
            'current_u_ms', 'current_v_ms', 'sea_state_bin', 'proximity_bin', 'traffic_bin',
            'draft_fwd_m', 'draft_aft_m', 'trim_m', 'displacement_t', 'load_bin',
            'eca_flag', 'tss_lane_flag', 'targets_in_5nm', 'min_cpa_nm', 'min_tcpa_min',
            'depth_under_keel_m', 'speed_limit_kn', 'est_power_kw', 'sfoc_gpkwh',
            'fuel_tph', 'fuel_per_nm_t', 'fuel_price_usd_per_ton', 'fuel_consumed_t',
            'fuel_cost_usd', 'delta_heading_cmd_deg', 'speed_setpoint_kn', 'remaining_distance_nm',
            'eta_error_min', 'mode_sea_state', 'mode_load', 'mode_proximity', 'mode_traffic']
    return np.array([obs_dict[k] for k in keys], dtype=np.float32)

# ----------------------------
# Helper: Gaussian log probability (for continuous actions)
# ----------------------------
def gauss_log_prob(mean, log_std, actions):
    # mean, log_std, actions are tensors of shape [B, act_dim]
    std = tf.exp(log_std)
    var = std ** 2
    log_prob = -0.5 * tf.reduce_sum(
        ((actions - mean) ** 2) / var + 2.0 * log_std + tf.math.log(2.0 * np.pi),
        axis=-1
    )
    return log_prob

# ----------------------------
# VAE-style Encoder (used to load your pre-trained encoder weights)
# ----------------------------
class Encoder(Model):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = layers.Dense(100, activation='relu', name='enc_dense1')
        self.fc2 = layers.Dense(100, activation='relu', name='enc_dense2')
        self.logits = layers.Dense(latent_dim, name='enc_logits')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.logits(x)

# ----------------------------
# Policy Network (Gaussian policy for continuous actions)
# ----------------------------
class PolicyNetwork(Model):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc1 = layers.Dense(100, activation='tanh', name='pi_fc1')
        self.fc2 = layers.Dense(100, activation='tanh', name='pi_fc2')
        self.mean = layers.Dense(action_dim, name='pi_mean')
        # one trainable log_std per action dim (state-independent std)
        self.log_std = tf.Variable(initial_value=tf.zeros(action_dim), trainable=True, name='pi_log_std')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        mean = self.mean(x)
        log_std = self.log_std
        # Broadcast log_std to batch dimension
        if len(mean.shape) == 2 and len(log_std.shape) == 1:
            log_std = tf.broadcast_to(log_std, tf.shape(mean))
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = tf.exp(log_std)
        noise = tf.random.normal(shape=tf.shape(mean))
        return mean + noise * std

# ----------------------------
# Discriminator (GAIL)
# ----------------------------
class Discriminator(Model):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = layers.Dense(100, activation='relu', name='disc_fc1')
        self.fc2 = layers.Dense(100, activation='relu', name='disc_fc2')
        self.out = layers.Dense(1, activation='sigmoid', name='disc_out')

    def call(self, obs, action):
        x = tf.concat([obs, action], axis=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)

# ----------------------------
# TRPO Agent (simplified updates; dataset-based training)
# ----------------------------
class TRPOAgent:
    def __init__(self, env, action_dimensions, latent_dimensions, observation_dimensions,
                 encoder=None, critic=None, discriminator=None, gamma=0.99, lr=3e-4):
        self.policy = PolicyNetwork(observation_dimensions, action_dimensions)
        # Your Critic wrapper (not a Keras Model directly)
        self.critic = critic if critic is not None else Critic(observation_dimensions)
        self.encoder = encoder
        self.discriminator = discriminator
        self.gamma = gamma
        self.env = env

        self.optimizer_policy = optimizers.Adam(lr)
        self.optimizer_critic = optimizers.Adam(lr)
        if self.discriminator is not None:
            self.optimizer_disc = optimizers.Adam(lr)

    # --- Losses ---
    def compute_policy_loss(self, obs, actions, advantages):
        mean, log_std = self.policy(obs)
        log_prob = gauss_log_prob(mean, log_std, actions)
        return -tf.reduce_mean(log_prob * advantages)

    def compute_discriminator_loss(self, obs_expert, actions_expert, obs_gen, actions_gen):
        if self.discriminator is None:
            return tf.constant(0.0, dtype=tf.float32)
        real_logits = self.discriminator(obs_expert, actions_expert)
        fake_logits = self.discriminator(obs_gen, actions_gen)
        real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_logits), real_logits)
        fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_logits), fake_logits)
        return tf.reduce_mean(real_loss + fake_loss)

    # --- One minibatch update ---
    def train_step(self, obs_batch, actions_batch, returns_batch):
        # Critic baseline values
        values = tf.convert_to_tensor(
            self.critic.predict(obs_batch).flatten(),  # uses your wrapper's .predict()
            dtype=tf.float32
        )
        advantages = returns_batch - values

        # Policy update
        with tf.GradientTape() as tape:
            loss_policy = self.compute_policy_loss(obs_batch, actions_batch, advantages)
        grads = tape.gradient(loss_policy, self.policy.trainable_variables)
        self.optimizer_policy.apply_gradients(zip(grads, self.policy.trainable_variables))

        # Critic update (use underlying Keras model)
        with tf.GradientTape() as tape:
            values_pred = self.critic.model(obs_batch, training=True)
            loss_critic = tf.reduce_mean(tf.square(returns_batch - tf.squeeze(values_pred, axis=-1)))
        grads_c = tape.gradient(loss_critic, self.critic.model.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(grads_c, self.critic.model.trainable_variables))

        return loss_policy, loss_critic

    # --- Optional discriminator update ---
    def train_discriminator(self, obs_expert, actions_expert, obs_gen, actions_gen):
        if self.discriminator is None:
            return 0.0
        with tf.GradientTape() as tape:
            loss_disc = self.compute_discriminator_loss(obs_expert, actions_expert, obs_gen, actions_gen)
        grads = tape.gradient(loss_disc, self.discriminator.trainable_variables)
        self.optimizer_disc.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return float(loss_disc.numpy())

    # --- Discounted returns for a single episode (utility) ---
    def discount_rewards(self, rewards, gamma=None):
        if gamma is None:
            gamma = self.gamma
        rewards = np.asarray(rewards, dtype=np.float32)
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            running = rewards[t] + gamma * running
            discounted[t] = running
        return discounted

    # --- Simple rollout (if your env supports it) ---
    def run(self, num_episodes, vae=False, bcloning=False, fname=None):
        actions_list, obs_list, discounted_rewards_list, total_rewards = [], [], [], []
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            ep_rewards, ep_actions, ep_obs = [], [], []

            while not done:
                obs_input = tf.convert_to_tensor([obs], dtype=tf.float32)
                action = self.policy.get_action(obs_input).numpy()[0]
                next_obs, reward, done, info = self.env.step(action)

                ep_actions.append(action)
                ep_obs.append(obs)
                ep_rewards.append(float(reward))
                obs = next_obs

            actions_list.append(np.array(ep_actions, dtype=np.float32))
            obs_list.append(np.array(ep_obs, dtype=np.float32))
            discounted_rewards_list.append(self.discount_rewards(ep_rewards))
            total_rewards.append(float(np.sum(ep_rewards)))

        return actions_list, obs_list, discounted_rewards_list, total_rewards

    # --- Dataset-based training (what main.py calls) ---
    def train(self, obs_train, actions_train, epochs=100, batch_size=256, returns=None,
              do_gail=True, gail_ratio=0.5):
        """
        Offline (behavior cloning-like) update with value baseline.
        - obs_train: np.array [N, obs_dim]
        - actions_train: np.array [N, act_dim]
        - returns: optional np.array [N] discounted returns; if None, zeros are used.
        - do_gail: if True and a discriminator is provided, also train it using the same data
                   and simple on-policy rollouts as "gen" data (very rough).
        """

        num_samples = obs_train.shape[0]
        obs_tensor_all = tf.convert_to_tensor(obs_train, dtype=tf.float32)
        actions_tensor_all = tf.convert_to_tensor(actions_train, dtype=tf.float32)

        if returns is None:
            returns_tensor_all = tf.zeros(shape=(num_samples,), dtype=tf.float32)
        else:
            returns_tensor_all = tf.convert_to_tensor(returns, dtype=tf.float32)

        for epoch in range(epochs):
            # shuffle indices
            idx = np.arange(num_samples)
            np.random.shuffle(idx)

            avg_pi_losses = []
            avg_v_losses = []
            avg_rew_proxy = []  # simple proxy: -policy_loss (just to have a number)

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                b = idx[start:end]

                obs_batch = tf.gather(obs_tensor_all, b)
                act_batch = tf.gather(actions_tensor_all, b)
                ret_batch = tf.gather(returns_tensor_all, b)

                loss_pi, loss_v = self.train_step(obs_batch, act_batch, ret_batch)
                avg_pi_losses.append(float(loss_pi.numpy()))
                avg_v_losses.append(float(loss_v.numpy()))
                avg_rew_proxy.append(float(-loss_pi.numpy()))

                # optional: lightweight GAIL update (uses same batch as "expert" and a quick rollout as "gen")
                if do_gail and self.discriminator is not None:
                    # generate a small batch of agent data with current policy (very rough)
                    gen_obs, gen_act = self._quick_rollout_batch(len(b))
                    if gen_obs is not None:
                        _ = self.train_discriminator(
                            tf.convert_to_tensor(gen_obs, dtype=tf.float32),  # treat as "expert" here if you prefer
                            tf.convert_to_tensor(gen_act, dtype=tf.float32),
                            obs_batch, act_batch
                        )

            print(f"[TRPOAgent.train] Epoch {epoch+1}/{epochs}\n"
                  f"Avg Policy Loss: {np.mean(avg_pi_losses):.4f} | "
                  f"Avg Critic Loss: {np.mean(avg_v_losses):.4f} | "
                  f"Avg Reward Proxy: {np.mean(avg_rew_proxy):.4f}")

    # --- tiny rollout helper for GAIL (optional) ---
    def _quick_rollout_batch(self, target_n):
        """
        Generate a quick batch of trajectories for GAIL-like discriminator updates.
        Converts environment states to feature vectors.
        """
        if self.env is None:
            return None, None

        obs_buf, act_buf = [], []
        obs = self.env.reset()
        done = False

        # Define the feature keys (excluding vessel_id, trajectory_id)
        feature_keys = [
            'timestamp', 'lat', 'lon', 'dlat', 'dlon', 'sog_kn', 'ax_kn_per_h', 'yaw_rate_deg_per_min',
            'cross_track_nm', 'along_track_nm', 'wind_u10_ms', 'wind_v10_ms', 'wind_speed_ms', 'wind_dir_deg',
            'wave_hs_m', 'wave_tp_s', 'wave_dir_deg', 'current_u_ms', 'current_v_ms',
            'sea_state_bin', 'proximity_bin', 'traffic_bin', 'draft_fwd_m', 'draft_aft_m', 'trim_m',
            'displacement_t', 'load_bin', 'eca_flag', 'tss_lane_flag', 'targets_in_5nm', 'min_cpa_nm',
            'min_tcpa_min', 'depth_under_keel_m', 'speed_limit_kn', 'est_power_kw', 'sfoc_gpkwh',
            'fuel_tph', 'fuel_per_nm_t', 'fuel_price_usd_per_ton', 'fuel_consumed_t', 'fuel_cost_usd',
            'eta_error_min', 'mode_sea_state', 'mode_load', 'mode_proximity', 'mode_traffic'
        ]

        while len(obs_buf) < target_n and not done:
            # Convert dict to feature vector
            obs_vec = np.array([obs[k] for k in feature_keys], dtype=np.float32)
            obs_buf.append(obs_vec)

            # Sample action from current policy
            act = self.policy.get_action(tf.convert_to_tensor([obs_vec], dtype=tf.float32)).numpy()[0]
            act_buf.append(act)

            # Step environment (no action argument)
            obs, done = self.env.step()

        if len(obs_buf) == 0:
            return None, None

        return np.array(obs_buf, dtype=np.float32), np.array(act_buf, dtype=np.float32)

    # --- Save helper (since main.py calls agent.save(path)) ---
    def save(self, save_dir):
        import os
        os.makedirs(save_dir, exist_ok=True)
        # Build variables before saving if not yet built
        # Dummy shape guesses: you can pass real examples if you want
        # (But usually you've already called the nets by now.)
        if not self.policy.built:
            _ = self.policy(tf.zeros((1, self.policy.fc1.input_shape[-1] if self.policy.fc1.input_shape else 1)))
        self.policy.save_weights(os.path.join(save_dir, "policy.h5"))

        # Save critic (underlying Keras model)
        self.critic.model.save_weights(os.path.join(save_dir, "critic.h5"))

        if self.encoder is not None:
            # Build encoder if needed
            if not self.encoder.built:
                _ = self.encoder(tf.zeros((1, self.encoder.fc1.input_shape[-1] if self.encoder.fc1.input_shape else 1)))
            self.encoder.save_weights(os.path.join(save_dir, "encoder.h5"))

        if self.discriminator is not None:
            # Build discriminator by calling once with dummy tensors
            try:
                _ = self.discriminator(tf.zeros((1, 1)), tf.zeros((1, 1)))
            except Exception:
                pass
            self.discriminator.save_weights(os.path.join(save_dir, "discriminator.h5"))
