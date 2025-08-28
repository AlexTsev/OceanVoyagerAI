import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import copy
from Source_Code.myconfig import myconfig
from Source_Code.utils import normalize_observations_MinMaxScaler, normalize_actions_MinMaxScaler, normalize_observation, normalize_action, unnormalize_observation, unnormalize_action, unnormalize_observations_MinMaxScaler, unnormalize_actions_MinMaxScaler, load_dataset



# ---------------------------
# GPU setup
# ---------------------------
physical_devices = tf.config.list_physical_devices('GPU')
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices('GPU'))
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"Using GPU: {physical_devices[0]}")
    except Exception as e:
        print("Could not set memory growth:", e)
else:
    print("GPU not found, using CPU.")

# ---------------------------
# Include environment folder
# ---------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), 'Environment'))
from Environment.environment_ATH_SG import VesselEnvironment

# ---------------------------
# Ensure folders
# ---------------------------
os.makedirs('../checkpoints/model', exist_ok=True)
os.makedirs('../plots', exist_ok=True)
os.makedirs('../outputs', exist_ok=True)

# =========================
# Dataset Loading
# =========================
def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    start_time = df['timestamp'].min()
    df['timestamp'] = (df['timestamp'] - start_time).dt.total_seconds() / 60.0

    drop_cols = ['vessel_id', 'trajectory_id', 'delta_heading_cmd_deg', 'speed_setpoint_kn', 'dlon', 'dlat']
    X_columns = [c for c in df.columns if c not in drop_cols]
    #pd.set_option("display.max_columns", None)
    #X2=df[X_columns]
    #print('X STATE:', X2)
    X = df[X_columns].values.astype(np.float32)
    action_cols = ['delta_heading_cmd_deg', 'speed_setpoint_kn', 'dlon', 'dlat']
    y = df[action_cols].values.astype(np.float32)

    print(f"Loaded dataset {csv_file}: X shape {X.shape}, y shape {y.shape}")
    return X, y, X_columns

# =========================
# Un-normalization
# =========================
def unnormalize_state(state_norm, scaler, X_columns):
    """
    Convert normalized state dict back to raw/original scale using fitted scaler.

    state_norm: dict with normalized state values (from env)
    scaler: MinMaxScaler fitted on training data
    X_columns: list of column names (used for ordering)
    """
    # Extract values from dict in the correct column order
    norm_vec = np.array([state_norm[c] for c in X_columns], dtype=np.float32).reshape(1, -1)

    # Inverse transform
    raw_vec = scaler.inverse_transform(norm_vec)[0]

    # Reconstruct dict with original scale values
    state_raw = {c: raw_vec[i] for i, c in enumerate(X_columns)}
    return state_raw

def unnormalize_observation(observation_dict, myconfig):
    """
    Unnormalize a dictionary of observations using myconfig for mean/std.

    Args:
        observation_dict (dict): dict of normalized feature values
        myconfig (dict): dict containing <feature>_avg and <feature>_std for each feature

    Returns:
        dict: unnormalized observation dictionary
    """
    unnormalized = {}
    for key, value in observation_dict.items():
        avg_key = f"{key}_avg"
        std_key = f"{key}_std"
        if avg_key in myconfig and std_key in myconfig:
            unnormalized[key] = value * myconfig[std_key] + myconfig[avg_key]
        else:
            # Keep the original value if no mean/std exists
            unnormalized[key] = value
    return unnormalized

def unnormalize_actions(y_scaled, scaler):
    """
    Inverse transform normalized actions back to original scale.
    y_scaled: np.ndarray (normalized actions)
    scaler: MinMaxScaler fitted on original y
    """
    return scaler.inverse_transform(y_scaled)

# =========================
# Encoder
# =========================
def build_encoder(input_dim, num_modes):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    logits = layers.Dense(num_modes)(x)
    return Model(inputs, logits, name='encoder')

# =========================
# Decoder
# =========================
def build_decoder(num_modes, output_dim):
    latent_inputs = layers.Input(shape=(num_modes,))
    x = layers.Dense(64, activation='relu')(latent_inputs)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(output_dim, activation='linear')(x)
    return Model(latent_inputs, outputs, name='decoder')

# =========================
# Gumbel-Softmax
# =========================
def sample_gumbel_softmax(logits, temperature=0.5, hard=False):
    uniform = tf.random.uniform(tf.shape(logits), 0, 1)
    gumbel = -tf.math.log(-tf.math.log(uniform + 1e-20) + 1e-20)
    y = logits + gumbel
    y_soft = tf.nn.softmax(y / temperature)
    if hard:
        y_hard = tf.cast(tf.equal(y_soft, tf.reduce_max(y_soft, axis=-1, keepdims=True)), y_soft.dtype)
        y = tf.stop_gradient(y_hard - y_soft) + y_soft
        return y
    return y_soft

# =========================
# VAE Loss (two versions)
# =========================

def vae_loss(x_true, x_pred, logits, recon_weight=1.0, kl_weight=0.1):
    """
    Compute VAE loss: reconstruction + KL divergence (no reward term).
    """
    # Reconstruction loss (MSE)
    recon = tf.reduce_mean(tf.reduce_sum(tf.square(x_true - x_pred), axis=-1))

    # KL divergence (discrete categorical via Gumbel-softmax)
    q_y = tf.nn.softmax(logits, axis=-1)
    eps = 1e-20
    log_q = tf.math.log(q_y + eps)
    kl_each = tf.reduce_sum(
        q_y * (log_q - tf.math.log(1.0 / tf.cast(tf.shape(q_y)[-1], tf.float32) + eps)),
        axis=-1
    )
    kl = tf.reduce_mean(kl_each)

    # Total VAE loss
    total_loss = recon_weight * recon + kl_weight * kl

    return total_loss, recon, kl

# =========================
# Training
# =========================
def train_vae(csv_file, num_modes=5, epochs=100, batch_size=128, learning_rate=1e-3):
    # --- Load & normalize data ---
    X, y, X_columns = load_dataset(csv_file)
    print("🤖🚢 Agent Vessel started training... 🚢")
    #USING SCALER NORMALIZ
    #X_scaled, X_scaler = normalize_observations(X)
    #y_scaled, y_scaler = normalize_actions(y)
    #input_dim = X_scaled.shape[1]

    # --- Normalize observations manually using your function ---
    # normalize_observation expects a single row, so vectorize over all rows
    X_scaled = np.array([normalize_observation(obs) for obs in X], dtype=np.float32)
    y_scaled = np.array([normalize_action(a) for a in y], dtype=np.float32)

    input_dim = X_scaled.shape[1]  # number of features, e.g., 49
    output_dim = input_dim  # VAE reconstructs same dimension

    # --- Build VAE model ---
    encoder = build_encoder(input_dim, num_modes)
    decoder = build_decoder(num_modes, output_dim)
    optimizer = Adam(learning_rate)

    # --- Prepare TensorFlow dataset ---
    dataset = (
        tf.data.Dataset.from_tensor_slices(X_scaled)
        .shuffle(10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # --- Environment (for reward logging only) ---
    env = VesselEnvironment()
    env.reset()
    latent_history = []

    # --- Training loop ---
    for epoch in range(epochs):
        total_loss = total_recon = total_kl = total_reward = 0.0
        steps = 0

        for step, batch_x in enumerate(dataset):
            steps += 1
            # Sample expert action for env (reward logging only)
            action_idx = min(env.time_step, y.shape[0] - 1)
            expert_action = y[action_idx]
            _, _, reward, done = env.step(expert_action)
            total_reward += float(reward)

            with tf.GradientTape() as tape:
                logits = encoder(batch_x)
                z = sample_gumbel_softmax(logits, temperature=0.5)
                x_recon = decoder(z)
                loss, recon, kl = vae_loss(batch_x, x_recon, logits)

            vars_ = encoder.trainable_variables + decoder.trainable_variables
            grads = tape.gradient(loss, vars_)
            optimizer.apply_gradients(zip(grads, vars_))

            # Track latent choice (for visualization)
            latent_mode = int(tf.argmax(z, axis=-1).numpy()[0])
            latent_history.append(latent_mode)

            total_loss += float(loss.numpy())
            total_recon += float(recon.numpy())
            total_kl += float(kl.numpy())

            if done:
                env.reset()

        steps = max(steps, 1)
        print(
            f"Epoch {epoch+1}/{epochs}  "
            f"Loss: {total_loss/steps:.6f}  "
            f"Recon: {total_recon/steps:.6f}  "
            f"KL: {total_kl/steps:.6f}  "
            f"AvgReward (log only): {total_reward/steps:.6f}"
        )

        # --- Plot latent trajectory ---
        plt.figure(figsize=(12, 3))
        plt.plot(latent_history, marker='o', linestyle='-')
        plt.title("Latent Modes During Training")
        plt.xlabel("Step")
        plt.ylabel("Latent Mode")
        plt.savefig(f"../plots/latent_modes_epoch{epoch+1}.png")
        plt.close()

        # --- Save checkpoint ---
        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            encoder.save_weights(f"../checkpoints/model/encoder_epoch{epoch+1}.h5")
            decoder.save_weights(f"../checkpoints/model/decoder_epoch{epoch+1}.h5")

    encoder.save_weights("../checkpoints/model/encoder_final.h5")
    decoder.save_weights("../checkpoints/model/decoder_final.h5")
    print("✅ VAE training complete")
    return encoder, decoder, X_columns


# =========================
# Testing with agent-computed actions
# =========================

def test_vae(encoder, decoder, X_columns, num_trajectories=50, trajectory_length=1000, output_excel='../outputs/vae_test_trajectories.xlsx'):
    print("🤖🚢 Agent Vessel started testing on: " + str(num_trajectories) + " trajectories")
    env = VesselEnvironment()

    all_trajectories = []        # keep everything in memory
    all_trajectoriescsv = []

    csv_path = output_excel.replace(".xlsx", ".csv")

    # Clear CSV if exists
    open(csv_path, 'w').close()

    for traj_id in range(num_trajectories):
        # reset environment, get initial state
        state_norm, state_raw = env.reset()
        traj_data = []
        traj_excel_data = []
        latent_modes = []
        print("trajectoryID:", traj_id)

        for t in range(trajectory_length):
            # --- normalize current observation manually ---
            obs_vec = np.array([state_raw[c] for c in X_columns], dtype=np.float32)
            obs_scaled = np.array([normalize_observation(obs_vec)], dtype=np.float32)  # shape (1, features)

            # --- forward pass through VAE ---
            logits = encoder(obs_scaled)
            z = sample_gumbel_softmax(logits, temperature=0.5, hard=True)
            x_recon = decoder(z)

            # --- convert reconstruction to action ---
            action = np.array([
                float(x_recon[0, 0]),  # delta_heading
                float(x_recon[0, 1]),  # speed_setpoint
                float(x_recon[0, 2]),  # dlon
                float(x_recon[0, 3])   # dlat
            ])

            # --- take step in environment ---
            state_norm, state_raw, reward, done = env.step(action)
            traj_data.append(state_raw)
            traj_excel_data.append(copy.deepcopy(state_raw))

            # track latent modes
            latent_mode = int(tf.argmax(z, axis=-1).numpy()[0])
            latent_modes.append(latent_mode)

            if done:
                break

        # Append trajectory to CSV incrementally
        df_traj = pd.DataFrame(traj_excel_data)
        df_traj.to_csv(csv_path, mode='a', header=(traj_id == 0), index=False)

        # Keep for plotting or further processing
        all_trajectories.extend(traj_data)
        all_trajectoriescsv.extend(traj_excel_data)

        # Plot latent modes for this trajectory
        plt.figure(figsize=(12, 3))
        plt.plot(latent_modes, marker='o', linestyle='-')
        plt.title(f"Latent Modes for Trajectory {traj_id + 1}")
        plt.xlabel("Step")
        plt.ylabel("Latent Mode")
        plt.savefig(f"../plots/latent_modes_test_traj{traj_id + 1}.png")
        plt.close()

    # Optional: save full Excel if needed
    df_all = pd.DataFrame(all_trajectoriescsv)
    df_all.to_excel(output_excel, index=False)

    print(f"✅ Saved {num_trajectories} test trajectories to {csv_path} and {output_excel}")

def test_withloaded_weights():
    # Define again your model architectures with SAME parameters
    num_modes = 5
    X, y, X_columns = load_dataset(csv_file)
    # --- Normalize observations manually using your function ---
    # normalize_observation expects a single row, so vectorize over all rows
    X_scaled = np.array([normalize_observation(obs) for obs in X], dtype=np.float32)
    y_scaled = np.array([normalize_action(a) for a in y], dtype=np.float32)

    input_dim = X_scaled.shape[1]
    output_dim = input_dim

    # --- Build model ---
    encoder = build_encoder(input_dim, num_modes)
    decoder = build_decoder(num_modes, output_dim)

    # Load trained weights
    encoder.load_weights("../checkpoints/model/encoder_epoch50.h5")
    decoder.load_weights("../checkpoints/model/decoder_epoch50.h5")

    print("✅ Encoder and decoder weights loaded successfully")
    test_vae(encoder, decoder, X_columns, num_trajectories=50)

# =========================
# Main
# =========================
if __name__ == '__main__':
    csv_file = '../Dataset/vessel_dataset_full_geo.csv'
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Dataset not found at {csv_file}")

    #encoder, decoder, X_columns = train_vae(csv_file, num_modes=5, epochs=2000, batch_size=128)
    #print("🤖🚢 Agent Vessel completed training!! 🚢")
    #test_vae(encoder, decoder, X_columns, num_trajectories=50)
    test_withloaded_weights()