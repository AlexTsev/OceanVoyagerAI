#!/usr/bin/python
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
import sys

from myconfig import myconfig
import trpo
import critic

# ----------------------------
# Environment import fix
# ----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from Environment.environment_ATH_SG import VesselEnvironment as environment
except Exception as e:
    raise ImportError(f"Could not import VesselEnvironment: {e}")

# ----------------------------
# Ensure TF GPU usage
# ----------------------------
if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ----------------------------
# Argument parsing
# ----------------------------
def read_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-e', '--experiment', default="0")
    parser.add_argument('-s', '--split_dataset', nargs='?', const=True, default=False)
    parser.add_argument('-i', '--input_dir', default="./")
    parser.add_argument('-o', '--output_dir', default="./")
    parser.add_argument('-l', '--log_dir', default="./")
    return parser.parse_args()

# ----------------------------
# Normalization helpers
# ----------------------------
def normalize_actions(actions):
    actions['delta_heading_cmd_deg'] = (actions['delta_heading_cmd_deg'] - myconfig['delta_heading_cmd_deg_avg']) / myconfig['delta_heading_cmd_deg_std']
    actions['speed_setpoint_kn'] = (actions['speed_setpoint_kn'] - myconfig['speed_setpoint_kn_avg']) / myconfig['speed_setpoint_kn_std']
    actions['remaining_distance_nm'] = (actions['remaining_distance_nm'] - myconfig['remaining_distance_nm_avg']) / myconfig['remaining_distance_nm_std']
    return actions

def normalize_observations(obs):
    # Convert timestamp to numeric minutes if exists
    if 'timestamp' in obs.columns:
        obs['timestamp'] = pd.to_datetime(obs['timestamp'])
        start_time = obs['timestamp'].min()
        obs['timestamp'] = (obs['timestamp'] - start_time).dt.total_seconds() / 60.0

    if 'vessel_id' in obs.columns:
        obs = obs.drop(['vessel_id'], axis=1)

    numeric_cols = obs.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        avg = myconfig.get(f'{col}_avg', obs[col].mean())
        std = myconfig.get(f'{col}_std', obs[col].std())
        obs[col] = (obs[col] - avg) / (std if std > 0 else 1)
    return obs

# ----------------------------
# Dataset splitting
# ----------------------------
def split_dataset(file, input_dir, splits=3):
    df = pd.read_csv(file)
    if 'vessel_id' not in df.columns:
        raise KeyError("CSV does not contain 'vessel_id' column")

    vessels = shuffle(df['vessel_id'].sort_values().unique())
    split_vessel_num = len(vessels) // splits
    train_idx = split_vessel_num * (splits - 2)
    train_vessels = vessels[:train_idx]
    validation_vessels = vessels[train_idx:train_idx + split_vessel_num]
    test_vessels = vessels[train_idx + split_vessel_num:]

    obs_train = df[df['vessel_id'].isin(train_vessels)].drop(['vessel_id', 'delta_heading_cmd_deg', 'speed_setpoint_kn', 'remaining_distance_nm'], axis=1)
    obs_validate = df[df['vessel_id'].isin(validation_vessels)].drop(['vessel_id', 'delta_heading_cmd_deg', 'speed_setpoint_kn', 'remaining_distance_nm'], axis=1)
    obs_test = df[df['vessel_id'].isin(test_vessels)].drop(['vessel_id', 'delta_heading_cmd_deg', 'speed_setpoint_kn', 'remaining_distance_nm'], axis=1)

    actions_train = df[df['vessel_id'].isin(train_vessels)][['delta_heading_cmd_deg', 'speed_setpoint_kn', 'remaining_distance_nm']]
    actions_validate = df[df['vessel_id'].isin(validation_vessels)][['delta_heading_cmd_deg', 'speed_setpoint_kn', 'remaining_distance_nm']]
    actions_test = df[df['vessel_id'].isin(test_vessels)][['delta_heading_cmd_deg', 'speed_setpoint_kn', 'remaining_distance_nm']]

    return obs_train, actions_train, obs_validate, actions_validate, obs_test, actions_test

# ----------------------------
# Main execution
# ----------------------------
args = read_args()
myconfig['input_dir'] = args.input_dir
myconfig['output_dir'] = args.output_dir
myconfig['exp'] = args.experiment
myconfig['log_dir'] = args.log_dir

dataset_file = os.path.join(myconfig['input_dir'], 'Dataset', 'vessel_dataset_full_geo.csv')
if args.split_dataset:
    obs_train, actions_train, obs_validate, actions_validate, obs_test, actions_test = split_dataset(dataset_file, myconfig['input_dir'])
else:
    obs_train = pd.read_csv(os.path.join(myconfig['input_dir'], 'obs_train.csv'))
    obs_validate = pd.read_csv(os.path.join(myconfig['input_dir'], 'obs_validate.csv'))
    obs_test = pd.read_csv(os.path.join(myconfig['input_dir'], 'obs_test.csv'))
    actions_train = pd.read_csv(os.path.join(myconfig['input_dir'], 'actions_train.csv'))
    actions_validate = pd.read_csv(os.path.join(myconfig['input_dir'], 'actions_validate.csv'))
    actions_test = pd.read_csv(os.path.join(myconfig['input_dir'], 'actions_test.csv'))

# Normalize observations & actions (keep timestamp)
obs_train = normalize_observations(obs_train)
obs_validate = normalize_observations(obs_validate)
obs_test = normalize_observations(obs_test)
actions_train = normalize_actions(actions_train)
actions_validate = normalize_actions(actions_validate)
actions_test = normalize_actions(actions_test)

# Keep only numeric columns for tensors
obs_train = obs_train.select_dtypes(include=[np.number])
obs_validate = obs_validate.select_dtypes(include=[np.number])
obs_test = obs_test.select_dtypes(include=[np.number])

##print(normalized first row)
#print("obs_train columns:", obs_train.columns.tolist())
#print("First row:", obs_train.iloc[0].to_dict())

# Convert to numpy arrays
obs_train, obs_validate, obs_test = obs_train.values, obs_validate.values, obs_test.values
actions_train, actions_validate, actions_test = actions_train.values, actions_validate.values, actions_test.values

# ----------------------------
# Initialize environment, encoder, discriminator, agent
# ----------------------------
env = environment()

encoder = trpo.Encoder(input_dim=obs_train.shape[1], latent_dim=5)
dummy_input = tf.convert_to_tensor(obs_train[:1], dtype=tf.float32)
_ = encoder(dummy_input)
encoder.load_weights('./checkpoints/model/encoder_final_0exp.h5')
print("✅ Loaded pre-trained encoder weights.")

discriminator = trpo.Discriminator(input_dim=obs_train.shape[1] + actions_train.shape[1])

agent = trpo.TRPOAgent(
    env=env,
    action_dimensions=3,
    latent_dimensions=5,
    observation_dimensions=obs_train.shape[1],
    encoder=encoder,
    critic=critic.Critic(observation_dimensions=obs_train.shape[1]),
    discriminator=discriminator
)

# ----------------------------
# Training Directed-Info GAIL
# ----------------------------
print("🤖Agent Starting Training...🚢")
agent.train(obs_train, actions_train)

# ----------------------------
# Save model weights
# ----------------------------
save_path = os.path.join(myconfig['output_dir'], f"directed_info_gail_{myconfig['exp']}")
os.makedirs(save_path, exist_ok=True)
agent.save(save_path)
print(f"✅ Saved model weights to {save_path}")

# ----------------------------
# Testing: 50 trajectories
# ----------------------------
num_test_trajectories = 50
actions_gail, obs_gail, discounted_rewards_gail, total_rewards_gail = agent.run(num_test_trajectories, vae=True, bcloning=True, fname='0%_test')

np.save(os.path.join(myconfig['output_dir'], 'obs_gail.npy'), obs_gail)
np.save(os.path.join(myconfig['output_dir'], 'actions_gail.npy'), actions_gail)
np.save(os.path.join(myconfig['output_dir'], 'rewards_gail.npy'), discounted_rewards_gail)

import pandas as pd
pd.DataFrame(obs_gail).to_excel("test_obs_50.xlsx", index=False)
pd.DataFrame(actions_gail).to_excel("test_act_50.xlsx", index=False)
pd.DataFrame(discounted_rewards_gail).to_excel("test_rew_50.xlsx", index=False)

print("✅ Saved test_obs_50.xlsx, test_act_50.xlsx, test_rew_50.xlsx")
print('Sum of Rewards Directed-Info GAIL:', sum(total_rewards_gail))
print('Mean Reward Directed-Info GAIL:', np.mean(total_rewards_gail))
