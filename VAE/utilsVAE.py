import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

sns.set(style="darkgrid")

def normalize_observation(obs, mean, std):
    return (obs - mean) / (std + 1e-8)

def unnormalize_observation(obs, mean, std):
    return obs * (std + 1e-8) + mean

def normalize_action(action, max_val, min_val):
    return (action - min_val) / (max_val - min_val)

def unnormalize_action(action, max_val, min_val):
    return action * (max_val - min_val) + min_val

def starting_points(random_choice=True, fname=None):
    # Load initial states or random starting points for episodes
    if fname:
        data = np.load(fname)
    else:
        data = np.zeros((50, 15))  # 50 episodes x 15 obs dims (example)
    if random_choice:
        np.random.shuffle(data)
    return data

def plot_vae(plot_x, latent_vars, e, save_dir="./plots"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8,8))
    plt.plot(plot_x, latent_vars)
    plt.xlabel("Time Steps")
    plt.ylabel("Latent Variables")
    plt.title(f"VAE Latents - Epoch {e}")
    plt.savefig(f"{save_dir}/epoch{e}_vae_latents.png", bbox_inches='tight')
    plt.close()

def rename_checkpoint(old_checkpoint_path, new_checkpoint_path, mapping):
    # mapping = {"old_layer_name":"new_layer_name", ...}
    reader = tf.train.load_checkpoint(old_checkpoint_path)
    var_map = {}
    for old_name, shape in reader.get_variable_to_shape_map().items():
        new_name = mapping.get(old_name, old_name)
        var_map[new_name] = tf.Variable(reader.get_tensor(old_name))
    saver = tf.train.Checkpoint(**var_map)
    saver.save(new_checkpoint_path)
    print(f"Checkpoint renamed and saved to {new_checkpoint_path}")