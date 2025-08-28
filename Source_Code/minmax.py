import numpy as np
import pandas as pd

obs_actions = pd.read_csv('./dataset/train_set.csv').drop(['trajectory_ID'], axis=1)

obs_actions_np = np.asarray(obs_actions)

min_obsact = np.min(obs_actions_np)
max_obsact = np.max(obs_actions_np)

print('observ-actions..min:',min_obsact,' max:',max_obsact)





