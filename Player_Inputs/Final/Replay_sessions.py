import numpy as np
import retro
import time

# Load the Super Mario World environment
env = retro.make(game="SuperMarioWorld-Snes", state='YoshiIsland2', obs_type=retro.Observations.IMAGE)

# Load training data from the .npy file
Training = np.load("/Users/sankalpssss/Documents/marioenv/mario/mario2/Data final/Session1.npy", allow_pickle=True)
Xtrain = Training[0]
Ytrain = Training[1]
print(Ytrain.shape)  # Print the shape of the training labels array

# Fix issues with numpy loading
Ytrain = np.array([np.array(val) for val in Ytrain])
Ytrain = np.array([val.reshape(1, 12) for val in Ytrain])  # Reshape to fit model

print(Ytrain.shape)  # Print the shape of the training labels array

# Reset the environment
obs = env.reset()

# Iterate over actions and render the environment
for action in Ytrain:
    for a in action:
        obs, rew, done, _info = env.step(a)  # Use the generated action
        env.render()  # Render the environment
        # Optionally, introduce a time delay to observe the game rendering
        # time.sleep(1)
