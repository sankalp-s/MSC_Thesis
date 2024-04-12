# Super Mario World RL Agent - PPO

This script utilizes the Retro and Stable Baselines3 libraries to create a reinforcement learning (RL) agent for playing the Super Mario World game. Below is a summary of its functionality:

## 1. Environment Setup
- Sets up the Super Mario World environment using Retro, specifying the game, state, and observation type.

## 2. Preprocessing
- Applies several preprocessing steps to the environment:
  - StochasticFrameSkip: Skips frames with a certain probability to reduce processing load.
  - WarpFrame: Resizes and converts the RGB image to grayscale for easier processing.

## 3. Model Loading
- Loads a pre-trained Proximal Policy Optimization (PPO) model from the specified file ("PPO_IMAGE_L1.zip").

## 4. Agent Interaction
- Runs a loop where the agent interacts with the environment:
  - Predicts an action using the loaded PPO model based on the current observation.
  - Executes the action in the environment and observes the resulting state, reward, and other information.
  - Renders the environment to visualize the agent's actions.
  - Resets the environment if the episode is finished.

## 5. Blank Action Handling
- Ensures that the agent does not hold any action (e.g., jump) between steps by executing a blank action.

This script demonstrates how to load a pre-trained RL model and use it to control an agent in the Super Mario World environment, showcasing the integration of Retro and Stable Baselines3 libraries for reinforcement learning tasks.

