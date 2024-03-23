# Interactive script

[Interactive_script.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/Interactive_script.py)

This script provides a framework for creating interactive gym environments, particularly designed for retro games. It leverages Pyglet for rendering the game environment and managing user input, enabling real-time interaction with the game.

### Overview

**Interactive Class:**  
The `Interactive` class serves as the base for interactive gym environments. It handles environment initialization, state updates, screen rendering, user input handling, and the main event loop.

**RetroInteractive Class:**  
The `RetroInteractive` class extends `Interactive` and is specialized for retro games. It facilitates the translation of keyboard inputs into gym actions suitable for retro game environments.

**Main Functionality:**  
The `main()` function orchestrates the script's operation. It parses command-line arguments to specify the retro game, state, scenario, and recording settings. Then, it initializes a `RetroInteractive` instance with these settings and runs the interactive game loop.

# Replay Sessions

[replay_sessions.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/replay_sessions.py)

This Python script loads training data from a NumPy .npy file, simulates gameplay in the Super Mario World environment using the loaded actions, and renders the environment after each action. It leverages the retro library to create the game environment and applies actions sequentially to simulate gameplay. Optionally, a time delay can be introduced to observe the game rendering more clearly.

# View files

[view_files.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/view_files.py)

This Python script provides a function `open_npy_file()` to load data from a NumPy `.npy` file. It allows users to specify the file path and returns the loaded data as a NumPy array. The function includes error handling to deal with cases where the file is not found or errors occur during loading.

## Summary of Functionality:

### `open_npy_file(file_path):`
- Opens a `.npy` file located at the specified `file_path`.
- Returns the loaded data as a NumPy array if successful.
- Handles errors gracefully, printing appropriate messages for file not found or other exceptions.



# ROM file

[rominfo.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Scripts/rominfo.py)

This script provides functions to extract attributes from the RAM memory of the game Super Mario World. It includes functions to retrieve the agent's position, obtain information about sprites displayed on the screen, check for obstacles, and gather inputs within a specified radius around the agent. These functions are useful for creating AI agents or analyzing game states in reinforcement learning tasks. Additionally, the script contains a function to retrieve the entire RAM memory from the game environment, which can be utilized for various debugging or analysis purposes.
