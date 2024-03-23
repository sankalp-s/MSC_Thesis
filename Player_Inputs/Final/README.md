# Random Agent

[Interactive_script.py](https://github.com/sankalp-s/MSC_Thesis/blob/main/Player_Inputs/Final/Interactive_script.py)

This script provides a framework for creating interactive gym environments, particularly designed for retro games. It leverages Pyglet for rendering the game environment and managing user input, enabling real-time interaction with the game.

### Overview

**Interactive Class:**  
The `Interactive` class serves as the base for interactive gym environments. It handles environment initialization, state updates, screen rendering, user input handling, and the main event loop.

**RetroInteractive Class:**  
The `RetroInteractive` class extends `Interactive` and is specialized for retro games. It facilitates the translation of keyboard inputs into gym actions suitable for retro game environments.

**Main Functionality:**  
The `main()` function orchestrates the script's operation. It parses command-line arguments to specify the retro game, state, scenario, and recording settings. Then, it initializes a `RetroInteractive` instance with these settings and runs the interactive game loop.

### Dependencies

- Pyglet
- NumPy
- Retro
- argparse

### Usage

To use the script:
1. Install the required dependencies.
2. Clone or download the repository.
3. Run the script from the command line, specifying the desired retro game, state, scenario, and recording settings using command-line arguments.
