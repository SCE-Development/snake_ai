# snake-ai

This directory contains the package with the code for training
an agent on the snake environment. It contains our own implementation
of the PPO algorithm, as well as utilities for collecting samples.

This directory also contains several scripts, including:

- `human_player.py` - a script for playing the snake game as a human
- `random_player.py` - a script for viewing a random agent play the snake game
- `ai_player.py` - a script for viewing an ai agent on the snake game
- `example_trainer.py` - an example script showing how to use this repository to train an agent for the snake game

## Install

To install this package, open the `snake-ai` directory and run the following command in your shell:

```sh
pip install .
```

If you want an editable install, meaning that changes you write will immediately be reflected without having to reinstall the package, run the following command instead:

```sh
pip install -e .
```
