# snake-env

This directory contains the package with the code for the Snake environment. After importing the package, you can create an environment
in your Python code by simply doing:

```python
import gymnasium as gym
import snake_env  # needed to register the environment

env = gym.make("snake_env/SnakeEnv-v0")
```

## Install

To install this package, open the `snake-env` directory and run the following command in your shell:

```sh
pip install .
```

If you want an editable install, meaning that changes you write will immediately be reflected without having to reinstall the package, run the following command instead:

```sh
pip install -e .
```
