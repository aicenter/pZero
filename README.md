# pZero - an attempt at implementing MuZero for POMDPs

See the report for the plan.

# Installation
After failures with previous packages, I am using LightZero in this attempt, which is a framework that contains multiple implementations of MuZero, is actively maintained, documented and can be installed.

- Install with Python3.11

Via uv install:

```bash
uv venv
source .venv/bin/activate
uv init
uv add --editable ./LightZero
uv add --editable minigrid==2.2.1 # this is the version of minigrid that is compatible with lightzero

uv pip install -e . # install the project in the venv, so that e.g. environments can be imported
```


Also installed, but not strictly necessary:

```bash
UV add numba
uv add pyecharts
uv add transformers
```

After adding numba, `uv run LightZero/zoo/classic_control/cartpole/config/cartpole_muzero_config.py` I saw a speedup from 244s to 57s

## Tools and such

- VS code offered to install tensorboard plugin, after installing `uv add torchvision` it loaded all logs and run within VS code. Neat!
- Code navigation (go to definition (F12)) was not working with lightzero functions (installed as editable from local directory). There was an easy fix, going to settings and changing python.languageServer to Jedi. This fixed it.

## Development
I was using VS code for remote development on the game theory group server. To be able to use git, I set up ssh-agent forwarding. To get the tensorboard to work, I ran tensorboard inside tmux on the server, and forwarded the tensorboard port to my browser.

# Project structure and implementation

I was trying to follow the structure of the lightzero project. I would usually start by copying the entrypoint from LightZero, such as `LightZero/zoo/minigrid/config/minigrid_muzero_config.py`, and modifying to first run, and then modifying it to my needs. If I needed to modify parts of the environment or the model, I would first copy them over, change them to make sure they work, and then modify them. That is how most of the files in `src/pzero` were created. I did a small change to the LightZero, it is a single commit in the LightZero [fork](https://github.com/aicenter/LightZero/tree/pZero).

```
pzero/
├── .venv
├── data
├── LightZero
├── Minigrid
├── scripts       # scripts for running the experiments from CLI
└── src/
    ├── pzero/    # pzero code
    │   └── zoo   # configs, minigrid environments, lightzero environments
    └── test      # test scripts
```

To run pZero, you can use the scripts in the `scripts` directory. For example, to run pZero on the registered `MiniGrid-WallEnvReset-5x5-v0` environment with debug-level logging, you can use the command:

```bash
uv run ./scripts/run_pomuzero_recursive_minigrid.py --env-id MiniGrid-WallEnvReset-5x5-v0 --log-level DEBUG
```

This script is just a CLI wrapper around the `/home/mrkosja1/pZero/src/pzero/zoo/pomuzero_recursive_minigrid_config.py` file, which is the main entry point, and which can be run also directly. 

The scripts generate logs in the `data` directory. These logs can be viewed with tensorboard, and the logs contain also the model checkpoints and gifs of the agent's behavior in the episodes. 

## State of things and next steps
I have merged everything into the `main` branch. The commit ["Attempt at recursive representation, not learning"](https://github.com/aicenter/pZero/commit/97b77b4b70ea8f42e9fd52614a449b7410d488c3) contains attempt at recursive representation network, which is not learning anything. This is one thick commit, and it may be better to start just before it from scratch, where I am somewhat confident things work as intended. 



