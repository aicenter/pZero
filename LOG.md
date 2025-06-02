Going with LightZero for this attempt

- Install with Python3.11

Via uv install:

```bash
uv venv
source .venv/bin/activate
uv init
uv add --editable ./LightZero
# uv add --editable ./Minigrid

uv pip install -e . # install the project in the venv, so that e.g. environments can be imported
```


Also installed:

```bash
UV add numba
uv add pyecharts
uv add transformers
```

On `uv run LightZero/zoo/classic_control/cartpole/config/cartpole_muzero_config.py` I saw a speedup from 244s to 57s

## Tools and such

- VS code offered to install tensorboard plugin, after installing `uv add torchvision` it loaded all logs and run within VS code. Neat!
- Code navigation (go to definition (F12)) was not working with lightzero functions (installed as editable from local directory). There was an easy fix, going to settings and changing python.languageServer to Jedi. This fixed it.
