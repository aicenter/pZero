Going with LightZero for this attempt

- Install with Python3.11

Via uv install:

```bash
uv venv
source .venv/bin/activate
uv add --editable ./LightZero
uv add --editable ./Minigrid
```

Note that I had to change LightZero/requiremnts.txt because of Minigrid, changing gymnasium from 2.8.0 to 2.8.1

Also installed:

```bash
UV add numba
uv add pyecharts
uv add transformers
```

On `uv run zoo/classic_control/cartpole/config/cartpole_muzero_config.py` I saw a speedup from 244s to 57s