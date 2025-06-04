
- mapa -> obrazek


When training:

Monitor Key Metrics:
- Episode Return/Reward: The primary indicator of performance. Is it increasing?
- Losses: Value loss, policy loss, reward loss (for the model), and SSL loss. Are they decreasing? Are they stable or fluctuating wildly?
- Exploration: How many unique states is the agent visiting? (This might require custom logging).


I set up tensorboard and I am looking at my metrics now. There are these tabs:
- Buffer
- collector_iter
- collector_step
- evaluator_iter
- evaluator_step
- learner_iter
- learner_step



## Minigrid

Observations are dictionaries containing:
     - an image (partially observable view of the environment)
     - the agent's direction/orientation (acting as a compass)
     - a textual mission string (instructions for the agent)

Image observations are tensors of shape (agent_view_size, agent_view_size, 3), (7,7,3) by default.

Each cell is encoded as 3-dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE)

The encoding uses three constant mappings defined in constants.py:
Object Types (OBJECT_TO_IDX):
0: unseen (cells not visible to agent)
1: empty (empty floor space)
2: wall
3: floor
4: door
5: key
6: ball
7: box
8: goal
9: lava
10: agent
Colors (COLOR_TO_IDX):
0: red
1: green
2: blue
3: purple
4: yellow
5: grey
States (STATE_TO_IDX):
0: open (for doors)
1: closed (for doors)
2: locked (for doors)
3. Direction Encoding
The direction field encodes the agent's orientation as an integer:
0: Pointing right (positive X)
1: Down (positive Y)
2: Pointing left (negative X)
3: Up (negative Y)







