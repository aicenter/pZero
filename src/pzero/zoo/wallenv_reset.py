from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from minigrid.minigrid_env import MiniGridEnv

from gymnasium.envs.registration import register

class WallEnvReset(MiniGridEnv):

    """
    ## Description

    Room split by a vertical hole that limits vision.

    ## Mission Space

    "get to the goal"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Unused                    |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Unused                    |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    Reward is 1 for reaching the goal, -0.01 for each step.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-DoorKey-5x5-v0`
    - `MiniGrid-DoorKey-6x6-v0`
    - `MiniGrid-DoorKey-8x8-v0`
    - `MiniGrid-DoorKey-16x16-v0`

    """

    def __init__(self, size=8, max_steps: int | None = None, **kwargs):
        if max_steps is None:
            max_steps = 10 * size**2
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space, grid_size=size, max_steps=max_steps, **kwargs
        )

    @staticmethod
    def _gen_mission():
        return "get to the goal behind the wall"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width - 2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a hole in the wall
        holeIdx = self._rand_int(1, width - 2)
        self.grid.set(splitIdx, holeIdx, None)
        # self.put_obj(Door("yellow", is_locked=True), splitIdx, doorIdx)

        # # Place a yellow key on the left side
        # self.place_obj(obj=Key("yellow"), top=(0, 0), size=(splitIdx, height))

        self.mission = "get to the goal"

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Copied and modified from minigrid.minigrid_env.MiniGridEnv.step"""
        self.step_count += 1

        reward = -0.01
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                # terminated = True
                # reward = self._reward()
                reward = 1
                self._teleport_agent_randomly()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}
    
    def _teleport_agent_randomly(self):
        """Teleport the agent to a random empty position in the grid."""
        # Use the same logic as place_agent but for teleportation
        self.agent_pos = (-1, -1)
        
        # Find all empty positions in the grid
        empty_positions = []
        for i in range(1, self.grid.width - 1):  # Exclude walls
            for j in range(1, self.grid.height - 1):  # Exclude walls
                if self.grid.get(i, j) is None:  # Empty cell
                    empty_positions.append((i, j))
        
        # Randomly select an empty position
        if empty_positions:
            new_pos = self._rand_elem(empty_positions)
            self.agent_pos = new_pos
            # Randomize direction too
            self.agent_dir = self._rand_int(0, 4)

register(
    id="MiniGrid-WallEnvReset-5x5-v0",
    entry_point="pzero.zoo.wallenv_reset:WallEnvReset",
    kwargs={"size": 5},
)
print("MiniGrid-WallEnvReset-5x5-v0")

register(
    id="MiniGrid-WallEnvReset-6x6-v0",
    entry_point="pzero.zoo.wallenv_reset:WallEnvReset",
    kwargs={"size": 6},
)
print("MiniGrid-WallEnvReset-6x6-v0")

register(
    id="MiniGrid-WallEnvReset-7x7-v0",
    entry_point="pzero.zoo.wallenv_reset:WallEnvReset",
    kwargs={"size": 7},
)
print("MiniGrid-WallEnvReset-7x7-v0")

register(
    id="MiniGrid-WallEnvReset-8x8-v0",
    entry_point="pzero.zoo.wallenv_reset:WallEnvReset",
    kwargs={"size": 8},
)
print("MiniGrid-WallEnvReset-8x8-v0")

if __name__ == "__main__":
    env = WallEnvReset()
    env.reset()
    # Get the frame and save it
    frame = env.get_frame(highlight=True, tile_size=8)
    from PIL import Image
    Image.fromarray(frame).save("wall_env.png")
    env.close()