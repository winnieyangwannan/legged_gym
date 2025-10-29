# Curriculum Learning System - Technical Report

## Executive Summary

This document provides a comprehensive overview of the curriculum learning system implemented in the legged robot reinforcement learning framework. The system employs **automatic curriculum learning** to progressively increase task difficulty as robots improve, enabling efficient training from simple to complex locomotion tasks.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Terrain Curriculum](#terrain-curriculum)
4. [Command Curriculum](#command-curriculum)
5. [Terrain Generation](#terrain-generation)
6. [Configuration Parameters](#configuration-parameters)
7. [Code Reference](#code-reference)
8. [Usage Guide](#usage-guide)

---

## Overview

### What is Curriculum Learning?

Curriculum learning is a training strategy where the difficulty of tasks gradually increases during training. Instead of immediately exposing robots to the hardest terrains and fastest velocities, they start with easier conditions and progress as they demonstrate competence.

### Benefits

- **Faster convergence**: Robots learn basic skills before tackling complex scenarios
- **Better final performance**: Progressive difficulty allows for more stable learning
- **Automatic adaptation**: No manual intervention required - the system adjusts based on performance

### Two Parallel Curricula

1. **Terrain Curriculum**: Adjusts environmental difficulty (slope steepness, obstacle height, etc.)
2. **Command Curriculum**: Increases velocity command ranges over time

---

## System Architecture

### High-Level Flow

```
Training Loop
    │
    └─→ Environment Step
          │
          └─→ post_physics_step()
                │
                └─→ reset_idx(env_ids)  ← Curriculum updates happen here
                      │
                      ├─→ Terrain Curriculum (ENABLED by default)
                      │   └─→ Adjust difficulty level based on distance walked
                      │
                      └─→ Command Curriculum (DISABLED by default)
                          └─→ Expand velocity ranges based on tracking performance
```

### Key Files

| File | Purpose |
|------|---------|
| `legged_gym/envs/base/legged_robot.py` | Core curriculum algorithms |
| `legged_gym/utils/terrain.py` | Terrain generation with difficulty scaling |
| `legged_gym/envs/base/legged_robot_config.py` | Configuration parameters |
| `legged_gym/scripts/play.py` | Testing/inference (curriculum disabled) |

---

## Terrain Curriculum

### Overview

The terrain curriculum automatically promotes robots to harder terrains when they succeed and demotes them to easier terrains when they struggle.

### Algorithm

**Location**: `legged_gym/envs/base/legged_robot.py:421-441`

```python
def _update_terrain_curriculum(self, env_ids):
    """ Implements the game-inspired curriculum.

    Args:
        env_ids (List[int]): ids of environments being reset
    """
    # Skip during initialization
    if not self.init_done:
        return

    # 1. Measure Progress: How far did the robot walk?
    distance = torch.norm(
        self.root_states[env_ids, :2] - self.env_origins[env_ids, :2],
        dim=1
    )

    # 2. Success Criterion: Walked more than half the terrain length
    move_up = distance > self.terrain.env_length / 2  # Default: 4 meters

    # 3. Failure Criterion: Walked less than half the commanded distance
    commanded_distance = torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s
    move_down = (distance < commanded_distance * 0.5) * ~move_up

    # 4. Update Difficulty Level
    self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

    # 5. Handle Maximum Level: Send to random level if beat hardest terrain
    self.terrain_levels[env_ids] = torch.where(
        self.terrain_levels[env_ids] >= self.max_terrain_level,
        torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
        torch.clip(self.terrain_levels[env_ids], 0)
    )

    # 6. Update Spawn Location for Next Episode
    self.env_origins[env_ids] = self.terrain_origins[
        self.terrain_levels[env_ids],
        self.terrain_types[env_ids]
    ]
```

### Decision Logic

| Condition | Distance Walked | Action |
|-----------|----------------|--------|
| **Success** | > 4m (50% of terrain) | Move up one difficulty level |
| **Failure** | < 50% of commanded distance | Move down one difficulty level |
| **Neutral** | Between thresholds | Stay at same level |
| **Mastery** | Reached max level (9) | Random assignment to level 0-8 |

### Terrain Grid Structure

Terrains are organized in a 2D grid:

```
       Column 0    Column 1    ...    Column 19
       (Slopes)    (Rough)            (Pits)
       ┌─────────┬─────────┬─────────┬─────────┐
Row 0  │ Easy    │ Easy    │ Easy    │ Easy    │  Difficulty: 0.0
       ├─────────┼─────────┼─────────┼─────────┤
Row 1  │         │         │         │         │  Difficulty: 0.1
       ├─────────┼─────────┼─────────┼─────────┤
Row 2  │         │         │         │         │  Difficulty: 0.2
       ├─────────┼─────────┼─────────┼─────────┤
  ...  │   ...   │   ...   │   ...   │   ...   │
       ├─────────┼─────────┼─────────┼─────────┤
Row 9  │ Hard    │ Hard    │ Hard    │ Hard    │  Difficulty: 1.0
       └─────────┴─────────┴─────────┴─────────┘
```

- **Rows (10)**: Difficulty levels (0 = easiest, 9 = hardest)
- **Columns (20)**: Different terrain types (slopes, stairs, obstacles, etc.)
- Each tile is 8m × 8m

### Initialization

**Location**: `legged_gym/envs/base/legged_robot.py:709-715`

```python
# Initialize terrain levels randomly
max_init_level = self.cfg.terrain.max_init_terrain_level  # Default: 5
self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)

# Distribute robots across terrain types (columns)
self.terrain_types = torch.div(
    torch.arange(self.num_envs, device=self.device),
    (self.num_envs / self.cfg.terrain.num_cols),
    rounding_mode='floor'
).to(torch.long)

# Set initial spawn positions
self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
```

With 4096 environments and 20 terrain types:
- ~205 robots per terrain type
- Each starts at random difficulty 0-5

### Logging

**Location**: `legged_gym/envs/base/legged_robot.py:182-183`

```python
if self.cfg.terrain.curriculum:
    self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
```

Average terrain level is logged to TensorBoard, allowing you to monitor curriculum progression during training.

---

## Command Curriculum

### Overview

The command curriculum gradually increases the range of velocity commands given to robots as they improve at tracking.

**Status**: **DISABLED by default** (can be enabled in config)

### Algorithm

**Location**: `legged_gym/envs/base/legged_robot.py:443-452`

```python
def update_command_curriculum(self, env_ids):
    """ Implements a curriculum of increasing commands

    Args:
        env_ids (List[int]): ids of environments being reset
    """
    # Check if tracking performance is good (80% of max reward)
    if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
        # Expand velocity range by ±0.5 m/s
        self.command_ranges["lin_vel_x"][0] = np.clip(
            self.command_ranges["lin_vel_x"][0] - 0.5,  # Expand backward
            -self.cfg.commands.max_curriculum,
            0.
        )
        self.command_ranges["lin_vel_x"][1] = np.clip(
            self.command_ranges["lin_vel_x"][1] + 0.5,  # Expand forward
            0.,
            self.cfg.commands.max_curriculum
        )
```

### Update Frequency

**Location**: `legged_gym/envs/base/legged_robot.py:161-162`

```python
# Only update periodically (not every reset)
if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
    self.update_command_curriculum(env_ids)
```

The command ranges are **shared across all environments**, so updates happen less frequently to ensure all robots experience the new range.

### Progression Example

| Training Stage | Tracking Performance | Velocity Range (m/s) |
|----------------|---------------------|----------------------|
| Initial | - | [-1.0, 1.0] |
| After success | > 80% | [-1.5, 1.5] |
| After success | > 80% | [-2.0, 2.0] |
| ... | ... | ... |
| Maximum | > 80% | [-max_curriculum, max_curriculum] |

### Logging

**Location**: `legged_gym/envs/base/legged_robot.py:184-185`

```python
if self.cfg.commands.curriculum:
    self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
```

---

## Terrain Generation

### Terrain Class

**Location**: `legged_gym/utils/terrain.py:38-73`

The `Terrain` class generates the curriculum terrain grid:

```python
class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.env_length = cfg.terrain_length  # 8m
        self.env_width = cfg.terrain_width    # 8m

        # Create empty height field
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        # Generate terrain based on mode
        if cfg.curriculum:
            self.curiculum()  # Generate difficulty progression
        elif cfg.selected:
            self.selected_terrain()  # Single terrain type
        else:
            self.randomized_terrain()  # Random difficulties
```

### Curriculum Generation Algorithm

**Location**: `legged_gym/utils/terrain.py:85-92`

```python
def curiculum(self):
    for j in range(self.cfg.num_cols):  # 20 terrain types
        for i in range(self.cfg.num_rows):  # 10 difficulty levels
            # Difficulty increases linearly with row
            difficulty = i / self.cfg.num_rows  # 0.0 → 1.0

            # Terrain type determined by column
            choice = j / self.cfg.num_cols + 0.001

            # Generate terrain tile
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
```

### Difficulty Scaling

**Location**: `legged_gym/utils/terrain.py:115-121`

Physical parameters scale with difficulty:

```python
def make_terrain(self, choice, difficulty):
    # Scale obstacle dimensions by difficulty (0.0 → 1.0)
    slope = difficulty * 0.4                          # 0° → 22° (0.4 rad)
    step_height = 0.05 + 0.18 * difficulty           # 5cm → 23cm
    discrete_obstacles_height = 0.05 + difficulty * 0.2  # 5cm → 25cm
    stepping_stones_size = 1.5 * (1.05 - difficulty) # 1.58m → 0.08m
    stone_distance = 0.05 if difficulty==0 else 0.1  # Gap between stones
    gap_size = 1. * difficulty                       # 0m → 1m
    pit_depth = 1. * difficulty                      # 0m → 1m

    # Generate appropriate terrain based on type...
```

### Terrain Types

**Location**: `legged_gym/utils/terrain.py:122-143`

Different terrain types are assigned to columns based on proportions:

| Terrain Type | Proportion | Description |
|--------------|-----------|-------------|
| Smooth slope | 10% | Clean uphill/downhill slopes |
| Rough slope | 10% | Slopes with random noise |
| Stairs up | 35% | Ascending stairs |
| Stairs down | 25% | Descending stairs |
| Discrete obstacles | 20% | Random rectangular obstacles |
| Stepping stones | - | Sparse footholds |
| Gaps | - | Holes to jump over |
| Pits | - | Deep depressions |

### Environment Origins

**Location**: `legged_gym/utils/terrain.py:147-164`

Each terrain tile stores its spawn location:

```python
def add_terrain_to_map(self, terrain, row, col):
    # Place terrain in grid
    start_x = self.border + row * self.length_per_env_pixels
    end_x = self.border + (row + 1) * self.length_per_env_pixels
    start_y = self.border + col * self.width_per_env_pixels
    end_y = self.border + (col + 1) * self.width_per_env_pixels
    self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

    # Calculate spawn position (center of tile, at terrain height)
    env_origin_x = (row + 0.5) * self.env_length
    env_origin_y = (col + 0.5) * self.env_width

    # Sample height at spawn location
    x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
    x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
    y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
    y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
    env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale

    self.env_origins[row, col] = [env_origin_x, env_origin_y, env_origin_z]
```

The `env_origins` array is later accessed by the curriculum system to update robot spawn locations.

---

## Configuration Parameters

### Terrain Curriculum Configuration

**Location**: `legged_gym/envs/base/legged_robot_config.py:43-67`

```python
class terrain:
    mesh_type = 'trimesh'  # Required for curriculum (or 'heightfield')
    horizontal_scale = 0.1  # [m] Resolution of terrain grid
    vertical_scale = 0.005  # [m] Height resolution
    border_size = 25  # [m] Flat border around terrain

    # CURRICULUM SETTINGS
    curriculum = True  # Enable terrain curriculum
    max_init_terrain_level = 5  # Starting difficulty (0-9)

    # TERRAIN GRID
    terrain_length = 8.  # [m] Size of each tile
    terrain_width = 8.   # [m]
    num_rows = 10  # Difficulty levels (0=easy, 9=hard)
    num_cols = 20  # Terrain type variety

    # TERRAIN TYPE DISTRIBUTION
    # [smooth_slope, rough_slope, stairs_up, stairs_down, discrete_obstacles]
    terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]

    # HEIGHT SENSING (for observations)
    measure_heights = True
    measured_points_x = [-0.8, -0.7, ..., 0.7, 0.8]  # 1m forward/back
    measured_points_y = [-0.5, -0.4, ..., 0.4, 0.5]  # 0.5m left/right

    # TERRAIN PHYSICS
    static_friction = 1.0
    dynamic_friction = 1.0
    restitution = 0.
    slope_treshold = 0.75  # Convert steep slopes to vertical walls
```

### Command Curriculum Configuration

**Location**: `legged_gym/envs/base/legged_robot_config.py:68-78`

```python
class commands:
    # CURRICULUM SETTINGS
    curriculum = False  # DISABLED by default
    max_curriculum = 1.  # [m/s] Maximum velocity range

    num_commands = 4  # [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]
    resampling_time = 10.  # [s] How often to change commands
    heading_command = True  # Use heading instead of yaw rate

    class ranges:
        lin_vel_x = [-1.0, 1.0]  # [m/s] Forward velocity range
        lin_vel_y = [-1.0, 1.0]  # [m/s] Lateral velocity range
        ang_vel_yaw = [-1, 1]    # [rad/s] Yaw rate range
        heading = [-3.14, 3.14]  # [rad] Target heading range
```

### Other Relevant Configuration

**Location**: `legged_gym/envs/base/legged_robot_config.py:34-41`

```python
class env:
    num_envs = 4096  # Number of parallel environments
    episode_length_s = 20  # [s] Episode duration
```

With 4096 environments and 20s episodes:
- Curriculum updates 4096 times per episode (when environments reset)
- Massive parallelism allows fast curriculum progression

---

## Code Reference

### Complete File Locations

#### 1. Core Curriculum Logic (`legged_robot.py`)

| Lines | Function/Section | Purpose |
|-------|------------------|---------|
| 145-189 | `reset_idx()` | Main reset function; triggers curriculum updates |
| 158-159 | Terrain curriculum check | Conditional call to terrain curriculum |
| 161-162 | Command curriculum check | Conditional call to command curriculum |
| 182-185 | Curriculum logging | Log terrain level and max velocity to TensorBoard |
| 421-441 | `_update_terrain_curriculum()` | **Main terrain curriculum algorithm** |
| 428-430 | Init check | Skip curriculum on first reset |
| 431 | Distance measurement | Calculate how far robot walked |
| 432-433 | Move up condition | Success criterion |
| 434-435 | Move down condition | Failure criterion |
| 436 | Level adjustment | Update difficulty |
| 438-440 | Max level handling | Randomize when reaching max difficulty |
| 441 | Spawn update | Set new environment origin |
| 443-452 | `update_command_curriculum()` | **Main command curriculum algorithm** |
| 450 | Performance check | Is tracking good enough? |
| 451-452 | Range expansion | Increase velocity limits |
| 709-715 | `_get_env_origins()` | Initialize terrain levels and types |
| 710 | Curriculum disable | Start at max level if curriculum off |
| 711 | Random initialization | Random starting difficulty per robot |
| 712 | Type distribution | Distribute robots across terrain types |
| 714-715 | Origin assignment | Set initial spawn locations |
| 734 | Auto-disable | Turn off curriculum for non-terrain meshes |

#### 2. Terrain Generation (`terrain.py`)

| Lines | Function/Section | Purpose |
|-------|------------------|---------|
| 38-73 | `Terrain.__init__()` | Initialize terrain system |
| 51 | `env_origins` | 2D array of spawn positions |
| 61-66 | Mode selection | Choose curriculum/random/selected |
| 85-92 | `curiculum()` | **Generate curriculum terrain grid** |
| 88 | Difficulty scaling | `i / num_rows` → 0.0 to 1.0 |
| 89 | Type selection | Column determines terrain type |
| 91 | Tile generation | Create individual terrain |
| 109-145 | `make_terrain()` | Generate single terrain tile |
| 115-121 | Difficulty parameters | Scale obstacle dimensions |
| 122-143 | Type generation | Create specific terrain types |
| 147-164 | `add_terrain_to_map()` | Place tile in grid and store origin |
| 157-164 | Origin calculation | Compute spawn position and height |

#### 3. Configuration (`legged_robot_config.py`)

| Lines | Section | Purpose |
|-------|---------|---------|
| 43-67 | `class terrain` | Terrain curriculum parameters |
| 48 | `curriculum` | Enable/disable flag |
| 58 | `max_init_terrain_level` | Starting difficulty |
| 59-62 | Grid size | Terrain dimensions and counts |
| 64 | `terrain_proportions` | Mix of terrain types |
| 68-78 | `class commands` | Command curriculum parameters |
| 69 | `curriculum` | Enable/disable flag (False) |
| 70 | `max_curriculum` | Maximum velocity range |
| 75-78 | `ranges` | Initial command ranges |

#### 4. Testing/Inference (`play.py`)

| Lines | Section | Purpose |
|-------|---------|---------|
| 42-52 | Test configuration | Override settings for evaluation |
| 48 | Disable curriculum | Turn off for fixed difficulty testing |

### Key Data Structures

```python
# Terrain state (per environment)
self.terrain_levels: torch.Tensor  # Shape: (num_envs,) - Current difficulty [0-9]
self.terrain_types: torch.Tensor   # Shape: (num_envs,) - Terrain type [0-19]
self.env_origins: torch.Tensor     # Shape: (num_envs, 3) - Current spawn position

# Terrain grid (global)
self.terrain_origins: torch.Tensor  # Shape: (num_rows, num_cols, 3) - All spawn positions
self.max_terrain_level: int         # Maximum difficulty level (10)

# Command ranges (global, shared by all envs)
self.command_ranges: Dict[str, List[float]]  # e.g., {"lin_vel_x": [-1.5, 1.5]}
```

---

## Usage Guide

### Enabling/Disabling Curriculum

#### Terrain Curriculum

**Enable** (default):
```python
# In your config file or override
env_cfg.terrain.curriculum = True
env_cfg.terrain.mesh_type = 'trimesh'  # or 'heightfield'
```

**Disable**:
```python
env_cfg.terrain.curriculum = False
# Robots start at max difficulty
```

#### Command Curriculum

**Enable**:
```python
env_cfg.commands.curriculum = True
env_cfg.commands.max_curriculum = 2.0  # Allow up to ±2 m/s
```

**Disable** (default):
```python
env_cfg.commands.curriculum = False
# Fixed velocity range throughout training
```

### Adjusting Difficulty Progression

#### Change Starting Difficulty

```python
# Start easier (level 0-2 instead of 0-5)
env_cfg.terrain.max_init_terrain_level = 2

# Start at maximum difficulty (no curriculum)
env_cfg.terrain.max_init_terrain_level = 9
env_cfg.terrain.curriculum = False
```

#### Add More Difficulty Levels

```python
# More granular difficulty progression
env_cfg.terrain.num_rows = 20  # 20 levels instead of 10
```

#### Modify Progression Speed

Edit `legged_robot.py:436`:
```python
# Slower progression (smaller steps)
self.terrain_levels[env_ids] += 0.5 * move_up - 0.5 * move_down

# Faster progression (only move up, never down)
self.terrain_levels[env_ids] += 1 * move_up
```

#### Change Success/Failure Criteria

Edit `legged_robot.py:432-435`:
```python
# Easier success criterion (walk 25% instead of 50%)
move_up = distance > self.terrain.env_length / 4

# Stricter failure criterion
move_down = (distance < commanded_distance * 0.75) * ~move_up
```

#### Modify Command Curriculum Progression

Edit `legged_robot.py:451-452`:
```python
# Slower velocity expansion
self.command_ranges["lin_vel_x"][0] -= 0.1  # Was 0.5
self.command_ranges["lin_vel_x"][1] += 0.1  # Was 0.5

# Change success threshold (currently 80%)
if tracking_reward > 0.9 * max_reward:  # Require 90% instead of 80%
```

### Monitoring Curriculum Progress

#### TensorBoard Metrics

During training, these metrics are logged:

```python
# Terrain curriculum
extras["episode"]["terrain_level"]  # Average difficulty level

# Command curriculum
extras["episode"]["max_command_x"]  # Current max forward velocity

# Performance (used for command curriculum)
extras["episode"]["rew_tracking_lin_vel"]  # Velocity tracking reward
```

**View in TensorBoard**:
```bash
tensorboard --logdir logs/<experiment_name>
```

Look for:
- `terrain_level` should gradually increase over training
- `max_command_x` should expand if command curriculum is enabled
- `rew_tracking_lin_vel` indicates how well robots track commands

#### Manual Inspection

Print curriculum state during training:
```python
# Add to your training script
if iteration % 100 == 0:
    print(f"Avg terrain level: {env.terrain_levels.float().mean():.2f}")
    print(f"Max terrain level: {env.terrain_levels.max()}")
    print(f"Velocity range: [{env.command_ranges['lin_vel_x'][0]:.2f}, "
          f"{env.command_ranges['lin_vel_x'][1]:.2f}]")
```

### Testing Without Curriculum

For evaluation, disable curriculum to test on fixed terrains:

```python
# In play.py or your test script
env_cfg.terrain.curriculum = False
env_cfg.terrain.num_rows = 5  # Smaller terrain grid
env_cfg.terrain.num_cols = 5
env_cfg.terrain.max_init_terrain_level = 9  # Test at max difficulty
```

This allows you to:
- Evaluate final policy on hardest terrains
- Test generalization to specific difficulty levels
- Compare performance across different terrain types

### Custom Curriculum Implementation

To implement your own curriculum logic:

1. **Override terrain curriculum**:
```python
def _update_terrain_curriculum(self, env_ids):
    # Your custom logic here
    # Example: Random curriculum
    self.terrain_levels[env_ids] = torch.randint(0, self.max_terrain_level,
                                                  (len(env_ids),), device=self.device)
    self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids],
                                                      self.terrain_types[env_ids]]
```

2. **Override command curriculum**:
```python
def update_command_curriculum(self, env_ids):
    # Your custom logic here
    # Example: Time-based curriculum
    progress = self.common_step_counter / self.max_iterations
    self.command_ranges["lin_vel_x"][1] = 1.0 + progress * 2.0  # 1 → 3 m/s
```

3. **Add new curriculum dimensions**:
- Modify reward scales over time
- Adjust action noise levels
- Change terrain type distributions
- Vary episode length

### Common Issues and Solutions

#### Issue: Robots don't progress past certain level

**Diagnosis**: Check if robots are walking far enough
```python
# Add to reset_idx()
print(f"Avg distance: {distance.mean():.2f}m, threshold: {self.terrain.env_length/2:.2f}m")
```

**Solutions**:
- Lower success threshold (edit line 432)
- Increase episode length in config
- Improve reward shaping for forward motion

#### Issue: Command curriculum doesn't activate

**Checks**:
1. Is it enabled? `env_cfg.commands.curriculum = True`
2. Is tracking reward good? Should be > 80% of scale
3. Is it updating? Only happens every `max_episode_length` steps

**Debug**:
```python
# Add to update_command_curriculum()
print(f"Tracking reward: {torch.mean(self.episode_sums['tracking_lin_vel'][env_ids]):.2f}")
print(f"Threshold: {0.8 * self.reward_scales['tracking_lin_vel'] * self.max_episode_length:.2f}")
```

#### Issue: Training unstable with curriculum

**Solutions**:
- Start easier: `max_init_terrain_level = 0`
- Slower progression: Reduce step size in line 436
- Disable command curriculum: Keep velocity ranges fixed
- Add more reward shaping for forward motion

---

## Appendix: Example Training Progression

### Typical Curriculum Timeline

| Iteration | Avg Terrain Level | Max Velocity (m/s) | Performance |
|-----------|-------------------|-------------------|-------------|
| 0 | 2.5 (random 0-5) | ±1.0 | Learning basics |
| 500 | 3.2 | ±1.0 | Progressing slowly |
| 1000 | 4.8 | ±1.0 | Near initial max |
| 1500 | 6.1 | ±1.5* | Passed init levels |
| 2000 | 7.3 | ±2.0* | Expert locomotion |
| 2500 | 8.2 | ±2.5* | Mastering hardest |
| 3000+ | 7.5-8.5 | ±3.0* | Converged |

*If command curriculum enabled

### Visual Representation

```
Training Progress:

Terrain Level
    9 │                                    ████████
    8 │                            ████████
    7 │                    ████████
    6 │            ████████
    5 │    ████████
    4 │████
    3 │
    2 │
    1 │
    0 └────────────────────────────────────────────
      0        500       1000      1500      2000
                    Training Iteration

Success Rate on Level 9:
  100%│                                        ████
   80%│                                   █████
   60%│                              █████
   40%│                         █████
   20%│                    █████
    0%│████████████████████
       └────────────────────────────────────────────
       0        500       1000      1500      2000
```

---

## References

### Related Papers

1. **Original Curriculum Learning**: Bengio et al., "Curriculum Learning" (ICML 2009)
2. **Game-Inspired Curriculum**: This implementation is inspired by video game difficulty progression
3. **Legged Locomotion**: Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep RL" (CoRL 2021)

### Implementation Credits

Based on the NVIDIA Isaac Gym legged robot environment:
- ETH Zurich, Nikita Rudin
- NVIDIA Corporation

---

## Conclusion

The curriculum learning system provides automatic, performance-based difficulty progression that:

1. **Accelerates training** by starting with easy tasks
2. **Improves final performance** through stable learning
3. **Requires no manual tuning** - adapts automatically
4. **Scales massively** - handles thousands of parallel environments

By understanding the code structure and configuration options documented here, you can customize the curriculum system for your specific robotic learning tasks.

---

**Document Version**: 1.0
**Last Updated**: 2025-10-29
**Codebase**: legged_gym (Isaac Gym)
