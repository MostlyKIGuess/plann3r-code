"""Episode helpers for start selection, geodesic distance, metrics and logging.

Holds the Habitat geodesic distance, travelled distance, SPL and soft SPL, the
start state choice from the recorded trajectory (easy, hard, full or a distance
in meters, with the reverse offset), and the writers for metadata.txt and
results.csv. Used by libs/experiments/task_setup.py and run_nav.py.
"""

import numpy as np
import magnum as mn
from pathlib import Path
from typing import Optional, List
import quaternion as qt

try:
    import habitat_sim
    from habitat_sim.utils.common import quat_from_magnum 
except:
    print("habitat_sim not found")

import logging
logger = logging.getLogger("[Episode Utils]")


def calculate_path_distance(
    sim: habitat_sim.Simulator,
    start_pos: np.ndarray,
    goal_pos: np.ndarray
) -> float:
    """Calculate geodesic distance between two positions."""
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array(start_pos, dtype=np.float32)
    path.requested_end = np.array(goal_pos, dtype=np.float32)
    
    found = sim.pathfinder.find_path(path)
    return path.geodesic_distance if found else np.inf


def compute_travelled_distance(
    positions: List[np.ndarray],
    remaining_distance: float,
) -> float:
    """Length of the agent's ground-plane trajectory plus the unclosed gap.

    The remaining geodesic distance to the goal is added so that an agent that
    stops far from the goal is not rewarded for the short path it did travel.
    The y axis is height in Habitat, so only x and z enter the path length.
    """
    if len(positions) < 2:
        return float(remaining_distance)

    total = 0.0
    for previous, current in zip(positions[:-1], positions[1:]):
        total += float(
            np.hypot(
                float(current[0]) - float(previous[0]),
                float(current[2]) - float(previous[2]),
            )
        )
    return total + float(remaining_distance)


def compute_spl(
    success: bool,
    shortest_path_distance: float,
    travelled_distance: float,
) -> float:
    """Success weighted by inverse Path Length (Anderson et al., 2018)."""
    if not success:
        return 0.0
    if not (shortest_path_distance > 0 and travelled_distance > 0):
        return 0.0
    return shortest_path_distance / max(shortest_path_distance, travelled_distance)


def compute_soft_spl(
    success: bool,
    shortest_path_distance: float,
    remaining_distance: float,
    travelled_distance: float,
) -> float:
    """Soft-SPL (Datta et al., 2021), with full credit on success.

    Failed episodes earn partial credit for the fraction of the geodesic
    distance they closed, weighted by the same path efficiency term as SPL.
    """
    if success:
        return 1.0
    if not shortest_path_distance > 0:
        return 0.0
    progress = max(0.0, 1.0 - remaining_distance / shortest_path_distance)
    if travelled_distance > 0:
        progress *= shortest_path_distance / max(shortest_path_distance, travelled_distance)
    return progress


def pick_random_start_state(
    sim: habitat_sim.Simulator,
    cfg,  # Expects flat config with max_start_distance, goal_distance_threshold
    final_goal_position: np.ndarray,
    agent_positions_in_map: np.ndarray,
    max_tries: int = 100
):
    """Sample random navigable start state with distance constraints."""
    for _ in range(max_tries):
        # Sample random point
        p = sim.pathfinder.get_random_navigable_point()
        p = np.array(p, dtype=np.float32)
        
        # keep agent at roughly the same floor height as the recorded trajectory (more stable)
        p[1] = float(agent_positions_in_map[:, 1].mean())
        p = sim.pathfinder.snap_point(p)
        
        # reject if cannot snap / not navigable
        if p is None or not sim.pathfinder.is_navigable(p):
            continue
        
        # Check distance constraint
        dist = calculate_path_distance(sim, p, final_goal_position)
        if not np.isfinite(dist):
            continue
        
        if not _satisfies_distance_constraint(dist, cfg):
            continue
        
        # Create state facing goal
        start_state = habitat_sim.AgentState()
        start_state.position = p
        # choose yaw to face the goal
        start_state.rotation = get_agent_rotation_from_two_positions(
            start_state.position, final_goal_position)
        
        # Verify path ahead is clear
        if _check_forward_navigability(sim, start_state):
            return start_state
    
    return None

def _satisfies_distance_constraint(distance: float, cfg) -> bool:
    """Check if distance satisfies difficulty setting."""
    max_dist_str = str(cfg.max_start_distance).lower()
    
    if max_dist_str == "easy":
        return 2.0 <= distance <= 7.0
    elif max_dist_str == "hard":
        return distance >= 7.0
    elif max_dist_str == "full":
        return distance >= cfg.goal_distance_threshold + 2
    else:
        try:
            max_val = float(max_dist_str)
            return distance <= max_val
        except ValueError:
            return True

def get_agent_rotation_from_two_positions(position_src, position_dst):
    tangent = position_src - position_dst
    theta = np.arctan2(tangent[0], tangent[2])
    # need to negate angle for habitat's coordinate system
    # TODO : remove mn dependency here later and use something else
    rotation = quat_from_magnum(mn.Quaternion.rotation(mn.Rad(theta), mn.Vector3([0,1,0])))
    return rotation

def _check_forward_navigability(
    sim: habitat_sim.Simulator,
    start_state,
    check_distance: float = 2.0,
    num_checks: int = 5
) -> bool:
    """Verify forward path is clear."""
    q = start_state.rotation
    R = np.array(mn.Quaternion(q.imag, q.real).to_matrix())
    forward = -R[:, 2]
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    
    for i in range(1, num_checks + 1):
        dist = (i / num_checks) * check_distance
        check_point = start_state.position + forward * dist
        if not sim.pathfinder.is_navigable(check_point):
            return False
    
    return True

def find_shortest_path(sim, p1, p2):
    path = habitat_sim.ShortestPath()
    path.requested_start = p1
    path.requested_end = p2
    # A failed search leaves geodesic_distance at inf, so the distance alone
    # tells callers the point is unreachable.
    sim.pathfinder.find_path(path)
    geodesic_distance = path.geodesic_distance
    path_points = path.points
    return geodesic_distance, path_points


def closest_trajectory_state(
    sim: habitat_sim.Simulator,
    agent_states: list,
    distance_threshold: float,
    goal_position: np.ndarray = None
) -> int:
    """
    Find trajectory index closest to target distance from goal.
    
    Args:
        sim: Habitat simulator instance
        agent_states: List of agent states from recorded trajectory
        distance_threshold: Target geodesic distance from goal (meters)
        goal_position: Goal position (defaults to last trajectory position)
        
    Returns:
        Index of trajectory state closest to target distance
    """
    if goal_position is None:
        goal_position = agent_states[-1].position
    
    distances = np.zeros(len(agent_states))
    for i, state in enumerate(agent_states):
        distances[i] = find_shortest_path(sim, goal_position, state.position)[0]
    
    # Find index with distance closest to threshold
    start_index = ((distances - distance_threshold) ** 2).argmin()
    return start_index


def set_reverse_orientation(
    agent_states: list,
    start_index: int
) -> habitat_sim.AgentState:
    """
    Set agent orientation to face backward along trajectory (for reverse navigation).
    
    Args:
        agent_states: List of agent states from recorded trajectory
        start_index: Index of the start state in trajectory
        
    Returns:
        Agent state with adjusted orientation facing backward
    """
    start_state = agent_states[start_index]
    
    # Look at the previous step in trajectory (toward original goal)
    lookat_index = start_index - 1
    if lookat_index < 0:
        logger.warning('Cannot reverse orientation at the start of the episode.')
        return start_state
    
    # Search for a position that's actually different (agent may not have moved)
    for k in range(lookat_index, -1, -1):
        if np.linalg.norm(start_state.position - agent_states[k].position) > 0.1:
            lookat_index = k
            break
    
    # Set rotation to face the previous position (reverse direction)
    start_state.rotation = get_agent_rotation_from_two_positions(
        start_state.position, agent_states[lookat_index].position
    )
    return start_state


def select_trajectory_start_state(
    sim: habitat_sim.Simulator,
    cfg,
    agent_states: list,
    goal_position: np.ndarray = None
) -> habitat_sim.AgentState:
    """
    Select start state from trajectory based on distance threshold.
    
    Args:
        sim: Habitat simulator instance
        cfg: Config with max_start_distance and reverse settings
        agent_states: List of agent states from recorded trajectory
        goal_position: Goal position (optional, defaults to last state)
        
    Returns:
        Selected agent state at appropriate distance from goal
    """
    # Determine distance threshold from config
    max_dist_str = str(cfg.max_start_distance).lower()
    reverse = getattr(cfg, 'reverse', False)
    
    # Offset for reverse navigation (ends 1m before original start)
    distance_offset = 1.0 if reverse else 0.0
    
    if max_dist_str == "easy":
        distance_threshold = 3.0 + distance_offset
    elif max_dist_str == "hard":
        distance_threshold = 5.0 + distance_offset
    elif max_dist_str == "full":
        # Use first position (or last for reverse)
        start_index = 0 if not reverse else len(agent_states) - 1
        start_state = agent_states[start_index]
        if reverse:
            start_state = set_reverse_orientation(agent_states, start_index)
        return start_state
    else:
        # Try parsing as numeric value
        try:
            distance_threshold = float(max_dist_str) + distance_offset
        except ValueError:
            raise ValueError(f"Invalid max_start_distance: {cfg.max_start_distance}")
    
    # Find trajectory index at target distance
    start_index = closest_trajectory_state(sim, agent_states, distance_threshold, goal_position)
    start_state = agent_states[start_index]
    
    # Adjust orientation for reverse navigation
    if reverse:
        start_state = set_reverse_orientation(agent_states, start_index)
    
    return start_state


# ==============================================================================
# Results Logging Functions
# ==============================================================================

def initialize_results(
    metadata_file: Path,
    results_csv: Path,
    method: str,
    goal_source: str,
    max_steps: int,
    goal_distance_threshold: float,
    pid_steer_values: List,
    hfov_degrees: float,
    time_delta: float,
    velocity_control: float,
    goal_position: Optional[np.ndarray],
) -> None:
    """
    Initialize metadata.txt and results.csv for an episode.
    
    Args:
        metadata_file: Path to metadata.txt
        results_csv: Path to results.csv
        method: Controller method name
        goal_source: Goal source type
        max_steps: Maximum steps allowed
        goal_distance_threshold: Distance threshold for success
        pid_steer_values: PID controller values
        hfov_degrees: Camera horizontal field of view in degrees
        time_delta: Time step for controller
        velocity_control: Default velocity control value
        goal_position: Goal position in world coordinates
    """
    # Write metadata
    with open(str(metadata_file), 'w') as f:
        f.writelines(
            f'method={method}\n'
            f'goal_source={goal_source}\n'
            f'max_steps={max_steps}\n'
            f'goal_distance_threshold={goal_distance_threshold}\n'
            f'steer_pid_values={pid_steer_values}\n'
            f'camera_fov={hfov_degrees:.2f}\n'
            f'time_delta={time_delta}\n'
            f'velocity_control={velocity_control}\n'
            f'goal_position={list(goal_position) if goal_position is not None else ""}\n'
        )

    # Write CSV header
    with open(str(results_csv), 'w') as f:
        f.writelines('step,x,y,z,yaw,distance_to_goal,velocity_control,theta_control,discrete_action,collided\n')


def write_results(
    results_csv: Path,
    step: int,
    current_robot_state,
    distance_to_goal: float,
    velocity_control: float,
    theta_control: float,
    collided: Optional[bool],
    discrete_action: int
) -> None:
    """
    Append one step's data to results.csv.
    
    Args:
        results_csv: Path to results.csv
        step: Current step number
        current_robot_state: Agent state (position, rotation) or None
        distance_to_goal: Current distance to goal
        velocity_control: Velocity command
        theta_control: Angular velocity command (radians)
        collided: Whether collision occurred
        discrete_action: Discrete action taken (-1 if continuous)
    """
    with open(str(results_csv), 'a') as f:
        if current_robot_state is not None:
            x = current_robot_state.position[0]
            y = current_robot_state.position[1]
            z = current_robot_state.position[2]
            # Extract yaw from rotation quaternion
            try:
                yaw = np.arccos(qt.as_rotation_matrix(current_robot_state.rotation)[0, 0]) * 180 / np.pi
            except:
                yaw = 0.0
        else:
            x, y, z, yaw = "", "", "", ""
        
        f.writelines(
            f'{step},'
            f'{x},'
            f'{y},'
            f'{z},'
            f'{yaw},'
            f'{distance_to_goal},'
            f'{velocity_control},'
            f'{theta_control * 180 / np.pi if not np.isnan(theta_control) else ""},'
            f'{discrete_action},'
            f'{int(collided) if collided is not None else ""}\n'
        )


def write_final_meta_results(
    metadata_file: Path,
    success_status: str,
    final_distance: float,
    step: int,
    distance_to_final_goal: float
) -> None:
    """
    Append final results to metadata.txt.
    
    Args:
        metadata_file: Path to metadata.txt
        success_status: Final status (success, exceeded_steps, stuck, etc.)
        final_distance: Final distance to goal
        step: Total steps taken
        distance_to_final_goal: Initial geodesic distance from start to goal
    """
    with open(str(metadata_file), 'a') as f:
        f.writelines(
            f'success_status={success_status}\n'
            f'final_distance={final_distance}\n'
            f'step={step}\n'
            f'distance_to_final_goal_from_start={distance_to_final_goal}'
        )