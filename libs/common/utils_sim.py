"""Habitat helpers for camera intrinsics and agent motion.

task_setup.py uses build_intrinsics for the simulator camera and apply_velocity
to move the agent by a linear and angular velocity, snapped to the NavMesh. The
simulator itself is built by libs/simulation/habitat_utils.py.
"""

import numpy as np
import torch
from typing import Optional

import habitat_sim
from habitat_sim.utils import common as utils


def build_intrinsics(image_width: int,
                     image_height: int,
                     field_of_view_radians_u: float,
                     field_of_view_radians_v: Optional[float] = None,
                     device='cpu') -> torch.Tensor:
    if field_of_view_radians_v is None:
        field_of_view_radians_v = field_of_view_radians_u
    center_u = image_width / 2
    center_v = image_height / 2
    fov_u = (image_width / 2.) / np.tan(field_of_view_radians_u / 2.)
    fov_v = (image_height / 2.) / np.tan(field_of_view_radians_v / 2.)
    intrinsics = np.array([
        [fov_u, 0., center_u],
        [0., fov_v, center_v],
        [0., 0., 1]
    ])
    intrinsics = torch.from_numpy(intrinsics).to(device)
    return intrinsics

def apply_velocity(vel_control, agent, sim, velocity, steer, time_step):
    # Update position
    forward_vec = habitat_sim.utils.quat_rotate_vector(agent.state.rotation, np.array([0, 0, -1.0]))
    new_position = agent.state.position + forward_vec * velocity

    # Update rotation
    new_rotation = habitat_sim.utils.quat_from_angle_axis(steer, np.array([0, 1.0, 0]))
    new_rotation = new_rotation * agent.state.rotation

    # Step the physics simulation
    # Integrate the velocity and apply the transform.
    # Note: this can be done at a higher frequency for more accuracy
    agent_state = agent.state
    previous_rigid_state = habitat_sim.RigidState(
        utils.quat_to_magnum(agent_state.rotation), agent_state.position
    )

    target_rigid_state = habitat_sim.RigidState(
        utils.quat_to_magnum(new_rotation), new_position
    )

    # manually integrate the rigid state
    target_rigid_state = vel_control.integrate_transform(
        time_step, target_rigid_state
    )

    # snap rigid state to navmesh and set state to object/agent
    # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
    end_pos = sim.step_filter(
        previous_rigid_state.translation, target_rigid_state.translation
    )

    # set the computed state
    agent_state.position = end_pos
    agent_state.rotation = utils.quat_from_magnum(
        target_rigid_state.rotation
    )
    agent.set_state(agent_state)

    # Check if a collision occured
    dist_moved_before_filter = (
            target_rigid_state.translation - previous_rigid_state.translation
    ).dot()
    dist_moved_after_filter = (
            end_pos - previous_rigid_state.translation
    ).dot()

    # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
    # collision _didn't_ happen. One such case is going up stairs.  Instead,
    # we check to see if the the amount moved after the application of the filter
    # is _less_ than the amount moved before the application of the filter
    EPS = 1e-5
    collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
    # run any dynamics simulation
    sim.step_physics(dt=time_step)

    return agent, sim, collided
