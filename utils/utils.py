import numpy as np
from numpy.typing import NDArray
from numpy import dtype
import torch
#
# RLlib from Ray
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from typing import Any, Dict, List, Type, Union


def wrap_to_pi(angles):
    """
    Wraps *angles* to **[-pi, pi]**
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi


def softmax(input, dim=None):
    """
    Applies a softmax function using NumPy.
    Args:
        input (numpy.ndarray): input array
        dim (int): A dimension along which softmax will be computed.
    Returns:
        numpy.ndarray: softmax applied array
    """
    # if dim is None, we assume the softmax should be applied to the last dimension
    if dim is None:
        dim = -1

    # Shift input for numerical stability
    input_shifted = input - np.max(input, axis=dim, keepdims=True)
    # Calculate the exponential of the input
    exp_input = np.exp(input_shifted)
    # Sum of exponentials along the specified dimension
    sum_exp_input = np.sum(exp_input, axis=dim, keepdims=True)
    # Compute the softmax
    softmax_output = exp_input / sum_exp_input

    return softmax_output


def compute_actions_and_probs(
        policy: Policy,
        obs: Dict[str, NDArray[dtype]],
        num_agents_: int,
        explore: bool = True,
        batch_mode: bool = False,
):
    """
    Compute actions using the given policy.
    Args:
        policy (Policy): The policy to use for computing the actions.
        obs: dict
        explore (bool): Whether to use exploration when computing the actions.
    Returns:
    """
    if batch_mode:
        assert isinstance(obs, list), "When batch_mode is True, obs must be a list of observations."
        batch_size = len(obs)
        input_dict = batch_observations(obs, get_input_dict=True)
        action_info = policy.compute_actions_from_input_dict(input_dict=input_dict, explore=explore)
        action_ = action_info[0]  # (batch_size, num_agents, num_agents)
        logits = action_info[2]['action_dist_inputs']  # flattened logits (batch_size, num_agents * num_agents * 2)
        logits = logits.reshape(batch_size, num_agents_, num_agents_, 2)  # (batch_size, num_agents, num_agents, 2)
        action_probs_ = softmax(logits, dim=-1)[:, :, :, 1]  # (batch_size, num_agents, num_agents)
    else:
        # Compute the action
        action_info = policy.compute_single_action(obs, explore=explore)
        action_ = action_info[0]
        logits = action_info[2]['action_dist_inputs']  # flattened logits (batch_size, num_agents * num_agents * 2)
        logits = logits.reshape(num_agents_, num_agents_, 2)  # (num_agents, num_agents, 2)
        action_probs_ = softmax(logits)[:, :, 1]  # (num_agents, num_agents)

    return action_, action_probs_


def batch_observations(
        obs_list: List[Dict[str, np.ndarray]],
        get_input_dict: bool = True,
        use_torch: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Batch a list of single observations into a format suitable for RLlib's compute_actions_from_input_dict method.
    Args:
        obs_list (List[Dict[str, np.ndarray]]): A list where each element is a single observation from the environment.
        get_input_dict (bool): A flag indicating whether to return the batched observations as a SampleBatch object.
        use_torch (bool): A flag indicating whether to use PyTorch tensors or numpy arrays for the batched observations.
    Returns:

    Use something like this:
    batch_size = 5

    obs_list = []
    env_list = []
    for seed in range(batch_size):
        env = LazyMsgListenersEnv(config)
        env.seed(seed)
        obs = env.reset()
        obs_list.append(obs)
        env_list.append(env)
    input_dict = batch_observations(obs_list, get_input_dict=True)
    actions = policy.compute_actions_from_input_dict(input_dict=input_dict, explore=False)[0]
    """
    batched_obs = {}

    # Iterate over each observation and batch them together
    for obs in obs_list:
        for key, value in obs.items():
            if key not in batched_obs:
                batched_obs[key] = []
            batched_obs[key].append(value)

    # Convert lists to tensors or appropriately shaped numpy arrays
    for key, value in batched_obs.items():
        if use_torch:  # Using PyTorch tensors
            batched_obs[key] = torch.tensor(value)
        else:  # Using numpy arrays
            batched_obs[key] = np.array(value)

    if get_input_dict:
        return {SampleBatch.OBS: SampleBatch(batched_obs)}
    else:
        return batched_obs


def wrap_to_rectangle(
        p: np.ndarray,
        width: float,
        height: float,
        center: np.ndarray = np.array([0.0, 0.0])
) -> np.ndarray:
    """
    Wraps the given coordinates to a rectangle with the given width and height.
    (x, y) goes within the rectangle area:
      x-axis range: [-width/2 + cen[0], width/2 + cen[0]],
      y-axis range: [-height/2 + cen[1], height/2 + cen[1]].
    :param p: (np.ndarray) The coordinates to wrap. 2D array of (num_agents, 2).
    :param width: (float) The width of the rectangle.
    :param height: (float) The height of the rectangle.
    :param center: (np.ndarray) The center of the rectangle.
    :return:
    """
    half_dims = np.array([width / 2, height / 2])

    # Translate points to the rectangle centered at the origin
    p_centered = p - center

    # Wrap coordinates within the rectangle dimensions
    p_centered = (p_centered + half_dims) % np.array([width, height]) - half_dims

    # Translate points back to the original center
    p_wrapped = p_centered + center

    return p_wrapped  # (num_agents, 2)


def get_rel_pos_dist_in_periodic_boundary(rel_pos_normal, width, height):
    """
    Calculate the relative positions and distances considering periodic boundary conditions.

    Parameters:
    rel_pos_normal (np.ndarray): Relative positions with shape (num_agents, num_agents, 2).
    width (float): The width of the boundary.
    height (float): The height of the boundary.

    Returns:
    rel_pos_periodic (np.ndarray): Relative positions considering periodic boundaries with shape (num_agents, num_agents, 2).
    rel_dist_periodic (np.ndarray): Relative distances considering periodic boundaries with shape (num_agents, num_agents).
    """

    # Apply periodic boundary conditions to the relative positions
    rel_positions_x = rel_pos_normal[:, :, 0]
    rel_positions_y = rel_pos_normal[:, :, 1]

    wrapped_diff_x = np.where(np.abs(rel_positions_x) <= width / 2, rel_positions_x, -np.sign(rel_positions_x) * (width - np.abs(rel_positions_x)))
    wrapped_diff_y = np.where(np.abs(rel_positions_y) <= height / 2, rel_positions_y, -np.sign(rel_positions_y) * (height - np.abs(rel_positions_y)))

    # Combine the wrapped differences into relative positions
    rel_pos_periodic = np.stack((wrapped_diff_x, wrapped_diff_y), axis=-1)  # (num_agents, num_agents, 2)

    # Compute distances
    rel_dist_periodic = np.linalg.norm(rel_pos_periodic, axis=2)  # (num_agents, num_agents)

    return rel_pos_periodic, rel_dist_periodic


def map_periodic_to_continuous_space(coordinates, width, height, center=np.array([0, 0])):
    """
    Maps periodic coordinates [x, y] to continuous inputs using cosine and sine transformations.

    Args:
    coordinates (np.ndarray): A numpy array of shape (num_agents, num_agents, 2) containing [x, y] coordinates.
    width (float): The width boundary for x coordinates.
    height (float): The height boundary for y coordinates.
    center (np.ndarray): A numpy array of shape (2,) representing the center [cx, cy] of the rectangular area.
                         Defaults to the origin [0, 0].

    Returns:
    np.ndarray: Transformed positions with shape (num_agents, num_agents, 4) containing [x_cos, x_sin, y_cos, y_sin].
    """
    # Normalize the coordinates to the range [-1, 1]
    normalized_positions = 2 * (coordinates - center) / np.array([width, height])

    # Apply cosine and sine transformations
    cos_positions = np.cos(np.pi * normalized_positions)
    sin_positions = np.sin(np.pi * normalized_positions)

    # Stack the transformed coordinates into a single array in the order [x_cos, x_sin, y_cos, y_sin]
    transformed_positions = np.concatenate(
        (cos_positions[..., 0:1], sin_positions[..., 0:1], cos_positions[..., 1:2], sin_positions[..., 1:2]),
        axis=-1
    )

    return transformed_positions  # (num_agents, num_agents, 4)


def compute_neighbors(wrapped_positions, width, height, radius, self_loops=True):
    """
    Compute the neighbors of each agent considering periodic boundary conditions.

    :param wrapped_positions: np.ndarray of shape (num_agents, 2) representing wrapped 2D positions of agents.
    :param width: float representing the width of the rectangular area.
    :param height: float representing the height of the rectangular area.
    :param radius: float representing the neighborhood radius.
    :param self_loops: bool indicating whether to consider self-loops.
    :return: np.ndarray of shape (num_agents_max, num_agents_max) representing neighbors within the radius.
    """

    # Calculate distance differences with periodic boundary conditions
    for dim in range(2):  # 0 for x, 1 for y
        diff = np.abs(wrapped_positions[:, dim][:, np.newaxis] - wrapped_positions[:, dim][np.newaxis, :])  # (num_agents, num_agents)
        diff = np.minimum(diff, width - diff) if dim == 0 else np.minimum(diff, height - diff)
        if dim == 0:
            dx = diff
        else:
            dy = diff

    distances = np.sqrt(dx ** 2 + dy ** 2)
    if not self_loops:
        np.fill_diagonal(distances, np.inf)  # Ignore self distances by setting them to infinity
    neighbors_array = distances <= radius

    return neighbors_array  # (num_agents, num_agents)
