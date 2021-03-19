# Author Lewis Stuart 201262348

from collections import namedtuple

# Stores the optimal parameters for each available atari game
OptimalParameters = namedtuple(
    'OptimalParameters',
    (
     'learning_rate',  # Learning rate of the policy network (how much each change effects the network)
     'epsilon_values',  # Exploration vs Exploration values
     'discount',  # How impactful future rewards are
     'resize',  # The final size of the screen to enter into the neural network
     'crop_values',  # Crop values to shrink down the size of the screen
     'screen_process_type',  # How the environment processes the screen
     'colour_type',  # Depicts how the screen should be processed with colour
     'prev_states_queue_size',  # How many states to store in the queue for returning the analysed state
     'policy',  # Chooses which policy to use (eg. DQN)
     'policy_parameters',  # The specific parameters for the selected policy (eg. Number of neurons)
     'batch_size',  # The number of batches to analyse per step
     'memory_size',  # How many experiences to save to memory
     'target_update',  # How many episodes before the target neural network should be updated with the policy networks weights
     'update_factor',  # How many steps per episode before performing a batch weight update
     'step_reward',  # The reward for the agent still being in play
     'ending_reward'  # The reward for if the episode finishes
     )
)

# Holds all the optimal parameters for all the available Atari games
optimal_game_parameters = {}


# Validates parameters that configure the agent, environment and policies
def validate_game_parameters(game_parameters: OptimalParameters):
    # Valid values to set colour, policy and processing values
    valid_colour_types = ["rgb", "gray", "binary"]
    available_policies = {"DQN_CNN": {"kernel_sizes": 3, "strides": 3, "neurons_per_layer": 3}, "DQN": {"neurons_per_layer": 3}}
    available_screen_processing_types = ["append", "difference", "standard", "morph"]

    # Learning rate must be between 1-0
    if not(1 >= game_parameters.learning_rate > 0):
        raise ValueError(f"Learning rate must be between 1 and 0")

    # At least 3 episilon values need to be included
    if len(game_parameters.epsilon_values) < 3:
        raise ValueError(f"Epsilon values must include start, finish and decay factors")

    # Each epsilon value must be between 1-0 to be valid
    for value in game_parameters.epsilon_values:
        if not(1 >= value > 0):
            raise ValueError(f"Epsilon values must be between 1 and 0")

    # Discount must be between 1 and 0 to be valid
    if not(1 >= game_parameters.discount > 0):
        raise ValueError(f"Discount value must be between 1 and 0")

    # The resized image must have a width and height size
    if len(game_parameters.resize) != 2:
        raise ValueError(f"Resize image must be a two dimensional (only {len(game_parameters.resize)} values given")

    if game_parameters.resize[0] <= 0 or game_parameters.resize[1] <= 0:
        raise ValueError(f"Resize image dimensions cannot be less than 0")

    # The crop values are a two dimensional array, each representing the percentage to crop the image, hence each must
    # dimension must have a size of two
    if len(game_parameters.crop_values) != 2:
        raise ValueError(f"Cropping values must include data for X and Y dimensions")

    if len(game_parameters.crop_values[0]) != 2 or len(game_parameters.crop_values[1]) != 2:
        raise ValueError(f"Cropping values must include percentages for both sides of X and Y dimension")

    # As the cropping works from left-right and top-bottom, the left and top values must be larger than the right and
    # bottom values, otherwise this will result in indexing errors
    for dimension in game_parameters.crop_values:
        if dimension[0] > dimension[1]:
            raise ValueError(f"Left/Top crop value ({dimension[0]}) cannot be larger than Right/Bottom crop value {dimension[1]}")
        if not(1 >= dimension[0] >= 0) or not(1 >= dimension[1] >= 0):
            raise ValueError(f"Crop values must be between 0-1")

    # Processing type must be supported
    if game_parameters.screen_process_type not in available_screen_processing_types:
        raise ValueError(f"Screen processing type {game_parameters.screen_process_type} must either: {available_screen_processing_types}")

    # The state queue must be between 1 and 10; 10 would be too large and not feasible to manage, while less than 1
    # would mean that no state is saved to the queue (as it would not exist)
    if not(10 >= game_parameters.prev_states_queue_size >= 1):
        raise ValueError(f"State queue size must be between 10-1")

    # Colour type must be supported
    if game_parameters.colour_type not in valid_colour_types:
        raise ValueError(f"Colour type {game_parameters.colour_type} must either: {valid_colour_types}")

    # Policy must be supported
    if game_parameters.policy not in available_policies.keys():
        raise ValueError(f"Policy {game_parameters.policy} must be one of the available policies: {available_policies}")

    required_policy_parameters = available_policies[game_parameters.policy]

    # Each policy must have the correct parameters, of both name and number of values
    for policy_parameters in game_parameters.policy_parameters.keys():
        if policy_parameters not in required_policy_parameters.keys():
            raise ValueError(f"Game policy parameters must include: {policy_parameters}")
        if len(game_parameters.policy_parameters[policy_parameters]) != required_policy_parameters[policy_parameters]:
            raise ValueError(f"Game policy parameters {policy_parameters}\
             of insufficient size: {len(policy_parameters)} != {required_policy_parameters[policy_parameters]}")

    # The memory size must be larger than the batch size, so that the memory can return enough experiences to match the
    # batch size parameter
    if game_parameters.batch_size > game_parameters.memory_size:
        raise ValueError(f"Replay memory size must be larger than the batch size")

    # The batch size should be large enough that it can actually update correctly with the rewards, but small enough
    # that is doesn't take the neural network too long to converge
    if not (10000 >= game_parameters.batch_size > 10):
        raise ValueError(f"Batch size must be between 10 and 10000")

    # Memory must be the correct size to avoid memory errors or not enough experiences being stored
    if not (10000000 >= game_parameters.memory_size > 50):
        raise ValueError(f"Replay memory size must be between 50 and 10,000,000")

    # The target update should be small enough that the policy and target networks update enough
    if not (100 >= game_parameters.target_update > 1):
        raise ValueError(f"Target update factor must be between 1 and 100")

    # The step reward cannot be too large otherwise this would produce poor results
    if not (1 >= game_parameters.step_reward >= -1):
        raise ValueError(f"Step reward must be between 0-1")

    # The ending reward refers to the reward the agent will receive for finishing an episode
    # This can be positive or negative
    if not (10 >= game_parameters.ending_reward >= -10):
        raise ValueError(f"Step reward must be between 10 and -10")


# Optimal Pong parameters
optimal_game_parameters["Pong-v0"] = OptimalParameters(
    0.01,
    [1, 0.1, 0.0005],
    0.999,
    [68, 40],
    [[0.06, 0.94], [0.17, 0.92]],
    "append",
    "gray",
    4,
    'DQN_CNN',
    {"kernel_sizes": [6, 3, 3], 'strides': [3, 2, 1], 'neurons_per_layer': [32, 64, 32]},
    100,
    10000,
    2,
    50,
    0,
    -0.01
)

# Optimal Breakout parameters
optimal_game_parameters["BreakoutDeterministic-v4"] = OptimalParameters(
    0.01,
    [1, 0.1, 0.0005],
    0.999,
    [68, 40],
    [[0.05, 0.95], [0.25, 0.95]],
    "append",
    "gray",
    4,
    'DQN_CNN',
    {"kernel_sizes": [6, 3, 2], 'strides': [3, 2, 1], 'neurons_per_layer': [32, 64, 32]},
    100,
    10000,
    2,
    50,
    0,
    -1
)

# Optimal Pacman parameters
optimal_game_parameters["MsPacman-v0"] = OptimalParameters(
    0.001,
    [1, 0.1, 0.0002],
    0.999,
    [100, 70],
    [[0, 1], [0, 0.83]],
    "standard",
    "rgb",
    4,
    'DQN_CNN',
    {"kernel_sizes": [6, 3, 2], 'strides': [3, 2, 1], 'neurons_per_layer': [32, 64, 32]},
    200,
    10000,
    2,
    25,
    0,
    -1
)

"""
# Optimal Breakout parameters
optimal_game_parameters["BreakoutDeterministic-v4"] = OptimalParameters(
    0.001,
    [1, 0.1, 0.0025],
    0.999,
    [80, 50],
    [[0.05, 0.95], [0.15, 0.95]],
    "append",
    4,
    'DQN_CNN'
)

"""


# Optimal Breakout parameters
optimal_game_parameters["CartPole-v0"] = OptimalParameters(
    0.001,
    [1, 0.01, 0.01],
    0.999,
    [40, 90],
    [[0, 1], [0.4, 0.8]],
    "difference",
    "rgb",
    2,
    'DQN_CNN',
    {"kernel_sizes": [8, 4, 3], 'strides': [4, 2, 1], 'neurons_per_layer': [24, 32, 48]},
    250,
    100000,
    10,
    10,
    0,
    0
)