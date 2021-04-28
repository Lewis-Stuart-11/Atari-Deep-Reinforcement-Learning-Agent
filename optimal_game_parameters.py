# Author Lewis Stuart 201262348

from collections import namedtuple
import json

# A template of the game parameters stored in the respective JSON file. Each of the features is explained below
OptimalParameters = namedtuple(
    'OptimalParameters',
    (
        'learning_technique',  # Whether to use Policy gradient or Deep Q Learning
        'learning_rate',  # Learning rate of the policy network (how much each change effects the network)
        'epsilon_strategy',  # The strategy for returning the epsilon values
        'epsilon_values',  # Exploration vs Exploration values
        'discount',  # How impactful future rewards are
        'resize',  # The final size of the screen to enter into the neural network
        'resize_interpolation_mode', # The pixel conversion technique for resizing the image
        'crop_values',  # Crop values to shrink down the size of the screen
        'screen_process_type',  # How the environment processes the screen
        'colour_type',  # Depicts how the screen should be processed with colour
        'prev_states_queue_size',  # How many states to store in the queue for returning the analysed state
        'policy',  # Chooses which policy to use (eg. DQN)
        'policy_parameters',  # The specific parameters for the selected policy (eg. Number of neurons)
        'batch_size',  # The number of batches to analyse per step
        'memory_size',  # How many experiences to save to memory
        'memory_size_start',  # The minimum memory size to start learning
        'target_update', # How many episodes before the target neural network should be updated with the policy networks weights
        'update_factor',  # How many steps per episode before performing a batch weight update
        'reward_scheme'  # An dictionary of custom rewards for different situations
    )
)

# Holds all the optimal parameters for all the available Atari games
optimal_game_parameters = {}


# Validates parameters that configure the agent, environment and policies
def validate_game_parameters(game_parameters: OptimalParameters):
    # Valid values to set colour, policy and processing values
    valid_colour_types = ["rgb", "gray", "binary"]

    available_policies = {"DQN_CNN_Advanced": {"kernel_sizes": 3, "strides": 3, "neurons_per_layer": 4},
                          "DQN_CNN_Basic": {"kernel_sizes": 3, "strides": 3, "neurons_per_layer": 3},
                          "DQN": {"neurons_per_layer": 3}}

    available_screen_processing_types = ["append", "difference", "standard", "morph"]

    available_strategies = {"epsilon greedy linear": ["start", "end", "decay_linear"],
                            "epsilon greedy linear advanced": ["start", "middle", "end", "decay_linear",
                                                               "end_decay_linear"],
                            "epsilon greedy reward": ["start", "end", "decay_linear", "reward_incrementation",
                                                      "reward_target", "reward_decay"]}

    deep_q_learning_methods = ["DQL", "DDQL"]
    policy_gradient_methods = ["reinforce"]

    resize_interpolation_modes = ["bilinear", "nearest", "bicubic"]

    # Learning rate must be between 1-0
    if not (1 >= game_parameters.learning_rate > 0):
        raise ValueError(f"Learning rate must be between 1 and 0")

    # Epsilon strategy must be valid
    if game_parameters.epsilon_strategy.lower() not in available_strategies.keys():
        raise ValueError(f"Epsilon strategy must be from the available section: {available_strategies.keys()} ")

    for epsilon_property in available_strategies[game_parameters.epsilon_strategy.lower()]:
        if epsilon_property not in game_parameters.epsilon_values.keys():
            raise ValueError(f"Epsilon strategy '{game_parameters.epsilon_strategy}' "
                             f"must include property: {epsilon_property}")

        if epsilon_property not in ["reward_target", "reward_incrementation"] \
                and not (0 <= int(game_parameters.epsilon_values[epsilon_property]) <= 1):
            raise ValueError(f"Epsilon value {epsilon_property} must be between 1-0")

    # Discount must be between 1 and 0 to be valid
    if not (1 >= game_parameters.discount > 0):
        raise ValueError(f"Discount value must be between 1 and 0")

    # The resized image must have a width and height size
    if len(game_parameters.resize) != 2:
        raise ValueError(f"Resize image must be a two dimensional (only {len(game_parameters.resize)} values given")

    if game_parameters.resize[0] <= 0 or game_parameters.resize[1] <= 0:
        raise ValueError(f"Resize image dimensions cannot be less than 0")

    if game_parameters.resize_interpolation_mode.lower() not in resize_interpolation_modes:
        raise ValueError(f"Resize Interpolation mode must be in set: {resize_interpolation_modes}")

    # The crop values are a two dimensional array, each representing the percentage to crop the image, hence each must
    # dimension must have a size of two
    if len(game_parameters.crop_values) != 2:
        raise ValueError(f"Cropping values must include data for X and Y dimensions")

    if len(game_parameters.crop_values["percentage_crop_width"]) != 2 or \
            len(game_parameters.crop_values["percentage_crop_height"]) != 2:
        raise ValueError(f"Cropping values must include percentages for both sides of X and Y dimension")

    # As the cropping works from left-right and top-bottom, the left and top values must be larger than the right and
    # bottom values, otherwise this will result in indexing errors
    for dimension in game_parameters.crop_values:
        if game_parameters.crop_values[dimension][0] > game_parameters.crop_values[dimension][1]:
            raise ValueError(
                f"Left/Top crop value ({game_parameters.crop_values[dimension][0]}) \
                cannot be larger than Right/Bottom crop value {game_parameters.crop_values[dimension][1]}")
        if not (1 >= game_parameters.crop_values[dimension][0] >= 0) or not (1 >= game_parameters.crop_values[dimension][1] >= 0):
            raise ValueError(f"Crop values must be between 0-1")

    # Processing type must be supported
    if game_parameters.screen_process_type not in available_screen_processing_types:
        raise ValueError(
            f"Screen processing type {game_parameters.screen_process_type} must either: {available_screen_processing_types}")

    # The state queue must be between 1 and 10; 10 would be too large and not feasible to manage, while less than 1
    # would mean that no state is saved to the queue (as it would not exist)
    if not (10 >= game_parameters.prev_states_queue_size >= 1):
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

    if not(game_parameters.learning_technique in deep_q_learning_methods
           or game_parameters.learning_technique in policy_gradient_methods):
        raise ValueError(f"Learning techniques must be one of the following available methods \
                        {policy_gradient_methods + deep_q_learning_methods}")

    if game_parameters.learning_technique in deep_q_learning_methods:

        # The memory size must be larger than the batch size, so that the memory can return enough experiences to match the
        # batch size parameter
        if game_parameters.batch_size > game_parameters.memory_size:
            raise ValueError(f"Replay memory size must be larger than the batch size")

        # The batch size should be large enough that it can actually update correctly with the rewards, but small enough
        # that is doesn't take the neural network too long to converge
        if not (10000 >= game_parameters.batch_size > 10):
            raise ValueError(f"Batch size must be between 10 and 10000")

        # Memory must be the correct size to avoid memory errors or not enough experiences being stored
        if not (1000000 >= game_parameters.memory_size > 50):
            raise ValueError(f"Replay memory size must be between 50 and 1,000,000")

        # Memory learning start size must be in between the max replay size and 0, this is because the replay memory will
        # only return experiences once the amount memory is greater than this value
        if game_parameters.memory_size_start > game_parameters.memory_size or game_parameters.memory_size_start < 0:
            raise ValueError(
                f"Replay memory start size must be less than the maximum memory size and greater or equal to 0")

        # The target update should be small enough that the policy and target networks update enough
        if not (100 >= game_parameters.target_update > 1):
            raise ValueError(f"Target update factor must be between 1 and 100")

    else:
        if not(0 < game_parameters.improve_episode_factor < 100):
            raise ValueError(f"Improve episode factor must be between 0-100")

    # Iterates through the reward scheme settings and checks each one is of the correct type
    bool_reward_types = ["use_given_reward", "one_life_game", "normalise_rewards", "end_on_negative"]

    for reward_type in bool_reward_types:
        if reward_type in game_parameters.reward_scheme.keys():
            if type(game_parameters.reward_scheme[reward_type]) is not bool:
                raise ValueError(f"Reward scheme {reward_type} must be boolean")
        else:
            raise ValueError(f"Reward scheme: {reward_type} cannot be found")

    float_types = ["lives_change_reward"]

    for reward_type in float_types:
        if reward_type in game_parameters.reward_scheme.keys():
            if type(game_parameters.reward_scheme[reward_type]) is not float and \
                    type(game_parameters.reward_scheme[reward_type]) is not int:
                raise ValueError(f"Reward scheme {reward_type} must be int or float")
        else:
            raise ValueError(f"Reward scheme: {reward_type} cannot be found")


def retrieve_game_parameters(atari_game: str, index: int):
    with open("game_parameters.json", "r") as settings_json_file:
        parameter_handler = json.load(settings_json_file)

    if atari_game not in parameter_handler:
        raise ValueError(f"Game parameters for {atari_game} could not be found")

    parameter_list = parameter_handler[atari_game]

    if index >= len(parameter_list) or index < 0:
        raise ValueError(
            f"Parameter index out of range, only {len(parameter_list)} parameter entries exist for the game {atari_game}")

    dict_game_parameters = parameter_list[str(index)]

    parameterConstructor = namedtuple('Game_Parameter_Tuple', ' '.join(dict_game_parameters.keys()))
    optimal_game_parameters = parameterConstructor(**dict_game_parameters)

    # Checks to see if the parameters used are in a valid format
    validate_game_parameters(optimal_game_parameters)

    print("Successfully returned agent game parameters")

    return optimal_game_parameters
