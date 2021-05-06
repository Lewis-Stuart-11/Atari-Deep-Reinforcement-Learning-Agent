# Author Lewis Stuart 201262348

import sys
import time
import random
from collections import namedtuple
import matplotlib.pyplot as plt
from itertools import count
import torch.optim as optim
from torchviz import make_dot
import json
from torch.distributions import Bernoulli

# Imports all policies that the agent can follow to achieve optimal actions
from policies import *

# Imports strategies for either following the policy or exploring
from strategies import *

# Imports agents that perform the actions based on the policy
from agents import *

# Imports environment handlers that manage interactions with the environment
from environment_handler import *

# Imports game parameters which configure what properties to use
from optimal_game_parameters import *

# Imports functions for handling and visualising results
from result_handlers import *

# Imports the names of all current available policies for referencing
from avaliable_policy_methods import *

# Ensures that graph vision can work correctly
import os

os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"

# Ensures that the results will be the same (same starting random seed each time)
random.seed(0)
np.random.seed(0)

# Default game to play
running_atari_game = "BreakoutDeterministic-v4"

# Rendering information
game_FPS = 30
actions_per_second = 15

# Episodes to train
num_training_episodes = 10000

# Updates the plot after so many episodes
plot_update_episode_factor = 150

# How many times to save the current agent progress (saves neural network weights)
save_policy_network_factor = 200

# Render's the agent performing the eps after a certain number of episodes
render_agent_factor = 1

# Will set whether to use the user menu
use_menu = False

# Shows the processed images every 100 steps
show_processed_screens = False

# Creates a diagram of the current neural network
show_neural_net = True

# Pass a file name to load in weights and test agent
test_agent_file = None

# States whether to store results to excel file
save_results_to_excel = False

# The parameters to use for the current game
game_parameters = None

# The index in the list of set parameters for each game
parameter_index = 0

# Stores the current parameters for the agent
agent_parameters = None


def load_settings():
    global running_atari_game, num_training_episodes, use_menu, show_processed_screens, show_neural_net, \
        test_agent_file, plot_update_episode_factor, save_policy_network_factor, render_agent_factor, \
        parameter_index, save_results_to_excel

    with open("settings.json", "r") as settings_json_file:
        try:
            settings = json.load(settings_json_file)
            running_atari_game = settings["running_atari_game"]
            num_training_episodes = settings["num_training_episodes"]
            plot_update_episode_factor = settings["plot_update_episode_factor"]
            save_policy_network_factor = settings["save_policy_network_factor"]
            render_agent_factor = settings["render_agent_factor"]
            use_menu = settings["use_menu"]
            show_processed_screens = settings["show_processed_screens"]
            show_neural_net = settings["show_neural_net"]
            test_agent_file = settings["test_agent_file"]
            parameter_index = settings["parameter_index"]
            save_results_to_excel = settings["save_results_to_excel"]
        except BaseException as e:
            print(e)
            print("WARNING: Failed to load settings- using default settings")
        else:
            print("Successfully loaded in settings")


# Creates a graph of a state screen
def display_processed_screens(next_state_screen, state_screen_cmap, step):
    plt.figure(1)
    plt.imshow(next_state_screen, interpolation='none', cmap=state_screen_cmap)
    plt.title(f'Computer edited screen: {step}')
    plt.show()
    plt.close()


# Trains the agent using deep Q learning
def train_agent(em, agent):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not agent_parameters:
        raise ValueError("Agent parameters are not defined")

    # Screen width and heights are returned
    screen_width = em.get_screen_width()
    screen_height = em.get_screen_height()

    learning_technique = agent_parameters.learning_technique

    # Uses a deep neural network (without convolution layers)
    if agent_parameters.policy == "DNN_Basic":
        # Sets up input sizes for the networks
        policy_net = BasicDeepNN(screen_height, screen_width, em.num_actions_available(),
                                 learning_technique).to(device)

        # Sets default weights
        policy_net.apply(initialise_weights)

        if learning_technique in VALUE_BASED_METHODS:
            target_net = BasicDeepNN(screen_height, screen_width, em.num_actions_available(),
                                     learning_technique).to(device)

    # Uses a deep neural network (with convolution layers)
    elif agent_parameters.policy == "CNN_Basic":
        # Establishes Policy and Target networks
        policy_net = BasicCNN(screen_height, screen_width, em.num_actions_available(), em.num_tensor_outputs,
                              agent_parameters.policy_parameters, learning_technique).to(device)

        # Sets default weights
        policy_net.apply(initialise_weights)

        if learning_technique in VALUE_BASED_METHODS:
            target_net = BasicCNN(screen_height, screen_width, em.num_actions_available(), em.num_tensor_outputs,
                                  agent_parameters.policy_parameters, learning_technique).to(device)

    elif agent_parameters.policy == "CNN_Advanced":
        policy_net = AdvancedCNN(em.get_screen_height(), em.get_screen_width(),
                                 em.num_actions_available(), em.num_tensor_outputs,
                                 agent_parameters.policy_parameters, learning_technique).to(device)

        policy_net.apply(initialise_weights)

        if learning_technique in VALUE_BASED_METHODS:
            target_net = AdvancedCNN(em.get_screen_height(), em.get_screen_width(),
                                     em.num_actions_available(), em.num_tensor_outputs,
                                     agent_parameters.policy_parameters, learning_technique).to(device)

    else:
        raise Exception("Policy and target networks not established")

    # Sets parameters for Deep Q Learning methods
    if learning_technique in VALUE_BASED_METHODS:
        # Sets the weights and biases to be the same for both networks
        target_net.load_state_dict(policy_net.state_dict())

        # Sets the network to not be in training mode (only be used for inference)
        target_net.eval()

        # Sets and optimiser with the values to optimised as the parameters of the policy network, with the learning rate
        optimizer = optim.Adam(params=policy_net.parameters(), lr=agent_parameters.learning_rate)

        # Establishes the replay memory
        memory = ReplayMemory(agent_parameters.memory_size, agent_parameters.memory_size_start)

        # Number of states in a batch
        batch_size = agent_parameters.batch_size

        # How often to improve the neural network weights (how many steps per episode to update)
        improve_step_factor = agent_parameters.update_factor

    # Sets parameters for Policy Gradient methods
    elif learning_technique in POLICY_GRADIENT_METHODS:
        target_net = None

        optimizer = optim.Adam(params=policy_net.parameters(), lr=agent_parameters.learning_rate)

        memory = ReplayMemory(50000, None)

        improve_episode_factor = agent_parameters.episode_update_factor
    else:
        raise ValueError("Learning technique not defined")

    # Stores episode durations
    episode_durations = []

    # Stores the index and reward of the best episode that occurred
    best_episode_index = (0, 0)

    # Creates a visual representation of the neural network and saves it as 'Policy Network diagram'
    if show_neural_net:
        print("Creating neural network diagram")
        returned_values = policy_net(em.get_state())
        make_dot(returned_values,
                 params=dict(list(policy_net.named_parameters()))).render("results/Policy_Network_diagram",
                                                                          format="png")

        # Remove extra unnecessary created config file
        os.remove("results/Policy_Network_diagram")

    # Iterates over each episode
    for episode in range(num_training_episodes):

        em.reset()

        # Return initial state
        state = em.get_state()

        # Total episode Reward
        episode_reward = 0

        # Start time of the episode
        start = time.time()

        # Iterates through the number of steps in each episode
        for step in count():

            # Returns action
            action = agent.select_action(state, policy_net, episode, episode_reward)

            # Returns reward
            env_reward, custom_reward = em.take_action(action)

            # The total episode reward is set as the environment reward, as some scenarios will set a constant
            # custom reward, which won't fairly show the agent's progress
            episode_reward += env_reward.cpu().numpy()[0]

            # Returns next state
            next_state = em.get_state()

            # Adds experience to list of memory
            memory.push(Experience(state, action, next_state, custom_reward))

            # Updates new state
            state = next_state

            # If set, shows how the states are visualised (used for debugging)
            if show_processed_screens and episode == 1 and step > 200:

                next_state_screen = next_state.squeeze(0).permute(1, 2, 0).cpu()

                if agent_parameters.colour_type == "RGB":
                    state_screen_cmap = "hsv"
                else:
                    state_screen_cmap = "gray"

                display_processed_screens(next_state_screen, state_screen_cmap, step)
                display_processed_screens(em.render('rgb_array'), state_screen_cmap, step)

            # If set, renders the environment on the screen
            if render_agent_factor and episode+1 >= render_agent_factor:
                em.render()

            # Retrieves a sample if possible to learn from if deep q learning is used
            if learning_technique in VALUE_BASED_METHODS:

                # Optimisation occurs if a sample can be provided and the number of steps matches the update factor
                if memory.can_provide_sample(batch_size) and step % improve_step_factor == 0:
                    experiences = memory.sample(batch_size)

                    # Extracts all states, actions, reward and next states into their own tensors
                    states, actions, rewards, next_states = extract_tensors(experiences)

                    # Sets the all gradients of all the weights and biases in the policy network to 0
                    # As pytorch accumulates gradients every time it is used, it needs to be reset as to not
                    # Factor in old gradients and biases
                    optimizer.zero_grad()

                    # Extracts the predicted Q-Values for the states and actions pairs
                    # (as predicted by the policy network)
                    current_q_values = QValues.get_current(policy_net, states, actions)

                    # Extracts the target Q-Values
                    target_q_values = QValues.get_target_Q_Values(policy_net, target_net, next_states,
                                                                  agent_parameters.discount,
                                                                  rewards, learning_technique)

                    # Calculates loss between the current Q values and the target Q values by using the
                    # mean squared error as the loss function
                    loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))

                    # Computes the gradients of loss (error) of all the weights and biases in the policy network
                    loss.backward()

                    # Updates the weights and biases of the policy network with the gradient computed from the loss
                    optimizer.step()

            # Checks if the episode has finished
            if em.done:
                # Performs optimisation if the method is policy gradient
                if learning_technique in POLICY_GRADIENT_METHODS:

                    if episode % improve_episode_factor == 0:
                        # Returns all the experiences from memory
                        experiences = memory.return_all_and_clear_memory()

                        states, actions, rewards, next_states = extract_tensors(experiences)

                        optimizer.zero_grad()

                        loss = PolicyGradient.get_policy_gradient_loss(policy_net, states, actions, rewards,
                                                                       agent_parameters.discount, learning_technique)

                        loss.backward()

                        optimizer.step()

                # Time taken is recorded
                total_time = str(time.time() - start)

                # Finds the 50 episode moving average
                if episode < 50:
                    prev_rewards = 0
                else:
                    prev_fifty_episodes = episode_durations[-49:]
                    prev_rewards = 0
                    for prev_episode in prev_fifty_episodes:
                        prev_rewards += prev_episode["total_reward"]

                    prev_rewards = round(((prev_rewards + episode_reward) / (len(prev_fifty_episodes) + 1)), 2)

                # Includes all episode information (time, reward, steps)
                episode_info = {"num_steps": step, "total_reward": episode_reward,
                                "total_time": total_time[0:total_time.find('.') + 3],
                                "moving_average": prev_rewards}

                if learning_technique in VALUE_BASED_METHODS:
                    episode_info["epsilon"] =  round(agent.return_exploration_rate(episode), 4)

                # Appends the episode information
                episode_durations.append(episode_info)

                # Prints the current episode information if set
                if not use_menu:
                    print(f"Current episode: {episode + 1}")
                    print(f"Reward: {round(episode_reward, 2)}")
                    print(f"Steps: {step}")
                    print(f"Moving_average: {prev_rewards}")
                    print(f"Time: {total_time[0:total_time.find('.') + 3]}")
                    if learning_technique in VALUE_BASED_METHODS:
                        print(f"Training: {memory.can_provide_sample(batch_size)}")
                        print(f"Current epsilon: {round(agent.return_exploration_rate(episode), 3)}")
                    else:
                        print("Training: True")
                    print()

                    # Updates the best episode if the reward was greater than the current best episode
                    if best_episode_index[1] < episode_reward:
                        best_episode_index = (episode, episode_reward)

                    # Displays the estimated time remaining based on the previous execution time
                    if (episode + 1) % 10 == 0 and episode > 50:
                        prev_time = 0
                        prev_episodes = episode_durations[-10:]
                        for prev_episode in prev_episodes:
                            prev_time += float(prev_episode["total_time"])

                        average_time = prev_time / 10
                        episodes_left = num_training_episodes - episode

                        estimated_time_remaining = average_time * episodes_left

                        print(f"Current estimated time left: {round(estimated_time_remaining / 60) // 60} hrs "
                              f"{round(estimated_time_remaining / 60) % 60} mins")
                        if learning_technique in VALUE_BASED_METHODS:
                            print(f"Current replay memory size: {memory.current_memory_size()}")
                        print()
                        print(f"Best episode number: {best_episode_index[0] + 1}")
                        print(f"Best episode reward: {best_episode_index[1]}")
                        print()

                # Draws graph depending on the plot update factor
                if (episode + 1) % plot_update_episode_factor == 0:
                    # Appends the number of steps
                    print("Creating graph plots")
                    print()
                    plot(episode_durations, False)

                # Episode is finished and breaks
                break

        # Checks to see if the target network needs updating by checking if the episode count is
        # a multiple of the target_update value
        if agent_parameters.learning_technique in VALUE_BASED_METHODS:
            if episode % agent_parameters.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Saves the current neural network weights depending on the save factor
        if episode % save_policy_network_factor == 0:
            torch.save(policy_net.state_dict(), f"network_weights/{running_atari_game}_Policy_Network_{episode}")

    print("Creating and saving final graph plots")
    print()

    # Plots performance before closing
    plot(episode_durations, True)

    # Writes data to excel file
    if save_results_to_excel:
        print("Saving results to Excel file:")
        print()
        write_final_results(episode_durations, running_atari_game, num_training_episodes, agent_parameters)

    # Prints average for the last 1000 episodes
    last_1000_episodes = [float(episode["total_reward"]) for episode in episode_durations]

    last_1000_episodes = np.array(last_1000_episodes[-1000:])

    print(f"Last 1000 episode average: {round(last_1000_episodes.mean(),2)}")
    print(f"Standard Diviation: {round(last_1000_episodes.std(),2)}")
    print()

    # Closes environment
    em.close()

    # Policy Network is returned
    return policy_net


# Agent plays against itself
def self_play(policy_net, em, agent):
    # Iterates through a series of steps until the agent either wins or loses
    current_frame = 1

    em.reset()

    # Return initial state
    state = em.get_state()
    while True:
        step_start_time = time.time()

        action = agent.select_exploitative_action(state, policy_net)  # if current_frame % (game_FPS/actions_per_second) or (game_FPS/actions_per_second) < 2 else torch.tensor([0]).to(device)

        print(f"taking action: {action}")
        print()

        # Returns reward
        em.take_action(action)

        # Renders environment
        em.render()

        # Executes if the agent either wins or loses
        if em.done: break

        # Syncs environment FPS
        computation_time = time.time() - step_start_time
        break_time = 1 / game_FPS - computation_time

        if break_time > 0:
            time.sleep(1 / game_FPS)

        current_frame += 1

    em.close()


# Lets the agent play the game either solo, against the user or against another agent
def play_game(play_type, policy_net, em, agent):
    em.reset()

    # Single-player agent
    if play_type == 0:
        print("Single player selected")
        self_play(policy_net, em, agent)

    # Agent vs User
    elif play_type == 1:
        pass

    # Agent vs Agent
    elif play_type == 2:
        pass

    else:
        raise ValueError("Type of play must be an int between 0-2")

    # Lets the user restart the game
    while True:
        restart = str(input("Restart game? (Enter Y or N) \n>")).lower().strip()
        if restart == "y" or restart == "yes":
            play_game(play_type, policy_net, em, agent)
            return
        elif restart == "n" or restart == "no":
            return
        else:
            print("Invalid input")
            print()


# Prints all information about the agent
def print_agent_information(em):
    print()
    print(f"New game: {running_atari_game}")

    # Outputs action and state space
    print(f"Action Space {em.env.action_space}")
    print(f"State Space {em.env.observation_space}")
    print()

    print(f"Learning technique: {agent_parameters.learning_technique}")
    print()

    # Parameters for how the agent is trained
    print("Parameters:")
    print(f"Episodes: {num_training_episodes}")
    print(f"Discount factor: {agent_parameters.discount}")
    print(f"Learning rate: {agent_parameters.learning_rate}")
    print(f"Policy used: {agent_parameters.policy.replace('_', ' ')}")
    print()

    # Custom reward values
    print("Custom rewards:")
    for scheme, reward in agent_parameters.reward_scheme.items():
        print(f"\t-{scheme.capitalize().replace('_', ' ')}: {reward}")
    print()

    # Screen values for how the agent views the environment
    print("Screen values:")
    print(f"\t-Crop width percentage: {agent_parameters.crop_values['percentage_crop_height']}")
    print(f"\t-Crop height percentage: {agent_parameters.crop_values['percentage_crop_width']}")
    print(f"Screen resize: {agent_parameters.resize}")
    print(f"Screen colour type: {agent_parameters.colour_type}")
    print(f"Screen Interpolation mode: {agent_parameters.resize_interpolation_mode}")
    print()

    # State processing types are displayed
    print("State processing types:")
    print(f"\tNumber of state queue: {agent_parameters.prev_states_queue_size}")
    print(f"\tStates analysis type: '{agent_parameters.screen_process_type}'")
    print()

    # The properties of the policy
    print("Neural network parameters:")
    print(f"\t-Network type: {agent_parameters.policy.replace('_', ' ')}")
    for current_property, value in agent_parameters.policy_parameters.items():
        print(f"\t-{current_property.capitalize().replace('_', ' ')}: {value}")
    print()

    if agent_parameters.learning_technique in VALUE_BASED_METHODS:
        # Parameters for how choices are made
        print("Epsilon values: ")
        print(f"\t-Start: {agent_parameters.epsilon_values['start']}")
        print(f"\t-Decay: {agent_parameters.epsilon_values['decay_linear']}")
        print(f"\t-End: {agent_parameters.epsilon_values['end']}")
        print()

        print("Replay parameters:")
        print(f"\tNumber of experiences saved replay memory: {agent_parameters.memory_size}")
        print(f"\tMemory start experience size: {agent_parameters.memory_size_start}")
        print()
        print(f"\tEpisodes to update target network (with policy network): {agent_parameters.target_update}")
        print(f"\tSteps per neural network update: {agent_parameters.update_factor}")
        print(f"\tExperience batch size: {agent_parameters.batch_size}")
    else:
        print(f"Episode update factor: {agent_parameters.episode_update_factor}")
    print()

    # CUDA Information is displayed
    print(f"{'GPU' if torch.cuda.is_available() else 'CPU'} used as primary training device")
    print(f"Torch version: {torch.version}")
    if torch.cuda.is_available():
        print(f"Torch Cuda version: {torch.version.cuda}")
    print()

    print("Running Agent")


def test(policy_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    atari_game_index = policy_name.find("_")

    if atari_game_index == -1:
        return ValueError("Policy_name must be a valid file produced by the program")

    atari_game = policy_name[0:atari_game_index]

    test_em = return_env_with_atari_game(atari_game)

    colour_type = optimal_game_parameters[atari_game].colour_type

    num_returned_states = 1
    if agent_parameters.screen_process_type == "append":
        num_returned_states = agent_parameters.prev_states_queue_size

    if colour_type == "rgb":
        input_channels = 3 * num_returned_states
    else:
        input_channels = 1 * num_returned_states

    if agent_parameters.policy == "DQN":
        # Sets up input sizes for the networks
        test_net = BasicDeepNN(test_em.get_screen_height(), test_em.get_screen_width(), test_em.num_actions_available()).to(
            device)

        # Uses a deep neural network (with convolution layers)
    elif optimal_game_parameters[running_atari_game].policy == "DQN_CNN":
        # Establishes Policy and Target networks
        test_net = BasicCNN(test_em.get_screen_height(), test_em.get_screen_width(),
                            test_em.num_actions_available(), input_channels,
                            agent_parameters.policy_parameters).to(device)

    test_net = AdvancedCNN(test_em.get_screen_height(), test_em.get_screen_width(),
                           test_em.num_actions_available(), input_channels,
                           agent_parameters.policy_parameters).to(device)

    storage = torch.load(f"network_weights/{policy_name}")
    if not storage:
        raise FileNotFoundError(
            f"Could not load in neural network for policy {policy_name}")
    test_net.load_state_dict(storage)

    episilon_values = agent_parameters.epsilon_values
    test_strategy = EpsilonGreedyStrategy(episilon_values[0], episilon_values[1], episilon_values[2])

    # Agent is created
    test_agent = Agent(test_strategy, test_em.num_actions_available(), agent_parameters.learning_technique)

    self_play(test_net, test_em, test_agent)


def return_env_with_atari_game(atari_game):
    try:
        # The percentages to crop the screen for returning a state
        crop_width = agent_parameters.crop_values["percentage_crop_width"]
        crop_height = agent_parameters.crop_values["percentage_crop_height"]

        # Resizes the image for output
        resize = agent_parameters.resize
        screen_process_type = agent_parameters.screen_process_type
        prev_states_queue_size = agent_parameters.prev_states_queue_size
        colour_type = agent_parameters.colour_type

        reward_scheme = agent_parameters.reward_scheme

        resize_interpolation_mode = agent_parameters.resize_interpolation_mode

        if atari_game == "MsPacmanDeterministic-v4":
            em = EnvironmentManagerPacMan(atari_game, [crop_width, crop_height], resize, screen_process_type,
                                    prev_states_queue_size, colour_type, resize_interpolation_mode, reward_scheme)
        elif atari_game == "EnduroDeterministic-v0":
            em = EnvironmentManagerEnduro(atari_game, [crop_width, crop_height], resize, screen_process_type,
                                          prev_states_queue_size, colour_type, resize_interpolation_mode, reward_scheme)
        else:
            em = EnvironmentManagerGeneral(atari_game, [crop_width, crop_height], resize, screen_process_type,
                                    prev_states_queue_size, colour_type, resize_interpolation_mode, reward_scheme)

    except BaseException:
        raise Exception("Failed to load gym environment and agent")

    return em


def main(arguements):
    # Loads settings from JSON file
    load_settings()

    global running_atari_game, agent_parameters
    agent_parameters = retrieve_game_parameters(running_atari_game, parameter_index)

    # Tests agent if a file name is given
    if test_agent_file is not None:
        test(test_agent_file)
        return True

    # If the user menu is set, passed parameters to the program are evaluated
    if use_menu:
        # If 5 arguments are not passed, then the default arguments are passed
        if len(arguements) != 5:
            print("No arguments: default settings being applied\n")
            play_type = 0
            train = True
            while True:
                reset = str(input("Would you like to reset agent's learning?\n>")).lower().strip()
                if reset == "y" or reset == "yes":
                    reset_agent = True
                    break
                elif reset == "n" or reset == "no":
                    reset_agent = False
                    break
                else:
                    print("Invalid input")
                    print()

        # Otherwise the game settings are set to the passed arguments
        else:
            running_atari_game = sys.argv[1].lower().strip()
            play_type = sys.argv[2]
            try:
                train = bool(sys.argv[3])
                reset_agent = bool(sys.argv[4])
            except BaseException:
                raise TypeError("Train and reset value must be either True or False")

        if play_type not in [0, 1, 2] or type(play_type) != int:
            raise ValueError("Type of play must be an int between 0-2")
        elif running_atari_game not in optimal_game_parameters.keys():
            raise ValueError(f"Passed Atari game '{running_atari_game}' could not be found")

    # Used if debugging and just training
    else:
        # Sets default settings
        play_type = 0
        train = True

    try:
        em = return_env_with_atari_game(running_atari_game)

        if agent_parameters.learning_technique in VALUE_BASED_METHODS:
            # Action strategy is set
            episilon_values = agent_parameters.epsilon_values

            if agent_parameters.epsilon_strategy.lower() == "epsilon greedy linear":
                strategy = EpsilonGreedyStrategy(episilon_values["start"], episilon_values["end"],
                                                 episilon_values["linear_decay"])

            elif agent_parameters.epsilon_strategy.lower() == "epsilon greedy linear advanced":
                strategy = EpsilonGreedyStrategyAdvanced(episilon_values["start"], episilon_values["middle"],
                                                         episilon_values["end"], episilon_values["decay_linear"],
                                                         episilon_values["end_decay_linear"])

            elif agent_parameters.epsilon_strategy.lower() == "epsilon greedy reward":
                strategy = EpsilonGreedyRewardStrategy(episilon_values["start"], episilon_values["end"],
                                                       episilon_values["decay_linear"],
                                                       episilon_values["reward_incrementation"],
                                                       episilon_values["reward_target"],
                                                       episilon_values["reward_decay"])

            else:
                raise ValueError("Could not find appropriate epsilon strategy")
        else:
            strategy = None

        # Agent is created
        agent = Agent(strategy, em.num_actions_available(), agent_parameters.learning_technique)

    except BaseException:
        raise Exception("Failed to load gym environment and agent")

    print_agent_information(em)

    # Trains the agent if the user has selected to do so
    if train:
        # Total time that the agent has been running
        start_run_time = time.time()
        policy_net = train_agent(em, agent)
        final_run_time = str(round(time.time() - start_run_time, 0))
        torch.save(policy_net.state_dict(), f"network_weights/{running_atari_game}_Policy_Network_Final")

        print(f"Final run time: {final_run_time}")
        print()

        pause = input("Agent has finished, enter to continue to normal play \n>")
        print()

    # Attempts to load in a previous deep Q network
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_net = BasicDeepNN(em.get_screen_height(), em.get_screen_width(), em.num_actions_available()).to(device)

        # Attempts to load the specific game DQN
        storage = torch.load(f"network_weights/{running_atari_game}_Policy_Network_Final")
        if not storage:
            raise FileNotFoundError(
                f"Could not load in neural network for game {running_atari_game}, please restart and train a new one")
        policy_net.load_state_dict(storage)

    # Agent plays the game according to the user play type input
    play_game(play_type, policy_net, em, agent)

    em.close()

    return True


if __name__ == '__main__':
    arguements = sys.argv
    main(arguements)
    print("Thank you for using Ataria")
    exit(0)
