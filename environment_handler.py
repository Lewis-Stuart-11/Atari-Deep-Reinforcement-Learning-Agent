# Author Lewis Stuart 201262348
import time

import torchvision.transforms as T
import gym
import torch
import numpy as np
import atari_py
import math

# Ensures that the results will be the same (same starting random seed each time)
np.random.seed(0)


# Handles the gym environment and all properties regarding game states
class EnvironmentManager():
    def __init__(self, game, crop_factors, resize, screen_process_type,
                 prev_states_queue_size, colour_type, resize_interpolation_mode, custom_rewards=None):

        if custom_rewards is None:
            custom_rewards = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(game).unwrapped
        self.env.seed(0)
        self.env.reset()
        self.done = False  # Episode has not finished
        self.state_info = None
        self.colour_type = colour_type
        self.current_screen = None

        if resize_interpolation_mode.lower() == "nearest":
            self.resize_interpolation_mode = T.InterpolationMode.NEAREST
        elif resize_interpolation_mode.lower() == "bicubic":
            self.resize_interpolation_mode = T.InterpolationMode.BICUBIC
        else:
            self.resize_interpolation_mode = T.InterpolationMode.BILINEAR

        self.current_game = game

        # The state return format
        self.screen_process_type = screen_process_type.lower().strip()

        # Tensor output
        # 3 RGB tensors are output for colour
        if self.colour_type == "rgb":
            self.num_tensor_outputs = 3
        else:
            self.num_tensor_outputs = 1

        # Append returns all prev states as tensors
        if self.screen_process_type == "append":
            self.num_tensor_outputs *= prev_states_queue_size

        # Properties for cropping/outputting the screens
        self.heights = crop_factors[1]
        self.widths = crop_factors[0]
        self.resize = resize

        # Internal state queue, for storing previous states
        self.state_queue = []
        self.num_states = prev_states_queue_size
        self.current_state_num = 0

        self.use_additional_info = True

        self.reward_scheme = {}
        self.establish_reward_scheme(custom_rewards)

        self.is_first_action = True

    # Sets a default scheme unless a given reward scheme is passed
    def establish_reward_scheme(self, custom_rewards):
        default_scheme = {"lives_change_reward": 0, "one_life_game": False,
                          "normalise_rewards": False, "use_given_reward": True,
                          "end_on_negative": False, "use_custom_env_rewards": False}

        for scheme, default in default_scheme.items():
            if scheme not in custom_rewards.keys():
                self.reward_scheme[scheme] = default
            else:
                self.reward_scheme[scheme] = custom_rewards[scheme]

    # Resets the environment and state number
    def reset(self):
        self.env.reset()
        self.current_state_num = 0
        self.state_queue = []
        self.current_screen = None
        self.is_first_action = True

    # Closes environment
    def close(self):
        self.env.close()

    # Renders environment
    def render(self, mode='human'):
        return self.env.render(mode)

    def just_starting(self):
        return self.current_screen is None

    # Finishes episode prematurely
    def set_episode_end(self):
        self.done = True

    # Returns the number of actions available
    def num_actions_available(self):
        return self.env.action_space.n

    # Returns all the avaliables atari games
    @staticmethod
    def return_avaliable_atari_games():
        return sorted(atari_py.list_games())

    # State queue is set to all black screens (as it is the start of a new episode)
    def reset_state_queue(self):
        self.current_screen = self.get_processed_screen()
        black_screen = torch.zeros_like(self.current_screen)
        for state in range(self.num_states):
            self.state_queue.append(black_screen)

    # Handles potential additional rewards (such as for losing a life)
    def additional_rewards(self, new_state_info, given_reward):
        if not self.state_info:
            return 0

        additional_reward = 0

        if self.reward_scheme["use_custom_env_rewards"]:
            additional_reward = self.return_custom_env_reward()

        if self.reward_scheme["end_on_negative"] and given_reward < 0:
            self.set_episode_end()
            additional_reward = int(self.reward_scheme["lives_change_reward"])

        # Checks if the current game has a lives counter
        elif 'ale.lives' in new_state_info.keys() and 'ale.lives' in self.state_info:

            prev_lives = self.state_info["ale.lives"]
            current_lives = new_state_info["ale.lives"]

            # If the previous lives is greater than the current lives, it means the agent has lost a life,
            # and thus the appropriate reward is given. If 'one life only' is set, then the environment terminates
            # early with a negative reward
            if prev_lives > current_lives and not self.is_first_action:
                if self.reward_scheme["one_life_game"]:
                    self.set_episode_end()
                additional_reward = int(self.reward_scheme["lives_change_reward"])

            # If the previous lives is less than the current lives, then the agent has gained a life, and thus the
            # negative of the 'life lost' reward is given
            elif prev_lives < current_lives and not self.is_first_action:
                additional_reward = int(self.reward_scheme["lives_change_reward"]) * -1

        return additional_reward

    # Takes an action in the environment
    def take_action(self, action):
        # Action is a tensor and thus is an item
        state, env_reward, self.done, state_info = self.env.step(action.item())

        # Copies environment reward to be returned unaltered
        actual_env_reward = env_reward

        if not self.reward_scheme["use_given_reward"]:
            env_reward = 0

        # If additional rewards are set, then this is returned and the additional state information is updated
        extra_reward = 0
        if self.use_additional_info:
            extra_reward = self.additional_rewards(state_info, env_reward)
        self.state_info = state_info

        final_reward = extra_reward + env_reward

        # If normalise rewards is set, then every reward will always be 1 or -1
        if self.reward_scheme["normalise_rewards"]:
            if final_reward != 0:
                final_reward = math.copysign(1, final_reward)  # Returns -1 or 1

        self.is_first_action = False

        # Returns the reward as a tensor
        return torch.tensor([actual_env_reward], device=self.device), torch.tensor([final_reward], device=self.device)

    # Returns the current state of the environment depending on the process type
    def get_state(self):
        # Queue is reset
        if self.just_starting():
            if self.current_game == "MsPacmanDeterministic-v0":
                self.last_ghost_positions = {"red_ghost": None, "blue_ghost": None,
                                             "pink_ghost": None, "yellow_ghost": None}
            self.reset_state_queue()

        # Each of the different state methods are activated
        if self.screen_process_type == "difference":
            return self.get_difference_state()
        elif self.screen_process_type == "append":
            return self.get_appended_state()
        elif self.screen_process_type == "morph":
            return self.get_morphed_state()
        elif self.screen_process_type == "standard":
            return self.get_standard_state()
        else:
            raise ValueError(f"Process type: '{self.screen_process_type}' not found")

    # Returns standard screen, which is the RGB screen pixel data extracted straight from the gym environment
    def get_standard_state(self):
        current_screen = self.get_processed_screen()
        self.state_queue.pop(0)
        self.state_queue.append(current_screen)
        return current_screen

    # Returns the difference between the first screen in the queue, and the last screen in the queue
    # This highlights the changes that have been made between these two states. This is advantageous
    # for environments where movement is key and static
    def get_difference_state(self):

        self.current_screen = self.get_processed_screen()

        # If the episode has finished, then the final screen should be all black
        if self.done:
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            # Chooses last state in the queue and subtracts from the latest state in the queue
            oldest_queue_state = self.state_queue.pop(0)
            self.state_queue.append(self.current_screen)
            return self.current_screen - oldest_queue_state

    # Returns the appended state, this involves appending each of the queues together to form a large
    # vertical image that is what is evaluated by the neural network
    def get_appended_state(self):

        self.current_screen = self.get_processed_screen()

        # If the episode has finished, add a black screen to the queue
        if self.done:
            black_screen = torch.zeros_like(self.current_screen)
            self.state_queue.pop(0)
            self.state_queue.append(black_screen)

        # Else append the current screen to the queue and remove oldest state
        else:
            self.state_queue.pop(0)
            self.state_queue.append(self.current_screen)

        return torch.stack(self.state_queue, dim=1).squeeze(2)

    def get_morphed_state(self):

        self.current_screen = self.get_processed_screen()

        #discount = 0.7
        discount = 1

        # Combines all states by adding all values together
        morphed_states = self.state_queue.pop(0)
        for i in range(0, self.num_states - 1):
            morphed_states = torch.add(morphed_states, self.state_queue[i]) * (discount ** (i + 1))

        self.state_queue.append(self.current_screen)

        # Takes the average of all taken states
        morphed_states /= self.num_states

        return morphed_states

    # Return the screen height of the processed state
    def get_screen_height(self):
        screen = self.get_state()
        return screen.shape[2]

    # Return the screen width of the processed state
    def get_screen_width(self):
        screen = self.get_state()
        return screen.shape[3]

    def return_custom_screen(self, screen):
        if self.current_game == "MsPacmanDeterministic-v0":
            return self.pac_man_state_converter(screen)
        else:
            return screen

    def return_custom_env_reward(self):
        if self.current_game == "MsPacmanDeterministic-v0":
            return self.pac_man_custom_reward()
        else:
            return 0

    # Returns the screen after it has been processed
    def get_processed_screen(self):
        # Renders the screen as an rgb_array and this is used as the current state
        # Is transposed into a format that can be used by the agent
        screen = self.render('rgb_array').transpose((2, 0, 1))

        # Screen is cropped
        screen = self.crop_screen(screen)

        # Screen is resized
        return self.transform_screen_data(screen)

    # Accepts a screen and returns a cropped version of it (makes it more efficient)
    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        screen_width = screen.shape[2]

        # Strip off top and bottom heights and widths
        height_top = int(screen_height * self.heights[0])
        height_bottom = int(screen_height * self.heights[1])

        width_top = int(screen_width * self.widths[0])
        width_bottom = int(screen_width * self.widths[1])

        # Returns cropped screen
        screen = screen[:, height_top:height_bottom, width_top:width_bottom]

        return screen

    # Converts image to a tensor form that can be used by the agent
    def transform_screen_data(self, screen):
        # Converts screen to a continuous array of float values and rescales by dividing by 255
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

        # Returns custom screen for each game
        screen = self.return_custom_screen(screen)

        # Converts screen to a tensor
        screen = torch.from_numpy(screen)

        if self.colour_type == "rgb":
            resize = T.Compose([
                T.ToPILImage()  # Firstly tensor is converted to a PIL image
                , T.Resize((self.resize[0], self.resize[1]), interpolation=self.resize_interpolation_mode)
                # Resized to the size specified by the resize property
                , T.ToTensor()  # Transformed to a tensor
            ])

        else:
            # Use torchvision package to compose image transforms
            resize = T.Compose([
                T.ToPILImage()  # Firstly tensor is converted to a PIL image
                , T.Resize((self.resize[0], self.resize[1]), interpolation=self.resize_interpolation_mode)
                # Resized to the size specified by the resize property
                , T.Grayscale(num_output_channels=1)  # Sets the image to grayscale
                , T.ToTensor()  # Transformed to a tensor
            ])

        # Returns a tensor from the image composition
        # resized_tensor = resize(screen).to(self.device)

        resized_tensor = resize(screen).to(self.device)

        # Converts all colour values to either 0 or 1 (very costly as had to make custom operation to perform this)
        cut_off = 0.1
        if self.colour_type == "binary":
            for width in range(self.resize[0]):
                for length in range(self.resize[1]):
                    resized_tensor[0, width, length] = 0 if resized_tensor[0, width, length] < cut_off else 1

        # An extra dimension is added as these will represent a batch of states
        batch_tensor = resized_tensor.unsqueeze(0).to(self.device)

        return batch_tensor

    # Accepts Numpy array and returns normalised pacman state
    def pac_man_state_converter(self, screen):

        screen_shape = screen.shape

        # All RGB values are summed together, giving a colour dimension of size 1
        summed_array = screen.sum(axis=0)

        # Array is converted into a single dimension
        resized_numpy = summed_array.reshape(-1)

        # As the colour scheme is relatively simple, each component of the pacman game has a specific RGB value
        # and when these values are summed together, each pixel has a unique float value between (0-3). Hence,
        # a series of conditions are employed which extract the specific components and store them in individual
        # arrays, which relate to the RGB arrays. The benefit of this is that all the ghosts can be expressed
        # through the red dimension by setting that all pixels between certain values (the summed RGB values
        # of the ghosts) are stored in the red array. This is also done for the pacman sprite as well as all
        # the points and walls in the game

        # Blue ghost: 1.65
        # Red ghost: 1.34
        # Pink ghost: 1.82
        # Yellow ghost: 1.37
        # Pacman ghost: 1.757

        iteration = 2

        if iteration == 1:
            # Ghosts array
            red_array = np.where((((resized_numpy > 1.2) & (resized_numpy <= 1.74)) |
                                  ((resized_numpy > 1.8) & (resized_numpy <= 3))), 1.0, 0.0)

            # Pacman array
            green_array = np.where((resized_numpy > 1.74) & (resized_numpy <= 1.76), 1.0, 0.0)

            #Points and walls array
            blue_array = np.where((resized_numpy > 1.76) & (resized_numpy <= 1.8), 1.0, 0.0)

        else:
            # Ghosts array
            red_array = np.where((((resized_numpy > 1.2) & (resized_numpy <= 1.4)) |
                                  ((resized_numpy > 1.6) & (resized_numpy <= 1.7)) |
                                  ((resized_numpy > 1.8) & (resized_numpy <= 3))), 1.0, 0.01)

            # Pacman array
            green_array = np.where((resized_numpy > 1.74) & (resized_numpy <= 1.76), 1.0, 0.01)

            # Vulnerable ghost array
            blue_array = np.where((resized_numpy > 1.4) & (resized_numpy <= 1.5), 1.0, 0.01)

        red_ghost_array = np.where((resized_numpy >= 1.33) & (resized_numpy <= 1.35), 1.0, 0.0)
        blue_ghost_array = np.where((resized_numpy >= 1.64) & (resized_numpy <= 1.66), 1.0, 0.0)
        pink_ghost_array = np.where((resized_numpy >= 1.81) & (resized_numpy <= 1.83), 1.0, 0.0)
        yellow_ghost_array = np.where((resized_numpy >= 1.36) & (resized_numpy <= 1.38), 1.0, 0.0)

        is_vulnerable_ghosts = np.where(blue_array == 1.0)[0].size != 0

        if not is_vulnerable_ghosts:
            if np.where(red_ghost_array == 1.0)[0].size != 0:
                self.last_ghost_positions["red_ghost"] = red_ghost_array
            elif self.last_ghost_positions["red_ghost"] is not None:
                red_array = np.add(red_array, self.last_ghost_positions["red_ghost"])
                red_array[red_array > 1.0] = 1.0

            if np.where(blue_ghost_array == 1.0)[0].size != 0:
                self.last_ghost_positions["blue_ghost"] = blue_ghost_array
            elif self.last_ghost_positions["blue_ghost"] is not None:
                red_array = np.add(red_array, self.last_ghost_positions["blue_ghost"])
                red_array[red_array > 1.0] = 1.0

            if np.where(pink_ghost_array == 1.0)[0].size != 0:
                self.last_ghost_positions["pink_ghost"] = pink_ghost_array
            elif self.last_ghost_positions["pink_ghost"] is not None:
                red_array = np.add(red_array, self.last_ghost_positions["pink_ghost"])
                red_array[red_array > 1.0] = 1.0

            if np.where(yellow_ghost_array == 1.0)[0].size != 0:
                self.last_ghost_positions["yellow_ghost"] = yellow_ghost_array
            elif self.last_ghost_positions["yellow_ghost"] is not None:
                red_array = np.add(red_array, self.last_ghost_positions["yellow_ghost"])
                red_array[red_array > 1.0] = 1.0

        # Arrays are reshaped back to the original dimensions
        red_array = red_array.reshape(1, screen_shape[1], screen_shape[2])
        green_array = green_array.reshape(1, screen_shape[1], screen_shape[2])
        blue_array = blue_array.reshape(1, screen_shape[1], screen_shape[2])

        # Arrays are combined together to form the final screen image
        final_array = np.concatenate((red_array, green_array, blue_array), axis=0)

        # Image is reshaped to form the same dimensions as the normal screen
        final_array = final_array.reshape(screen_shape[0], screen_shape[1], screen_shape[2])

        # Removes all elements inside of the inner square on the centre of the board, as ghosts inside
        # are not a threat
        final_array[:, 60:96, 64:96] = 0

        return final_array

    def pac_man_custom_reward(self):

        def return_pac_distances(ghost_pixels):

            if pac_man_pixels[0].size == 0 or pac_man_pixels[1].size == 0:
                return 0

            average_horizontal_pos = np.mean(pac_man_pixels[0])
            average_vertical_pos = np.mean(pac_man_pixels[1])

            horizontal_distances = np.absolute(np.subtract(ghost_pixels[0], average_horizontal_pos))
            vertical_distances = np.absolute(np.subtract(ghost_pixels[1], average_vertical_pos))

            closest_horizontal_distance = np.amin(horizontal_distances)
            closest_vertical_distance = np.amin(vertical_distances)

            relative_distance = math.sqrt(closest_horizontal_distance ** 2 + closest_vertical_distance ** 2)

            return round(relative_distance, 1)

        current_state = self.state_queue[-1].squeeze(0)

        current_state = current_state.cpu().numpy()

        pac_man_pixels = np.where(current_state[1] == 1.0)

        hostile_ghost_pixels = np.where((current_state[0] == 1.0))

        vulnerable_ghost_pixels = np.where((current_state[2] == 1.0))

        max_distance = round(math.sqrt((current_state[0].shape[0] ** 2) + (current_state[0].shape[1] ** 2)), 1)

        if hostile_ghost_pixels[0].size == 0 or hostile_ghost_pixels[1].size == 0:
            if vulnerable_ghost_pixels[0].size == 0 or vulnerable_ghost_pixels[1].size == 0:
                return 0
            pac_man_distance_reward = (max_distance - return_pac_distances(vulnerable_ghost_pixels))/4
        else:
            pac_man_distance_reward = return_pac_distances(hostile_ghost_pixels)

        return pac_man_distance_reward

