# Author Lewis Stuart 201262348

import torchvision.transforms as T
import gym
import torch
import numpy as np
import atari_py

# Ensures that the results will be the same (same starting random seed each time)
np.random.seed(0)

# Handles the gym environment and all properties regarding game states
class EnvironmentManager():
    def __init__(self, game, crop_factors, resize, screen_process_type,
                 prev_states_queue_size, colour_type, custom_rewards):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(game).unwrapped
        self.env.seed(0)
        self.env.reset()
        self.done = False # Episode has not finished
        self.state_info = None
        self.colour_type = colour_type
        self.current_screen = None

        # The state return format
        self.screen_process_type = screen_process_type.lower().strip()

        # Properties for cropping/outputting the screens
        self.heights = crop_factors[1]
        self.widths = crop_factors[0]
        self.resize = resize

        # Internal state queue, for storing previous states
        self.state_queue = []
        self.num_states = prev_states_queue_size
        self.current_state_num = 0

        self.use_additional_info = True
        self.life_change_reward = int(custom_rewards["lives_change_reward"])
        self.one_life_only = custom_rewards["one_life_game"]
        self.normalise_rewards = custom_rewards["normalise_rewards"]

    # Resets the environment and state number
    def reset(self):
        self.env.reset()
        self.current_state_num = 0
        self.state_queue = []
        self.current_screen = None

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
    def additional_rewards(self, new_state_info):
        if not self.state_info:
            return 0

        additional_reward = 0

        # Checks if the current game has a lives counter
        if 'ale.lives' in new_state_info.keys() and 'ale.lives' in self.state_info:

            prev_lives = self.state_info["ale.lives"]
            current_lives = new_state_info["ale.lives"]

            # If the previous lives is greater than the current lives, it means the agent has lost a life,
            # and thus the appropriate reward is given. If 'one life only' is set, then the environment terminates
            # early with a negative reward
            if prev_lives > current_lives:
                if self.one_life_only:
                    self.set_episode_end()
                additional_reward = self.life_change_reward

            # If the previous lives is less than the current lives, then the agent has gained a life, and thus the
            # negative of the 'life lost' reward is given
            elif prev_lives < current_lives:
                additional_reward = self.life_change_reward * -1

        return additional_reward

    # Takes an action in the environment
    def take_action(self, action):
        # Action is a tensor and thus is an item
        state, reward, self.done, state_info = self.env.step(action.item())

        # If additional rewards are set, then this is returned and the additional state information is updated
        extra_reward = 0
        if self.use_additional_info:
            extra_reward = self.additional_rewards(state_info)

        self.state_info = state_info

        final_reward = extra_reward + reward

        # Returns the reward as a tensor
        return torch.tensor([final_reward], device=self.device)

    # Returns the current state of the environment depending on the process type
    def get_state(self):
        # Queue is reset
        if self.just_starting():
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
        return self.get_processed_screen()

    # Returns the difference between the first screen in the queue, and the last screen in the queue
    # This highlights the changes that have been made between these two states. This is advantageous
    # for environments where movement is key and static
    def get_difference_state(self):

        self.current_screen = self.get_processed_screen()

        # If the episode has finisehd, then the final screen should be all black
        if self.done:
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            # Chooses last state in the queue and subtracts from the latest state in the queue
            oldest_queue_state = self.state_queue.pop(0)
            current_state = self.state_queue[self.num_states-1]
            self.state_queue.append(self.current_screen)
            return current_state - oldest_queue_state

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

        # Combines all states by adding all values together
        morphed_states =  self.state_queue.pop(0)
        for i in range(1, self.num_states):
            morphed_states = torch.add(morphed_states,  self.state_queue[i])

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
        # Converts screen to a tensor
        screen = torch.from_numpy(screen)

        if self.colour_type == "rgb":
            resize = T.Compose([
                T.ToPILImage()  # Firstly tensor is converted to a PIL image
                , T.Resize((self.resize[0], self.resize[1]))  # Resized to the size specified by the resize property
                , T.ToTensor()  # Transformed to a tensor
            ])

        else:
            # Use torchvision package to compose image transforms
            resize = T.Compose([
                T.ToPILImage()  # Firstly tensor is converted to a PIL image
                , T.Resize((self.resize[0], self.resize[1])) # Resized to the size specified by the resize property
                , T.Grayscale(num_output_channels=1) # Sets the image to grayscale
                , T.ToTensor() # Transformed to a tensor
            ])

        # Returns a tensor from the image composition
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

