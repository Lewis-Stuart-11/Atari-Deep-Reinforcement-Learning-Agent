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
    def __init__(self, game, crop_factors, resize, screen_process_type, prev_states_queue_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(game).unwrapped
        self.env.seed(0)
        self.env.reset()
        self.done = False # Episode has not finished

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

    # Resets the environment and state number
    def reset(self):
        self.env.reset()
        self.current_state_num = 0

    # Closes environment
    def close(self):
        self.env.close()

    # Renders environment
    def render(self, mode='human'):
        return self.env.render(mode)

    # Returns the number of actions available
    def num_actions_available(self):
        return self.env.action_space.n

    # Returns all the avaliables atari games
    def return_avaliable_atari_games(self):
        return sorted(atari_py.list_games())

    # Updates the state in queue and increases current step number
    def update_state_queue(self):
        self.state_queue[self.current_state_num % self.num_states] = self.get_processed_screen()
        self.current_state_num += 1

    # Resets the state queue to all black, or to the first state that is returned
    def reset_queue(self, reset_type="normal"):
        current_screen = self.get_processed_screen()

        # Iterates through the queue size and adds the first state to the queue
        for i in range(self.num_states):
            # Sets queue full of black screens (array of zeros)
            if reset_type == "black":
                self.state_queue.append(torch.zeros_like(current_screen))

            # Sets queue full of first screen
            elif reset_type == "normal":
                self.state_queue.append(current_screen)

            # Returns error if screen type is not valid
            else:
                raise ValueError("Reset type not valid")

    # Takes an action in the environment
    def take_action(self, action):
        # Action is a tensor and thus is an item
        state, reward, self.done, info = self.env.step(action.item())

        # Increases number of states that have been traversed in this episode
        self.update_state_queue()

        # Returns the reward as a tensor
        return torch.tensor([reward], device=self.device)

    # Returns the current state of the environment depending on the process type
    def get_state(self):
        if self.screen_process_type == "difference":
            return self.get_difference_state()
        elif self.screen_process_type == "append":
            return self.get_appended_state()
        else:
            return self.get_standard_state()

    # Returns standard screen, which is the RGB screen pixel data extracted straight from the gym environment
    def get_standard_state(self):
        return self.state_queue[self.current_state_num % self.num_states]

    # Returns the difference between the first screen in the queue, and the last screen in the queue
    # This highlights the changes that have been made between these two states. This is advantageous
    # for environments where movement is key and static
    def get_difference_state(self):
        # Check to see if the screen is just starting or just ended (cannot be the difference of two screens)
        if len(self.state_queue) == 0 or self.done:
            # Fills queue with black screens and returns the first state
            self.reset_queue("black")
            return self.state_queue[0]

        # Returns the difference of the current and previous screens
        else:
            # Chooses last state in the queue and subtracts from the latest state in the queue
            oldest_queue_state = self.state_queue[self.current_state_num % self.num_states - 1]
            current_state = self.state_queue[self.current_state_num % self.num_states]
            return current_state - oldest_queue_state

    # Returns the appended state, this involves appending each of the queues together to form a large
    # vertical image that is what is evaluated by the neural network
    def get_appended_state(self):
        # Check to see if the screen is just starting or just ended, if so the queue is reset to the first state
        if len(self.state_queue) == 0:
            self.reset_queue("normal")

        # Combine all the states into one large vertical image
        combined_states = self.state_queue[0]
        for i in range(1, self.num_states):
            combined_states = torch.cat((combined_states, self.state_queue[i]), 2)

        # If the episode has finished, set the queue to empty and return the latest queue
        if self.done and self.current_state_num != 0:
            self.state_queue = []

        return combined_states

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

        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()  # Firstly tensor is converted to a PIL image
            , T.Resize((self.resize[0], self.resize[1])) # Resized to the size specified by the resize property
            , T.ToTensor() # Transformed to a tensor
        ])

        # Returned and an extra dimension is added as these will represent a batch of states
        return resize(screen).unsqueeze(0).to(self.device)

