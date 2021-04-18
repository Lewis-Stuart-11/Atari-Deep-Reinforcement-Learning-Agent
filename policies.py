# Author Lewis Stuart 201262348

import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple
import torch
import numpy as np

# Ensures that the results will be the same (same starting random seed each time)
random.seed(0)

# Deep Q network class
# nn.Module- the base class for all neural networks. Hence all aspects of the neural network will
# extend from the module class
# Image height and width are needed as these will be the number of pixels passed in as a state


# A normal deep QNetwork, takes in the image sizes and uses these as the inputs to the neural network
class DQN(nn.Module):
    def __init__(self, img_height, img_width, num_actions):
        super().__init__()

        # Three fully connected hidden layers- fully connected layers are known as 'linear' layers
        self.fc1 = nn.Linear(in_features=img_height*img_width*3, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=68)
        self.fc3 = nn.Linear(in_features=68, out_features=32)

        # One output layer (avaliable actions)
        self.out = nn.Linear(in_features=32, out_features=num_actions)

    # A forward pass through the network with an image sensor T
    def forward(self, t):
        # Tensor will need to be flattened before being passed to the first layer
        t = t.flatten(start_dim=1)

        # Tensor is passed through each layer
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = self.out(t)
        return t


# A deep neural network with convoluted layers to process the image
class DQN_CNN_Basic(nn.Module):
    def __init__(self, h, w, outputs, input_channels,  nn_structure: dict):
        super(DQN_CNN_Basic, self).__init__()

        # Properties for how the convoluted neural network
        kernel_sizes = nn_structure["kernel_sizes"]
        strides = nn_structure["strides"]
        neurons_per_layer = nn_structure["neurons_per_layer"]

        # CNN layers
        self.conv1 = nn.Conv2d(input_channels, neurons_per_layer[0],
                               kernel_size=kernel_sizes[0], stride=strides[0])
        self.bn1 = nn.BatchNorm2d(neurons_per_layer[0])

        self.conv2 = nn.Conv2d(neurons_per_layer[0], neurons_per_layer[1],
                               kernel_size=kernel_sizes[1], stride=strides[1])
        self.bn2 = nn.BatchNorm2d(neurons_per_layer[1])

        self.conv3 = nn.Conv2d(neurons_per_layer[1], neurons_per_layer[2],
                               kernel_size=kernel_sizes[2], stride=strides[2])
        self.bn3 = nn.BatchNorm2d(neurons_per_layer[2])

        # Calculates the size of the neural network head
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = w
        convh = h
        for i in range(3):
            convw = conv2d_size_out(convw, kernel_sizes[i], strides[i])
            convh = conv2d_size_out(convh, kernel_sizes[i], strides[i])

        # Head input size
        linear_input_size = convw * convh * neurons_per_layer[2]
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


# More complex CNN with multiple linear layers
class DQN_CNN_Advanced(nn.Module):
    def __init__(self, h, w, outputs, input_channels, nn_structure: dict):
        super().__init__()

        kernel_sizes = nn_structure["kernel_sizes"]
        strides = nn_structure["strides"]
        neurons_per_layer = nn_structure["neurons_per_layer"]

        self.cnn = nn.Sequential(nn.Conv2d(input_channels, neurons_per_layer[0], kernel_size=kernel_sizes[0], stride=strides[0]),
                                        nn.ReLU(True),
                                        nn.Conv2d(neurons_per_layer[0], neurons_per_layer[1], kernel_size=kernel_sizes[1], stride=strides[1]),
                                        nn.ReLU(True),
                                        nn.Conv2d(neurons_per_layer[1], neurons_per_layer[2], kernel_size=kernel_sizes[2], stride=strides[2]),
                                        nn.ReLU(True)
                                        )

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = w
        convh = h
        for i in range(3):
            convw = conv2d_size_out(convw, kernel_sizes[i], strides[i])
            convh = conv2d_size_out(convh, kernel_sizes[i], strides[i])

        # Head input size
        linear_input_size = convw * convh * neurons_per_layer[2]

        self.classifier = nn.Sequential(nn.Linear(linear_input_size, neurons_per_layer[3]),
                                        nn.ReLU(True),
                                        nn.Linear(neurons_per_layer[3], outputs)
                                        )

    def forward(self, x):
        x = self.cnn(x)

        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


# Sets the weights between -0.01 and 0.01
def initialise_weights(model):
    if type(model) == nn.Conv2d or type(model) == nn.Linear:
        torch.nn.init.uniform(model.weight, -0.01, 0.01)
        model.bias.data.fill_(0.01)


# Class for calculating the Q values
class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Method can be called without creating an instance of the class
    @staticmethod
    # Takes the state action pairs and returns the predicted Q-Values from the policy network that
    # were passed in
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    # Accepts the target network and next states
    def get_next(target_net, next_states):

        # The latest state in the batch
        last_screens_of_state = next_states[:, -1, :, :]

        # Finds the locations of all the final states (these are the states that occur after the
        # the state is occured that ended the episode)
        # In this context, a final state is a state that represents an all black screen
        # Hence all the final states are found (if there are any in a given batch), so that it is known
        # not to pass these final states to the target network to get a predicted value. These final states
        # do not have a reward, hence all of the next states are checked, and if it has a reward of 0, then
        # it is a final state. These final states are represented as True in this case (as it is converted to a boolean)
        # so it is then known not to include them

        final_state_locations = last_screens_of_state.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)

        # All non-final state_locations are the opposites of the final state locations
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]

        # Finds the batch size by checking to see how many next states are in the next states tensor
        batch_size = next_states.shape[0]

        # Creates a list of QValues equal to the batch size
        values = torch.zeros(batch_size).to(QValues.device)

        # Sets corresponding values of all the next state locations as the
        # maximum predicted Q values of the target network for each actions
        # (returns the maximum Q values for all actions) for each non-final state
        with torch.no_grad():
            values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()

        # These Q values are then returned
        return values

    @staticmethod
    def get_target_Q_Values(target_net, next_states, discount, rewards):

        # Extracts next Q values of the best corresponding actions of the target network
        # The target network is used for finding the next best actions
        next_q_values = QValues.get_next(target_net, next_states)

        # Uses formula E[reward + gamma * maxarg(next state)] to update Q values
        target_q_values = (next_q_values * discount) + rewards

        return target_q_values

    @staticmethod
    def get_next_DDQN(policy_net, target_net, next_states):
        last_screens_of_state = next_states[:, -1, :, :]  # (B,H,W)
        final_state_locations = last_screens_of_state.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]  # (B',4,H,W)
        batch_size = next_states.shape[0]
        # print("# of none terminal states = ", batch_size)
        values = torch.zeros(batch_size).to(QValues.device)
        if non_final_states.shape[0] == 0:  # BZX: check if there is survival
            print("EXCEPTION: this batch is all the last states of the episodes!")
            return values
        # BZX: different from DQN
        with torch.no_grad():
            argmax_a = policy_net(non_final_states).detach().max(dim=1)[1]
            values[non_final_state_locations] = target_net(non_final_states).detach().gather(dim=1,
                                                                                             index=argmax_a.unsqueeze(
                                                                                                 -1)).squeeze(-1)
        return values


class PolicyGradient():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_probabilities(target_net):
        pass


# An experience represents a transaction that the agent took and is what is used for training the network
# Represents replay memory
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

# Stores all the previous experiences and is used for helping the agent learn
class ReplayMemory():
    def __init__(self, capacity, start_size):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        self.start_size = start_size

        if start_size > capacity:
            raise ValueError("Replay capacity cannot be smaller than the start size")

    # Pushes the new experience to the replay memory queue
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    # Returns a random sample from the memory
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # Checks if a sample can be provided, the amount of experiences in memory must be larger than the
    # Batch size, and the memory must be at least 1/10 full of experiences
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size and len(self.memory) >= self.start_size

    def current_memory_size(self):
        return len(self.memory)
