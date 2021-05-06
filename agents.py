# Author Lewis Stuart 201262348

import torch
import random
import numpy as np
from torch.distributions import Bernoulli
from torch.nn import Softmax
# Ensures that the results will be the same (same starting random seed each time)
random.seed(0)

# Imports the names of all current available policies for referencing
from avaliable_policy_methods import *

# Handles what actions to take in the environment
class Agent():
    def __init__(self, strategy, num_actions, learning_technique):
        # The strategy for choosing which action to take
        self.strategy = strategy

        self.learning_technique = learning_technique

        # Number of actions of the current game
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chooses the new action to take for the agent
    def select_action(self, state, policy_net, episode, reward):

        if self.learning_technique in POLICY_GRADIENT_METHODS:
            return self.return_action_from_probs(state, policy_net)

        # Returns exploration rate
        rate = self.strategy.get_exploration_rate(episode, reward) if self.strategy.use_reward else \
            self.strategy.get_exploration_rate(episode)

        # Chooses a random action if agent decides to explore
        if rate > random.random():
            return self.select_random_action()

        # Chooses the most optimal action, exploiting the policy
        else:
            return self.select_exploitative_action(state, policy_net)

    def return_action_from_probs(self, state, policy_net):
        with torch.no_grad():
            action_space = np.arange(self.num_actions)
            action_probs = policy_net(state).to("cpu").squeeze().detach().numpy()
            action = np.random.choice(action_space, p=action_probs)
            return torch.tensor([action], dtype=torch.int64).to(self.device)

            # Returns random action
    def select_random_action(self):
        action = random.randrange(self.num_actions)
        return torch.tensor([action]).to(self.device)

    # Returns optimal action
    def select_exploitative_action(self, state, policy_net):
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).to(self.device)  # exploit

    # Returns the default action (which is always 0)
    def select_default_action(self):
        return torch.tensor([0]).to(self.device)

    # Returns the current exploration rate
    def return_exploration_rate(self, episode):
        if not self.strategy:
            return False

        if self.strategy.use_reward:
            return self.strategy.get_exploration_rate(episode, 0)
        else:
            return self.strategy.get_exploration_rate(episode)