# Author Lewis Stuart 201262348

# Basic Linear strategy for choosing the action depending on exploration or exploitation
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):

        # Start exploration rate
        self.start = start

        # End exploration rate
        self.end = end

        # Linear episodic decay of exploration rate
        self.decay = decay

        # Sets if the strategy uses a reward, in this case no reward is used in determining the exploration rate
        self.use_reward = False

    # Returns the current exploration rate
    def get_exploration_rate(self, episode):
        # Sets the rate as the start, subtracted by the number of episodes multiplied by the decay
        # Hence, the exploration rate decreases linearly with time
        episodic_rate = self.start - (self.decay * episode)

        # Returns the end exploration rate if the current rate passes this
        return episodic_rate if episodic_rate > self.end else self.end


# A more advanced linear epsilon greedy strategy that slows rate of decay after a middle point is passed
class EpsilonGreedyStrategyAdvanced():
    def __init__(self, start, middle, end, start_decay, end_decay):
        self.start = start
        self.middle = middle
        self.end = end

        # Decay to use before decay passes middle exploration rate
        self.start_decay = start_decay

        # Decay to use after decay passes middle exploration rate
        self.end_decay = end_decay

        # The episode that the middle exploration rate was passed
        self.episode_switch = None

        # Sets if the strategy uses a reward, in this case no reward is used in determining the exploration rate
        self.use_reward = False

    # Returns the current exploration rate
    def get_exploration_rate(self, episode):
        # Sets the rate as the start, subtracted by the number of episodes multiplied by the decay
        # Hence, the exploration rate decreases linearly with time
        episodic_rate = self.start - (self.start_decay * episode)

        # If the exploration rate has passed a middle threshold, then a new linear decay is used with a new decay
        if episodic_rate < self.middle:
            if not self.episode_switch:
                self.episode_switch = episode
            episodic_rate = self.middle - (episode - self.episode_switch) * self.end_decay

        return episodic_rate if episodic_rate > self.end else self.end


# Uses both linear episodic decay as well as reward decay
class EpsilonGreedyRewardStrategy():
    def __init__(self, start, end, decay, reward_incrementation, reward_target, reward_decay):
        self.start = start
        self.decay = decay
        self.end = end

        # The maximum reward that the agent must achieve before exploration rate is no longer altered by reward
        self.reward_target = reward_target

        # Current reward threshold to pass before increasing the reward
        self.current_reward_threshold = 0

        # How much to increment to the threshold by
        self.reward_incrementation = reward_incrementation

        # First exploration rate is set to 1
        self.reward_rate = 1

        # How much to decay the reward exploration rate
        self.reward_decay = reward_decay

        # Sets if the strategy uses a reward, in this case reward is used in determining the exploration rate
        self.use_reward = True

    # Returns the current exploration rate
    def get_exploration_rate(self, episode, current_reward=None):

        # Episode decay rate is derived, same as linear episodic decay
        episodic_rate = self.start - (self.decay * episode)
        episodic_rate = episodic_rate if episodic_rate > self.end else self.end

        # If the current reward passes a reward threshold, then the new decay is formulated based on the current
        # reward rate and decay
        if current_reward:
            while self.current_reward_threshold <= current_reward < self.reward_target:
                self.current_reward_threshold += self.reward_incrementation
                self.reward_rate = (self.reward_rate - self.reward_decay) if self.reward_rate > self.end else self.end

        # Returns multiplication of episodic and reward rate
        return episodic_rate * self.reward_rate if episodic_rate * self.reward_rate > 0 else 0
