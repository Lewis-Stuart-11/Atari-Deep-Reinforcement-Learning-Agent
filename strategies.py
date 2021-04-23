# Author Lewis Stuart 201262348

# Strategy for choosing the action depending on exploration or exploitation
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.use_reward = False

    # Returns the current exploration rate
    def get_exploration_rate(self, episode):
        # Sets the rate as the start, subtracted by the number of episodes multiplied by the decay
        # Hence, the exploration rate decreases linearly with time
        episodic_rate = self.start - (self.decay * episode)
        return episodic_rate if episodic_rate > self.end else self.end


class EpsilonGreedyStrategyAdvanced():
    def __init__(self, start, middle, end, start_decay, end_decay):
        self.start = start
        self.middle = middle
        self.end = end
        self.start_decay = start_decay
        self.end_decay = end_decay
        self.episode_switch = None
        self.use_reward = False

    # Returns the current exploration rate
    def get_exploration_rate(self, episode):
        # Sets the rate as the start, subtracted by the number of episodes multiplied by the decay
        # Hence, the exploration rate decreases linearly with time
        episodic_rate = self.start - (self.start_decay * episode)

        if episodic_rate < self.middle:
            if not self.episode_switch:
                self.episode_switch = episode
            episodic_rate = self.middle - (episode - self.episode_switch) * self.end_decay

        return episodic_rate if episodic_rate > self.end else self.end


class EpsilonGreedyRewardStrategy():
    def __init__(self, start, end, decay, reward_incrementation, reward_target, reward_decay):
        self.start = start
        self.decay = decay
        self.end = end
        self.reward_target = reward_target
        self.current_reward_threshold = 0
        self.reward_incrementation = reward_incrementation
        self.reward_rate = 1
        self.reward_decay = reward_decay
        self.use_reward = True

    # Returns the current exploration rate
    def get_exploration_rate(self, episode, current_reward=None):
        episodic_rate = self.start - (self.decay * episode)
        episodic_rate = episodic_rate if episodic_rate > self.end else self.end

        if current_reward:
            while self.current_reward_threshold <= current_reward < self.reward_target:
                self.current_reward_threshold += self.reward_incrementation
                self.reward_rate = (self.reward_rate - self.reward_decay) if self.reward_rate > self.end else self.end

        return episodic_rate * self.reward_rate
