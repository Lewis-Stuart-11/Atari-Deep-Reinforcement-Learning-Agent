# Author Lewis Stuart 201262348

# Strategy for choosing the action depending on exploration or exploitation
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    # Returns the current exploration rate
    def get_exploration_rate(self, episode):
        # Sets the rate as the start, subtracted by the number of episodes multiplied by the decay
        # Hence, the exploration rate decreases linearly with time
        rate = self.start - (self.decay * episode)
        return rate if rate > self.end else self.end


class EpsilonGreedyStrategyAdvanced():
    def __init__(self, start, middle, end, start_decay, end_decay):
        self.start = start
        self.middle = middle
        self.end = end
        self.start_decay = start_decay
        self.end_decay = end_decay
        self.episode_switch = None

    # Returns the current exploration rate
    def get_exploration_rate(self, episode):
        # Sets the rate as the start, subtracted by the number of episodes multiplied by the decay
        # Hence, the exploration rate decreases linearly with time
        rate = self.start - (self.start_decay * episode)

        if rate < self.middle:
            if not self.episode_switch:
                self.episode_switch = episode
            rate = self.middle - (episode - self.episode_switch) * self.end_decay

        return rate if rate > self.end else self.end