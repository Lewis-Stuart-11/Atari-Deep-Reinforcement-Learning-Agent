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

    def get_step_dependant_exploration_rate(self, episode, episode_step):
        pass