class ExperiencePool:
    """
    Experience pool for collecting trajectories.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, reward, done):
        self.states.append(state)  # Sometimes state is also called obs (observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def __getstate__(self):
        """
        Custom method to serialize the state without relying on the class or module.
        """
        # Return only the necessary data for pickling
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
        }

    def __setstate__(self, state):
        """
        Custom method to restore the state when unpickling, without requiring the class/module.
        """
        self.states = state['states']
        self.actions = state['actions']
        self.rewards = state['rewards']
        self.dones = state['dones']
