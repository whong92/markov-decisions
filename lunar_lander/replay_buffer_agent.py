import numpy as  np
from agent import BaseAgent
from abc import ABCMeta, abstractmethod


class ReplayBuffer:
    def __init__(self, size, minibatch_size, seed):
        """
        Args:
            size (integer): The size of the replay buffer.
            minibatch_size (integer): The sample size.
            seed (integer): The seed for the random number generator.
        """
        self.buffer = []
        self.minibatch_size = minibatch_size
        self.rand_generator = np.random.RandomState(seed)
        self.max_size = size

    def append(self, state, action, reward, terminal, next_state, timestep):
        """
        Args:
            state (Numpy array): The state.
            action (integer): The action.
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
            next_state (Numpy array): The next state.
        """
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, terminal, next_state, timestep])

    def sample(self):
        """
        Returns:
            A list of transition tuples including state, action, reward, terinal, and next_state
        """
        idxs = self.rand_generator.choice(np.arange(len(self.buffer)), size=self.minibatch_size)
        return [self.buffer[idx] for idx in idxs]

    def size(self):
        return len(self.buffer)


class ReplayBufferAgent(BaseAgent):

    __metaclass__ = ABCMeta

    def init_replay_buffer(self, replay_buffer_size, minibatch_sz, seed, num_replay_updates_per_step):
        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            replay_buffer_size,
            minibatch_sz,
            seed
        )
        self.num_replay = num_replay_updates_per_step

    @abstractmethod
    def _agent_init(self, agent_config):
        pass

    # Work Required: No.
    def agent_init(self, agent_config={}):
        """Setup variables to track state
        """
        self.last_state = None
        self.last_action = None

        self.sum_rewards = 0
        self.episode_steps = 0

        self.train = True  # whether or not to perform optimization

        self._agent_init(agent_config)
        self.init_replay_buffer(
            agent_config['replay_buffer_size'],
            agent_config['minibatch_sz'],
            agent_config['seed'],
            agent_config['num_replay_updates_per_step']
        )

    # Work Required: No.
    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array([state])
        self.last_action = self.policy(self.last_state)
        return self.last_action

    @abstractmethod
    def policy(self, state):
        """
        run this with provided state to get action
        """
        pass

    @abstractmethod
    def agent_optimize(self, experiences):
        """
        run this with provided experiences to run one step of optimization
        """
        pass

    @abstractmethod
    def agent_pre_replay(self):
        """
        run this to set things up before the replay buffer runs
        """
        pass

    @abstractmethod
    def agent_post_replay(self):
        """
        run this to cleanup after replay buffer runs finishes
        """
        pass

    # weights update using optimize_network, and updating last_state and last_action (~5 lines).
    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

        self.sum_rewards += reward
        self.episode_steps += 1

        # Make state an array of shape (1, state_dim) to add a batch dimension and
        # to later match the get_action_values() and get_TD_update() functions
        state = np.array([state])

        # Select action
        action = self.policy(state)
        # Append new experience to replay buffer
        self.replay_buffer.append(self.last_state, self.last_action, reward, False, state, self.episode_steps)

        if self.train:
            # Perform replay steps:
            if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
                self.agent_pre_replay()
                for _ in range(self.num_replay):
                    # Get sample experiences from the replay buffer
                    experiences = self.replay_buffer.sample()
                    self.agent_optimize(experiences)
                self.agent_post_replay()

        # Update the last state and last action.
        self.last_state = state
        self.last_action = action
        # your code here

        return action

    # Work Required: Yes. Fill in the replay-buffer update and
    # update of the weights using optimize_network (~2 lines).
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1

        # Set terminal state to an array of zeros
        state = np.zeros_like(self.last_state)

        # Append new experience to replay buffer
        self.replay_buffer.append(self.last_state, self.last_action, reward, True, state, self.episode_steps)

        if self.train:
            # Perform replay steps:
            if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
                self.agent_pre_replay()
                for _ in range(self.num_replay):
                    # Get sample experiences from the replay buffer
                    experiences = self.replay_buffer.sample()
                    self.agent_optimize(experiences)
                self.agent_post_replay()