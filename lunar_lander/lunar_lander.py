#!/usr/bin/env python

"""RandomWalk environment class for RL-Glue-py.
"""

from environment import BaseEnvironment
import numpy as np
import gym
from gym import wrappers
from time import time
import functools

def capped_cubic_video_schedule(episode_id, cap):
    if episode_id < cap:
        return int(round(episode_id ** (1. / 3))) ** 3 == episode_id
    else:
        return episode_id % cap == 0


class LunarLanderEnvironment(BaseEnvironment):
    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.
        """
        self.env_to_wrap = gym.make("LunarLander-v2")
        self.env_to_wrap._max_episode_steps = env_info["timeout"] - 1
        max_num_episodes = env_info["num_episodes"]
        self.env = wrappers.Monitor(self.env_to_wrap, './results/videos/' + str(time()), video_callable=functools.partial(capped_cubic_video_schedule, cap=max_num_episodes))
        self.env_to_wrap.seed(0)
        self.env.seed(0)

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """        
        
        reward = 0.0
        observation = self.env.reset()
        is_terminal = False
                
        self.reward_obs_term = (reward, observation, is_terminal)
        
        # return first state observation from the environment
        return self.reward_obs_term[1]
        
    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        last_state = self.reward_obs_term[1]
        current_state, reward, is_terminal, _ = self.env.step(action)
        
        self.reward_obs_term = (reward, current_state, is_terminal)
        
        return self.reward_obs_term
    
    def env_cleanup(self):
        self.env.close()
        self.env_to_wrap.close()