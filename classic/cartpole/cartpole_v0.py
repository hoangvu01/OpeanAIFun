import gym, random, math, json
import matplotlib.pyplot as plt
import numpy as np

from classic.common import GenericAgent

class CartpoleAgent(GenericAgent):
  _ENV = 'CartPole-v0'
  def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.5, bins=(12, 16), 
               upper_bounds=[0.4, 0.5], lower_bounds=[-0.4, -0.5], 
               num_episodes=30, min_epsilon=0.05, graphics=True):
    
    GenericAgent.__init__(self, alpha, gamma, epsilon, min_epsilon,
                          bins, upper_bounds, lower_bounds,
                          num_episodes, graphics)

  
  def create_agent_from_config(config_path):
    config = json.loads(config_path)
    return CartpoleAgent(**config)

  def discretise(self, prev_obs, next_obs):
    temp_obs = next_obs[2:]
    new_obs = [max(min(temp_obs[i], self.upper_bounds[i]), self.lower_bounds[i]) for i in range(len(temp_obs))]
    range_bounds = [self.upper_bounds[i] - self.lower_bounds[i] for i in range(len(self.lower_bounds))]
    res = tuple([int(round((new_obs[i] - self.lower_bounds[i]) / range_bounds[i] * (self.bins[i] - 1))) for i in range(len(new_obs))])
    return res  
  
 
  def calculate_reward(self, prev_obs, curr_obs):
    return - abs(curr_obs[2]) * abs(curr_obs[3])
   

