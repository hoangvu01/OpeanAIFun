import gym, random, math, json
import matplotlib.pyplot as plt
import numpy as np

from classic.common import GenericAgent

class MountainCarAgent(GenericAgent):
  """
  Observation:
   0: Position [-1.2, 0.6]
   1: Velocity [-0.07, 0.07]

  Actions:
   0:
   1:
   2:

  """ 
  _ENV = 'MountainCar-v0'
  
  def __init__(self, alpha=0.1, gamma=1, epsilon=0.7, min_epsilon=0.1,                 #Hyperparam
               bins=(16, 12, 3), upper_bounds=[0.6, 0.07], lower_bounds=[-1.2, -0.07], #Discretisation
               num_episodes=200, graphics=True):                                       #Training
    GenericAgent.__init__(self, alpha, gamma, epsilon, min_epsilon, bins, 
                         upper_bounds, lower_bounds, num_episodes, graphics)

  def create_agent_from_config(config_path):
    """
       Create agent from pre-defined config
    """
    config = json.loads(config_path)
    return MountainCarAgent(**config)

  def discretise(self, prev_obs, curr_obs):
    range_bounds = [self.upper_bounds[i] - self.lower_bounds[i] for i in range(curr_obs.size)]    
    discrete = [int(round((curr_obs[i] - self.lower_bounds[i]) / range_bounds[i]) * (self.bins[i] - 1)) 
                for i in range(curr_obs.size)]
    
    return tuple(discrete)

  def calculate_reward(self, prev_obs, curr_obs):
    """
      Calculate reward given by a observation change
    """
    distance_from_flag = self.env.observation_space.high[0] - curr_obs[0]
    distance_from_wall = curr_obs[0] - self.env.observation_space.low[0] 
    beta = self.moves / 200
    reward = (1 - beta) * 36 * abs(curr_obs[1]) - 0.8 * (beta ** 1.05) * distance_from_flag + curr_obs[0]  * beta / 1000 

    if (curr_obs[0] >= 0.5):
      reward += 20 - self.moves / 20
      print("Reached flag")
    return reward
  
