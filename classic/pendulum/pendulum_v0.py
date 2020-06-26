import gym, random, math, json
import matplotlib.pyplot as plt
import numpy as np

from classic.common import GenericAgent

class PendulumAgent(GenericAgent):
  """
  Observation:
    0: cos(theta) [-1,1]
    1: sin(theat) [-1,1]
    2: theta dot [-8, 8]
  
  Actions:
   0: effort [-2, 2]

  """ 
  _ENV = 'Pendulum-v0'
  
  def __init__(self, alpha=0.1, gamma=1, epsilon=0.7, min_epsilon=0.1,                 
               bins=(16, 16, 16, 16), upper_bounds=(1, 1, 8, 2), lower_bounds=(-1, -1, -8 , -2),
               num_episodes=200, graphics=True):                        
    GenericAgent.__init__(self, alpha, gamma, epsilon, min_epsilon, bins, 
                         upper_bounds, lower_bounds, num_episodes, graphics)

  def create_agent_from_config(config_path):
    """
       Create agent from pre-defined config
    """
    config = json.loads(config_path)
    return PendulumAgent(**config)

  def discretise(self, prev_obs, curr_obs):
    range_bounds = [self.upper_bounds[i] - self.lower_bounds[i] for i in range(curr_obs.size)] 
    discrete = [int(round((curr_obs[i] - self.lower_bounds[i]) / 
                           range_bounds[i] * (self.bins[i] - 1))) 
                for i in range(curr_obs.size)]
    return tuple(discrete)

  def get_action(self, state):
    if (random.uniform(0, 1) < self.get_epsilon()): 
      action = self.env.action_space.sample()
    else:
      discrete_action = np.argmax(self.qtable[state])
      action = [discrete_action / (self.bins[3] - 1) * (self.upper_bounds[3] - self.lower_bounds[3]) 
                + self.lower_bounds[3]]
    return action

  def update_score(self, curr_state, next_state, action, reward):
    """
      Evaluate a move that has just been made
    """
    next_max = np.max(self.qtable[next_state])
    denormalised_action = int(round((action[0] - self.lower_bounds[3]) / 
                          (self.upper_bounds[3] - self.lower_bounds[3]) * (self.bins[3] - 1)))
    old_score = self.qtable[curr_state][denormalised_action]
    new_score = (1 - self.alpha) * old_score + self.alpha * (reward + self.gamma * next_max)
    self.qtable[curr_state][denormalised_action] = new_score
  
  def calculate_reward(self, prev_obs, curr_obs, env_reward):
    """
      Calculate reward given by a observation change
    """
    acceleration = curr_obs[2] - prev_obs[2]
    reward = curr_obs[0] * 7 - abs(curr_obs[2])
    return env_reward + reward
