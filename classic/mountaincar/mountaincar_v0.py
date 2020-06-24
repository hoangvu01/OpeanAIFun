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
  def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.5, min_epsilon=0.05,              #Hyperparam
               bins=(16, 16, 3), upper_bounds=[0.6, 0.07], lower_bounds=[-1.2, -0.07], #Discretisation
               num_episodes=200, graphics=True):                                        #Training
    GenericAgent.__init__(self, alpha, gamma, epsilon, min_epsilon, bins, 
                         upper_bounds, lower_bounds, num_episodes, graphics)
    self.env = gym.make('MountainCar-v0')

  def create_agent_from_config(config_path):
    """
       Create agent from pre-defined config
    """
    config = json.loads(config_path)
    return MountainCarAgent(**config)

  def discretise(self, obs):
    new_obs = [max(min(obs[i], self.upper_bounds[i]), self.lower_bounds[i]) for i in range(len(obs))]
    range_bounds = [self.upper_bounds[i] - self.lower_bounds[i] for i in range(len(self.lower_bounds))]
    res = tuple([int(round((new_obs[i] - self.lower_bounds[i]) / range_bounds[i] 
                 * (self.bins[i] - 1))) for i in range(len(new_obs))])
    return res  

  def calculate_reward(self, prev_obs, curr_obs):
    """
      Calculate reward given by a observation change
    """
    return abs(curr_obs[1]) * 10 - self.env.observation_space.high[0] + curr_obs[1]

  @staticmethod
  def test(agent):
  # Test the agent 
    test_results = []
    env = gym.make('MountainCar-v0')
    for t in range(100):
      obs = env.reset()
      done = False
      moves = 1
      while not done:
        action = agent.get_action(agent.discretise(obs), moves)
        obs, _, done, _ = env.step(action)
        moves += 1
      test_results.append(moves)  
    return test_results


