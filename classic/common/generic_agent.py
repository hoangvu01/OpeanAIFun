import random, gym
import numpy as np

from abc import ABC, abstractmethod 

class GenericAgent(ABC):

  def __init__(self, alpha, gamma, epsilon, min_epsilon, bins, upper_bounds, 
               lower_bounds, num_episodes, graphics):
    # Hyperparam
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.min_epsilon = min_epsilon
    
    # Param
    self.upper_bounds = upper_bounds
    self.lower_bounds = lower_bounds
    self.num_episodes = num_episodes
    self.bins = bins
    self.qtable = np.zeros(bins, dtype=np.float64)
    
    # Environment
    self.graphics = graphics
    self.env = gym.make('MountainCar-v0')
    self.performance = []

  @abstractmethod
  def create_agent_from_config(config_path):
    pass

  @abstractmethod
  def discretise(self, obs):
    pass 

  def get_epsilon(self, moves):
    """
      Calculate epsilon after decay based on number of moves made
    """
    return min(self.min_epsilon, self.epsilon ** moves)
  
  def get_action(self, state, moves):
    """
      Choose an action given the state of the cartpole and the number of moves made 
    """
    if (random.uniform(0, 1) < self.get_epsilon(moves)): 
      action = self.env.action_space.sample()
    else:
      action = np.argmax(self.qtable[state])
    return action
  
  def update_score(self, curr_state, next_state, action, reward):
    """
      Evaluate a move that has just been made
    """
    next_max = np.max(self.qtable[next_state])
    old_score = self.qtable[curr_state][action]
    new_score = (1 - self.alpha) * old_score + self.alpha * (reward + self.gamma * next_max)
    self.qtable[curr_state][action] = new_score
 
  @abstractmethod
  def calculate_reward(self, prev_obs, next_obs):
    pass
  
  def train(self):
    """
      Start the training process 
    """
    for episode in range(self.num_episodes):
      curr_obs = self.env.reset()
      curr_state = self.discretise(curr_obs)
      # Start game
      done = False
      moves = 1
      while not done:
        if (self.graphics):
          self.env.render()
        action = self.get_action(curr_state, moves)
        next_obs, _, done, _ = self.env.step(action)
        next_state = self.discretise(next_obs)
        reward = self.calculate_reward(curr_obs, next_obs)
        self.update_score(curr_state, next_state, action, reward)
        
        curr_obs = next_obs 
        curr_state = next_state
        moves += 1
      self.performance.append(moves) 
      print("Episode {} finished after {} moves \r".format(episode, moves), end="")
    self.env.close()  

  @staticmethod
  @abstractmethod 
  def test(agent):
    pass
