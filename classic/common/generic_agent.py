import random, gym
import numpy as np

from abc import ABC, abstractmethod 

class GenericAgent(ABC):
  _ENV = '' 
 
  def __init__(self, alpha, gamma, epsilon, min_epsilon, bins, upper_bounds, 
               lower_bounds, num_episodes, graphics):
    # Hyperparam
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.min_epsilon = min_epsilon
    
    # Environment
    self.graphics = graphics
    self.env = gym.make(self._ENV)
    self.bins = bins
    self.qtable = np.zeros(self.bins, dtype=np.float64)
    self.performance = []
    self.moves = 0
    self.env_reward = 0

    # Param 
    self.upper_bounds = (upper_bounds 
                         if upper_bounds != None 
                         else list(self.env.observation_space.high))
    self.lower_bounds = (lower_bounds
                         if lower_bounds != None
                         else list(self.env.observation_space.low)) 
    self.num_episodes = num_episodes
    
  @staticmethod
  @abstractmethod
  def create_agent_from_config(config_path):
    pass

  @abstractmethod
  def discretise(self, prev_obs, next_obs):
    pass 

  def get_epsilon(self):
    """
      Calculate epsilon after decay based on number of moves made
    """
    return max(self.min_epsilon, self.epsilon ** self.moves)
  
  def get_action(self, state):
    """
      Choose an action given the state of the cartpole and the number of moves made 
    """
    if (random.uniform(0, 1) < self.get_epsilon()): 
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
  def calculate_reward(self, prev_obs, next_obs, env_reward):
    pass
  
  def train(self):
    """
      Start the training process 
    """
    for episode in range(self.num_episodes):
      prev_obs = next_obs = self.env.reset()
      curr_state = self.discretise(prev_obs, next_obs)
      # Start game
      done = False
      self.moves = 0
      while not done:
        if (self.graphics):
          self.env.render()
        action = self.get_action(curr_state)
        curr_obs, env_reward, done, _ = self.env.step(action)
        self.env_reward += env_reward
        next_state = self.discretise(prev_obs, curr_obs)
        reward = self.calculate_reward(prev_obs, curr_obs, env_reward)
        self.update_score(curr_state, next_state, action, reward)
        
        prev_obs = curr_obs 
        curr_state = next_state
        self.moves += 1
      self.performance.append(self.env_reward) 
      print("Episode {} finished, env_reward {} ".format(episode, self.env_reward))
    self.env.close()  

  @staticmethod
  def test(agent):
  # Test the agent 
    test_results = []
    env = gym.make(agent._ENV)
    for t in range(100):
      prev_obs = curr_obs = env.reset()
      done = False
      agent.moves = 1
      while not done:
        action = agent.get_action(agent.discretise(prev_obs, curr_obs))
        prev_obs = curr_obs
        curr_obs, _, done, _ = env.step(action)
        agent.moves += 1
      test_results.append(agent.moves)  
    return test_results

class HyperParamTuner():
  """
    hyper_[upper/lower] = "alpha, gamma, epsilon"
  """
  def __init__(self, hyper_lower, hyper_upper, bins_lower, bins_upper,
               min_epsilon, upper_bounds, num_episodes, graphics, classname):
    self.hyper_lower = hyper_lower
    self.hyper_upper = hyper_upper
    self.bins_lower = bins_lower
    self.bins_upper = bins_upper
    self.min_epsilon = min_epsilon
    self.lower_bounds = lower_bounds
    self.upper_bounds = upper_bounds
    self.num_epsiodes = num_episodes
    self.graphics = graphics
    self.classname = classname


  def run(self):
    pass 



