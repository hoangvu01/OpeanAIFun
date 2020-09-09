import itertools
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
      self.env_reward = 0
      while not done:
        if (self.graphics):
          self.env.render()
        action = self.get_action(curr_state)
        curr_obs, env_reward, done, _ = self.env.step(action)
        self.env_reward += env_reward
        next_state = self.discretise(prev_obs, curr_obs)
        reward = self.calculate_reward(prev_obs, curr_obs, env_reward)
        self.env_reward += env_reward
        self.update_score(curr_state, next_state, action, reward)
        
        prev_obs = curr_obs 
        curr_state = next_state
        self.moves += 1
      self.performance.append(self.env_reward) 
      print("Episode {} finished, env_reward {} \r".format(episode, self.env_reward), end='')
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
      agent.env_reward = 0
      while not done:
        action = agent.get_action(agent.discretise(prev_obs, curr_obs))
        prev_obs = curr_obs
        curr_obs, env_reward, done, _ = env.step(action)
        agent.env_reward += env_reward
        agent.moves += 1
      test_results.append(agent.env_reward)  
    return test_results

class HyperParamTuner():
  """
    hyper_[upper/lower] = "alpha, gamma, epsilon"
  """
  def __init__(self, classname):
    self.classname = classname


  def run(self):
    params = itertools.product(np.linspace(0, 1, 11), repeat=3) 
    performance = np.zeros((11, 11, 11), dtype=np.float64)
    for param in params:
      agent = self.classname(alpha=param[0], gamma=param[1], epsilon=param[2], graphics=False)
      agent.train()
      test_res = self.classname.test(agent)
      test_avg = np.average(np.array(test_res))
      test_std = np.std(np.array(test_res))
      print('=======================================================================')
      print("Alpha: {}, Gamma: {}, Epsilon: {}".format(param[0], param[1], param[2]))
      print("Avergage: {}, Standard Deviation: {}".format(test_avg, test_std))
      print('=======================================================================')
      indices = tuple(map(lambda x: int(x * 10), param))
      performance[indices] = test_avg
      np.save("testfile.txt", performance)
      print("Best params: {}".format(np.argmax(performance)))

 
