import gym, random, math, json
import matplotlib.pyplot as plt
import numpy as np


class CartpoleAgent:
  def __init__(self, alpha, gamma, epsilon, bins, upper_bounds, lower_bounds, num_episodes, min_epsilon=0.05, graphics=True):
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.performance = []
    self.upper_bounds = upper_bounds
    self.lower_bounds = lower_bounds
    self.num_episodes = num_episodes
    self.bins = bins
    self.qtable = np.zeros(bins, dtype=np.float64)
    self.graphics = graphics
    self.min_epsilon = min_epsilon
    self.env = gym.make('CartPole-v0')

  @staticmethod
  def load_from_file():
    with open(JSONFILE) as jsonfile:
      qtable_dict_list = json.load(jsonfile)  
  
    for jdict in qtable_dict_list:
      for _ in jdict.items():
        qtable[tuple(jdict["key"])] = np.array(jdict["value"])  

  @staticmethod
  def write_to_file():    
     with open(JSONFILE, "w+") as jsonfile:
        json.dump([ { 'key' : list(k), 
                      'value' : list(v) } for (k,v) in qtable.items()], 
                jsonfile, indent=2)
 
  def get_epsilon(self, moves):
    return min(self.min_epsilon, self.epsilon ** moves)
   
  def discretise(self, obs):
    temp_obs = obs[2:]
    new_obs = [max(min(temp_obs[i], self.upper_bounds[i]), self.lower_bounds[i]) for i in range(len(temp_obs))]
    range_bounds = [self.upper_bounds[i] - self.lower_bounds[i] for i in range(len(self.lower_bounds))]
    res = tuple([int(round((new_obs[i] - self.lower_bounds[i]) / range_bounds[i] 
                 * (self.bins[i] - 1))) for i in range(len(new_obs))])
    return res  
  
  def get_action(self, state, moves):
    if (random.uniform(0, 1) < self.get_epsilon(moves)): 
      action = self.env.action_space.sample()
    else:
      action = np.argmax(self.qtable[state])
      print(self.qtable[state], action)
    return action

  def update_score(self, curr_state, next_state, action, reward):
    next_max = np.max(self.qtable[next_state])
    old_score = self.qtable[curr_state][action]
    new_score = (1 - self.alpha) * old_score + self.alpha * (reward + self.gamma * next_max)
    self.qtable[curr_state][action] = new_score
  
  def train(self):
    for episode in range(self.num_episodes):
      if (self.graphics):
        self.env.render()
      curr_obs = self.env.reset()
      curr_state = self.discretise(curr_obs)
      # Start game
      done = False
      moves = 1
      while not done:
        action = self.get_action(curr_state, moves)
        next_obs, survived , done, _ = self.env.step(action)
        next_state = self.discretise(next_obs)
        self.update_score(curr_state, next_state, action, - abs(next_obs[2]) * abs(next_obs[3]))
        
        curr_obs = next_obs 
        curr_state = next_state
        moves += 1
      self.performance.append(moves) 
    self.env.close()  


def test(agent : CartpoleAgent):
  test_results = []
  env = gym.make('CartPole-v0')
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

def main():
  agent = CartpoleAgent(0.1, 0.9, 0.5, (10, 16, 2), 
                        [ 0.4,  0.5], 
                        [-0.4, -0.5], 150, min_epsilon=0.1, graphics=True)
  agent.train()
  test_results = test(agent)
  plt.plot(test_results)
  plt.plot(agent.performance)
  plt.show()

if __name__ == "__main__":
  main()
