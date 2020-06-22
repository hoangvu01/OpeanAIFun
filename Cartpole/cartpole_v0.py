import gym, random
import matplotlib.pyplot as plt
import numpy as np
import json

MAX_ACTIONS = 200
NUM_EPISODES = 1000 

ALPHA = 0.1
GAMMA = 0.8
EPSILON = 0.1 

JSONFILE = "cartpole_res.txt"
PERFFILE = "cartpole_data.txt"
qtable = {}
qtable_dict_list = []
performance = []

def load_from_file():
  with open(JSONFILE) as jsonfile:
    qtable_dict_list = json.load(jsonfile)  

  for jdict in qtable_dict_list:
    for _ in jdict.items():
      qtable[tuple(jdict["key"])] = np.array(jdict["value"])  

def write_to_file():    
  with open(JSONFILE, "w+") as jsonfile:
      json.dump([ { 'key' : list(k), 
                    'value' : list(v) } for (k,v) in qtable.items()], 
              jsonfile, indent=2)

def evaluate_env(obs):
  #TODO
  return - np.array(map(abs, obs)).sum()

def get_actions_space(state):
  actions_space = np.zeros(2, dtype=np.float64)
  # Create action space if not exists already
  if random.uniform(0, 1) < EPSILON or (not state in qtable): 
    # Exploration  
    action = random.randint(0, 1)
    qtable[state] = actions_space
  else: 
    #Exploitation 
    actions_space = qtable[state] 
    action = np.argmax(actions_space)
  return action, actions_space     

def get_state(obs):
  #TODO: 
  tilt_sign = (obs[2] > 0) ^ (obs[3] > 0)  
  angular_v = 1 if abs(obs[3]) > 10 else 0 
  return (round(abs(obs[2]), 1), angular_v, int(tilt_sign))


def train():
  env = gym.make('CartPole-v0')
 
  for episode in range(NUM_EPISODES):
    obs = env.reset()
    old_state = get_state(obs) 
    # Start game
    for t in range(MAX_ACTIONS):
      env.render()
      
      # Get the next action given current state
      action, old_actions_space = get_actions_space(old_state)  
      old_score = old_actions_space[action] 
      obs, reward, done, info = env.step(action)
      
      # Evaluate the state after action
      new_state = get_state(obs) 
      action, new_actions_space = get_actions_space(new_state)   
      next_max = new_actions_space[action]   
      
      # Update score and prepare for the iteration
      old_actions_space[action] = (1 - ALPHA) * old_score + ALPHA * ((-obs[2])+ GAMMA * next_max)
      old_state = new_state
      
      if done or t == MAX_ACTIONS - 1:
        performance.append(t)  
        print("Done! Episode {} after {} steps \n".format(episode, t + 1))
        break
    if episode % 100 == 0:
      write_to_file()
  env.close()  

def main():
  load_from_file()
  train()
  write_to_file()
  plt.plot(performance)
  plt.show()

main()  
