import matplotlib.pyplot as plt
import numpy as np

from classic.cartpole.cartpole_v0 import CartpoleAgent
from classic.mountaincar.mountaincar_v0 import MountainCarAgent
from classic.pendulum.pendulum_v0 import PendulumAgent

from classic.common.generic_agent import HyperParamTuner

def cartpole_run():
  agent = CartpoleAgent()
  agent.train()
  test_results = CartpoleAgent.test(agent)
  plt.plot(agent.performance)
  plt.plot(test_results)


def mountaincar_run():
  agent = MountainCarAgent() 
  agent.train()
  test_results = MountainCarAgent.test(agent)
  print(np.average(test_results), np.std(test_results))
  plt.plot(agent.performance)
  plt.plot(test_results)
  

def pendulum_run():
  agent = PendulumAgent(graphics=True) 
  agent.train()
  test_results = MountainCarAgent.test(agent)
  print(np.average(test_results), np.std(test_results))
  plt.plot(agent.performance)
  plt.plot(test_results)

if __name__ == '__main__':
  tuner = HyperParamTuner(PendulumAgent)
  tuner.run()
