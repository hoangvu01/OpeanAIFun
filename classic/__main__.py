import matplotlib.pyplot as plt
import numpy as np

from classic.cartpole.cartpole_v0 import CartpoleAgent
from classic.mountaincar.mountaincar_v0 import MountainCarAgent

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
  plt.plot(agent.performance)
  plt.plot(test_results)

if __name__ == '__main__':
  mountaincar_run()
  plt.show()
