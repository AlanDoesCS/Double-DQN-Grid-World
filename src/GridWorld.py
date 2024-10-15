import numpy as np
import random

np.random.seed(111)
random.seed(111)

class GridWorld:
    def __init__(self, size):
        self.size = size
        self.reset()

    def reset(self):
        self.grid = np.random.rand(self.size, self.size)

        self.position = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]

        while True:
            self.goal = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
            if self.goal != self.position:
                break

        return self._get_state()

    def step(self, action):
        if action == 0 and self.position[0] > 0: # 0: up
            self.position[0] -= 1
        elif action == 1 and self.position[1] < self.size - 1: # 1: right
            self.position[1] += 1
        elif action == 2 and self.position[0] < self.size - 1: # 2: down
            self.position[0] += 1
        elif action == 3 and self.position[1] > 0: # 3: left
            self.position[1] -= 1
        # 4: no-op

        done = (self.position == self.goal)
        reward = -self.grid[self.position[0], self.position[1]]
        if done:
            reward += 10  # goal reward

        return self._get_state(), reward, done

    def _get_state(self):
        state = np.zeros((self.size, self.size, 3))
        state[:, :, 0] = self.grid
        state[self.position[0], self.position[1], 1] = 1  # Agent position
        state[self.goal[0], self.goal[1], 2] = 1  # Goal position
        return state
