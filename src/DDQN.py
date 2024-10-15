import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

random.seed(111)
np.random.seed(111)
tf.random.set_seed(111)

class DoubleDQN:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.priorities = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.loss = []

        # PER
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.state_shape),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(1.0)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0

        # sample
        priorities = np.array(self.priorities)
        scaled_priorities = priorities ** self.alpha
        probabilities = scaled_priorities / np.sum(scaled_priorities)

        minibatch_indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        minibatch = [self.memory[i] for i in minibatch_indices]

        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch]).astype(int)

        # precicted q values for the current state
        current_q_values = self.model.predict(states, verbose=0)
        # predicted q values for the next states from main and target networks
        next_q_values_main = self.model.predict(next_states, verbose=0)
        next_q_values_target = self.target_model.predict(next_states, verbose=0)

        next_actions = np.argmax(next_q_values_main, axis=1)

        # target q vals
        target_q_values = current_q_values.copy()
        target = rewards + self.gamma * next_q_values_target[np.arange(batch_size), next_actions] * (1 - dones)
        target_q_values[np.arange(batch_size), actions] = target

        # TD errs
        td_errors = np.abs(target - current_q_values[np.arange(batch_size), actions])
        for i, idx in enumerate(minibatch_indices):
            self.priorities[idx] = td_errors[i] + 1e-5

        weights = (len(self.memory) * probabilities[minibatch_indices] + 1e-8) ** -self.beta
        weights = np.clip(weights / np.max(weights), 0, 1) # clipping

        history = self.model.fit(states, target_q_values, sample_weight=weights, batch_size=batch_size, verbose=0)

        # Updates
        if self.beta < 1:
            self.beta += self.beta_increment

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history.history['loss'][0]
