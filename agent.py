from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminated'))

writer = SummaryWriter('logs')
BATCH_SIZE = 64

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity
        self.episode = None

    def push(self, state, action, reward, next_state, terminated, episode):
        """Save a transition."""
        transition = Transition(state, action, reward, next_state, terminated)
        if len(self.memory) == self.capacity:
            self.memory.popleft()
        self.memory.append(transition)
        self.episode = episode

    def sample(self, batch_size):
        """Randomly sample a batch of transitions from memory."""
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        transitions = [self.memory[i] for i in indices]

        # Convertir chaque transition en une liste de tableaux NumPy (à compléter)
        state_list, action_list, reward_list, next_state_list, terminated_list = zip(*transitions)
        state_array = np.array(state_list)
        action_array = np.array(action_list)
        reward_array = np.array(reward_list)
        next_state_array = np.array(next_state_list)
        terminated_array = np.array(terminated_list)

        return state_array, action_array, reward_array, next_state_array, terminated_array

        
    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, n_hid_layers=4, hid_units=64, dropout_rate=0.1):
        super(QNetwork, self).__init__()
        self.n_hid_layers = n_hid_layers
        self.hid_units = hid_units

        self.dropout_rate = dropout_rate

        self.input_layer = nn.Linear(state_size, hid_units)
        self.output_layer = nn.Linear(hid_units, action_size)

        self.hid_layers = nn.ModuleList()
        for i in range(n_hid_layers):
            self.hid_layers.append(nn.Linear(hid_units, hid_units))

    def forward(self, state):
        x = torch.relu(self.input_layer(state))
        x = nn.Dropout(self.dropout_rate)(x)

        for i in range(self.n_hid_layers):
            x = torch.relu(self.hid_layers[i](x))
            x = nn.Dropout(self.dropout_rate)(x)

        q_values = self.output_layer(x)
        return q_values

    
class Agent():
    def __init__(self, model, memory, epsilon):
        self.model = model
        self.memory = memory
        self.epsilon = epsilon 
        self.velocity = 0  # Vitesse initiale de l'agent
        self.acceleration = 0.5  # Accélération/deccélération
        self.learning_rate_a = 0.01 # alpha or learning rate
        self.discount_factor_g = 0.9 # gamma or discount factor.
        self.state_vector = []
        self.max_score = 2211
        self.action_space = [0, 1, 2, 3]
        self.action_space_n = len(self.action_space)
        self.Q = None # Q-Table
        self.max_episodes = 500
        self.rng = np.random.default_rng()   # random number generator

    
    def update_velocity(self, action):
        # Mise à jour de la vitesse en fonction de l'action
        if action == 0:
            self.velocity = max(self.velocity - self.acceleration, -50)  # Mouvement vers la gauche
        elif action == 2:
            self.velocity = min(self.velocity + self.acceleration, 50)  # Mouvement vers la droite
        else:
            # Si aucune action n'est effectuée, réduisez la vitesse (traînée)
            self.velocity = 0 if self.velocity == 0 else (self.velocity - np.sign(self.velocity) * self.acceleration)

        return self.velocity

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = self.memory.sample(batch_size)
        # Create the optimizer outside the function and reuse it at each iteration
        optimizer = optim.Adam(self.model.parameters())
        q_values = self.model(state_batch)
        next_q_values = self.model(next_state_batch)
        next_q_values[terminated_batch] = 0.0
        target_q_values = reward_batch + self.discount_factor_g * torch.max(next_q_values, dim=1)[0]
        loss = F.mse_loss(q_values[range(batch_size), action_batch], target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
    @staticmethod
    def normalize_reward(reward, alpha=0.01, max_reward=100, min_reward=10):
        """Normalize the reward between 1 and 100"""
        normalized_reward = ((reward) / (max_reward - (reward * alpha))) / min_reward
        normalized_reward = int(normalized_reward)
        return normalized_reward
