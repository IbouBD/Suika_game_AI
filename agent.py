import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminated'))

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

        # Convert each transition to a list of NumPy arrays 
        state_list, action_list, reward_list, next_state_list, terminated_list = zip(*transitions)
        

        return state_list, action_list, reward_list, next_state_list, terminated_list

        
    def __len__(self):
        return len(self.memory)
    

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, n_hid_layers=4, hid_units=64, dropout_rate=0.1):
        super(QNetwork, self).__init__()
        self.n_hid_layers = n_hid_layers
        self.hid_units = hid_units
        self.dropout_rate = dropout_rate
        self.input_layer = nn.Linear(state_size, hid_units)
        self.hid_layers = nn.ModuleList([nn.Linear(hid_units, hid_units) for _ in range(n_hid_layers)])
        self.output_layer = nn.Linear(hid_units, action_size)

        
    def forward(self, state):
        """Forward pass through the network. Returns the predicted Q-values for each possible action."""
        state = torch.stack(list(state), dim=0).to(torch.float32)
        x = torch.relu(self.input_layer(state))
        x = nn.Dropout(self.dropout_rate)(x)

        for hid_layer in self.hid_layers:
            x = torch.relu(hid_layer(x)) 
            x = nn.Dropout(self.dropout_rate)(x)

        q_values = self.output_layer(x)

        return q_values

    
    def loss(self, rewards, gamma, next_state, predictQ):
        """
        Compute the loss function. 
        The target for each sample is to choose the action that maximizes (in the future) the expected Q-value. 
        Parameters:
        - rewards: (torch.Tensor) The reward of taking an action in a particular state.
        - gamma:   (float) The discount factor.
        - next_state: (torch.Tensor) The state we transitioned to from the current state.
        - predictQ: (torch.Tensor) The Q-values output by the target network.
        Returns:
        - loss: (torch.Tensor) The computed loss.
        """
        predictQ = self.forward(next_state)
        with torch.no_grad():
            maxQ_next = torch.max(predictQ, dim=1)[0] # Get the maximum Q-value from the next states
        
        Qtarget = rewards + gamma * maxQ_next # Expected future rewards
        criterion = nn.MSELoss()
        loss = criterion(predictQ, Qtarget)
        
        return loss

MAX_STATE_SIZE = 23  
action_space = [0, 1, 2, 3]
action_space_n = len(action_space)       
    
q_net = QNetwork(state_size=MAX_STATE_SIZE, action_size=action_space_n)
target_net = QNetwork(state_size=MAX_STATE_SIZE, action_size=action_space_n)
optimizer = optim.Adam(q_net.parameters(), lr=0.001)
    

class Agent():
    def __init__(self):
        self.epsilon = 16
        self.velocity = 0  # Agent initial speed
        self.acceleration = 0.5  # Acceleration/deceleration
        self.discount_factor_g = 0.9 # gamma or discount factor.
        self.state_vector = []
        self.max_score = 2000


    def select_action(self, state):
            
            # Convert state to torch tensor
            state = torch.FloatTensor(state)

            # Define the model in eval mode for determining actions
            q_net.eval()

            # Obtain Q values ​​predicted by the model for this state
            with torch.no_grad():
                q_values = q_net(state)
    
            # epsilon-greedy policy
            if np.random.rand() < self.epsilon:

                # Exploration: choose a random action
                action = np.random.choice(len(q_values))
            else:

                # Exploitation: choose the action with the maximum Q value
                action = torch.argmax(q_values).item()
            
            return action

    
    def update_velocity(self, action): 
        """ Update the velocity of the agent using the given action. """
        # Update speed based on action
        if action == 0:
            self.velocity = max(self.velocity - self.acceleration, -50)  # Movement to the left
        elif action == 2:
            self.velocity = min(self.velocity + self.acceleration, 50)  # Movement to the right
        else:
            # If no action is taken, reduce speed (drag)
            self.velocity = 0 if self.velocity == 0 else (self.velocity - np.sign(self.velocity) * self.acceleration)

        return self.velocity

    
    def compute_state_vector(particles, next_particle,):
        while True :
            state_vector = []
            unique_particles = []

            # Get the current particle
            current_particle = particles[-1] if particles else None

            # Position in x and radius of the current particle
            if current_particle and not any(p.pos[1] < current_particle.pos[1] for p in particles if p is not current_particle):
                state_vector.extend([current_particle.pos[0], current_particle.radius])
            else:
                state_vector.extend([0.0, 0.0])

            # Information about adjacent particles
            e = 5
            for particle in particles:
                
                if not any(
                    abs(p.pos[0] - particle.pos[0]) < e 
                    and abs(p.pos[1] - particle.pos[1]) < e
                    for p in particles if p is not particle
                ):
                    if len(state_vector) < MAX_STATE_SIZE:
                        state_vector.extend([particle.pos[0], particle.pos[1], particle.radius, int(any(p.pos[1] < particle.pos[1] for p in particles if p is not particle))])
                        unique_particles.append(particle)
                        
            # Next particle radius
            state_vector.append(next_particle.radius)

            if len(unique_particles) < 5:
                # If we have less than 5 unique particles, we complete with default values
                state_vector.extend([0.0]*((5-len(unique_particles))*4))

            state_vector = state_vector[:MAX_STATE_SIZE]  # Limit vector size to MAX_STATE_SIZE
            
        
            return np.array(state_vector)
        
    def reward(self, particles, score, fusion, game_over, prev_score, i, time_survived, PAD, WIDTH, action, best_score):

        reward = score / 100
        time_bonus = min(0.1 * time_survived, 10)

        # Reward for each fusion
        reward += fusion * 0.1

        # Penalty for particles close to edges (unless they have a small radius)
        for adjacent_particle in particles:
            if adjacent_particle.radius <= 25:
                if adjacent_particle.pos[0] == PAD[0] or adjacent_particle.pos[0] == WIDTH - PAD[0]:
                    reward -= 0.1

        # Reward for particles far from edges (unless they have a large radius)
        for adjacent_particle in particles:
            if adjacent_particle.radius > 25:
                if abs(PAD[0] - adjacent_particle.pos[0]) >= WIDTH / 4:
                    reward += 0.1


        # Reward for a score higher than the previous best score
        if score > best_score:
            reward  += 5

        # Reward for score above 1000
        if score > 1000:
            reward += 5
        
        # Bonus for survival time
        reward += time_bonus
        if score >= self.max_score: reward += 10

        if game_over == True: 
            reward -= 20 
            if score > prev_score: 
                reward += 5 
                if score <= prev_score: 
                    reward += -5

        prev_score = score

        return reward
    
    def update_target_network(self):
        target_net.load_state_dict(q_net.state_dict())
    
    def update_q_network(self, batch_state, batch_action, batch_reward, batch_next_state, batch_terminated):
        """
        Perform a gradient descent step on the Q-Network.
        Parameters:
        - batch_state: (torch.Tensor) A tensor containing all of the states in the current batch.
                    Shape is [batch size, num channels, height, width].
        - batch_action: (torch.LongTensor) An array where each element is an integer representing which
                        action was taken in that particular state. Shape is [batch size].
        - batch_reward: (torch.FloatTensor) An array where each element is a float representing the reward
                        given after taking the action in the corresponding state. Shape is [batch size].
        - batch_next_state: (torch.Tensor) A tensor containing all of the next states in the current batch.
                            Shape is same as `batch_state`.
        - batch_terminated: (torch.BoolTensor) An array where each element indicates whether or not the episode 
        """
        batch_next_q_values = target_net(batch_next_state) 
        batch_q_values = q_net(batch_state) 
        batch_next_maxQ = torch.max(batch_next_q_values, dim=1)[0]
        batch_reward = torch.stack(list(batch_reward), dim=0)
        batch_terminated_tensor = torch.cat(batch_terminated).squeeze()
        # Extract boolean values ​​from tensor
        batch_terminated_bool = batch_terminated_tensor.tolist()
        
        if batch_terminated_bool[0] == False:
            batch_terminated_bool = 0
        else:
            batch_terminated_bool = 1
        batch_q_target = batch_reward + self.discount_factor_g * batch_next_maxQ * (1 - batch_terminated_bool)
        
        #criterion = nn.SmoothL1Loss
        # We use here the MSE instead of Huber loss
        loss = F.mse_loss(batch_q_values[range(BATCH_SIZE), batch_action], batch_q_target)
        q_net.zero_grad() 
        loss.backward() 
        optimizer.step()
