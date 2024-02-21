from SuikaGameEnv import*
from agent import*

if __name__ == "__main__":
    
    ENV = env(max_episode=607, max_score=2000, epsilon_decay=0.99, terminated=False, speed_up=True)
    memory_capacity = 180000
    memory = ReplayMemory(capacity=memory_capacity)
    agent = Agent()

    for state, rewards, terminated, episode, action, next_state in ENV:

        state = torch.tensor(state)
        action_t = torch.tensor(action)
        rewards = torch.tensor(rewards)
        next_state_t = torch.tensor(next_state)
        terminated = torch.tensor([terminated])
        memory.push(state, action_t, rewards, next_state_t, terminated, episode)
        
        state = next_state 
        if len(memory) >= memory_capacity:

            batch_state, batch_action, batch_reward, batch_next_state, batch_terminated = memory.sample(BATCH_SIZE) 
            agent.update_q_network(batch_state, batch_action, batch_reward, batch_next_state, batch_terminated)
            

        if episode % 100 == 0: 
            agent.update_target_network()