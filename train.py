from SuikaGameEnv import*
from agent import*


if __name__ == "__main__":
    
    ENV = env(max_episode=300, max_score=20000, terminated=False)
    memory_capacity = 2000
    # Initialisez la mémoire de relecture
    memory = ReplayMemory(capacity=memory_capacity)
    model = QNetwork(state_size=23, action_size=4)
    agent = Agent(model, memory)

    # Boucle pour obtenir l'état en temps réel
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
            model.update_q_network(batch_state, batch_action, batch_reward, batch_next_state, batch_terminated)


        if episode % 50 == 0: 
            model.update_target_network()