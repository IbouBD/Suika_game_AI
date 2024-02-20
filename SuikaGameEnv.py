from agent import Agent, QNetwork, ReplayMemory
from SuikaGame import*

    
def env(max_score, max_episode, terminated):
    best_score = 0
    rewards = 0
    particles = []
    next_particle = PreParticle(WIDTH//2, rng.integers(0, 5))

    state_vector = GetState(particles, next_particle)
    
    terminated=False

    memory_capacity = 10000
        # Initialisez la mémoire de relecture
    memory = ReplayMemory(capacity=memory_capacity)
    model = QNetwork(state_size=len(state_vector), action_size=4)
    
    AGENT = Agent(model, memory)

    epsilon = AGENT.epsilon
    for i in range(max_episode):
        
        if terminated == True: break

        # Initialisation du jeu
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))

        # Initialisation de l'état initial du jeu 
        pygame.display.set_caption("PySuika DQL")
        clock = pygame.time.Clock()
        pygame.font.init()
        clock = pygame.time.Clock()

        # Main game loop
        game_over = False
        
        scorefont = pygame.font.SysFont("monospace", 32)
        overfont = pygame.font.SysFont("monospace", 72)
        
        space = pymunk.Space()
        space.gravity = (0, GRAVITY)
        space.damping = DAMPING
        space.collision_bias = BIAS
            
            # Walls
        pad = 20
        left = Wall(A, B, space)
        bottom = Wall(B, C, space)
        right = Wall(C, D, space)
        walls = [left, bottom, right]

            # List to store particles
        wait_for_next = 0
        next_particle = PreParticle(WIDTH//2, rng.integers(0, 5))
        particles = []

            # Collision Handler
        handler = space.add_collision_handler(1, 1)
        def collide(arbiter, space, data):
            sh1, sh2 = arbiter.shapes
            _mapper = data["mapper"]
            pa1 = _mapper[sh1]
            pa2 = _mapper[sh2]
            cond = bool(pa1.n != pa2.n)
            pa1.has_collided = cond
            pa2.has_collided = cond
            if not cond:
                new_particle = resolve_collision(pa1, pa2, space, data["particles"], _mapper)
                data["particles"].append(new_particle)
                data["score"] += POINTS[pa1.n]
                data["fusion"] += 1
            return cond

        handler.begin = collide
        handler.data["mapper"] = shape_to_particle
        handler.data["particles"] = particles
        handler.data["score"] = 0
        handler.data["fusion"] = 0
      
        episode = i+1
        print("Beginning of Episode n°", episode)
        
        while terminated == False:
           
            
            state = Agent.compute_state_vector(q=0, MAX_SIZE=23, particles=particles, next_particle=next_particle)

            action = select_action(model, state, training=True, epsilon=epsilon)

            
           

            for event in pygame.event.get():
                if any([
                        event.type == pygame.QUIT,
                        event.type == pygame.KEYDOWN and event.key in [pygame.K_q, pygame.K_ESCAPE],
                ]):
                    pygame.quit()
                    sys.exit()
            

            if action == 3 and wait_for_next == 0:
                particles.append(next_particle.release(space, shape_to_particle))
                wait_for_next = NEXT_DELAY                
            
            velocity = AGENT.update_velocity(action=action)

            # Mise à jour progressive de la position
            next_particle.set_x(next_particle.x + velocity)

            # Restreindre la position dans les limites de l'environnement
            next_particle.set_x(np.clip(next_particle.x, 0, WIDTH - 1))
            if wait_for_next > 1:
                wait_for_next -= 1
            elif wait_for_next == 1:
                next_particle = PreParticle(next_particle.x, rng.integers(0, 5))
                wait_for_next -= 1

            # Draw background and particles
            screen.fill(BG_COLOR)
            if wait_for_next == 0:
                next_particle.draw(screen)

                next_state = Agent.compute_state_vector(q=0, MAX_SIZE=23, particles=particles, next_particle=next_particle)

                yield state, rewards, terminated, episode, action, next_state
                
            for w in walls:
                w.draw(screen)
            for p in particles:
                pos = p.pos[0] / WIDTH
                p.draw(screen)

                if p.pos[1] < PAD[1] and p.has_collided:
                    label = overfont.render("Game Over!", 1, (0, 0, 0))
                    screen.blit(label, PAD)
                    game_over = True
                    print(f"Fail, final score: {score}")
                    current_score = score
                    filename = 'scores.json'
                    append_to_json_file(filename, current_score)

            score = handler.data['score']
            current_score = score

            loaded_scores = load_scores()
            
            if isinstance(loaded_scores, list):
                
                if loaded_scores:
                    prev_score = loaded_scores[-1]
                    
                else:
                    print("No previous scores available.")
                    prev_score = [0]
            else:
                print("Invalid scores format. Expecting a list.")
                prev_score = [0]

        
            rewards = AGENT.reward(
                fusion = handler.data["fusion"], 
                max_score=AGENT.max_score,
                prev_score=prev_score, 
                game_over= game_over,
                particles=particles,
                best_score = best_score, 
                time_survived=1,
                action=action, 
                WIDTH=WIDTH,
                score=score,
                PAD=PAD, 
                i=i
                )
            
            if game_over:
                # Gestion de fin de jeu
                for particle in particles:
                    particle.kill(particles=particles, space=space)
                particles = []
                walls = []
                next_particle = PreParticle(next_particle.x, rng.integers(0, 5))
                wait_for_next = 0
                game_over = False
                score = 0
                print('Rewards :',rewards)
                epsilon *= 0.99
                AGENT.epsilon = epsilon
                print('Epslion : ',AGENT.epsilon)
                handler._reset
                break
            
            if score >= max_score:
                print(f"Succes, final score: {score}")
                current_score = score
                filename = 'scores.json'
                append_to_json_file(filename, current_score)
                terminated = True
                print('Rewards :',rewards)
                break

            if current_score > best_score: 
                best_score = current_score

            
            label = scorefont.render(f"Score: {score}", 1, (0, 0, 0))
            label3 = scorefont.render(f"Best Score: {best_score}", 1, (0, 0, 0))
            label2 = scorefont.render(f"Episode: {episode}", 1, (0, 0, 0))
            screen.blit(label, (10, 10))
            screen.blit(label3, (10, 50))
            screen.blit(label2, (10, 90))

            i +=1

            space.step(1/FPS)
            pygame.display.update()
            clock.tick(FPS)
        
    return state