import sys
import numpy as np
import pygame
import pymunk
import json
from PIL import Image
from agent import Agent, QNetwork, ReplayMemory
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


pygame.init()

rng = np.random.default_rng()

# Constants
SIZE = WIDTH, HEIGHT = np.array([570, 770])
PAD = (24, 160)
A = (PAD[0], PAD[1])
B = (PAD[0], HEIGHT - PAD[0])
C = (WIDTH - PAD[0], HEIGHT - PAD[0])
D = (WIDTH - PAD[0], PAD[1])
BG_COLOR = (250, 240, 148)
W_COLOR = (250, 190, 58)
COLORS = [
    (245, 0, 0),
    (250, 100, 100),
    (150, 20, 250),
    (250, 210, 10),
    (250, 150, 0),
    (245, 0, 0),
    (250, 250, 100),
    (255, 180, 180),
    (255, 255, 0),
    (100, 235, 10),
    (0, 185, 0),
]
FPS = 240
RADII = [17, 25, 32, 38, 50, 63, 75, 87, 100, 115, 135]
THICKNESS = 14
DENSITY = 0.002
ELASTICITY = 0.1
IMPULSE = 10000
GRAVITY = 2000
DAMPING = 0.75
NEXT_DELAY = FPS
BIAS = 0.00001
POINTS = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66]
BATCH_SIZE = 128
shape_to_particle = dict()


class Particle:
    def __init__(self, pos, n, space, mapper):
        self.n = n % 11
        self.radius = RADII[self.n]
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = tuple(pos)
        self.shape = pymunk.Circle(body=self.body, radius=self.radius)
        self.shape.density = DENSITY
        self.shape.elasticity = ELASTICITY
        self.shape.collision_type = 1
        self.shape.friction = 0.2
        self.has_collided = False
        mapper[self.shape] = self
        self.name = f"Particle_{self.n}"
        self.fusion_count = 0
        

        space.add(self.body, self.shape)
        self.alive = True
        #print(f"{self.name} created {self.shape.elasticity=}")

    def draw(self, screen):
        if self.alive:
            global c1
            c1 = np.array(COLORS[self.n])
            c2 = (c1 * 0.8).astype(int)
            pygame.draw.circle(screen, tuple(c2), self.body.position, self.radius)
            pygame.draw.circle(screen, tuple(c1), self.body.position, self.radius * 0.9)

        
    def kill(self, space): 
        space.remove(self.body, self.shape) 
        self.alive = False 
        particles.remove(self)
        self.fusion_count += 1
        #print(f"Particle {id(self)} killed")


    @property
    def pos(self):
        return np.array(self.body.position)
    
class PreParticle:
    def __init__(self, x, n):
        self.n = n % 11
        self.name = f"PreParticle_{self.n}"
        self.radius = RADII[self.n]
        self.x = x
        #print(f"PreParticle {(self.name)} created")

    def draw(self, screen):
        c1 = np.array(COLORS[self.n])
        c2 = (c1 * 0.8).astype(int)
        pygame.draw.circle(screen, tuple(c2), (self.x, PAD[1] // 2), self.radius)
        pygame.draw.circle(screen, tuple(c1), (self.x, PAD[1] // 2), self.radius * 0.9)

    def set_x(self, x):
        lim = PAD[0] + self.radius + THICKNESS // 2
        self.x = np.clip(x, lim, WIDTH - lim)
    
        
    def release(self, space, mapper):
        return Particle((self.x, PAD[1] // 2), self.n, space, mapper)

class Wall:
    thickness = THICKNESS

    def __init__(self, a, b, space):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Segment(self.body, a, b, self.thickness // 2)
        self.shape.friction = 10
        space.add(self.body, self.shape)
        #print(f"wall {self.shape.friction=}")


    def draw(self, screen):
        pygame.draw.line(screen, W_COLOR, self.shape.a, self.shape.b, self.thickness)

def resolve_collision(p1, p2, space, particles, mapper):
    
    
    if p1.n == p2.n:
        distance = np.linalg.norm(p1.pos - p2.pos)
        if distance < 2 * p1.radius:
            p1.kill(space)
            p2.kill(space)
            
            pn = Particle(np.mean([p1.pos, p2.pos], axis=0), p1.n+1, space, mapper)
            pn.fusion_count = p1.fusion_count + p2.fusion_count  
            for p in particles:
                if p.alive:
                    vector = p.pos - pn.pos
                    distance = np.linalg.norm(vector)
                    if distance < pn.radius + p.radius:
                        impulse = IMPULSE * vector / (distance ** 2)
                        p.body.apply_impulse_at_local_point(tuple(impulse))
                        #print(f"{impulse=} was applied to {id(p)}")
                        
            return pn
    
    return None

def save_scores(score):
    try:
        # Charger les scores existants
        loaded_scores = load_scores()

        # Ajouter le nouveau score
        loaded_scores.append(score)

        # Sauvegarder la liste mise à jour dans le fichier
        with open('scores.json', 'w') as file:
            json.dump(loaded_scores, file)
    except Exception as e:
        # Gérer les erreurs lors de la sauvegarde
        print(f"Error saving score: {e}")

def load_scores():
    try:
        with open('scores.json', 'r') as file:
            scores = json.load(file)
        return scores
    except (FileNotFoundError, json.JSONDecodeError):
        # Gérer le cas où le fichier n'existe pas ou n'est pas un JSON valide
        return []
def clear_json_file(filename):
    with open(filename, 'w') as f:
        f.truncate(0)
def append_to_json_file(filename, new_data):
    try:
        # Charger le fichier JSON existant
        with open(filename, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        # Si le fichier n'existe pas, créer une liste vide
        data = []

    # Ajouter de nouvelles données à la liste
    data.append(new_data)

    # Écrire la liste mise à jour dans le fichier JSON
    with open(filename, 'w') as file:
        json.dump(data, file)



epsilon = 1.5
terminated = False

for i in range(500):
    
    if terminated == True: break

    # Initialisation du jeu
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # Initialisation de l'état initial du jeu 
    pygame.display.set_caption("PySuika DQN")
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


    state_vector = (Agent.compute_state_vector(particles, next_particle, q=0, MAX_SIZE=23))

    state = state_vector

    memory_capacity = 10000
    # Initialisez la mémoire de relecture
    memory = ReplayMemory(capacity=memory_capacity)
    model = QNetwork(state_size=len(state_vector), action_size=4)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    AGENT = Agent(model, memory, epsilon)

    max_score = AGENT.max_score
    
    def reward(particles, score, terminated, fusion, game_over, prev_score, state_vector):

        reward = score
        # Récompense pour chaque fusion
        reward += 20 * fusion
        
        # Récompense en fonction de la position des petites particules
        
        adjacent_particle = particles[-i - 1] if len(particles) > i else None
            #distance_to_edge = PAD[1] - state_vector[-3]
        for adjacent_particle in particles: 
            if adjacent_particle.radius <= 25 and (adjacent_particle.pos[0] == PAD[0]
                                                    or adjacent_particle.pos[0] == WIDTH - PAD[0]):
                reward += 10

        # Punir l'inaction
        if action == 1:
            reward -= 10

        # Récompense si partie terminée (score maximal atteint)    
        if score >= max_score:
            reward += 100
        
        # pénalité en cas de game over
        if game_over == True:  
            reward -= 100
            # Récompense en fonction du score
            if score > prev_score:
                reward += 10
            if score <= prev_score:
                reward += -10
            

        prev_score = score
        reward = Agent.normalize_reward(reward)
        
        return reward

    print("Beginning of Episode n°", i+1)

    print(f'Episode {i+1}, Epsilon : {epsilon}')
    print(model)

    
    def select_action(state):
        # Convertir l'état en tenseur torch
        state = torch.FloatTensor(state)

        # Mettre le modèle en mode eval pour la détermination des actions
        model.eval()

        # Obtenir les valeurs Q prédites par le modèle
        with torch.no_grad():
            q_values = model(state)

        # Politique epsilon-greedy
        if np.random.rand() < epsilon:
            # Exploration : choisir une action aléatoire
            action = np.random.choice(len(q_values))
        else:

            # Exploitation : choisir l'action avec la valeur Q maximale
            action = torch.argmax(q_values).item()
        # Mettre le modèle en mode train pour l'apprentissage
        model.train()
        return action


    while terminated == False:
        
        for event in pygame.event.get():
            if any([
                    event.type == pygame.QUIT,
                    event.type == pygame.KEYDOWN and event.key in [pygame.K_q, pygame.K_ESCAPE],
            ]):
                pygame.quit()
                sys.exit()
        current_state = state
        
        if wait_for_next == 0:
                current_state = (Agent.compute_state_vector(particles, next_particle, q=0, MAX_SIZE=23))
                #print(f'current state : {current_state}')
                pass
        
        
        action = select_action(current_state)

        next_state = current_state
        

        if action == 3 and wait_for_next == 0:
            particles.append(next_particle.release(space, shape_to_particle))
            wait_for_next = NEXT_DELAY
            
        
        velocity = AGENT.update_velocity(action)

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
            next_state = (Agent.compute_state_vector(particles, next_particle, q=0, MAX_SIZE=23))
            #print(f'new state :{next_state}')
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

        

        if game_over:
            # Gestion de fin de jeu
            for particle in particles:
                particle.kill
            particles = []
            walls = []
            next_particle = PreParticle(next_particle.x, rng.integers(0, 5))
            wait_for_next = 0
            game_over = False
            score = 0
            print('Rewards :',rewards)
            epsilon *= 0.99
            break
        
        if score >= max_score:
            print(f"Succes, final score: {score}")
            current_score = score
            filename = 'scores.json'
            append_to_json_file(filename, current_score)
            terminated = True
            print('Rewards :',rewards)
            break
            
        loaded_scores = load_scores()
        # Vérifier si loaded_scores est une liste
        if isinstance(loaded_scores, list):
            # Si oui, accéder au dernier score
            if loaded_scores:
                prev_score = loaded_scores[-1]
                
            else:
                print("No previous scores available.")
                prev_score = [0]
        else:
            print("Invalid scores format. Expecting a list.")
            prev_score = [0]

        rewards = reward(score = score, terminated = terminated, fusion = handler.data["fusion"], prev_score=prev_score, game_over= game_over, state_vector=state_vector, particles=particles)
        #print(f"Episode : {i+1}, Score : {score}, Rewards : {rewards}, state : {state}")


        if len(memory) >= BATCH_SIZE:
            experiences = AGENT.memory.sample(BATCH_SIZE)
        else:
            experiences = memory
        state_r = np.array(current_state)
        action_r = np.array(action)
        reward_r = np.array([[reward]])
        next_state_r = np.vstack(next_state)
        terminated_r = np.array([terminated])

         # Sauvegarder la transition dans la mémoire de relecture
        memory.push(state_r, action_r, reward_r, next_state_r, terminated_r, episode=i+1)

        i += 1
        # Former le modèle toutes les 100 étapes
        if i % 100 == 0:
            model.train()

        # Mettre à jour l'état actuel
        current_state = next_state
        AGENT.epsilon = epsilon
       
        
        space.step(1/FPS)
        pygame.display.update()
        data = pygame.image.tostring(pygame.display.get_surface(), 'RGB')
        img = Image.frombytes('RGB', tuple(SIZE), data)
        clock.tick(FPS)