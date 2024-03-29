"""Environment based on github.com/Ole-Batting/suika"""
import numpy as np
import pygame
import pymunk
import json
import torch
import sys
from agent import*

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


rng = np.random.default_rng()

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
        

    def draw(self, screen):
        if self.alive:
            global c1
            c1 = np.array(COLORS[self.n])
            c2 = (c1 * 0.8).astype(int)
            pygame.draw.circle(screen, tuple(c2), self.body.position, self.radius)
            pygame.draw.circle(screen, tuple(c1), self.body.position, self.radius * 0.9)

        
    def kill(self, space, particles):
        space.remove(self.body, self.shape)
        self.alive = False
        if self in particles:
            particles.remove(self)
        else:
            pass
        self.fusion_count += 1


    @property
    def pos(self):
        return np.array(self.body.position)
    
class PreParticle:
    def __init__(self, x, n):
        self.n = n % 11
        self.name = f"PreParticle_{self.n}"
        self.radius = RADII[self.n]
        self.x = x

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
            p1.kill(space, particles)
            p2.kill(space, particles)
            
            pn = Particle(np.mean([p1.pos, p2.pos], axis=0), p1.n+1, space, mapper)
            pn.fusion_count = p1.fusion_count + p2.fusion_count  
            for p in particles:
                if p.alive:
                    vector = p.pos - pn.pos
                    distance = np.linalg.norm(vector)
                    if distance < pn.radius + p.radius:
                        impulse = IMPULSE * vector / (distance ** 2)
                        p.body.apply_impulse_at_local_point(tuple(impulse))
                        
            return pn
    
    return None

def save_scores(score):
    try:
        loaded_scores = load_scores()

        loaded_scores.append(score)

        with open('scores.json', 'w') as file:
            json.dump(loaded_scores, file)
    except Exception as e:
        print(f"Error saving score: {e}")

def load_scores():
    try:
        with open('scores.json', 'r') as file:
            scores = json.load(file)
        return scores
    except (FileNotFoundError, json.JSONDecodeError):
        
        return []
def clear_json_file(filename):
    with open(filename, 'w') as f:
        f.truncate(0)
def append_to_json_file(filename, new_data):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        
        data = []

    data.append(new_data)

    with open(filename, 'w') as file:
        json.dump(data, file)

def GetState(particles, next_particle):
    state = Agent.compute_state_vector(particles, next_particle)
    return np.array(state)
