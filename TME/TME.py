import pycxsimulator
import random
import matplotlib.pyplot as plt
import numpy as np


total_steps = 250  
initial_population = 0 
inject_population = 10  
division_rate_high_sial = 0.4  
division_rate_low_sial = 0.2 
death_rate = 0.02  
fusion_rate = 0.005  
suppressive_effect = 0.7 
permissive_effect = 1.8  
neutral_effect = 1.0  
mutation_rate = 0.01  


class Cell:
    def __init__(self, id, position, environment, sial_level):
        self.id = id
        self.position = position  
        self.environment = environment  
        self.sial_level = sial_level  

    def act(self):
        action = "none"
        division_rate = division_rate_high_sial if self.sial_level == 'high' else division_rate_low_sial
        effective_rate = division_rate * self.environment_effect()
        
        if random.random() < effective_rate:
            action = "divide"
        elif random.random() < death_rate:
            action = "die"
        elif random.random() < fusion_rate:
            action = "fuse"
        return action

    def mutate(self):
        
        if random.random() < mutation_rate:
            self.sial_level = 'high' if self.sial_level == 'low' else 'low'

    def environment_effect(self):
        if self.environment == 'suppressive':
            return suppressive_effect
        elif self.environment == 'permissive':
            return permissive_effect
        else:
            return neutral_effect


cells = []
cell_id_counter = 0
current_step = 0
history_high_sial = []
history_low_sial = []

def initialize():
    global cells, cell_id_counter, current_step, history_high_sial, history_low_sial
    cells = []
    cell_id_counter = 0
    current_step = 0
    history_high_sial = []
    history_low_sial = []

def inject_initial_cells():
    global cells, cell_id_counter
    for _ in range(inject_population):
        position = (random.uniform(0, 10), random.uniform(0, 10))
        environment = random.choice(['suppressive', 'permissive', 'neutral'])
        sial_level = random.choice(['high', 'low'])
        cells.append(Cell(cell_id_counter, position, environment, sial_level))
        cell_id_counter += 1

def step():
    global cells, cell_id_counter, current_step, history_high_sial, history_low_sial
    current_step += 1
    if current_step == 1:  
        inject_initial_cells()

    new_cells = []
    for cell in cells:
        cell.mutate()
        action = cell.act()
        if action == "divide":
            new_position = tuple(coord + random.uniform(-0.5, 0.5) for coord in cell.position)
            new_cells.append(Cell(cell_id_counter, new_position, cell.environment, cell.sial_level))
            cell_id_counter += 1
        elif action == "die":
            continue
        elif action == "fuse":
            cell.sial_level = random.choice(['high', 'low'])
        new_cells.append(cell)
    cells = new_cells

    
    high_sial_count = sum(1 for cell in cells if cell.sial_level == 'high')
    low_sial_count = sum(1 for cell in cells if cell.sial_level == 'low')
    history_high_sial.append(high_sial_count)
    history_low_sial.append(low_sial_count)

def draw():
    global cells, current_step, history_high_sial, history_low_sial
    
    plt.figure(figsize=(10, 5))
    plt.plot(history_high_sial, label='High a2-6Sia Cells', color='red')
    plt.plot(history_low_sial, label='Low a2-6Sia Cells', color='blue')
    plt.xlabel('Time (Days)')
    plt.ylabel('Cell Count')
    plt.title('Tumor Growth Dynamics Over Time')
    plt.yscale('log')  
    plt.legend()
    plt.show()

    
    plt.figure()
    colors = {
        ('high', 'suppressive'): 'red',
        ('high', 'permissive'): 'orange',
        ('high', 'neutral'): 'yellow',
        ('low', 'suppressive'): 'blue',
        ('low', 'permissive'): 'green',
        ('low', 'neutral'): 'purple'
    }
    for cell in cells:
        color = colors[(cell.sial_level, cell.environment)]
        plt.scatter(cell.position[0], cell.position[1], color=color, s=10)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'2D Tumor Cell Simulation - Day {current_step}')
    plt.show()

pycxsimulator.GUI().start(func=[initialize, step, draw])
