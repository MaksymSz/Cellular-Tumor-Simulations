import pycxsimulator
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import tkinter as tk
from tkinter import simpledialog

# Default Simulation Parameters
parameters = {
    "total_steps": 200,
    "initial_population": 50,
    "division_rate_high_sial": 0.3,
    "division_rate_low_sial": 0.1,
    "death_rate": 0.05,
    "fusion_rate": 0.02,
    "suppressive_effect": 0.5,
    "permissive_effect": 1.5,
    "neutral_effect": 1.0,
    "mutation_rate": 0.02,
}

# Cell and TME Properties
class Cell:
    def __init__(self, id, position, environment, sial_level):
        self.id = id
        self.position = position  # (x, y) coordinates
        self.environment = environment  # 'suppressive', 'permissive', or 'neutral'
        self.sial_level = sial_level  # 'high' or 'low' a2-6Sia expression

    def act(self):
        action = "none"
        division_rate = parameters["division_rate_high_sial"] if self.sial_level == 'high' else parameters["division_rate_low_sial"]
        effective_rate = division_rate * self.environment_effect()
        
        if random.random() < effective_rate:
            action = "divide"
        elif random.random() < parameters["death_rate"]:
            action = "die"
        elif random.random() < parameters["fusion_rate"]:
            action = "fuse"
        return action

    def mutate(self):
        if random.random() < parameters["mutation_rate"]:
            self.sial_level = 'high' if self.sial_level == 'low' else 'low'

    def environment_effect(self):
        if self.environment == 'suppressive':
            return parameters["suppressive_effect"]
        elif self.environment == 'permissive':
            return parameters["permissive_effect"]
        else:
            return parameters["neutral_effect"]

# Simulation Setup
cells = []
cell_id_counter = 0
current_step = 0

def initialize():
    global cells, cell_id_counter, current_step
    cells = []
    cell_id_counter = 0
    current_step = 0
    for _ in range(int(parameters["initial_population"])):  # Ensure integer conversion here
        position = (random.uniform(0, 10), random.uniform(0, 10))
        environment = random.choice(['suppressive', 'permissive', 'neutral'])
        sial_level = random.choice(['high', 'low'])
        cells.append(Cell(cell_id_counter, position, environment, sial_level))
        cell_id_counter += 1

def step():
    global cells, cell_id_counter, current_step
    new_cells = []
    current_step += 1
    for cell in cells:
        cell.mutate()
        action = cell.act()
        if action == "divide":
            new_position = tuple(max(0, min(10, coord + random.uniform(-0.5, 0.5))) for coord in cell.position)
            new_cells.append(Cell(cell_id_counter, new_position, cell.environment, cell.sial_level))
            cell_id_counter += 1
        elif action == "die":
            continue  # Cell dies, do not append it to the new list
        elif action == "fuse":
            cell.sial_level = random.choice(['high', 'low'])
        new_cells.append(cell)
    cells = new_cells

def draw():
    global cells, current_step
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
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='High-Suppressive', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='High-Permissive', markerfacecolor='orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='High-Neutral', markerfacecolor='yellow', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Low-Suppressive', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Low-Permissive', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Low-Neutral', markerfacecolor='purple', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.show()

# GUI for Parameter Adjustment
def adjust_parameters():
    for key in parameters:
        new_value = simpledialog.askfloat("Parameter Adjustment", f"Set value for {key} (current: {parameters[key]}):")
        if new_value is not None:
            # Cast to int if the parameter should be an integer
            if key in ["initial_population", "total_steps"]:
                parameters[key] = int(new_value)
            else:
                parameters[key] = new_value

# GUI Setup
root = tk.Tk()
root.withdraw()  # Hide the main window
if tk.messagebox.askyesno("Adjust Parameters", "Do you want to adjust simulation parameters?"):
    adjust_parameters()

pycxsimulator.GUI().start(func=[initialize, step, draw])
