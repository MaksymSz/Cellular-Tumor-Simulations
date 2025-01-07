import numpy as np
from models.model import Model
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numba import cuda
import math
import time

@cuda.jit(device=True)
def get_neighbors(row, col, lattice_size):
    neighbors = cuda.local.array((8, 2), dtype=np.int32)
    count = 0
    for dx in range(-1,2):
        for dy in range(-1,2):
            if dx == 0 and dy == 0:
                continue
            nx = row + dx
            ny = col + dy
            if nx >= 0 and nx < lattice_size and ny >= 0 and ny < lattice_size:
                neighbors[count,0] = nx
                neighbors[count,1] = ny
                count += 1
    return neighbors, count

@cuda.jit(device=True)
def calculate_nutrient_concentration(C_nut, neighbors, count, I_nut):
    nutrient_sum = 0.0
    for i in range(count):
        nx = neighbors[i,0]
        ny = neighbors[i,1]
        nutrient_sum += C_nut[nx, ny]
    initial_sum = count * I_nut
    if initial_sum > 0:
        return nutrient_sum / initial_sum
    else:
        return 0.0

@cuda.jit(device=True)
def calculate_probabilities(lattice, neighbors, count, K_CC, K_NN, K_HH):
    cancer_count = 0
    necrotic_count = 0
    healthy_count = 0
    for i in range(count):
        nx = neighbors[i,0]
        ny = neighbors[i,1]
        state = lattice[nx, ny]
        if state == 1:
            cancer_count += 1
        elif state == 2:
            necrotic_count += 1
        else:
            healthy_count += 1

    E_cancer   = -K_CC * cancer_count
    E_necrotic = -K_NN * necrotic_count
    E_healthy  = -K_HH * healthy_count

    exp_cancer = math.exp(-E_cancer)
    exp_necrotic = math.exp(-E_necrotic)
    exp_healthy = math.exp(-E_healthy)
    total_exp = exp_cancer + exp_necrotic + exp_healthy

    P_quiescent = exp_cancer / total_exp
    P_necrosis = exp_necrotic / total_exp
    P_apoptosis = exp_healthy / total_exp

    if (exp_cancer + exp_healthy) > 0 and cancer_count > 0:
        P_proliferation = exp_cancer / (exp_cancer + exp_healthy)
    else:
        P_proliferation = 0.0

    return P_quiescent, P_necrosis, P_apoptosis, P_proliferation

@cuda.jit
def update_lattice_kernel(lattice, C_nut, new_lattice, random_vals,
                          lattice_size, I_nut, K_CC, K_HH, K_NN):
    row, col = cuda.grid(2)
    if row < lattice_size and col < lattice_size:
        cell = lattice[row, col]
        neighbors, count = get_neighbors(row, col, lattice_size)
        N_conc = calculate_nutrient_concentration(C_nut, neighbors, count, I_nut)
        P_quiescent, P_necrosis, P_apoptosis, P_proliferation = calculate_probabilities(lattice, neighbors, count, K_CC, K_NN, K_HH)

        r = random_vals[row, col]

        if cell == 1:
            new_val = C_nut[row, col] - 6.0
            if new_val < 0.0:
                new_val = 0.0
            C_nut[row, col] = new_val

            if N_conc >= 0.5:
                if r <= P_quiescent:
                    new_lattice[row, col] = 1
                elif r <= P_quiescent + P_necrosis:
                    new_lattice[row, col] = 2
                else:
                    new_lattice[row, col] = 0
            else:
                new_lattice[row, col] = 2

        elif cell == 0:  
            new_val = C_nut[row, col] - 3.0
            if new_val < 0.0:
                new_val = 0.0
            C_nut[row, col] = new_val

            if N_conc >= 0.5:
                if r <= P_proliferation: 
                    new_lattice[row, col] = 1
                else:
                    new_lattice[row, col] = 0
            else:
                new_lattice[row, col] = 0
        else:
            new_lattice[row, col] = 2
    



class GhaemiModel(Model):
    def __init__(self, lattice_size, params):
        self.lattice_size = lattice_size
        self.params = params
        self.lattice = None
        self.c_nut = None
        self.cur_step = 0
        self.HEALTHY = 0
        self.CANCEROUS = 1
        self.NECROTIC = 2
        self.cancers_at_time_step = []
        self.necrotic_at_time_step = []
        self.healthy_at_time_step = []
        self.sum_of_nutrients_at_time_step = []
        self._init_matrices()

    def reset(self):
        self._init_matrices()
        self.cur_step = 0

    def make_step(self):
        self.log_data()
        start = time.time()
        random_vals = np.random.random((self.lattice_size, self.lattice_size))
        
        d_lattice = cuda.to_device(self.lattice)
        d_c_nut = cuda.to_device(self.c_nut)
        d_new_lattice = cuda.to_device(self.lattice.copy())
        d_random_vals = cuda.to_device(random_vals)

        threadsperblock = (16, 16)
        blockspergrid_x = (self.lattice_size + (threadsperblock[0] - 1)) // threadsperblock[0]
        blockspergrid_y = (self.lattice_size + (threadsperblock[1] - 1)) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        update_lattice_kernel[blockspergrid, threadsperblock](
            d_lattice, d_c_nut, d_new_lattice, d_random_vals,
            self.lattice_size, self.params['I_nut'],
            self.params['K_CC'], self.params['K_HH'], self.params['K_NN']
        )

        self.lattice = d_new_lattice.copy_to_host()
        self.c_nut = d_c_nut.copy_to_host()
        end = time.time()


        self._nutrient_diffusion()
        self.cur_step += 1

        self.cancers_at_time_step.append(np.sum(self.lattice == self.CANCEROUS))
        self.necrotic_at_time_step.append(np.sum(self.lattice == self.NECROTIC))
        self.healthy_at_time_step.append(np.sum(self.lattice == self.HEALTHY))
        self.sum_of_nutrients_at_time_step.append(np.sum(self.c_nut))

     
        

    def plot_lattice(self):
        start = time.time()
        plt.clf()
        cmap = ListedColormap(['green', 'red', 'black'])
        plt.imshow(self.lattice, cmap=cmap, vmin=0, vmax=2, origin='upper')
        plt.colorbar(ticks=[0, 1, 2], label='Cell State') 
        plt.title("Tumor Growth Simulation, Step {}".format(self.cur_step))
        plt.pause(0.01)
        end = time.time()

    def plot_lattice_nutrients(self):
        start = time.time()
        plt.clf()
        # Plot nutrient concentration (0 to 400)
        plt.imshow(self.c_nut, cmap='viridis', vmin=0, vmax=400, origin='upper')
        cbar = plt.colorbar()
        cbar.set_label('Nutrient Concentration')
        plt.title("Nutrient Distribution, Step {}".format(self.cur_step))
        plt.pause(0.01)
        end = time.time()

    def _init_matrices(self):
        self.lattice = np.zeros((self.lattice_size, self.lattice_size), dtype=np.uint8)
        self.c_nut = np.full((self.lattice_size, self.lattice_size), self.params['I_nut'], dtype=np.float64)
        self.lattice[self.lattice_size // 2, self.lattice_size // 2] = self.CANCEROUS
        self.lattice[self.lattice_size // 2 + 1, self.lattice_size // 2] = self.CANCEROUS
        self.lattice[self.lattice_size // 2, self.lattice_size // 2 + 1] = self.CANCEROUS
        self.lattice[self.lattice_size // 2 + 1, self.lattice_size // 2 + 1] = self.CANCEROUS

    def _nutrient_diffusion(self):
        for _ in range(self.params['k']):
            self._diffuse_nutrients()

    def _diffuse_nutrients(self):
        block_size = int(self.lattice_size / 15)
        rows, cols = self.c_nut.shape
        if rows < block_size:
            raise ValueError("Lattice dimensions must be at least the size of a single block.")

        self.c_nut[0:block_size, :] = self.params["I_nut"] * np.random.uniform(0.9, 1, size=(block_size, cols))
        new_nut = self.c_nut.copy()
        step_size = block_size // 2  
        for row in range(0, rows, step_size):
            row_start = max(row - block_size // 2, 0)
            row_end = min(row + block_size // 2, rows)

            
            block = new_nut[row_start:row_end, :]
            block_mean = np.mean(block)

            new_nut[row_start:row_end, :] = block_mean

        self.c_nut = new_nut

    def plot_simulation(self):
        plt.figure(figsize=(10, 6))  
        plt.plot(self.cancers_at_time_step, label='Cancerous')
        plt.plot(self.necrotic_at_time_step, label='Necrotic')
        plt.plot(self.healthy_at_time_step, label='Healthy')
        plt.grid()
        plt.xlabel('Time Steps')
        plt.ylabel('Number of Cells')
        plt.legend()
        plt.title('Cell Population Over Time')
        plt.show()
        plt.savefig('plots/cells.png')

        plt.figure(figsize=(10, 6))  
        plt.plot(self.sum_of_nutrients_at_time_step)
        plt.xlabel('Time Steps')
        plt.ylabel('Sum of Nutrient Concentrations')
        plt.grid()
        plt.ylim(bottom=0)
        plt.title('Nutrient Concentration Over Time')
        plt.show()  
        plt.savefig('plots/nutrients.png')
    
    def log_data(self):
        if self.cur_step in {0, 50, 100, 200, 300, 400}:
            print(f"Time step: {self.cur_step}")
            print(f"Number of cancerous cells: {np.sum(self.lattice == self.CANCEROUS)}")
            print(f"Number of necrotic cells: {np.sum(self.lattice == self.NECROTIC)}")
            print(f"Number of healthy cells: {np.sum(self.lattice == self.HEALTHY)}")
            print(f"Sum of nutrient concentrations: {np.sum(self.c_nut)}")

