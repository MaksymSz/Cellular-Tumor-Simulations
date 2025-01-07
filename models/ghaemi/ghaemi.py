import numpy as np
from models.model import Model
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class GhaemiModel(Model):
    """
    A probabilistic cellular automaton model for tumor growth simulation.

    Parameters:
    - lattice_size (int): The size of the 2D lattice representing the tissue.
    - params (dict): A dictionary containing the following keys:
        - K_CC (float): Adhesion coefficient for cancerous cells.
        - K_HH (float): Adhesion coefficient for healthy cells.
        - K_NN (float): Adhesion coefficient for necrotic cells.
        - I_nut (float): Initial nutrient concentration for all cells.
        - k (int): Number of nutrient diffusion iterations per time step.

    Attributes:
    - lattice (np.ndarray): 2D array representing the cell states on the lattice.
    - C_nut (np.ndarray): 2D array representing the nutrient concentration at each cell.
    - HEALTHY (int): State value for healthy cells.
    - CANCEROUS (int): State value for cancerous cells.
    - NECROTIC (int): State value for necrotic cells.
    - cancers_at_time_step (list): Number of cancerous cells at each time step.
    - necrotic_at_time_step (list): Number of necrotic cells at each time step.
    - healthy_at_time_step (list): Number of healthy cells at each time step.
    - sum_of_nutrients_at_time_step (list): Sum of nutrient concentrations at each time step.

    Methods:
    - reset(): Resets the lattice and nutrient matrix to their initial states.
    - make_step(): Advances the simulation by one time step.
    - plot_lattice(): Visualizes the current state of the lattice.
    """
    def __init__(self, lattice_size, params):
        self.lattice_size = lattice_size
        self.params = params
        self.lattice = None
        self.c_nut = None
        self.cur_step = 0
        self.HEALTHY = 0
        self.CANCEROUS = 1
        self.NECROTIC = 2
        self.cancers_at_time_step = [4]
        self.necrotic_at_time_step = [0]
        self.healthy_at_time_step = [self.lattice_size ** 2 - 4]
        self.sum_of_nutrients_at_time_step = [self.lattice ** 2 * self.params['I_nut']]
        self._init_matrices()



    def reset(self):
        self._plot_simulation()
        self._init_matrices()

    def make_step(self):
        new_lattice = self.lattice.copy()

        for row in range(self.lattice_size):
            for col in range(self.lattice_size):
                cell = self.lattice[row, col]
                neighbors = self._get_neighbors(row, col)
                N_conc = self._calculate_nutrient_concentration(neighbors)
                P = self._calculate_probabilities(neighbors)

                if cell == self.CANCEROUS:
                    self.C_nut[row, col] = max(self.C_nut[row, col] - 6, 0)
                    if N_conc >= 0.5:
                        r = np.random.random()
                        if r <= P[0]: 
                            continue
                        elif r <= P[0] + P[1]:  
                            new_lattice[row, col] = self.NECROTIC
                        else:  
                            new_lattice[row, col] = self.HEALTHY
                    else:
                        new_lattice[row, col] = self.NECROTIC

                elif cell == self.HEALTHY:
                    self.C_nut[row, col] = max(self.C_nut[row, col] - 3, 0)
                    if N_conc >= 0.5 and np.random.random() <= P[3]: 
                        new_lattice[row, col] = self.CANCEROUS

        self.C_nut = self._nutrient_diffusion(self.C_nut)
        self.lattice = new_lattice
        self.cur_step += 1

        self.cancers_at_time_step.append(np.sum(self.lattice == self.CANCEROUS))
        self.necrotic_at_time_step.append(np.sum(self.lattice == self.NECROTIC))
        self.healthy_at_time_step.append(np.sum(self.lattice == self.HEALTHY))
        self.sum_of_nutrients_at_time_step.append(np.sum(self.C_nut))

        self.log_data()

    def plot_lattice(self):
        plt.clf()
        cmap = ListedColormap(['green', 'red', 'black'])
        plt.imshow(self.lattice, cmap=cmap, vmin=0, vmax=2, origin='upper')
        plt.colorbar(ticks=[0, 1, 2], label='Cell State') 
        plt.title("Tumor Growth Simulation, Step {}".format(self.cur_step))
        plt.pause(0.01)

    def _plot_simulation(self):
        plt.clf()
        plt.plot(self.cancers_at_time_step, label='Cancerous')
        plt.plot(self.necrotic_at_time_step, label='Necrotic')
        plt.plot(self.healthy_at_time_step, label='Healthy')
        plt.xlabel('Time Steps')
        plt.ylabel('Number of Cells')
        plt.legend()
        plt.title('Cell Population Over Time')

        plt.plot(self.sum_of_nutrients_at_time_step)
        plt.xlabel('Time Steps')
        plt.ylabel('Sum of Nutrient Concentrations')
        plt.title('Nutrient Concentration Over Time')
        plt.savefig('plots/plots.png')

        plt.pause(0.01)

    def _init_matrices(self):
        self.lattice = np.zeros((self.lattice_size, self.lattice_size), dtype=np.uint8)
        self.C_nut = np.full((self.lattice_size, self.lattice_size), self.params['I_nut'], dtype=np.float64)
        self.lattice[len(self.lattice) // 2, len(self.lattice) // 2] = self.CANCEROUS
        self.lattice[len(self.lattice) // 2 + 1, len(self.lattice) // 2] = self.CANCEROUS
        self.lattice[len(self.lattice) // 2, len(self.lattice) // 2 + 1] = self.CANCEROUS
        self.lattice[len(self.lattice) // 2 + 1, len(self.lattice) // 2 + 1] = self.CANCEROUS
        self.cancers_at_time_step = [4]



    def _get_neighbors(self, row, col):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = row + dx, col + dy
                if 0 <= nx < self.lattice_size and 0 <= ny < self.lattice_size:
                    neighbors.append((nx, ny))
        return neighbors

    def _calculate_nutrient_concentration(self, neighbors):
        nutrient_sum = sum(self.C_nut[nx, ny] for nx, ny in neighbors)
        initial_sum = len(neighbors) * self.params['I_nut']
        return nutrient_sum / initial_sum if initial_sum > 0 else 0

    def _calculate_probabilities(self, neighbors):
        cancer_count = sum(1 for nx, ny in neighbors if self.lattice[nx, ny] == self.CANCEROUS)
        necrotic_count = sum(1 for nx, ny in neighbors if self.lattice[nx, ny] == self.NECROTIC)
        healthy_count = sum(1 for nx, ny in neighbors if self.lattice[nx, ny] == self.HEALTHY)

        E_cancer = -self.params['K_CC'] * cancer_count
        E_necrotic = -self.params['K_NN'] * necrotic_count
        E_healthy = -self.params['K_HH'] * healthy_count

        exp_cancer = np.exp(-E_cancer)
        exp_necrotic = np.exp(-E_necrotic)
        exp_healthy = np.exp(-E_healthy)
        total_exp = exp_cancer + exp_necrotic + exp_healthy

        P_quiescent = exp_cancer / total_exp
        P_necrosis = exp_necrotic / total_exp
        P_apoptosis = exp_healthy / total_exp
        if exp_cancer + exp_healthy > 0 and cancer_count > 0:
            P_proliferation = exp_cancer / (exp_cancer + exp_healthy)
        else:
            P_proliferation = 0

        return [P_quiescent, P_necrosis, P_apoptosis, P_proliferation]

    def _nutrient_diffusion(self, C_nut):
        for _ in range(self.params['k']):
            C_nut = self._diffuse_nutrients(C_nut)
        return C_nut

    def _diffuse_nutrients(self, C_nut):
        new_nut = C_nut.copy()
        for row in range(1, self.lattice_size - 1):
            for col in range(1, self.lattice_size - 1):
                neighborhood = C_nut[row-1:row+2, col-1:col+2]
                new_nut[row, col] = np.mean(neighborhood)
        return new_nut
    
    def log_data(self):
        if self.cur_step in {50, 100, 200}:
            print(f"Time step: {self.time_step}")
            print(f"Number of cancerous cells: {np.sum(self.lattice == self.CANCEROUS)}")
            print(f"Number of necrotic cells: {np.sum(self.lattice == self.NECROTIC)}")
            print(f"Number of healthy cells: {np.sum(self.lattice == self.HEALTHY)}")
            print(f"Sum of nutrient concentrations: {np.sum(self.C_nut)}")