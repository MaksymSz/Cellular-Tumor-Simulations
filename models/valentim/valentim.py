from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import numpy as np
from random import shuffle, random
from tqdm import tqdm
from models.model import Model

from numba import njit


@njit
def _displacement_capacity(lattice, x, y):
    """
    Calculate the number of empty neighboring cells around a given cell.

    Parameters:
        lattice (np.ndarray): The lattice grid.
        x (int): X-coordinate of the cell.
        y (int): Y-coordinate of the cell.

    Returns:
        int: The displacement capacity (number of empty neighbors).
    """
    return 8 - np.count_nonzero(lattice[x - 1: x + 2, y - 1: y + 2])


@njit
def _roll_probab(n, expand=False):
    """
    Generate random probability values for actions, proliferation, and movement.

    Parameters:
        n (int): The number of probabilities to generate.
        expand (bool): Whether to expand the range (optional).

    Returns:
        tuple: Three arrays of random probabilities (action, proliferation, movement).
    """
    if expand:
        probab_a = np.random.random(n)
        probab_p = np.random.random(n)
        probab_mu = np.random.random(n)
        return probab_a, probab_p, probab_mu

    probab_a = np.random.random(n)
    probab_p = np.random.random(n)
    probab_mu = np.random.random(n)
    return probab_a, probab_p, probab_mu


@njit
def _generate_shuffled_indices(lattice):
    """
    Generate a shuffled list of indices where the lattice is non-zero.

    Parameters:
        lattice (np.ndarray): The lattice grid.

    Returns:
        np.ndarray: Shuffled indices of non-zero elements.
    """
    x, y = np.nonzero(lattice)
    indices = np.stack((x, y), axis=-1)
    shuffled_indices = np.random.permutation(indices)
    return shuffled_indices


@njit
def _change_state(lattice, probab_a, probab_p, probab_mu, neighbors_array, P_A, P_P, P_S, P_MU, P_STC, p_rtc_init, x,
                  y):
    """
    Update the state of a single cell based on probabilities and neighbors.

    Parameters:
        lattice (np.ndarray): The lattice grid.
        probab_a, probab_p, probab_mu (np.ndarray): Probabilities for action, proliferation, and movement.
        neighbors_array (np.ndarray): Array of neighbor offsets.
        P_A, P_P, P_S, P_MU, P_STC, p_rtc_init (float): State and probability parameters.
        x, y (int): Coordinates of the cell.

    Returns:
        np.ndarray: Updated lattice grid.
    """
    if lattice[x, y] != P_STC and probab_a[x, y] < P_A:
        lattice[x, y] = 0
        return lattice

    displacement_capacity = 8 - np.count_nonzero(lattice[x - 1:x + 2, y - 1:y + 2])
    if displacement_capacity == 0:
        return lattice

    if probab_p[x, y] < P_P:
        neighbors = np.random.permutation(neighbors_array)
        for n_x, n_y in neighbors:
            if lattice[x + n_x, y + n_y] == 0:
                if lattice[x, y] == P_STC:
                    lattice[x + n_x, y + n_y] = P_STC if random() < P_S else p_rtc_init
                else:
                    print(lattice[x, y], lattice[x, y] - 1)
                    lattice[x, y] = lattice[x, y] - 1
                    lattice[x + n_x, y + n_y] = lattice[x, y]
                return lattice
    elif probab_mu[x, y] < P_MU:
        neighbors = np.random.permutation(neighbors_array)
        for n_x, n_y in neighbors:
            if lattice[x + n_x, y + n_y] == 0:
                lattice[x + n_x, y + n_y], lattice[x, y] = lattice[x, y], 0
                return lattice
    return lattice


@njit
def _make_step(lattice, neighbors_array, P_A, P_P, P_S, P_MU, P_STC, p_rtc_init):
    """
    Perform a single step of the simulation, updating the lattice.

    Parameters:
        lattice (np.ndarray): The lattice grid.
        neighbors_array (np.ndarray): Array of neighbor offsets.
        P_A, P_P, P_S, P_MU, P_STC, p_rtc_init (float): State and probability parameters.

    Returns:
        np.ndarray: Updated lattice grid.
    """
    x, y = np.nonzero(lattice)
    indices = np.stack((x, y), axis=-1)
    shuffled_indices = np.random.permutation(indices)
    probab_a, probab_p, probab_mu = _roll_probab(len(shuffled_indices))

    for idx, (x, y) in enumerate(shuffled_indices):
        if lattice[x, y] != P_STC and probab_a[idx] < P_A:
            lattice[x, y] = 0
            continue

        displacement_capacity = 8 - np.count_nonzero(lattice[x - 1:x + 2, y - 1:y + 2])
        if displacement_capacity == 0:
            continue
        elif probab_p[idx] < P_P:
            neighbors = np.random.permutation(neighbors_array)
            for n_x, n_y in neighbors:
                if 0 == lattice[x + n_x, y + n_y]:
                    if lattice[x, y] == P_STC:
                        lattice[x + n_x, y + n_y] = P_STC if random() < P_S else p_rtc_init
                    else:
                        lattice[x, y] = lattice[x, y] - 1
                        lattice[x + n_x, y + n_y] = lattice[x, y]
                    break
        elif probab_mu[idx] < P_MU:
            neighbors = np.random.permutation(neighbors_array)
            for n_x, n_y in neighbors:
                if 0 == lattice[x + n_x, y + n_y]:
                    lattice[x + n_x, y + n_y], lattice[x, y] = lattice[x, y], 0
                    break
        continue

    return lattice


class ValentimModel(Model):
    """
    A simulation model representing dynamic cellular processes on a lattice grid.
    Inherits from the ABC `Model`.

    Parameters:
    - lattice_size (int): The size of the square lattice (number of rows/columns).
    - run_params (dict): Dictionary of simulation parameters including:
        - DELTA_T (float): Time increment per simulation step.
        - CCT (float): Cell cycle time for proliferation.
        - MU (float): Movement coefficient.
        - P_MAX (float): Maximum proliferation potential for cells.
        - P_A (float): Probability of cell death per step.
        - P_P (float): Probability of cell proliferation.
        - P_S (float): Probability of symmetric division for stem-like cells.
        - INIT_CELL (str): Initial cell state or type (e.g., 'healthy', 'cancerous').

    Attributes:
    - DELTA_T (float): Time increment per step, scaled for probabilities.
    - CCT (float): Cell cycle time affecting proliferation timing.
    - MU (float): Movement coefficient determining cell motility.
    - P_MAX (float): Maximum proliferation potential for cells.
    - P_A (float): Probability of cell death.
    - P_P (float): Probability of cell proliferation.
    - P_S (float): Probability of symmetric division.
    - P_MU (float): Probability of cell movement.
    - P_STC (float): State identifier for stem-like cells.
    - p_rtc_init (float): Initial proliferation potential for proliferating cells.
    - neighbors_array (np.ndarray): Array of coordinate offsets for neighbors.
    - lattice (np.ndarray): 2D array representing the cell states.
    - M (int): Size of the lattice (number of rows/columns).
    - init_cell (int): Initial state/type of the central cell.
    - STC_count (list): Count of stem-like cells at each time step.
    - RTC_count (list): Count of total cells at each time step.
    - step (int): Current simulation step.
    - newly_born (NoneType or np.ndarray): Placeholder for tracking newly born cells.

    Methods:
    - make_step(): Advances the simulation by one step, updating the lattice.
    - reset(): Resets the simulation to its initial state.
    - _expand_lattice(): Expands the lattice size, keeping the current state centered.
    - run(n_steps): Runs the simulation for a specified number of steps.
    - plot_lattice(plot_bar): Visualizes the current state of the lattice.
    """
    __slots__ = [
        'DELTA_T', 'CCT', 'MU', 'P_MAX', 'P_A', 'P_P', 'P_S', 'P_MU',
        'neighbors_array', 'p_rtc_init', 'P_STC', 'newly_born', 'lattice',
        'M', 'probab_a', 'probab_p', 'probab_mu', 'init_cell', 'STC_count', 'RTC_count', 'step'
    ]

    def __init__(self, lattice_size, run_params):
        """
        Initialize the ValentimModel with lattice size and simulation parameters.

        Parameters:
            lattice_size (int): The size of the square lattice.
            run_params (dict): Dictionary of simulation parameters (e.g., P_A, P_P).
        """
        self.DELTA_T = 1 / run_params['DELTA_T']
        self.CCT = run_params['CCT']
        self.MU = run_params['MU']
        self.P_MAX = run_params['P_MAX']
        self.P_A = run_params['P_A'] * self.DELTA_T
        self.P_P = run_params['P_P']
        self.P_S = run_params['P_S']
        self.P_MU = self.MU * self.DELTA_T
        self.neighbors_array = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
        self.p_rtc_init = self.P_MAX + 1
        self.P_STC = self.P_MAX + 2
        self.init_cell = self.__getattribute__(run_params['INIT_CELL'])
        self.newly_born = None
        self.lattice = np.zeros((lattice_size, lattice_size), dtype=np.uint8)
        self.M = lattice_size
        self.probab_a = None
        self.probab_p = None
        self.probab_mu = None
        self.STC_count = []
        self.RTC_count = []
        self.lattice[self.M // 2, self.M // 2] = self.init_cell
        self.step = 0

    def make_step(self):
        """
        Perform a single simulation step, updating the lattice state.
        """
        self.step += 1
        self.lattice = _make_step(
            self.lattice,
            self.neighbors_array, self.P_A, self.P_P, self.P_S,
            self.P_MU, self.P_STC, self.p_rtc_init
        )

    def reset(self):
        """
        Reset the simulation to its initial state.
        """
        self.step = 0
        self.STC_count = []
        self.RTC_count = []
        self.lattice = np.zeros(self.lattice.shape, dtype=int)
        self.lattice[self.M // 2, self.M // 2] = self.init_cell

    def _expand_lattice(self):
        """
        Expand the lattice size, keeping the current state at the center.
        """
        lattice = self.lattice
        new_shape = (lattice.shape[0] + 10, lattice.shape[1] + 10)
        new_arr = np.zeros(new_shape, dtype=lattice.dtype)

        start_row = new_shape[0] // 2 - lattice.shape[0] // 2
        start_col = new_shape[1] // 2 - lattice.shape[1] // 2

        new_arr[start_row:start_row + lattice.shape[0], start_col:start_col + lattice.shape[1]] = lattice
        self.lattice = new_arr
        self.M = new_shape[0]

    def run(self, n_steps: int):
        """
        Run the simulation for a specified number of steps.

        Parameters:
            n_steps (int): Number of steps to simulate.
        """
        for i in tqdm(range(n_steps), desc=f"\033[32mSimulation progress"):
            self.step = i
            self.make_step()
            self.STC_count.append((self.lattice == (self.P_MAX + 2)).sum())
            self.RTC_count.append(np.count_nonzero(self.lattice))

    def plot_lattice(self, plot_bar=False, dst=None):
        """
        Visualize the current lattice state using a colormap.

        Parameters:
            plot_bar (bool): Whether to include a color bar in the plot (optional).
            dst (str): Destination where the plot should be saved (optional).
        """
        cmap = LinearSegmentedColormap.from_list('name', ['black', 'red'])
        cmap.set_under('white')
        cmap.set_over('yellow')

        plt.cla(), plt.clf()
        plt.imshow(self.lattice, cmap=cmap, vmin=1, vmax=self.p_rtc_init)

        if plot_bar:
            cbar = plt.colorbar(label="Proliferation Potential")
            tick_positions = [0, self.P_MAX // 2 + 1, self.P_MAX + 2]
            tick_labels = ['Empty', 'Medium', r'STC']

            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels)
        _stc_count = (self.lattice == (self.P_MAX + 2)).sum()
        plt.title(f"Day {self.step // 24}\n {_stc_count}")
        plt.tight_layout()
        if dst:
            plt.savefig(dst)
        else:
            plt.show()
        plt.pause(0.01)

    def plot_simulation(self, plot_bar=False, dst=None):
        self.plot_lattice(plot_bar=plot_bar, dst=dst)
