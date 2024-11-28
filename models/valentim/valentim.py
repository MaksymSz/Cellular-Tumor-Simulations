from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import numpy as np
from random import shuffle, random
from tqdm import tqdm
from models.model import Model

from numba import njit


@njit
def _displacement_capacity(lattice, x, y):
    return 8 - np.count_nonzero(lattice[x - 1: x + 2, y - 1: y + 2])


@njit
def _roll_probab(n, expand=False):
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
    x, y = np.nonzero(lattice)
    indices = np.stack((x, y), axis=-1)
    shuffled_indices = np.random.permutation(indices)
    return shuffled_indices


@njit
def _change_state(lattice, probab_a, probab_p, probab_mu, neighbors_array, P_A, P_P, P_S, P_MU, P_STC, p_rtc_init, x,
                  y):
    if lattice[x, y] != P_STC and probab_a[x, y] < P_A:
        print('dead')
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
def _make_step(lattice, probab_a, probab_p, probab_mu, neighbors_array, P_A, P_P, P_S, P_MU, P_STC, p_rtc_init):
    x, y = np.nonzero(lattice)
    indices = np.stack((x, y), axis=-1)
    shuffled_indices = np.random.permutation(indices)

    for x, y in shuffled_indices:
        if lattice[x, y] != P_STC and probab_a[x, y] < P_A:
            lattice[x, y] = 0
            continue

        displacement_capacity = 8 - np.count_nonzero(lattice[x - 1:x + 2, y - 1:y + 2])
        if displacement_capacity == 0:
            continue

        if probab_p[x, y] < P_P:
            neighbors = np.random.permutation(neighbors_array)
            for n_x, n_y in neighbors:
                if 0 == lattice[x + n_x, y + n_y]:
                    if lattice[x, y] == P_STC:
                        lattice[x + n_x, y + n_y] = P_STC if random() < P_S else p_rtc_init
                    else:
                        lattice[x, y] = max(lattice[x, y] - 1, 0)
                        lattice[x + n_x, y + n_y] = lattice[x, y]
                    break
        elif probab_mu[x, y] < P_MU:
            neighbors = np.random.permutation(neighbors_array)
            for n_x, n_y in neighbors:
                if lattice[x + n_x, y + n_y] == 0:
                    lattice[x + n_x, y + n_y], lattice[x, y] = lattice[x, y], 0
                    break
        continue

    return lattice


class ValentimModel(Model):
    __slots__ = [
        'DELTA_T', 'CCT', 'MU', 'P_MAX', 'P_A', 'P_P', 'P_S', 'P_MU',
        'neighbors_array', 'p_rtc_init', 'P_STC', 'newly_born', 'lattice',
        'M', 'probab_a', 'probab_p', 'probab_mu', 'init_cell', 'STC_count', 'RTC_count', 'step'
    ]

    def __init__(self, lattice_size, run_params):
        self.DELTA_T = 1 / run_params['DELTA_T']
        self.CCT = run_params['CCT']
        self.MU = run_params['MU']
        self.P_MAX = run_params['P_MAX']
        self.P_A = self.DELTA_T * run_params['P_A']
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
        self.step += 1
        self.probab_a, self.probab_p, self.probab_mu = _roll_probab(self.lattice.shape)
        self.lattice = _make_step(
            self.lattice, self.probab_a, self.probab_p, self.probab_mu,
            self.neighbors_array, self.P_A, self.P_P, self.P_S,
            self.P_MU, self.P_STC, self.p_rtc_init
        )

    def reset(self):
        self.STC_count = []
        self.RTC_count = []
        self.lattice = np.zeros(self.lattice.shape, dtype=int)
        self.lattice[self.M // 2, self.M // 2] = self.init_cell

    def _expand_lattice(self):
        print('expanding lattice')

        lattice = self.lattice
        new_shape = (lattice.shape[0] + 10, lattice.shape[1] + 10)
        new_arr = np.zeros(new_shape, dtype=lattice.dtype)

        start_row = new_shape[0] // 2 - lattice.shape[0] // 2
        start_col = new_shape[1] // 2 - lattice.shape[1] // 2

        new_arr[start_row:start_row + lattice.shape[0], start_col:start_col + lattice.shape[1]] = lattice
        self.lattice = new_arr
        self.M = new_shape[0]

    def run(self, n_steps: int):
        for i in tqdm(range(n_steps), desc=f"\033[32mSimulation progress"):
            self.step = i
            self.make_step()
            self.STC_count.append((self.lattice == (self.P_MAX + 2)).sum())
            self.RTC_count.append(np.count_nonzero(self.lattice))

    def plot_lattice(self, plot_bar=False):
        # colors = [(1 / i, 0, 0, 1) for i in range(self.P_MAX, 0, -1)]
        # colors.append((240 / 255, 1, 0, 1))
        # colors.insert(0, (0, 0, 0, 0))
        # cmap = ListedColormap(colors)
        # norm = BoundaryNorm(list(range(0, self.P_MAX + 2)), cmap.N)

        cmap = LinearSegmentedColormap.from_list('name', ['black', 'red'])
        cmap.set_under('white')
        cmap.set_over('yellow')

        plt.cla()
        plt.imshow(self.lattice, cmap=cmap, vmin=1, vmax=self.p_rtc_init)

        if plot_bar:
            cbar = plt.colorbar(label="Proliferation Potential")
            tick_positions = [0, self.P_MAX // 2 + 1, self.P_MAX + 2]
            tick_labels = ['Empty', 'Medium', r'STC']

            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels)

        plt.title(f"Day {self.step // 24}")
        plt.tight_layout()
        plt.show()
