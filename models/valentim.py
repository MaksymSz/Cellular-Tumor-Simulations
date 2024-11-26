from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap, BoundaryNorm
import numpy as np
from random import shuffle, random
from tqdm import tqdm


def _is_edge_nonzero(arr: np.ndarray):
    top_edge = arr[0, :]
    bottom_edge = arr[-1, :]
    left_edge = arr[:, 0]
    right_edge = arr[:, -1]

    return np.all(top_edge != 0) and np.all(bottom_edge != 0) and np.all(left_edge != 0) and np.all(right_edge != 0)


class ValentimModel(object):
    __slots__ = [
        'DELTA_T', 'CCT', 'MU', 'P_MAX', 'P_A', 'P_P', 'P_S', 'P_MU',
        'neighbors_array', 'p_rtc_init', 'p_stc', 'newly_born', 'lattice',
        'M', 'probab_a', 'probab_p', 'probab_mu', 'init_cell', 'STC_count', 'RTC_count', 'step'
    ]

    def __init__(self, lattice_size, run_params):
        self.DELTA_T = run_params['DELTA_T']
        self.CCT = run_params['CCT']
        self.MU = run_params['MU']
        self.P_MAX = run_params['P_MAX']
        self.P_A = run_params['P_A']
        self.P_P = run_params['P_P']
        self.P_S = run_params['P_S']
        self.P_MU = self.MU * self.DELTA_T
        self.neighbors_array = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.p_rtc_init = self.P_MAX
        self.p_stc = self.P_MAX + 2
        self.init_cell = self.__getattribute__(run_params['INIT_CELL'])
        self.newly_born = None
        self.lattice = np.zeros((lattice_size, lattice_size), dtype=np.int8)
        self.M = lattice_size
        self.probab_a = None
        self.probab_p = None
        self.probab_mu = None
        self.STC_count = []
        self.RTC_count = []
        self.lattice[self.M // 2, self.M // 2] = self.init_cell
        self.step = 0

    def reset(self):
        self.STC_count = []
        self.RTC_count = []
        self.lattice = np.zeros(self.lattice.shape, dtype=int)
        self.lattice[self.M // 2, self.M // 2] = self.init_cell

    def _neighbors(self, x, y):
        return self.lattice[x - 1: x + 2, y - 1: y + 2]

    def _displacement_capacity(self, x, y):
        return 8 - np.count_nonzero(self.lattice[x - 1: x + 2, y - 1: y + 2])

    def _roll_probab(self):
        self.probab_a = np.random.random(self.lattice.shape)
        self.probab_p = np.random.random(self.lattice.shape)
        self.probab_mu = np.random.random(self.lattice.shape)

    def _expand_lattice(self):
        print('expanding lattice')

        lattice = self.lattice
        new_shape = (lattice.shape[0] + 10, lattice.shape[1] + 10)
        new_arr = np.zeros(new_shape, dtype=lattice.dtype)

        start_row = new_shape[0] // 2 - lattice.shape[0] // 2
        start_col = new_shape[1] // 2 - lattice.shape[1] // 2

        new_arr[start_row:start_row + lattice.shape[0], start_col:start_col + lattice.shape[1]] = lattice
        self.lattice = new_arr

    def _change_state(self, x, y):
        if self.step % 24 == 0 and self.lattice[x, y] != self.p_stc and self.probab_a[x, y] < self.P_A:
            self.lattice[x, y] = 0
            return

        if not self._displacement_capacity(x, y):
            return

        if self.probab_p[x, y] < self.P_P:
            shuffle(self.neighbors_array)
            for n_x, n_y in self.neighbors_array:
                if self.lattice[x + n_x, y + n_y] == 0:
                    if self.lattice[x, y] == self.p_stc:
                        if random() < self.P_S:
                            self.lattice[x + n_x, y + n_y] = self.lattice[x, y]
                        else:
                            self.lattice[x + n_x, y + n_y] = self.p_rtc_init
                    else:
                        self.lattice[x, y] = max(self.lattice[x, y] - 1, 0)
                        self.lattice[x + n_x, y + n_y] = self.lattice[x, y]
                    # self.newly_born.append((x + n_x, y + n_y))
                    return

        if self.step % 24 == 0 and self.probab_mu[x, y] < self.P_MU:
            for n_x, n_y in self.neighbors_array:
                if self.lattice[x + n_x, y + n_y] == 0:
                    self.lattice[x + n_x, y + n_y] = self.lattice[x, y]
                    self.lattice[x, y] = 0
                    return
        return

    def make_step(self):
        self.step += 1
        self.newly_born = []
        self._roll_probab()
        tmp_x, tmp_y = np.nonzero(self.lattice)
        self.newly_born = list(zip(tmp_x, tmp_y))
        shuffle(self.newly_born)
        repeat = True
        # while repeat:
        for x, y in self.newly_born:
            try:
                self._change_state(x, y)
            except IndexError:
                self._expand_lattice()
                self._change_state(x, y)
                self._roll_probab()
            # else:
            #     repeat = False

    def run(self, n_steps: int):
        for i in tqdm(range(n_steps), desc=f"\033[32mSimulation progress"):
            self.step = i
            self.make_step()
            self.STC_count.append((self.lattice == (self.P_MAX + 2)).sum())
            self.RTC_count.append(np.count_nonzero(self.lattice))

    def plot_lattice(self, plot_bar=False):
        colors = [(1 / i, 0, 0, 1) for i in range(self.P_MAX, 0, -1)]
        colors.append((240 / 255, 1, 0, 1))
        colors.insert(0, (0, 0, 0, 0))
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(list(range(0, self.P_MAX + 3)), cmap.N)

        plt.cla()
        plt.imshow(self.lattice, cmap=cmap, norm=norm)

        if plot_bar:
            cbar = plt.colorbar(label="Proliferation Potential")
            tick_positions = [0, self.P_MAX // 2 + 1, self.P_MAX + 2]
            tick_labels = ['Empty', 'Medium', r'STC']

            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels)

        plt.title(f"Day {self.step // 24}")
        plt.tight_layout()
        plt.show()
