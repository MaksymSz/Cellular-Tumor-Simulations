import numpy as np
from models.model import Model


class GhaemiModel(Model):
    """
    Ghaemi model for simulating cancer growth on a lattice.
    """
    def __init__(self, scenario):
        super().__init__(self, scenario)
        self.lattice_size = scenario['lattice_size']
        self.k_cc = scenario['K_cc']
        self.k_nn = scenario['K_nn']
        self.k_hh = scenario['K_hh']
        self.lattice = np.zeros((self.lattice_size, self.lattice_size), dtype=np.int8)


    def reset(self):
        pass

    def make_step(self):
        pass

    def plot_lattice(self):
        pass
