from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract class for models
    """

    @abstractmethod
    def plot_lattice(self):
        """
        Visualize the current lattice state using a colormap.
        """
        raise NotImplementedError

    @abstractmethod
    def make_step(self):
        """
        Perform a single simulation step, updating the lattice state.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Reset the simulation to its initial state.
        """
        raise NotImplementedError
