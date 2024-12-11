from abc import ABC, abstractmethod

class Model(ABC):
    """
    Abstract class for models
    """
    @abstractmethod
    def plot_lattice(self):
        raise NotImplementedError
    
    @abstractmethod
    def make_step(self):
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError