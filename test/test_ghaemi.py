from models.ghaemi.ghaemi_gpu import GhaemiModel
from pycx import pycxsimulator

def main():
    params = {
        'K_CC': 3,  
        'K_HH': 0.5, 
        'K_NN': 3,  
        'I_nut': 400.0, 
        'k': 7
    }


    lattice_size = 400

    model = GhaemiModel(lattice_size, params)

    def initialize():
        model.reset()

    def observe():
        model.plot_lattice()

    def update():
        model.make_step()

    pycxsimulator.GUI().start(func=[initialize, observe, update])

if __name__ == "__main__":
    main()