from models.ghaemi.ghaemi_gpu import GhaemiModel
from pycx import pycxsimulator

def main():
    params = {
        'K_CC': 3,  
        'K_HH': 1, 
        'K_NN': 3,  
        'I_nut': 400.0, 
        'k': 2
    }


    lattice_size = 400

    model = GhaemiModel(lattice_size, params)

    def initialize():
        model.reset()

    def observe():
        model.plot_lattice()

    def update():
        model.make_step()
    
    def plot():
        model.plot_simulation()

    pycxsimulator.GUI().start(func=[initialize, observe, update, plot])

if __name__ == "__main__":
    main()