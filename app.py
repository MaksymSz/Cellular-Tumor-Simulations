from pycx.inter_gui import get_simulation_params
from pycx import pycxsimulator
from models.ghaemi.ghaemi_gpu import GhaemiModel
from models.valentim.valentim import ValentimModel
from models.sato.sim import initialize, step, draw, plot_simulation


def main():
    model = None
    params = get_simulation_params()
    match params["simulation_type"]:
        case 'Probabilistic Cellular Automata':
            model = GhaemiModel(400, params)
        case 'Model accounting for different types of cancer cells':
            model = ValentimModel(64, params)
    if model:
        pycxsimulator.GUI().start(func=[model.reset, model.plot_lattice, model.make_step, model.plot_simulation])
    else:
        pycxsimulator.GUI().start(func=[initialize, draw, step, plot_simulation])


    

if __name__ == "__main__":
    main()