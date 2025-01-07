from pycx.inter_gui import get_simulation_params
from pycx import pycxsimulator
from models.ghaemi.ghaemi_gpu import GhaemiModel
from models.valentim.valentim import ValentimModel
from models.sato.sim import initialize, step, draw


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
        pycxsimulator.GUI().start(func=[initialize, draw, step, lambda x: x])


    

if __name__ == "__main__":
    main()