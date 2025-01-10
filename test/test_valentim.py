# from pycx import pycxsimulator
from models.valentim import valentim
import json
from models.model import Model

with open('../models/valentim/valentim_scenarios.json', 'rb') as fh:
    scenarios = json.load(fh)

scenario = scenarios["scenario_4"]["case_1"]

M = 512
model= valentim.ValentimModel(M, scenario)
model.run(3000)
model.plot_simulation()

# def initialize():
#     model.reset()
#
#
# def observe():
#     model.plot_lattice()
#
#
# def update():
#     model.make_step()
#
#
# def plot():
#     model.plot_simulation()
#
#
# pycxsimulator.GUI().start(func=[initialize, observe, update])
