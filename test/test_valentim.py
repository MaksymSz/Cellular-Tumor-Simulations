from pycx import pycxsimulator
from models.valentim import valentim
import json
from models.model import Model

with open('../models/valentim/valentim_scenarios.json', 'rb') as fh:
    scenarios = json.load(fh)

scenario = scenarios["scenario_3"]["case_1"]

M = 300
model: Model = valentim.ValentimModel(M, scenario)


def initialize():
    model.reset()


def observe():
    model.plot_lattice()


def update():
    model.make_step()


pycxsimulator.GUI().start(func=[initialize, observe, update])
