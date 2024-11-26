from pycx import pycxsimulator
from models import valentim
import json

with open('../models/valentim_scenarios.json', 'rb') as fh:
    scenarios = json.load(fh)

scenario = scenarios["scenario_3"]["case_1"]

M = 128
model = valentim.ValentimModel(M, scenario)


def initialize():
    model.reset()


def observe():
    model.plot_lattice()


def update():
    model.make_step()


pycxsimulator.GUI().start(func=[initialize, observe, update])
