from collections import defaultdict

from models import valentim
import json
import numpy as np
import matplotlib.pyplot as plt

with open('models/valentim_scenarios.json', 'rb') as fh:
    scenarios = json.load(fh)

M = 128
days = 208
steps = 24 * days

colors = 'red blue orange green'.split()
results = {f"case_{i}": defaultdict(list) for i in range(1, 5)}

for i in range(1, 5):
    scenario = scenarios["scenario_3"][f"case_{i}"]
    for _ in range(5):
        model = valentim.ValentimModel(M, scenario)
        model.run(steps)

        rtc = np.array(model.RTC_count)
        stc = np.array(model.STC_count)
        rtc -= stc
        results[f"case_{i}"]["rtc"].append(rtc)
        results[f"case_{i}"]["stc"].append(rtc)


def plot(cell_type):
    for i in range(1, 5):
        scenario = scenarios["scenario_3"][f"case_{i}"]

        case = results[f"case_{i}"]
        arr = np.array(case[cell_type])

        arr = np.sort(arr, axis=0)
        std = np.std(arr, axis=0)
        arr = np.mean(arr, axis=0)
        plt.plot(np.arange(steps), arr, label=r"$p_{max}=$" + f"{scenario["P_MAX"]}", color=colors[i - 1])
        plt.fill_between(np.arange(steps), arr - std, arr + std, color=colors[i - 1], alpha=0.3)

    plt.yscale("log")
    plt.xticks(np.arange(0, days, 25) * 24, labels=np.arange(0, days, 25))
    plt.xlabel("Time (days)")
    plt.ylabel(f"{cell_type.capitalize()} count")
    plt.title(f"Evolution of {cell_type.capitalize()}s")
    plt.legend()
    plt.show()


plot("rtc")
plot("stc")