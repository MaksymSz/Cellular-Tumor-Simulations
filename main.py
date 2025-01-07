import pickle
from collections import defaultdict
from time import time
from models.valentim import valentim
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

with open('models/valentim/valentim_scenarios.json', 'rb') as fh:
    scenarios = json.load(fh)

M = 64
days = 20
steps = 24 * days
SCENARIO_NUM = 4
CASES_NUM = 4
CASES_OFFSET = 0
TRIALS_NUM = 3

colors = 'red blue orange green'.split()
results = {f"case_{i}": defaultdict(list) for i in range(1, 5)}

for i in range(1 + CASES_OFFSET, CASES_NUM + 1):
    scenario = scenarios[f"scenario_{SCENARIO_NUM}"][f"case_{i}"]
    if i == 1:
        M = 350
    else:
        M = 260
    for _ in range(TRIALS_NUM):
        model = valentim.ValentimModel(M, scenario)
        model.run(steps)

        rtc = np.array(model.RTC_count)
        stc = np.array(model.STC_count)
        rtc -= stc
        results[f"case_{i}"]["rtc"].append(rtc)
        results[f"case_{i}"]["stc"].append(stc)


def plot(cell_type, dst=None, logx=False):
    for i in range(1 + CASES_OFFSET, CASES_NUM + 1):
        scenario = scenarios[f"scenario_{SCENARIO_NUM}"][f"case_{i}"]

        case = results[f"case_{i}"]
        arr = np.array(case[cell_type])

        arr = np.sort(arr, axis=0)
        std = np.std(arr, axis=0)
        arr = np.mean(arr, axis=0)
        plt.plot(np.arange(steps), arr, label=r"$p_{max}=$" + f"{scenario["P_MAX"]}", color=colors[i - 1])
        plt.fill_between(np.arange(steps), arr - std, arr + std, color=colors[i - 1], alpha=0.2)

    plt.yscale("log")
    plt.ylim(bottom=0.9)
    plt.xticks(np.arange(0, days, 25) * 24, labels=np.arange(0, days, 25))
    if logx:
        plt.xscale("log")
    plt.xlabel("Time (days)")
    plt.ylabel(f"{cell_type.upper()} count")
    plt.title(f"Evolution of {cell_type.upper()}s")
    plt.legend()
    if dst:
        plt.savefig(dst)
    plt.show()


def make_report(path, logx=False):
    timestamp = int(time())
    report = {
        "timestamp": timestamp,
        "scenario_num": SCENARIO_NUM,
    }
    os.mkdir(path / f"{str(timestamp)}")
    for i in range(1 + CASES_OFFSET, CASES_NUM + 1):
        res = results[f"case_{i}"]
        case = {}
        for cc in 'rtc stc'.split():
            arr = np.array(res[cc])

            # arr = np.sort(arr, axis=0)
            std = np.std(arr, axis=0)
            mean = np.mean(arr, axis=0)
            # print(np.max(arr[:, -1]))
            case[f"{cc}_max"] = int(np.max(arr))
            case[f"{cc}_max_std"] = float(np.max(arr, axis=1).std())
            case[f"{cc}_final"] = float(mean[-1])
            case[f"{cc}_final_std"] = float(std[-1])
            with open(path / f"{str(timestamp)}" / f"case_{i}__{cc.upper()}.pickle", "wb") as fh:
                pickle.dump(case, fh)

        # print(f"## CASE {i}")
        # print(case["rtc_max"], case["rtc_max_std"])
        # print(case["rtc_final"], case["rtc_final_std"])
        # print(case["stc_max"], case["stc_max_std"])
        # print(case["stc_final"], case["stc_final_std"])
        report[f"case_{i}"] = case
    with open(path / f"{str(timestamp)}" / f"{str(timestamp)}.json", "w") as fh:
        json.dump(report, fh)
    plot("rtc", path / f"{str(timestamp)}" / f"rtc.pdf", logx=logx)
    plot("stc", path / f"{str(timestamp)}" / f"stc.pdf", logx=logx)


# plot("rtc")
# plot("stc")

# print(model.P_A)
# print(model.P_P)
# print(model.P_MU)

# model.plot_lattice()

dst = Path("results")
make_report(dst, logx=False)
