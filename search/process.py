import gc
import os
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np
from mass_automation.deisotoping.process import find_spec_peaks
from mass_automation.experiment import Experiment, Spectrum
from mass_automation.plot import plot_spectrum
from tqdm import tqdm


def index_big_spectra(exp, mass_dict, filepath, window_size, min_distance=0.001):
    edges = np.concatenate((np.arange(0, exp.len, window_size), np.array([exp.len])))
    for i, edge in enumerate(edges[:-1]):
        spectrum = exp.summarize(edge, edges[i + 1])
        peaks = find_spec_peaks(
            spectrum, algorithm="quantile", min_distance=min_distance
        )
        masses = np.round(peaks[0], 4)
        filename = filepath.split("/")[-1]
        for mass in masses:
            if mass in mass_dict.keys():
                filelist = mass_dict[mass]
                filelist.append(filepath)
                mass_dict[mass] = filelist
            else:
                mass_dict[mass] = [filepath]

        del spectrum
        gc.collect()

    del exp
    gc.collect()

    return mass_dict


mass_dict = {}
batch_dir = sys.argv[1]
window_size = int(sys.argv[2])
min_distance = float(sys.argv[3])

print(f"Window size: {window_size}, Min. distance: {min_distance} \n")
batch_name = batch_dir.split("/")[-1]
non_read_files = []
for line in tqdm(sys.stdin):
    filepath = str(line).rstrip("\n")
    filename = filepath.split("/")[-1].rstrip(".mzXML")
    try:
        exp = Experiment(filepath)
        if exp.len > 60 and window_size > 0:
            print(f"Monitoring spectrum. Cut on windows with size {window_size}\n")

            mass_dict = index_big_spectra(
                exp, mass_dict, filepath, window_size, min_distance
            )

        else:
            spectrum = exp.summarize()
            peaks = find_spec_peaks(
                spectrum, algorithm="quantile", min_distance=min_distance
            )
            masses = np.round(peaks[0], 4)
            filename = filepath.split("/")[-1]
            for mass in masses:
                if mass in mass_dict.keys():
                    filelist = mass_dict[mass]
                    filelist.append(filepath)
                    mass_dict[mass] = filelist
                else:
                    mass_dict[mass] = [filepath]

            del exp, spectrum
            gc.collect()

    except:
        non_read_files.append(filepath)

with open(f"errors/non_read_files_{batch_name}.txt", "w") as f:
    for item in non_read_files:
        f.write(f"{item}\n")

with open(f"index_pickles/{batch_name}.pkl", "wb") as f:
    pkl.dump(mass_dict, f)
