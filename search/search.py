import csv
import gc
import os
import pickle as pkl
import sys
import warnings
from argparse import ArgumentParser

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")
from mass_automation.experiment import Experiment, Spectrum
from mass_automation.formula import Formula
from mass_automation.formula.check_formula import (
    check_presence,
    check_presence_isotopologues,
    del_isotopologues,
)
from mass_automation.formula.plot import plot_compare
from scipy.signal import find_peaks

INTERVAL = 0.02
AVERAGE_MASS = 1.00121
MAX_MASS_DELTA = 10
MIN_PEAK_PERCENTAGE = 0.3

parser = ArgumentParser()
parser.add_argument("formula", type=str, help="Formula name")
parser.add_argument("charge", type=str, help="Ion charge")
parser.add_argument("path", type=str, help="Path to the index dictionary")
parser.add_argument(
    "n_jobs", type=int, help="Maximum number of concurrently running workers"
)
parser.add_argument(
    "report_name", type=str, help="Directory, where report will be saved"
)
parser.add_argument(
    "autoclb", type=str, help='If "Yes", auto-calibration algorithm will be preformed'
)
parser.add_argument("threshold", type=str, help="Maximal possible cos. dist")
parser.add_argument(
    "fp_filter",
    type=str,
    help='If "Yes", false positive filtering model will be performed.',
)
parser.add_argument(
    "window_size", type=str, help="If not 0, sliding windows will be used"
)
parser.add_argument("make_plots", type=str, help='If "Yes", plotting will be performed')
args = parser.parse_args()

formula_name = args.formula
charge = int(args.charge)
path = args.path
n_jobs = args.n_jobs
report_name = args.report_name
autoclb = args.autoclb
threshold = args.threshold
fp_filter = args.fp_filter
window_size = int(args.window_size)
make_plots = args.make_plots

if fp_filter == "Yes":
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "models", "fp_filter.pkl"
        ),
        "rb",
    ) as f:
        filter_model = pkl.load(f)

    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "models", "std_scaler.pkl"
        ),
        "rb",
    ) as f:
        scaler = pkl.load(f)
else:
    filter_model = None
    scaler = None


def check_in_spectrum(
    spectrum,
    formula,
    deisotopic_masses,
    deisotopic_peaks,
    filepath,
    threshold=0.052,
    cal_error=0.05,
    dist_error=0.01,
):
    if autoclb == "Yes":
        pass

    z = formula.charge

    res, real_coords, peak_percentage, mass_delta = check_presence_isotopologues(
        spectrum,
        deisotoped_masses,
        deisotoped_peaks,
        cal_error=cal_error,
        dist_error=dist_error,
        distance=1,
        max_peaks=2,
    )

    if fp_filter == "Yes":
        masses_1 = real_coords[0][0]
        ints_1 = real_coords[1][0]

        left_thresh = masses_1 - (AVERAGE_MASS / z) - cal_error
        right_thresh = masses_1 - (AVERAGE_MASS / z) + cal_error

        peak_slice_masses = spectrum.masses[
            (spectrum.masses > left_thresh) & (spectrum.masses < right_thresh)
        ]
        peak_slice_ints = spectrum.ints[
            (spectrum.masses > left_thresh) & (spectrum.masses < right_thresh)
        ]

        pos_peak_indices, _ = find_peaks(
            peak_slice_ints, height=np.quantile(spectrum.ints, 0.95), distance=1
        )
        if len(pos_peak_indices) != 0:
            ints_0 = peak_slice_ints[pos_peak_indices].max()
            masses_0 = peak_slice_masses[np.argmax(peak_slice_ints[pos_peak_indices])]
            features = np.array(
                [ints_0 / ints_1, int(real_coords[0].size), (masses_1 - masses_0) * z]
            ).reshape(1, -1)
            scaled_features = scaler.transform(features)
            fp = int(filter_model.predict(scaled_features))

            if fp:
                print("False Positive prediction is detected")

        else:
            fp = 0
    else:
        fp = 0

    if (
        (res <= threshold)
        and (mass_delta < MAX_MASS_DELTA)
        and (peak_percentage > MIN_PEAK_PERCENTAGE)
    ):
        file_path_noext = formula_name + os.path.splitext(filepath)[0].replace(
            "/", "--"
        )
        if not fp:
            plot_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "reports",
                report_name,
                "plots",
                file_path_noext + ".png",
            )
        else:
            plot_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "reports",
                report_name,
                "plots",
                file_path_noext + "_FP_.png",
            )

        if make_plots == "Yes":
            plot_compare(
                spectrum,
                formula,
                cal_error=cal_error,
                dist_error=dist_error,
                figsize=(5, 7),
                off_axis=True,
                distance=1,
                max_peaks=2,
                path=plot_path,
            )

    del spectrum
    gc.collect()

    if (
        (res <= threshold)
        and (mass_delta < MAX_MASS_DELTA)
        and (peak_percentage > MIN_PEAK_PERCENTAGE)
        and not fp
    ):
        row = [formula_name, filepath, res]
        return row
    else:
        return None


def convolute(ints, kernel_size=1):
    pre_conv = np.zeros(26)
    start_ind = 13 - int(ints.shape[0] / 2)
    pre_conv[start_ind : start_ind + ints.shape[0]] = ints
    w = np.ones(kernel_size)
    conv = np.convolve(pre_conv, w)
    conv = conv / conv.max()
    return conv


if threshold == "auto":
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "models",
            "threshold_estimator_conv.pkl",
        ),
        "rb",
    ) as f:
        threshold_estimator = pkl.load(f)

    try:
        formula = Formula(formula_name, charge)
        masses, ints = formula.isodistribution()
        _, ints = del_isotopologues(masses, ints)
        conv = convolute(ints, kernel_size=3)
        threshold = threshold_estimator.predict(conv)
        print(
            f"Estimated threshold is equal to {round(threshold, 6)}. Continuing the search..."
        )
    except ValueError:
        print(
            "Automated threshold estimation has failed. Please, set the threshold manually: \n"
        )
        threshold = float(input())

else:
    threshold = float(threshold)

batch_name = os.path.basename(path)

with open(path, "rb") as f:
    d = pkl.load(f)

formula = Formula(formula_name, charge)
masses, ints = formula.isodistribution()
deisotoped_masses, deisotoped_peaks = del_isotopologues(masses, ints)
intensive_masses = masses[np.argpartition(-ints, 2)][:2]
keys = np.array(list(d.keys()))

nice_keys = [[], []]
for i, mass in enumerate(intensive_masses):
    nice_keys[i] = keys[(keys > mass - INTERVAL) & (keys < mass + INTERVAL)]

pot_files_1 = []
for key in nice_keys[0]:
    pot_files_1.extend(d[key])

pot_files_2 = []
for key in nice_keys[1]:
    pot_files_2.extend(d[key])

A = set(pot_files_1)
B = set(pot_files_2)
pot_files = A.intersection(B)

if len(pot_files) == 0:
    print(
        f"{len(pot_files)} potential files are collected in {batch_name}. Checking presence is not required"
    )
    print(f"Search finished in {batch_name}")
else:
    print(
        f"{len(pot_files)} potential files are collected in {batch_name}. Start checking presence"
    )

    def check_in_experiment(
        filepath,
        formula,
        deisotoped_masses,
        deisotoped_peaks,
        window_size,
        threshold=0.052,
        cal_error=0.05,
        dist_error=0.01,
    ):
        try:
            exp = Experiment(filepath, verbose=False)
        except RuntimeError:
            return None

        if window_size == 0 or exp.len < window_size + 1:
            spectrum = exp.summarize()
            del exp
            gc.collect()

            return check_in_spectrum(
                spectrum,
                formula,
                deisotoped_masses,
                deisotoped_peaks,
                filepath,
                threshold,
                cal_error,
                dist_error,
            )

        else:
            edges = np.concatenate(
                (np.arange(0, exp.len, window_size), np.array([exp.len]))
            )
            rows = []
            for iter_, edge in enumerate(edges[:-1]):
                spectrum = exp.summarize(edge, edges[iter_ + 1])
                filepath_spec = f"/spec_{edge}_{edges[iter_ + 1]}" + filepath
                row = check_in_spectrum(
                    spectrum,
                    formula,
                    deisotoped_masses,
                    deisotoped_peaks,
                    filepath_spec,
                    threshold,
                    cal_error,
                    dist_error,
                )
                if row is not None:
                    rows.append(row)

            del exp
            gc.collect()
            if len(rows) == 0:
                return None
            else:
                best_row = rows[0]
                best_res = 100
                for row in rows:
                    if row[2] < best_res:
                        best_row = row
                        best_res = row[2]
                return best_row

    my_data = Parallel(n_jobs=n_jobs)(
        delayed(check_in_experiment)(
            filepath,
            formula,
            deisotoped_masses,
            deisotoped_peaks,
            window_size,
            threshold=threshold,
        )
        for filepath in tqdm(pot_files)
    )

    my_data = list(filter(lambda x: not (x is None), my_data))

    rel_path = (
        f"reports/{report_name}/{report_name}_{os.path.splitext(batch_name)[0]}.csv"
    )
    my_file = open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path), "w"
    )
    with my_file:
        writer = csv.writer(my_file)
        writer.writerows(my_data)

    print(f"Search finished in {batch_name}")
