import copy
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import sklearn.gaussian_process as gp
from mass_automation.deisotoping.process import find_spec_peaks
from mass_automation.experiment import Experiment, Spectrum
from mass_automation.plot import plot_spectrum
from matplotlib import ticker
from scipy.optimize import curve_fit
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def lorentzian(x, x0, a, gam):
    return a * gam**2 / (gam**2 + (x - x0) ** 2)


def plot_clb_peaks(spectrum, clb_peaks, window=0.1):
    clb_peaks = np.array(clb_peaks)

    clb_peaks = clb_peaks[clb_peaks < spectrum.masses.max()]

    nrows = int(len(clb_peaks) / 2) + int((len(clb_peaks) % 2) != 0)

    fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(15, int(nrows * 3)))

    for i, clb_peak in enumerate(clb_peaks):
        cond = (spectrum.masses > (clb_peak - window)) & (
            spectrum.masses < (clb_peak + window)
        )

        ax = plt.subplot(nrows, 2, i + 1)

        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1E}"))
        plt.plot(spectrum.masses[cond], spectrum.ints[cond])
        plt.axvline(x=clb_peak, color="b", label="axvline - full height")

    plt.show()


def clb_classifier(
    spectrum,
    clb_peaks,
    radius,
    threshold=0.3,
    match_method="max",
    plot_lorentzians=True,
):
    matched_masses = {}
    peak_masses, peak_ints, indices = find_spec_peaks(
        spectrum, min_distance=0.01, algorithm="quantile"
    )
    counter = 0
    for clb_peak in clb_peaks:
        int_masses = peak_masses[
            (peak_masses > clb_peak - radius) & (peak_masses < clb_peak + radius)
        ]
        int_ints = peak_ints[
            (peak_masses > clb_peak - radius) & (peak_masses < clb_peak + radius)
        ]

        if int_masses.size != 0:
            if match_method == "max":
                matched_masses[clb_peak] = int_masses[int_ints == int_ints.max()][0]
            elif match_method == "lorentzian":
                spec_masses = spectrum.masses[
                    (spectrum.masses > clb_peak - radius)
                    & (spectrum.masses < clb_peak + radius)
                ]
                spec_ints = spectrum.ints[
                    (spectrum.masses > clb_peak - radius)
                    & (spectrum.masses < clb_peak + radius)
                ]
                if len(spec_masses) > 25:
                    try:
                        popt, _ = curve_fit(
                            lorentzian,
                            spec_masses,
                            spec_ints / spec_ints.max(),
                            p0=[clb_peak, 1, 1e-2],
                            maxfev=5000,
                        )

                        exp_masses = np.linspace(
                            clb_peak - radius, clb_peak + radius, 100
                        )
                        matched_masses[clb_peak] = exp_masses[
                            lorentzian(exp_masses, *popt).argmax()
                        ]

                        if plot_lorentzians:
                            fig, ax = plt.subplots()
                            plt.title(clb_peak)
                            plt.scatter(spec_masses, spec_ints)
                            plt.plot(
                                exp_masses,
                                spec_ints.max() * lorentzian(exp_masses, *popt),
                            )
                            ax.xaxis.set_major_formatter(
                                ticker.StrMethodFormatter("{x:.2f}")
                            )
                            plt.show()
                    except RuntimeError:
                        matched_masses[clb_peak] = int_masses[
                            int_ints == int_ints.max()
                        ][0]

                else:
                    matched_masses[clb_peak] = int_masses[int_ints == int_ints.max()][0]
        else:
            counter += 1

    percentage = counter / clb_peaks.size
    if percentage >= 1 - threshold:
        print("Low matched peaks percentage")
        return matched_masses

    else:
        return matched_masses


def find_cal_errors(spectrum, m_m, algo="gpr", radius=0.1, length_scale=1e-5):
    x_train = np.fromiter(m_m.values(), dtype=float)
    y_train = np.fromiter(m_m.keys(), dtype=float) - x_train
    test_spec = spectrum.masses

    if algo == "gpr":
        kernel = 1 * gp.kernels.RBF(length_scale=length_scale)
        model = gp.GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=20,
            alpha=1e-10,
            normalize_y=False,
            random_state=42,
        )

        model.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
        deviation, std = model.predict(test_spec.reshape(-1, 1), return_std=True)
        deviation, std = deviation.squeeze(), std.squeeze()

        deviation[np.abs(deviation) > radius] = radius

        return deviation, std

    elif algo == "poly":
        degree = 2
        polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        polyreg.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
        poly_pred = polyreg.predict(test_spec.reshape(-1, 1))
        poly_pred_ = poly_pred.squeeze()

        poly_pred_[np.abs(poly_pred_) > radius] = radius

        return poly_pred_


def autocalibrate(
    spectrum,
    clb_peaks,
    radius,
    threshold=0.3,
    method="max",
    algo="gpr",
    return_std=True,
    plot=False,
    clb_peaks_plot=True,
    figsize=(9, 5),
    length_scale=1e-5,
):
    # import ipdb; ipdb.set_trace();

    m_m = clb_classifier(spectrum, clb_peaks, radius, threshold, method)

    if len(m_m) / clb_peaks.size <= threshold:
        if return_std and algo == "gpr":
            return spectrum, None
        else:
            return spectrum

    x_train = np.fromiter(m_m.values(), dtype=float)
    y_train = np.fromiter(m_m.keys(), dtype=float) - x_train
    calibrated_spectrum = copy.copy(spectrum)

    if algo == "gpr":
        deviation, std = find_cal_errors(
            spectrum, m_m, algo="gpr", radius=radius, length_scale=length_scale
        )
        calibrated_spectrum.masses = deviation + spectrum.masses

        if plot:
            plt.figure(figsize=figsize)
            plt.scatter(x_train, y_train, label="calibrated peaks")
            plt.plot(spectrum.masses, deviation, color="darkblue")

            plt.fill_between(
                spectrum.masses,
                deviation - 1.96 * std,
                deviation + 1.96 * std,
                alpha=0.35,
                label=r"95% confidence interval",
            )

            plt.legend(fontsize=12, loc="best")
            plt.xlabel(r"$m_{exp}$", fontsize=12)
            plt.ylabel(r"$m_{theor} - m_{exp}$", fontsize=12)
            plt.show()

        if clb_peaks_plot:
            plot_clb_peaks(calibrated_spectrum, clb_peaks, window=0.1)

        if return_std:
            return calibrated_spectrum, std
        else:
            return calibrated_spectrum

    elif algo == "poly":
        deviation = find_cal_errors(spectrum, m_m, algo="poly", radius=radius)
        calibrated_spectrum.masses = deviation + spectrum.masses
        if plot:
            plt.figure(figsize=figsize)
            plt.scatter(x_train, y_train, label="calibrated peaks")
            plt.plot(spectrum.masses, deviation, color="darkblue")
            plt.legend(fontsize=12, loc="best")
            plt.xlabel(r"$m_{exp}$", fontsize=12)
            plt.ylabel(r"$m_{theor} - m_{exp}$", fontsize=12)
            plt.show()

        if clb_peaks_plot:
            plot_clb_peaks(calibrated_spectrum, clb_peaks, window=0.1)

        return calibrated_spectrum
