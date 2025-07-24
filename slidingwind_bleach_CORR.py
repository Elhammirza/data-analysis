import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import fft
import matplotlib as mpl

# Set global font sizes for all matplotlib figures
mpl.rcParams.update({
    "axes.labelsize": 18,      # x/y label font size
    "axes.titlesize": 20,      # title font size
    "xtick.labelsize": 16,     # x-tick number font size
    "ytick.labelsize": 16,     # y-tick number font size
    "legend.fontsize": 15,     # legend font size
    "figure.titlesize": 22,    # overall figure title font size
    "font.size": 16,           # default font size for text
    "lines.linewidth": 2,      # line width (optional)
    "lines.markersize": 8      # marker size (optional)
})
# ==== PARAMETERS ====
folder_path = r"C:\Users\mirza010\OneDrive - Universiteit Utrecht\Desktop\UU data\Projects\Cellular_uptake_endosomal_escape\living cells experiments\2025_07_01_Hek_pamps particles_after1h\NEWdata_FULL\NEWdata_FULL\try"
suffix = "Countrates.csv"
bin_size = 100  # this is the bin size 

# Window sizes to test (in seconds)
window_sizes_seconds = np.array([0.01, 0.02, 0.04, 0.08, 0.1, 0.3, 0.5, 1, 2, 4, 8, 10]) #different 
# windows in secondes to check the bleaching correction 

#%% Functions that I'm going to use in this code, BIN_data for binning data, calculating autocorrelation function
#wih fft, and local averaging with sliding window time, and at the end the merit function to find the best window

def bin_data(t, F, bin_size):
    F = np.asarray(F)
    t = np.asarray(t)
    n_bins = len(F) // bin_size
    F = F[:n_bins * bin_size]
    t = t[:n_bins * bin_size]
    F_binned = F.reshape(n_bins, bin_size).mean(axis=1)
    t_binned = t.reshape(n_bins, bin_size).mean(axis=1)
    return t_binned, F_binned

def autocorrelation_fft(f):
    f = np.asarray(f, dtype=float)
    N = len(f)
    f_mean = np.mean(f)
    if f_mean == 0:
        raise ValueError("Mean of the signal is zero. Cannot normalize autocorrelation.")
    df = f - f_mean
    fft_f = fft.fft(df, n=2*N)
    acf = fft.ifft(fft_f * np.conjugate(fft_f))[:N].real
    acf /= np.arange(N, 0, -1)
    G = acf / (f_mean**2)
    return G

def local_average_acf(F_binned, window_size_bins, max_lag_bins):
    N = len(F_binned)
    acf_sum = np.zeros(max_lag_bins)
    n_windows = 0
    step = max(1, window_size_bins // 5)
    for start in range(0, N - window_size_bins + 1, step):
        window = F_binned[start : start + window_size_bins]
        if len(window) < max_lag_bins:
            continue
        acf = autocorrelation_fft(window) # compute acf for each windows
        acf_sum += acf[:max_lag_bins]  # sum all the acf of different windows
        n_windows += 1
    if n_windows == 0:
        raise RuntimeError("No valid windows for ACF computation!")
    return acf_sum / n_windows # return the local averaging 

def merit_function(acf, min_lag_bin, max_lag_bin, sampling_time):
    region = acf[min_lag_bin:max_lag_bin]
    lag_window = (max_lag_bin - min_lag_bin) * sampling_time
    return np.sum(region ** 2) / lag_window
# "min_lag_bin" minimum lag time for the analysis the "b" value in the paper 
#he time width of each bin in your binned data (e.g., if you binned 10 × 1 μs points, sampling_time = 10 μs = 0.00001 s).

#%% 
# ==== DATA LOADING (for a single file) ====
fn = [fn for fn in os.listdir(folder_path) if fn.endswith(suffix)][0]
print("Working on file:", fn)
df_raw = pd.read_csv(os.path.join(folder_path, fn), sep='\t', header=None, skiprows=2)
t = df_raw.iloc[:, 0].astype(float).reset_index(drop=True)
F = df_raw.iloc[:, 1].astype(float).reset_index(drop=True)

# ==== BINNING ====
t_binned, F_binned = bin_data(t, F, bin_size)
sampling_time = t_binned[1] - t_binned[0]
min_lag_time =sampling_time    # [s] The shortest lag time to analyze in the autocorrelation
print("\nWindow size (s)    Merit function")
print("-" * 35)
merit_results = []

for window_size_sec in window_sizes_seconds:
    window_size_bin = int(window_size_sec / sampling_time)
    min_lag_bin = int(np.ceil(min_lag_time / sampling_time))
    max_lag_bin = window_size_bin  # Only up to window length

    if window_size_bin <= min_lag_bin:
        print(f"{window_size_sec:8.3f}         Skipped: window too small for min lag")
        continue

    try:
        acf = local_average_acf(F_binned, window_size_bin, max_lag_bin)
        merit = merit_function(acf, min_lag_bin, max_lag_bin, sampling_time)
        print(f"{window_size_sec:8.3f}         {merit:12.6g}")
        merit_results.append((window_size_sec, merit))
    except Exception as e:
        print(f"{window_size_sec:8.3f}         Error: {str(e)}")

# (Optional) Save results as CSV
df_merit = pd.DataFrame(merit_results, columns=["window_sec", "merit"])
df_merit.to_csv("merit_results.csv", index=False)

# ==== PLOT MERIT FUNCTION ====
plt.figure()
plt.plot(df_merit['window_sec'], df_merit['merit'], marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Window size (s)')
plt.ylabel('Normalized merit function')
plt.title("Merit function vs. window size")
plt.tight_layout()
plt.show()


# ==== PLOT INTENSITY FLUCTUATIONS (Downsampled for clarity) ====
plt.figure(figsize=(10, 4))
downsample_factor = 1000      #Change this number to 100 or higher for sparser plotting

# Use binned data for a reasonable size plot (or use raw data if you want)
plt.plot(t_binned[::downsample_factor], F_binned[::downsample_factor], 
         lw=0.7, color='tab:blue')
plt.xlabel('Time (s)')
plt.ylabel('Count Rate (a.u.)')
plt.title(f'Intensity Trace (every {downsample_factor}th point)')
plt.tight_layout()
plt.show()







# ==== COMPUTE & PLOT GLOBAL AND BEST LOCAL ACF, COMPATIBLE WITH REST OF CODE ====

# ----- Global (full) ACF -----
max_lag_plot_seconds = 30.0  # adjust as desired for plotting
full_lag_bins = int(max_lag_plot_seconds / sampling_time)
global_acf_full = autocorrelation_fft(F_binned)[:full_lag_bins]
lags_full = np.arange(full_lag_bins) * sampling_time

# ----- Find minimum merit function result -----
if len(merit_results) > 0:
    # Find best (minimum) merit window
    min_index = np.argmin(df_merit['merit'])
    best_window_sec = df_merit['window_sec'].iloc[min_index]
    best_window_bin = int(best_window_sec / sampling_time)
    max_lag_bin = best_window_bin
    min_lag_bin = int(np.ceil(min_lag_time / sampling_time))
    # Local-averaged ACF for this window
    best_acf = local_average_acf(F_binned, best_window_bin, max_lag_bin)
    lags_best = np.arange(max_lag_bin) * sampling_time
    plot_max_lag_bin_best = int(0.5 * max_lag_bin)
else:
    best_acf = None

# ----- Plot all together -----
plt.figure(figsize=(8, 5))

# Plot raw global ACF
plot_max_lag_bin = int(0.5 * full_lag_bins)
plt.plot(
    lags_full[min_lag_bin:plot_max_lag_bin],
    global_acf_full[min_lag_bin:plot_max_lag_bin],
    label="Global ACF (raw)", lw=2
)

# Plot local-averaged (best merit) ACF, if available
if best_acf is not None:
    plt.plot(
        lags_best[min_lag_bin:plot_max_lag_bin_best],
        best_acf[min_lag_bin:plot_max_lag_bin_best],
        label=f"Local avg ACF (raw), window={best_window_sec:.3f} s", lw=2
    )

# Overlay all software ACFs (_CF.csv)
for fn_cf in os.listdir(folder_path):
    if fn_cf.endswith("CF.csv"):
        df_cf = pd.read_csv(os.path.join(folder_path, fn_cf), sep='\t', header=None, skiprows=2)
        t_cf = df_cf.iloc[:, 0].astype(float).reset_index(drop=True).values
        G_cf = df_cf.iloc[:, 1].astype(float).reset_index(drop=True).values
        t_cf=t_cf/1000
        plt.plot(t_cf, G_cf, '--', label=f"Software CF ({fn_cf})", alpha=0.7)

plt.xscale('log')
plt.xlabel("Lag time (s)")
plt.ylabel("ACF")
plt.title("Global ACF, Local-Averaged (Best), and Software Overlay")
plt.legend()
plt.tight_layout()
plt.show()








#%% Fitting the autocorrelation to anomalous diffusion model with custom range and colored plot
from scipy.optimize import curve_fit

# ----- Set your fit range here (in seconds) -----
fit_lower = 5e-6     # e.g., 10 microseconds
fit_upper =0.1    # e.g., 1 second

kappa = 4.7  # Structure parameter
r0 = 0.23    # [um] Confocal waist
V_eff = 0.32e-15  # [L], adjust for your setup!
NA = 6.022e23    # Avogadro's number
def G_anom_2c_triplet(tau, p1, ln_tau1, ln_tau2, alpha1, alpha2, T, ln_tauT, G_inf, amp):
    tau1 = np.exp(ln_tau1)
    tau2 = np.exp(ln_tau2)
    tauT = np.exp(ln_tauT)
    p2 = 1.0 - p1
    trip = 1.0 + (-T + T * np.exp(-tau / tauT))
    def diff_comp(tau, taui, alphai):
        x = (tau / taui) ** alphai
        return (1 + x) ** (-1) * (1 + x / kappa ** 2) ** (-0.5)
    diff1 = diff_comp(tau, tau1, alpha1)
    diff2 = diff_comp(tau, tau2, alpha2)
    model = amp * trip * (p1 * diff1 + p2 * diff2) + G_inf
    return model

def G_anom_1c_triplet(tau, ln_tauD, alpha, T, ln_tauT, G_inf, amp):
    tauD = np.exp(ln_tauD)
    tauT = np.exp(ln_tauT)
    trip = 1.0 + (-T + T * np.exp(-tau / tauT))
    x = (tau / tauD) ** alpha
    diff = (1 + x) ** (-1) * (1 + x / kappa ** 2) ** (-0.5)
    model = amp * trip * diff + G_inf
    return model




import matplotlib.pyplot as plt
import numpy as np

# ... your existing code above ...

if best_acf is not None:
    # --- Build masks for the selected fit range ---
    fit_mask = (lags_best >= fit_lower) & (lags_best <= fit_upper)
    tau_fit = lags_best[fit_mask]
    acf_fit = best_acf[fit_mask]

    print(f"Fitting ACF between {tau_fit[0]:.2e} s and {tau_fit[-1]:.2e} s ({len(tau_fit)} points)")

    # --- Fit both models regardless of choice ---
    # 1-component initial values and bounds
    p0_1c = [np.log(0.01), 0.9, 0.05, np.log(0.001), 0.0, acf_fit[0]]
    lb_1c = [np.log(1e-5), 0.1, 0.0, np.log(1e-6), -np.inf, 0.1 * acf_fit[0]]
    ub_1c = [np.log(fit_upper), 1.0, 1.0, np.log(0.01), np.inf, 10 * acf_fit[0]]

    # 2-component initial values and bounds
    p0_2c = [0.5, np.log(0.01), np.log(0.1), 0.9, 0.9, 0.05, np.log(0.001), 0.0, acf_fit[0]]
    lb_2c = [0.0, np.log(1e-5), np.log(1e-5), 0.1, 0.1, 0.0, np.log(1e-6), -np.inf, 0.1 * acf_fit[0]]
    ub_2c = [1.0, np.log(fit_upper), np.log(fit_upper), 1.0, 1.0, 1.0, np.log(0.01), np.inf, 10 * acf_fit[0]]

    tau_model = np.logspace(np.log10(fit_lower), np.log10(fit_upper), 200)

    # --- Fit 1C model ---
    popt_1c, pcov_1c = curve_fit(G_anom_1c_triplet, tau_fit, acf_fit, p0=p0_1c, bounds=(lb_1c, ub_1c), maxfev=20000)
    G_model_1c = G_anom_1c_triplet(tau_model, *popt_1c)
    G_fit_1c = G_anom_1c_triplet(tau_fit, *popt_1c)
    residuals_1c = acf_fit - G_fit_1c

    # --- Fit 2C model ---
    popt_2c, pcov_2c = curve_fit(G_anom_2c_triplet, tau_fit, acf_fit, p0=p0_2c, bounds=(lb_2c, ub_2c), maxfev=20000)
    G_model_2c = G_anom_2c_triplet(tau_model, *popt_2c)
    G_fit_2c = G_anom_2c_triplet(tau_fit, *popt_2c)
    residuals_2c = acf_fit - G_fit_2c

    # --- Chi2 calculation ---
    ndat = len(acf_fit)
    n1c = len(popt_1c)
    n2c = len(popt_2c)
    chi2_1c = np.sum(residuals_1c**2)
    chi2_red_1c = chi2_1c / (ndat - n1c)
    chi2_2c = np.sum(residuals_2c**2)
    chi2_red_2c = chi2_2c / (ndat - n2c)

    # --- Plot for 1C model (ACF + fit + residual) ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 7), gridspec_kw={'height_ratios':[3,1]})
    ax1.semilogx(lags_best, best_acf, 'o', color='lightgray', label="All ACF data", markersize=4, alpha=0.7)
    ax1.semilogx(tau_fit, acf_fit, 'o', color='tab:orange', label="Fit region", markersize=6)
    ax1.semilogx(tau_model, G_model_1c, '-', color='crimson', label="1C anom+triplet fit", lw=2)
    ax1.set_ylabel("ACF")
    ax1.set_ylim([-0.009,0.12])
    ax1.set_title(f"1C+triplet fit | Range: {fit_lower:.1e} to {fit_upper:.1e} s\n"
                  f"chi2_red = {chi2_red_1c:.3g}")
    ax1.legend()
    # Residual plot
    ax2.semilogx(tau_fit, residuals_1c, 'o-', color='tab:blue', label="Residuals")
    ax2.axhline(0, ls='--', color='k', lw=1)
    ax2.set_xlabel("Lag time (s)")
    ax2.set_ylabel("Residual")
    ax2.legend()
    plt.tight_layout()
    plt.show()

    # --- Plot for 2C model (ACF + fit + residual) ---
    fig2, (ax3, ax4) = plt.subplots(2, 1, sharex=True, figsize=(7, 7), gridspec_kw={'height_ratios':[3,1]})
    ax3.semilogx(lags_best, best_acf, 'o', color='lightgray', label="All ACF data", markersize=4, alpha=0.7)
    ax3.semilogx(tau_fit, acf_fit, 'o', color='tab:blue', label="Fit region", markersize=6)
    ax3.semilogx(tau_model, G_model_2c, '-', color='crimson', label="2C anom+triplet fit", lw=2)
    ax3.set_ylabel("ACF")
    ax3.set_ylim([-0.009,0.12])
    ax3.set_title(f"2C+triplet fit | Range: {fit_lower:.1e} to {fit_upper:.1e} s\n"
                  f"chi2_red = {chi2_red_2c:.3g}")
    ax3.legend()
    # Residual plot
    ax4.semilogx(tau_fit, residuals_2c, 'o-', color='tab:orange', label="Residuals")
    ax4.axhline(0, ls='--', color='k', lw=1)
    ax4.set_xlabel("Lag time (s)")
    ax4.set_ylabel("Residual")
    ax4.legend()
    plt.tight_layout()
    plt.show()

    # --- Print results for both fits ---
    print(f"\n1C fit results:")
    print(f"  tauD (s):     {np.exp(popt_1c[0]):.4g}")
    print(f"  alpha:        {popt_1c[1]:.3f}")
    print(f"  T:            {popt_1c[2]:.3f}")
    print(f"  tauT (s):     {np.exp(popt_1c[3]):.4g}")
    print(f"  offset G_inf: {popt_1c[4]:.3g}")
    print(f"  amplitude:    {popt_1c[5]:.3f}")
    print(f"  D: {r0**2/(4*np.exp(popt_1c[0])):.3f} μm²/s")
    print(f"  chi2:         {chi2_1c:.3g}")
    print(f"  chi2_red:     {chi2_red_1c:.3g}")

    print(f"\n2C fit results:")
    print(f"  tau1 (s):     {np.exp(popt_2c[1]):.4g}")
    print(f"  tau2 (s):     {np.exp(popt_2c[2]):.4g}")
    print(f"  alpha1:       {popt_2c[3]:.3f}")
    print(f"  alpha2:       {popt_2c[4]:.3f}")
    print(f"  T:            {popt_2c[5]:.3f}")
    print(f"  tauT (s):     {np.exp(popt_2c[6]):.4g}")
    print(f"  offset G_inf: {popt_2c[7]:.3g}")
    print(f"  amplitude:    {popt_2c[8]:.3f}")
    print(f"  D1: {r0**2/(4*np.exp(popt_2c[1])):.3f} μm²/s")
    print(f"  D2: {r0**2/(4*np.exp(popt_2c[2])):.3f} μm²/s")
    print(f"  chi2:         {chi2_2c:.3g}")
    print(f"  chi2_red:     {chi2_red_2c:.3g}")

    Neff = 1 / popt_2c[8]
    N1_eff = popt_2c[0] * Neff
    N2_eff = (1 - popt_2c[0]) * Neff
    conc1_eff = N1_eff / (NA * V_eff)
    conc2_eff = N2_eff / (NA * V_eff)
    print(f"Fractional effective N1: {N1_eff:.2f}  C1: {conc1_eff*1e9:.3f} nM")
    print(f"Fractional effective N2: {N2_eff:.2f}  C2: {conc2_eff*1e9:.3f} nM")
else:
    print("No valid windows to fit ACF.")















