import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Robust FCS-style autocorrelation with proper normalization
def autocorrelation_fft(f):
    f = np.asarray(f, dtype=float)
    N = len(f)
    f_mean = np.mean(f)
    if f_mean == 0:
        raise ValueError("Mean of the signal is zero. Cannot normalize autocorrelation.")
    df = f - f_mean
    fft_f = np.fft.fft(df, n=2*N)
    acf = np.fft.ifft(fft_f * np.conjugate(fft_f))[:N].real
    acf /= np.arange(N, 0, -1)  # normalize by number of points at each lag
    G = acf / (f_mean**2)   # normalize by mean^2, per FCS convention
    return G

# --- User-specified ACF window (in seconds) ---
start_time = 0.0
end_time = 30.0

folder_path = r"C:\Users\mirza010\OneDrive - Universiteit Utrecht\Desktop\UU data\Projects\data\Bleachiing\Bleachiing\try"
suffix = "Countrates.csv"
data = {}

for fn in os.listdir(folder_path):
    if fn.endswith(suffix):
        # Tab-separated, skip first two lines of metadata/header
        df_raw = pd.read_csv(os.path.join(folder_path, fn), sep='\t', header=None, skiprows=2)
        t = df_raw.iloc[:,0].astype(float).reset_index(drop=True)
        c = df_raw.iloc[:,1].astype(float).reset_index(drop=True)
        key = os.path.splitext(fn)[0]
        data[key] = pd.DataFrame({'time': t, 'count_rate': c})

# --- Plot count rates (intensity traces) ---
plt.figure(figsize=(10, 4))
for label, df in data.items():
    mask = (df['time'] >= start_time) & (df['time'] <= end_time)
    plt.plot(df['time'][mask], df['count_rate'][mask], label=label)
plt.xlabel('Time (s)')
plt.ylabel('Count Rate (kCounts/s)')
plt.title('Count Rate Trace')
plt.legend()
plt.tight_layout()
plt.show()

# --- Continue with autocorrelation as before ---
plot_curves = []

for fn in os.listdir(folder_path):
    if fn.endswith("CF.csv"):
        df_cf = pd.read_csv(os.path.join(folder_path, fn), sep='\t', header=None, skiprows=2)
        t_cf = df_cf.iloc[:,0].astype(float).reset_index(drop=True).values
        G_cf = df_cf.iloc[:,1].astype(float).reset_index(drop=True).values

        for label, df in data.items():
            time = df['time'].values
            F = df['count_rate'].values
            mask = (time >= start_time) & (time <= end_time)
            t_win = time[mask]
            F_win = F[mask]
            if len(F_win) < 2:
                continue

            print(f"{label}: Mean={np.mean(F_win)}, Max={np.max(F_win)}, Min={np.min(F_win)}, N={len(F_win)}")
            try:
                acf = autocorrelation_fft(F_win)
            except ValueError as e:
                print(f"Skipping {label} due to error: {e}")
                continue

            dt = np.median(np.diff(t_win))
            tau_all = np.arange(1, len(acf)) * dt * 1000  # in ms
            G = acf[1:]
            # G = G / G[0]  # Uncomment if you want G(0) == 1

            G_interp = np.interp(t_cf, tau_all, G)
            plot_curves.append((f'Calculation_FFT_{label}', t_cf, G_interp))

        plot_curves.append(('Software', t_cf, G_cf))

# --- Plot autocorrelation curves ---
plt.figure()
for label, x, y in plot_curves:
    plt.semilogx(x, y, label=label)
plt.xlabel('Lag time (ms)')
plt.ylabel(r'$G(\tau)$')
plt.title('Autocorrelation via FFT (Normalized)')
plt.legend()
plt.tight_layout()
plt.show()
