# -*- coding: utf-8 -*-
"""
FCS Microrheology Analysis:
 1) Plot raw autocorrelation G(t)
 2) Invert Rathgeber et al. eq 4 → Calculate MSD
 3) Fit MSD in certain range to extract power-law slope
 4) Compute and plot |G*| over that range
 5) Compute and plot G′, G″ over that range

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma

# 1) Plot styling & colors
plt.style.use({
    'font.size': 18,
    'figure.figsize': (7, 5),
    'lines.linewidth': 2,
    'lines.markersize': 6,
})
colors = {
    'auto': "#004488",
    'fit':  "#DD3333",
    'Gp':   "#33AA33",
    'Gpp':  "#AA3333",
    'Gabs': "#3333AA",
}

# 2) the correct path of the data 
folder_path = r"C:\Users\mirza010\OneDrive - Universiteit Utrecht\Desktop\UU data\Projects\nanoparticles\try"
#INPUTS
R_bead = 210e-9               # tracer radius [m]
k_B, T  = 1.380e-23, 295     # J/K, K
gamma_i= 1.0
delta_i= 0
rho_bead = 2200           # polystyrene density [kg/m^3]
m_bead   = (4/3)*np.pi*R_bead**3 * rho_bead

# 3) Load PSF parameters & N from the fit workbook
w_xy = w_z = N = None
for fn in os.listdir(folder_path):
    if fn.endswith("1expfit.xlsx"):
        df   = pd.read_excel(os.path.join(folder_path, fn), header=None)
        w_xy = df.iloc[10, 1] * 1e-6  # [μm → m]
        w_z  = df.iloc[11, 1] * 1e-6  # [μm → m]
        N    = df.iloc[7, 1]          # dimensionless
        break
    
# correction of size of the particle in comparison to the beam waist
# w_xy_C = np.sqrt(w_xy**2 + R_bead**2)
# N_C = N * (1 + R_bead/w_xy)

w_xy_C = w_xy
N_C = N
# 4) Load raw autocorrelation G(t) from CF.xlsx (no normalization)
t      = None
G_exp  = None
skip   = 2
for fn in os.listdir(folder_path):
    if fn.endswith("CF.xlsx"):
        df   = pd.read_excel(os.path.join(folder_path, fn), header=None)
        t    = df.iloc[skip:, 0].astype(float).values  # maybe in ms
        G_exp= df.iloc[skip:, 1].astype(float).values  # raw G(t)
        break

# 5) Sanity-check (avoid ambiguous truth on arrays)
if any(v is None for v in (w_xy_C, w_z, N_C, t, G_exp)):
    raise RuntimeError("Failed to load one of: w_xy, w_z, N, t or G_exp.")

# 6) Convert to SI: if t read in ms, convert to seconds
t = t * 1e-3  # comment out if already in seconds

# 7)  Plot raw G(t)
plt.figure()
plt.loglog(t, G_exp, 'o-', color=colors['auto'])
plt.xlabel('Lag time $t$ (s)')
plt.ylabel('Autocorrelation $G(t)$')
plt.title('Raw FCS autocorrelation')
plt.tight_layout()

# 8) Invert Rathgeber eq 4 → MSD(t)
def compute_msd(G, w_xy_C, w_z, N_C):
    a = 2.0 / (3.0 * w_xy_C**2)
    b = 2.0 / (3.0 * w_z**2)
    msd = np.empty_like(G)
    for i, Gi in enumerate(G):
        C2    = (1.0/(N_C*Gi))**2
        coefs = [a*a*b, a*a + 2*a*b, 2*a + b, 1 - C2]
        roots = np.roots(coefs)
        realr = roots[np.isreal(roots)].real
        pos   = realr[realr >= 0]
        msd[i]= pos.min() if pos.size else np.nan
    return msd
msd_full = compute_msd(G_exp, w_xy_C, w_z, N_C)

# 2) Approximate MSD from eq 6:
def compute_msd_approx(G, w_xy_C, N_C):
    return 1.5 * w_xy_C**2 * (1.0/(N_C * G) - 1.0)

msd_approx = compute_msd_approx(G_exp, w_xy_C, N_C)



# 9) Fit power-law in [4e-5, 5e-2] s
mask       = (t >=2e-4) & (t <= 2e-2)
t_fit      = t[mask]
msd_fit    = msd_full[mask]
log_t_fit  = np.log10(t_fit)
log_msd_fit= np.log10(msd_fit)
slope, intercept = np.polyfit(log_t_fit, log_msd_fit, 1)

# 10) Plot MSD + fit
plt.figure()
plt.loglog(t, msd_full, 'o-', color=colors['auto'], label='MSD')
plt.loglog(t, msd_approx, 'g*', label="Approx (eq 6)")

t_line = np.array([t_fit.min(), t_fit.max()])
msd_line = 10**intercept * t_line**slope
plt.loglog(t_line, msd_line, '--', color=colors['fit'],
           label=f'slope={slope:.2f}')
plt.xlabel('Lag time $t$ (s)')
plt.ylabel('MSD ⟨∆r²(t)⟩ (m²)')
plt.title('MSD & power-law fit')
plt.legend()
plt.tight_layout()
print(f"Power-law exponent (4e-5–5e-2 s): {slope:.3f}")

# 11) Restrict arrays to fit range
t_m   = t[mask]
msd_m = msd_full[mask]

# 12) Compute R(ω)=d ln MSD/d ln t and β(ω)=d² ln MSD/d (ln t)²
#there factors coming from the taylor expansion of MSD 
ln_t   = np.log(t_m)
ln_msd = np.log(msd_m)
R      = np.gradient(ln_msd, ln_t)
beta   = np.gradient(R, ln_t)
max_beta = np.nanmax(np.abs(beta))
# 13) Compute |G*| (eq 14) with γ, δ (eqs 15–16)
max_beta    = np.nanmax(np.abs(beta))
max_corr    = np.nanmax((1+R+R**2)/2 * np.abs(beta))
max_delta   = np.nanmax((R+1)*np.abs(beta))
print("max |β| =", max_beta)
print("max γ-1 correction ≈", max_corr)
print("max |δ| ≈", max_delta)
norm   = np.sqrt(gamma_i**2 + delta_i**2)
Gamma_t= gamma(1.0 + R)
G_abs  = (k_B*T)/(np.pi*R_bead*msd_m*Gamma_t) * norm

G_prime = G_abs * ( gamma_i*np.cos(np.pi*R/2) + delta_i*np.sin(np.pi*R/2) )/norm
G_double= G_abs * (-delta_i*np.cos(np.pi*R/2) + gamma_i*np.sin(np.pi*R/2))/norm
omega   = 1.0 / t_m

# 14) Sort by ω
idx     = np.argsort(omega)
ω       = omega[idx]
Gp      = G_prime[idx]
Gpp     = G_double[idx]
Gabs    = G_abs[idx]




# 2) compute inertia term G_iner(ω) = m ω² / (6 π R)
G_iner = m_bead * ω**2 / (6*np.pi*R_bead)

# 3) add it to your previously computed Gp_s (Mason–Weitz storage)
Gp_corrected = Gp + G_iner
# … after sorting omega_s, Gp_s, Gpp_s, Gabs_s …

# --- Fit G'' to a power law: G'' ∝ ω^n ---
# (you can also restrict to a sub-range if desired)
# compute logs
log_ω   = np.log10(ω)
log_Gpp = np.log10(Gpp)

# mask out any non-finite values
mask_valid = np.isfinite(log_ω) & np.isfinite(log_Gpp)

# also ensure you have at least two different x-values
if mask_valid.sum() < 2:
    raise RuntimeError("Not enough valid points to do a regression!")
if np.ptp(log_ω[mask_valid]) == 0:
    raise RuntimeError("All log(ω) are identical — can't fit a slope!")

# now safely perform the fit on the valid subset
slope_Gpp, intercept_Gpp = np.polyfit(log_ω[mask_valid],
                                      log_Gpp[mask_valid], 1)
print(f"Fitted G'' exponent: {slope_Gpp:.3f}")

# 16) Plot |G*| vs ω (magnitude) like Fig. 4
plt.figure()
plt.loglog(ω, Gabs, 's-', color=colors['Gabs'], label="|G*|")
plt.xlabel('Angular frequency ω (rad/s)')
plt.ylabel('|G*(ω)| (Pa)')
plt.title("Magnitude of complex modulus")
plt.legend(); plt.tight_layout()


# 15) Plot G' & G''
plt.figure()
plt.loglog(ω, Gp_corrected,  'o',    markerfacecolor='none',   markeredgecolor=colors['Gpp'], color=colors['Gpp'],  label="G' (storage)",markersize=9)
plt.loglog(ω, Gpp, 'o', color=colors['Gpp'], label="G'' (loss)",markersize=9)
plt.xlabel('Angular frequency ω (rad/s)')
plt.ylabel('Modulus (Pa)')
plt.title("Dynamic shear moduli G' and G''")
plt.legend();  plt.tight_layout()
Gpp_fit = 10**intercept_Gpp * ω**slope_Gpp
eta = 10**intercept_Gpp*1e3
print('Viscosity =', round(eta, 2), 'mPa.s')
# 6) Plot G'' and the fit
plt.loglog(ω, Gpp_fit, '--', color='k', label=f"Fit: n={slope_Gpp:.2f}")
plt.xlabel('Angular frequency ω (rad/s)')
plt.ylabel("G''(ω) (Pa)")
plt.title("Loss modulus G'' with power-law fit")
plt.legend()
plt.tight_layout()
plt.show()












#%% CHECK POINTS







