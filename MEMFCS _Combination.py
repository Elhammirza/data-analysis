import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import minimize



plt.style.use({
    'font.size': 18,
    'figure.figsize': (7, 5),
    'lines.linewidth': 2,
    'lines.markersize': 6,
})


# Define your custom color
my_color1 ="#004488"  # Hexadecimal color (a shade of blue)
my_color2 = "#DDAA33"  # Hexadecimal color (a shade of blue)
my_color3 = "#BB5566"  # Hexadecimal color (a shade of blue)

#############################################################################
# 1) Inputs of the experiments
###############################################################################

num_components = 100 # number of alpha_i points we will use for MEMFCS fit
t_min, t_max =0.001, 1000   # Specify the begining and end point for fitting autocorrelation F
sigma=0.01       #standard deviation 
alpha = 0.1  # Adjust regularization factor, which specify the contribution of entropy
etha=0.89*10**(-3) #Pa.s
k_B=1.380*10**(-23) #joules per kelvin    1 Joule = 1 Pa.m3
T= 295
###############################################################################
# 2) Load experimental data from excel 
###############################################################################

# Ask user for fit type
fit_type = None
while fit_type not in (1, 2):
    try:
        fit_type = int(input("Select exponent fitting (1 for single, 2 for double): "))
    except ValueError:
        pass

# Folder containing data
folder_path = r"C:\Users\mirza010\OneDrive - Universiteit Utrecht\Desktop\UU data\Projects\Nanogel-polymer encapsulation_Neshat\PIESA_Nanogel_encapsulation\PIESA_Nanogel_encapsulation\2025_03_04_PAA47_Rb_pink_Rawdata\test"

# Read Excel fit file based on choice
for filename in os.listdir(folder_path):
    if fit_type == 1 and filename.endswith("1expfit.xlsx"):
        df_fit = pd.read_excel(os.path.join(folder_path, filename), header=None)
        r0 = df_fit.iloc[10,1]
        t_D_fit = df_fit.iloc[3,1] * 1e-3
        t_D_fit_err = df_fit.iloc[3,2] * 1e-3
        amp = 0.04
        r_over_l = df_fit.iloc[10,1] / df_fit.iloc[11,1]
        break
    elif fit_type == 2 and filename.endswith("2expfit.xlsx"):
        df_fit = pd.read_excel(os.path.join(folder_path, filename), header=None)
        r0=df_fit.iloc[13,1]  # radius of the PSF (micro meter)
        t_D_fit1=df_fit.iloc[3,1]*10**(-3) # in ms 
        amp_fit1=df_fit.iloc[2,1] # in ms
        t_D_fit_err1=df_fit.iloc[3,2]*10**(-3) # in ms 
        t_D_fit2=df_fit.iloc[5,1]*10**(-3) # in ms 
        amp_fit2=df_fit.iloc[4,1] # in ms
        t_D_fit_err2=df_fit.iloc[5,2]*10**(-3) # in ms 
        amp=0.04
        triple_time=df_fit.iloc[8,1]*10**(-3)
        r_over_l = df_fit.iloc[13,1]/df_fit.iloc[14,1]
        break



data_list = []
x = 2 # Skip first rows 
for filename in os.listdir(folder_path):
    if filename.endswith("CF.xlsx"):
        excel_path = os.path.join(folder_path, filename)

        # Read correlation function data
        df = pd.read_excel(excel_path, header=None)
        t = np.array(df.iloc[x:, 0], dtype=float)  # Time lag
        G_exp = np.array(df.iloc[x:, 1], dtype=float)  # Autocorrelation
        G_exp = G_exp / np.max(G_exp)  # Normalize to 1

        # Extract nM concentration from filename
        filename_no_ext, _ = os.path.splitext(filename)
        parts = filename_no_ext.split("_")
        nM_label = next((part for part in parts if "nM" in part), None)

        # Extract numeric concentration
        numeric_part = float(nM_label.replace("nM", "")) if nM_label else float('inf')

        # Store data
        data_list.append((numeric_part, t, G_exp, nM_label if nM_label else filename))
        break  # Load only the first matching file

# Sort by concentration (if needed)
data_list.sort(key=lambda x: x[0])

# Extract experimental time and correlation data
_, t_exp, G_exp, _ = data_list[0]

##########################################################
# 3)  THE MEMFCS FIT 
########################################################


# Log-spaced diffusion time, because the diffusion time of experiment can be pretty wide
t_D = np.logspace(np.log10(t_min), np.log10(t_max), num_components)

#  initialization points for amplitudes
a_init = np.exp(-np.log(t_D / np.median(t_D))**2)  # Gaussian-like initial guess

a_init /= np.sum(a_init)  # Normalize to sum=1
scale_init = np.max(G_exp)

# Combine initial parameters
params_init = np.concatenate([a_init, [scale_init]])

#############################################################
# 3) Define MEMFCS method
########################################################

# Use r_over_l as a default parameter
def autocorrelation_model(t, t_D, a, scale_factor, r_over_l=r_over_l):
    t_2d = t[:, np.newaxis]
    tD_2d = t_D[np.newaxis, :]
    denom = (1.0 + t_2d/tD_2d) * np.sqrt(1.0 + (r_over_l**2)*(t_2d/tD_2d))
    basis = 1.0 / denom
    G_calc = scale_factor * np.sum(a * basis, axis=1)
    return G_calc
########################################################
# 4) DEFINE Entropy and chi-swuares 
#########################################################

def entropy(a):
    """ Shannon entropy (smoothness constraint) """
    return -np.sum(a * np.log(a + 1e-12))

def chi_square(params, t, G_data, sigma, t_D, alpha):
    """ Chi-square error function """
    a = params[:-1]
    scale_factor = params[-1]
    G_calc = autocorrelation_model(t, t_D, a, scale_factor)
    residuals = (G_calc - G_data) / sigma
    return np.sum(residuals**2) / len(t)

###########################################################
# 5) DEFINE COST FUNCTION (Chi-Square + Entropy Regularization)
###########################################################

def cost_function(params, t, G_data, sigma, t_D, alpha=alpha):
    """
    cost = chi-square - alpha * entropy
    Minimizing this ensures best fit with maximum smoothness in 'a'.
    """
    a = params[:-1]
    scale_factor = params[-1]
    c2 = chi_square(params, t, G_data, sigma, t_D, alpha)
    ent = entropy(a)
    return c2 - alpha * ent

###########################################################
# 6) CONSTRAINTS AND BOUNDS (SLSQP) - it is a gradient based method
##########################################################

def constraint_sum_a(params):
    """ Ensures sum of all a_i equals 1 """
    return np.sum(params[:-1]) - 1.0

cons = {
    'type': 'eq',
    'fun': constraint_sum_a
}

# Bounds: a_i >= 0, scale_factor >= 0
bounds = [(0.0, None)] * num_components + [(0.0, None)]

############################################################
# 7) OPTIMIZE MEMFCS PARAMETERS
############################################################

# Estimate sigma from noise level
# sigma = np.std(G_exp[-10:])  # Use last few points as baseline noise estimate

res = minimize(
    fun=cost_function,
    x0=params_init,
    args=(t_exp, G_exp, sigma, t_D, alpha),
    method='SLSQP',
    constraints=[cons],
    bounds=bounds,
    options={'maxiter': 1000, 'ftol': 1e-12, 'disp': True}
)

# Extract optimized parameters
params_opt = res.x
a_opt = params_opt[:-1]
scale_opt = params_opt[-1]

print("Optimization success:", res.success)
print("Message:", res.message)

#######################################################
#%% 8) PLOT AND ANALYZE RESULTS
#######################################################

# 8.1) Fit the experimental autocorrelation data
G_fit = autocorrelation_model(t_exp, t_D, a_opt, scale_opt)
# Remove triplet contributions: set amplitude to zero for t_D values below 0.05 ms
threshold = 0.05  # in ms
a_opt[t_D < threshold] = 0
a_opt = a_opt / np.sum(a_opt)  # Re-normalize



#%% Plotting section 
plt.figure()
plt.plot(t_exp, G_exp, 'o', color='gray', label='Experimental Data')
plt.plot(t_exp, G_fit, 'r-', label='Fitted Curve')
plt.xscale('log')
plt.xlabel('Time lag (s)')
plt.ylabel('Autocorrelation G(τ)')
plt.title('FCS - MEMFCS Fit (Experimental Data)')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()

# 8.2) Plot the recovered diffusion time distribution
plt.figure()
plt.plot(t_D, a_opt, 'o-')
plt.xscale('log')
plt.xlabel('Diffusion Time (ms)')
plt.ylabel(' $a_i$')
# plt.grid(True, which='both', ls='--', alpha=0.5)


# Show the final distribution
plt.tight_layout()

if fit_type==1:
    plt.errorbar(x=t_D_fit, y=amp,  xerr=t_D_fit_err, yerr=0, fmt='o', color='red',  ecolor='black',elinewidth=1,  capsize=4)
    t_D=t_D*10**(3) # to be in us 
    #diffusion calculation: 
    D= (r0**2)/(4*t_D*10**(-6))   #um^2/s
    #plot of iffusion Coefficient Distribution 
    plt.figure(figsize=(8, 6))
    plt.plot(D, a_opt, marker='*', linestyle='-', label='size',color=my_color3,markersize=12)
    plt.xscale('log')
    # plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xlabel('Diffusion coefficient ($\mu$m$^2$/s)', fontsize=24)
    plt.ylabel('Normalized amplitude', fontsize=24)
    plt.title('Diffusion Coefficient Distribution')
    # #size distribution 
    s_p=(k_B*T)/(6*np.pi*etha*D*10**(-12)) #size of the particle in meter
    s_p=s_p* 10**(9) #size in nm
    plt.figure(figsize=(8, 6))
    plt.plot(s_p, a_opt, marker='D', linestyle='-', label='size',color=my_color2)
    plt.xscale('log')
    # plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xlabel('size (nm)', fontsize=32)
    plt.ylabel('Normalized amplitude', fontsize=32)

elif fit_type==2:
# Scatter plot with error bars
    plt.errorbar(x=t_D_fit1, y=amp,  xerr=t_D_fit_err1, yerr=0, fmt='o', color='red',  ecolor='black',elinewidth=1,  capsize=4)
    plt.errorbar(x=t_D_fit2, y=amp,  xerr=t_D_fit_err2, yerr=0, fmt='o', color='red',  ecolor='black',elinewidth=1,  capsize=4)
    t_D=t_D*10**(3) # to be in us
    t_D_fit1=t_D_fit1*10**(3)
    t_D_fit2=t_D_fit2*10**(3)

    #diffusion calculation: 
    D= (r0**2)/(4*t_D*10**(-6))   #um^2/s
    D_fit1= (r0**2)/(4*t_D_fit1*10**(-6))
    D_fit2= (r0**2)/(4*t_D_fit2*10**(-6))

    #plot of iffusion Coefficient Distribution 
    plt.figure(figsize=(8, 6))
    plt.plot(D, a_opt/np.max(a_opt), marker='*', linestyle='-', label='size',color=my_color3,markersize=12)
    plt.errorbar(x=D_fit1, y=amp_fit1/(amp_fit1+amp_fit2),  xerr=0, yerr=0, fmt='o', color='red',  ecolor='black',elinewidth=1,  capsize=4)
    plt.errorbar(x=D_fit2, y=amp_fit2/(amp_fit1+amp_fit2),  xerr=0, yerr=0, fmt='o', color='red',  ecolor='black',elinewidth=1,  capsize=4)

    plt.xscale('log')
    # plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xlabel('Diffusion coefficient ($\mu$m$^2$/s)')
    plt.ylabel('Normalized amplitude')

    s_p=(k_B*T)/(6*np.pi*etha*D*10**(-12)) #size of the particle in meter
    s_p_fit1=(k_B*T)/(6*np.pi*etha*D_fit1*10**(-12)) #size of the particle in meter
    s_p_fit2=(k_B*T)/(6*np.pi*etha*D_fit2*10**(-12)) #size of the particle in meter
    t_D_fit_err1=t_D_fit_err1*10**(-11)
    s_p=s_p* 10**(9) #size in nm
    s_p_fit1=s_p_fit1* 10**(9) #size in nm
    s_p_fit2=s_p_fit2* 10**(9) #size in nm

    plt.figure(figsize=(8, 6))
    plt.plot(s_p, a_opt/np.max(a_opt), marker='D', linestyle='-', label='size',color=my_color2)
    plt.errorbar(x=s_p_fit1, y=amp_fit1/(amp_fit1+amp_fit2),  xerr=t_D_fit_err1, yerr=0, fmt='o', color='red',  ecolor='black',elinewidth=1,  capsize=4)
    plt.errorbar(x=s_p_fit2, y=amp_fit2/(amp_fit1+amp_fit2),  xerr=t_D_fit_err1, yerr=0, fmt='o', color='red',  ecolor='black',elinewidth=1,  capsize=4)
    print(s_p_fit1)
    print(s_p_fit2)

    plt.xscale('log')
    # plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xlabel('size (nm)')
    plt.ylabel('Normalized amplitude')



plt.show()



















