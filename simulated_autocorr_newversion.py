import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.style.use({
    'font.size': 13,
    'figure.figsize': (7, 5),
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

###############################################################################
# 1) SIMULATE A TWO-COMPONENT FCS AUTOCORRELATION
###############################################################################

# "True" parameters
r_over_l = 0.14
tau_D1_true = 0.01  # seconds
tau_D2_true = 1.0   # seconds
a1_true = 0.6
a2_true = 0.4

noise_amplitude = 0.001  # level of random noise

# Time lag values (log-spaced)
tau = np.logspace(-4, 1, 500)  # from 1e-4 to 10 seconds

def true_autocorrelation(t):
    """
    Two-component FCS model:
    G(t) = a1 * f(tau_D1) + a2 * f(tau_D2),
    where f(tau_D) includes 3D diffusion factor + shape factor (r_over_l).
    """
    term1 = a1_true / ((1 + t/tau_D1_true)
                       * np.sqrt(1 + (r_over_l**2)*(t/tau_D1_true)))
    term2 = a2_true / ((1 + t/tau_D2_true)
                       * np.sqrt(1 + (r_over_l**2)*(t/tau_D2_true)))
    return term1 + term2

# Simulate "ideal" G(t)
G_ideal = true_autocorrelation(tau)

# Add random noise
np.random.seed(123)  # it produce same set of randome numbers for reproducibility 
G_sim = G_ideal + noise_amplitude * (np.random.rand(len(tau)) - 0.5)

###############################################################################
# 2) SET UP THE MEMFCS FIT (a_i DISTRIBUTION + SEPARATE scale_factor)
###############################################################################

num_components = 100
t_min, t_max = 1e-5, 10

# Log-spaced grid for tau_D
t_D = np.logspace(np.log10(t_min), np.log10(t_max), num_components)

# Initial guess for a_i and scale_factor
a_init = np.ones(num_components) / num_components  # sum=1
scale_init = 1.0
params_init = np.concatenate([a_init, [scale_init]])  # shape = (num_components+1,)

# The FCS model
def autocorrelation_model(t, t_D, a, scale_factor, r_over_l=0.14):
    """
    G_model(t) = scale_factor * sum_i [ a_i * basis_i(t) ],
    where sum_i a_i = 1 (the distribution),
    basis_i(t) = 1 / [(1 + t/tD_i) * sqrt(1 + (r_over_l^2)*(t/tD_i))].
    """
    t_2d = t[:, np.newaxis]
    tD_2d = t_D[np.newaxis, :]

    denom = (1.0 + t_2d/tD_2d) * np.sqrt(1.0 + (r_over_l**2)*(t_2d/tD_2d))
    basis = 1.0 / denom

    G_calc = scale_factor * np.sum(a * basis, axis=1)
    return G_calc

def entropy(a):
    # Shannon entropy with a small offset to avoid log(0)
    return -np.sum(a * np.log(a + 1e-12))

def chi_square(params, t, G_data, sigma, t_D, alpha):
    a = params[:-1]
    scale_factor = params[-1]
    G_calc = autocorrelation_model(t, t_D, a, scale_factor, r_over_l)
    residuals = (G_calc - G_data)/sigma
    return np.sum(residuals**2) / len(t)

def cost_function(params, t, G_data, sigma, t_D, alpha=1.0):
    """
    cost = chi-square - alpha * entropy
    Minimizing => best fit with maximum smoothness in 'a'.
    """
    a = params[:-1]
    scale_factor = params[-1]
    c2 = chi_square(params, t, G_data, sigma, t_D, alpha)
    # sum(a_i) should be = 1 by constraint => 'a' is a probability distribution
    ent = entropy(a)
    return c2 - alpha*ent

# We estimate sigma from noise_amplitude (roughly)
sigma = noise_amplitude if noise_amplitude > 0 else 1e-3
alpha = 1  # You can try alpha=0.1 if you want sharper peaks

###############################################################################
# 3) CONSTRAINTS AND BOUNDS (SLSQP)
###############################################################################

def constraint_sum_a(params):
    # sum of all a_i minus 1
    return np.sum(params[:-1]) - 1.0

cons = {
    'type': 'eq',
    'fun': constraint_sum_a
}

# Bounds: a_i >= 0, scale_factor >= 0
bounds = [(0.0, None)]*num_components + [(0.0, None)]

###############################################################################
# 4) OPTIMIZE
###############################################################################

res = minimize(
    fun=cost_function,
    x0=params_init,
    args=(tau, G_sim, sigma, t_D, alpha),
    method='SLSQP',
    constraints=[cons],
    bounds=bounds,
    options={'maxiter': 1000, 'ftol': 1e-12, 'disp': True}
)

# Extract solution
params_opt = res.x
a_opt = params_opt[:-1]
scale_opt = params_opt[-1]

print("Optimization success:", res.success)
print("Message:", res.message)

###############################################################################
# 5) PLOT AND ANALYZE THE RESULTS
###############################################################################

# 5.1) Plot the fitted vs. simulated autocorrelation
G_fit = autocorrelation_model(tau, t_D, a_opt, scale_opt, r_over_l=r_over_l)

# plt.figure()
# plt.plot(tau, G_sim, 'o', color='gray', label='Simulated Data')
# plt.plot(tau, G_fit, 'r-', label='Fitted Curve')
# plt.xscale('log')
# plt.xlabel('Time lag (s)')
# plt.ylabel('Autocorrelation G(τ)')
# plt.title('Two-Component FCS - MEMFCS Fit')
# plt.grid(True, which='both', ls='--', alpha=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()

# 5.2) Plot the distribution a_opt vs. t_D (sums to 1)
plt.figure()
plt.plot(t_D, a_opt, 'o-', label='Recovered distribution')
plt.xscale('log')
plt.xlabel('Diffusion Time (s)')
plt.ylabel('Fraction a_i')
plt.title('Distribution (sum(a_i)=1)')
plt.grid(True, which='both', ls='--', alpha=0.5)

# Mark the "true" diffusion times
# plt.axvline(tau_D1_true, color='green', linestyle='--', label='True τD1=0.01')
# plt.axvline(tau_D2_true, color='blue', linestyle='--', label='True τD2=1.0')
# plt.scatter(tau_D1_true, a1_true, color='red', s=200)  # s controls the marker size
# plt.scatter(tau_D2_true, a2_true, color='red', s=200)  # s controls the marker size
plt.legend()
plt.tight_layout()
plt.show()

print("Sum of a_i =", np.sum(a_opt))
print("scale_factor =", scale_opt)



