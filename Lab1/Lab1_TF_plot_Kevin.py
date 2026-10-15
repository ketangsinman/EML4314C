import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


# ============================================================
# HELPER FUNCTION
# ============================================================

def calculate_r2(y_exp, y_model):
    """
    Calculate coefficient of determination (R^2)
    between experimental and modeled data.
    """
    ss_res = np.sum((y_exp - y_model)**2)
    ss_tot = np.sum((y_exp - np.mean(y_exp))**2)

    return 1 - ss_res / ss_tot


# ============================================================
# READ EXPERIMENTAL DATA
# ============================================================

step_data = pd.read_csv("Kevin_Step_Response.csv")
bode_data = pd.read_csv("Kevin_Bode_Plot_Data.csv")


# ============================================================
# STEP-RESPONSE MODEL
# G_step(s) = A_step / (tau_step*s + 1)
# ============================================================

# Experimental voltage step
delta_V = (300 - (-300)) * (9 / 1023)  # V

# Experimental initial/final angular velocities
omega_i = -4.56021299   # rad/s
omega_f =  4.511413392  # rad/s

delta_omega = omega_f - omega_i

# Times used to determine time constant
t_i = 0.000  # s
t_63 = 0.030  # s

# Step-derived parameters
A_step = delta_omega / delta_V
tau_step = t_63 - t_i
p_step = -1 / tau_step

# Step-derived transfer function
G_step = signal.TransferFunction(
    [A_step],
    [tau_step, 1]
)


# ============================================================
# BODE-RESPONSE MODEL
# G_bode(s) = A_bode / (tau_bode*s + 1)
# ============================================================

# Experimental Bode data
w_exp = bode_data["Frequency (Rad/s)"].to_numpy()
mag_exp = bode_data["Gain (dB)"].to_numpy()
phase_exp = bode_data["Phase (deg)"].to_numpy()


# ------------------------------------------------------------
# Bode-derived parameters
# ------------------------------------------------------------

# Approximate DC gain using lowest-frequency experimental point
DC_gain_dB = mag_exp[0]

# Pole/corner frequency determined from experimental Bode plot
omega_p_bode = 32.0  # rad/s

# Convert Bode magnitude from dB to linear gain
A_bode = 10**(DC_gain_dB / 20)

# For a first-order system:
# omega_p = 1/tau
tau_bode = 1 / omega_p_bode
p_bode = -omega_p_bode

# Bode-derived transfer function
G_bode = signal.TransferFunction(
    [A_bode],
    [tau_bode, 1]
)


# ============================================================
# PRINT MODEL PARAMETERS
# ============================================================

print("STEP-DERIVED MODEL")
print(f"A_step   = {A_step:.4f} (rad/s)/V")
print(f"tau_step = {tau_step:.4f} s")
print(f"p_step   = {p_step:.4f} rad/s")

print("\nBODE-DERIVED MODEL")
print(f"DC gain  = {DC_gain_dB:.4f} dB")
print(f"A_bode   = {A_bode:.4f} (rad/s)/V")
print(f"tau_bode = {tau_bode:.4f} s")
print(f"p_bode   = {p_bode:.4f} rad/s")


# ============================================================
# STEP-RESPONSE COMPARISON
# ============================================================

# Experimental step-response data
t_exp = step_data["Time"].to_numpy()
omega_exp = step_data["Response"].to_numpy()

# signal.step() requires equally spaced time values
t_model = np.linspace(
    0,
    t_exp.max(),
    1000
)

# Calculate unit-step responses
t_step, y_step_unit = signal.step(
    G_step,
    T=t_model
)

t_bode, y_bode_unit = signal.step(
    G_bode,
    T=t_model
)

# Scale unit-step responses to actual experimental voltage step
# and add initial angular velocity
omega_step_model = omega_i + delta_V * y_step_unit
omega_bode_model = omega_i + delta_V * y_bode_unit


# ============================================================
# STEP RESPONSE R^2
# ============================================================

# Interpolate the model onto the experimental timestamps
omega_step_at_exp = np.interp(
    t_exp,
    t_step,
    omega_step_model
)

# Calculate R^2
R2_step = calculate_r2(
    omega_exp,
    omega_step_at_exp
)

print("\nMODEL FIT")
print(f"Step Response R^2  = {R2_step:.4f}")


# ============================================================
# PLOT STEP RESPONSE
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    t_step,
    omega_step_model,
    label=f"Step-Derived Model ($R^2$ = {R2_step:.3f})"
)

plt.scatter(
    t_exp,
    omega_exp,
    label="Experimental Data",
    s=15,
    color="Orange"
)

plt.title("Kevin Experimental and Modeled Step Responses")
plt.xlabel("Time [s]")
plt.ylabel("Angular Velocity [rad/s]")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# BODE-PLOT COMPARISON
# ============================================================

# Smooth logarithmically spaced frequency vector
w_model = np.logspace(
    np.log10(w_exp.min()),
    np.log10(w_exp.max()),
    500
)

# Bode response of step-derived model
_, mag_step, phase_step = signal.bode(
    G_step,
    w=w_model
)

# Bode response of Bode-derived model
_, mag_bode, phase_bode = signal.bode(
    G_bode,
    w=w_model
)


# ============================================================
# BODE R^2
# ============================================================

# Interpolate Bode model onto experimental frequencies.
# Use log10 frequency because the Bode frequency axis is logarithmic.
mag_bode_at_exp = np.interp(
    np.log10(w_exp),
    np.log10(w_model),
    mag_bode
)

phase_bode_at_exp = np.interp(
    np.log10(w_exp),
    np.log10(w_model),
    phase_bode
)

# Calculate magnitude R^2
R2_mag = calculate_r2(
    mag_exp,
    mag_bode_at_exp
)

# Calculate phase R^2
R2_phase = calculate_r2(
    phase_exp,
    phase_bode_at_exp
)

print(f"Bode Magnitude R^2 = {R2_mag:.4f}")
print(f"Bode Phase R^2     = {R2_phase:.4f}")


# ============================================================
# PLOT BODE RESPONSE
# ============================================================

fig, ax = plt.subplots(
    2, 1,
    figsize=(9, 7),
    sharex=True
)


# ---------------- Magnitude ----------------

ax[0].semilogx(
    w_model,
    mag_bode,
    label=f"Bode-Derived Model ($R^2$ = {R2_mag:.3f})"
)

ax[0].scatter(
    w_exp,
    mag_exp,
    label="Experimental Data",
    color="Orange"
)

ax[0].set_ylabel("Magnitude [dB]")
ax[0].set_title("Kevin Experimental and Modeled Bode Responses")
ax[0].legend()
ax[0].grid(True, which="both")


# ---------------- Phase ----------------

ax[1].semilogx(
    w_model,
    phase_bode,
    label=f"Bode-Derived Model ($R^2$ = {R2_phase:.3f})"
)

ax[1].scatter(
    w_exp,
    phase_exp,
    label="Experimental Data",
    color="Orange"
)

ax[1].set_xlabel("Angular Frequency [rad/s]")
ax[1].set_ylabel("Phase [deg]")
ax[1].legend()
ax[1].grid(True, which="both")


plt.tight_layout()
plt.show()