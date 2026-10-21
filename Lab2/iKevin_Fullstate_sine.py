import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

subset = pd.read_csv("Lab2/Kevin_fullstate_sine.csv")

plt.plot(
    figsize=(16,10),
    sharex=True,
    sharey=True
)

 # Falling Edge Detection
desired = subset["Desired Position"].to_numpy()
time = subset["Time"].to_numpy()

signs = np.sign(desired)

falling_edges = np.where((signs[:-1] >= 0) & (signs[1:]<0))[0]+1 

subset = subset.iloc[falling_edges[1]:falling_edges[2]].copy()

subset["Time"] = subset["Time"] - subset["Time"].iloc[0]


# Error
error = subset["Desired Position"] - subset["Filtered Platen Rotation Angle"]
mean_abs_error = np.mean(np.abs(error))
std_abs_error = np.std(np.abs(error)) 
rms_error = np.sqrt((sum(error**2))/len(error))

# Command 
command = np.clip(subset["Command"], -1023, 1023)
mean_abs_command = np.mean(np.abs(command))
std_abs_command = np.std(np.abs(command))
rms_command = np.sqrt((sum(command**2))/len(command))

# Percent Overshoot
final_desired_angle = subset["Desired Position"].iloc[-1]
max_angle = max(subset["Filtered Platen Rotation Angle"])
initial_desired_angle = subset["Desired Position"].iloc[1]
percent_overshoot = (max_angle - final_desired_angle)/(final_desired_angle-initial_desired_angle)*(100)

# Steady_state_error 
steady_state_error = final_desired_angle - subset["Filtered Platen Rotation Angle"].iloc[-1] 

# Rise Time:
# Extract response after second step
response = subset[subset["Time"] >= 1].copy()

time = response["Time"].to_numpy()
angle = response["Filtered Platen Rotation Angle"].to_numpy()

# Define 10% and 90% thresholds
lower_threshold = -subset["Filtered Platen Rotation Angle"].iloc[-1] + 2 * subset["Filtered Platen Rotation Angle"].iloc[-1] * 0.1
upper_threshold = -subset["Filtered Platen Rotation Angle"].iloc[-1] + 2 * subset["Filtered Platen Rotation Angle"].iloc[-1] * 0.9

# Find first crossing of 10%
idx_10 = np.where(angle >= lower_threshold)[0]

if len(idx_10) > 0:
    i10 = idx_10[0]

    # Find first crossing of 90% after 10%
    idx_90 = np.where(angle[i10:] >= upper_threshold)[0]

    if len(idx_90) > 0:
        i90 = i10 + idx_90[0]
        rise_time = time[i90] - time[i10]
    else:
        rise_time = np.nan
else:
    rise_time = np.nan

# Setting Time:
final_angle = np.mean(angle[-50:])
lower_threshold = final_angle - (2*final_angle * 0.02)
upper_threshold = final_angle + (2*final_angle * 0.02)

# Find first crossing into band:
idx_first = np.where(angle >= lower_threshold)[0]

#Find last crossing into band after exiting the band:
idx_last = np.where((angle >= upper_threshold) | (angle <= lower_threshold))[-1]

if len(idx_first) > 0:
    i_first = idx_first[0]
    
    if len(idx_last) > 0:
        i_last = idx_last[-1]

        settling_time = time[i_last] - time[i_first]
    else:
        settling_time = np.nan
else: 
    settling_time = np.nan

# Print

print()
print(f"Kp: {subset["kp"].iloc[0]}")

print(f"Mean Abs Error: {mean_abs_error}")
print(f"Std Abs Error: {std_abs_error}")
print(f"RMS Error: {rms_error}")

print(f"Mean Abs Command: {mean_abs_command}")
print(f"Std Abs Command: {std_abs_command}")
print(f"RMS Command: {rms_command}")

print(f"Percent Overshoot: {percent_overshoot}")
print(f"Steady State Error: {steady_state_error}")
print(f"Rise Time: {rise_time}")
print(f"Settling Time: {settling_time}")
print()

# Plot:
plt.scatter(subset["Time"], subset["Filtered Platen Rotation Angle"],
    color="blue", label="Filtered Platen Rotation Angle", s=5)
plt.scatter(subset["Time"], subset["Desired Position"],
    color="red", label="Desired Position", s=5)
plt.scatter(subset["Time"], (command*26/1023),
        color="green", label="Command", s=5)
plt.xlabel("Time (s)")
plt.ylabel("Position (deg)")
plt.title(f"Tuned PID Controller Kp: {subset["kp"].iloc[0]}, Ki: {subset["ki"].iloc[0]}, Kd: {subset["kd"].iloc[0]}")

plt.text(1.15, 2.5, f"Mean Abs Error: {mean_abs_error:.3g} deg",fontsize=7)
plt.text(1.15, 0, f"Std Abs Error: {std_abs_error:.3g} deg",fontsize=7)
plt.text(1.15, -2.5, f"RMS Error: {rms_error:.3g} deg",fontsize=7)

plt.text(1.15, -7.5, f"Mean Abs Command: {mean_abs_command:.3g}",fontsize=7)
plt.text(1.15, -10, f"Std Abs Command: {std_abs_command:.3g}",fontsize=7)
plt.text(1.15, -12.5, f"RMS Command: {rms_command:.3g}",fontsize=7)

plt.text(1.15, -17.5, f"Percent Overshoot: {percent_overshoot:.3g}",fontsize=7)
plt.text(1.15, -20, f"Steady State Error: {steady_state_error:.3g}",fontsize=7)
    
plt.text(1.15, -22.5, f"Rise Time: {rise_time:.3g}",fontsize=7)
plt.text(1.15, -25, f"Settling Time: {settling_time:.3g}",fontsize=7)

plt.tight_layout()
plt.show()