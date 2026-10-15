import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Lab2/Kevin_pid_lab2-10-12-26-26-deg.csv")

kp_list = [10, 35, 110]

fig, axes = plt.subplots(
    1, len(kp_list),
    figsize=(16,5),
    sharex=True,
    sharey=True
)

for i, kp in enumerate(kp_list):
    subset = df[
        (df["kp"] == kp)
    ].copy()

    # Falling Edge Detection:

    desired = subset["Desired Position"].to_numpy()
    time = subset["Time"].to_numpy()

    signs = np.sign(desired)

    falling_edges = np.where((signs[:-1] >= 0) & (signs[1:]<0))[0]+1

    subset = subset.iloc[falling_edges[1]:falling_edges[2]].copy()

    subset["Time"] = subset["Time"] - subset["Time"].iloc[0] #Initialize Time


    # Error
    error = subset["Desired Position"] - subset["Filtered Platen Rotation Angle"]
    mean_abs_error = np.mean(np.abs(error))
    std_abs_error = np.std(np.abs(error))
    rms_error = np.sqrt((sum(error**2))/len(error))

    # Command
    command = subset["Command"]
    mean_abs_command = np.mean(np.abs(command))
    std_abs_command = np.std(np.abs(command))
    rms_command = np.sqrt((sum(command**2))/len(command))

     # Percent Overshoot
    final_desired_angle = subset["Desired Position"].iloc[-1]
    max_angle = max(subset["Filtered Platen Rotation Angle"])
    initial_desired_angle = subset["Desired Position"].iloc[1]
    percent_overshoot = (max_angle - final_desired_angle)/(final_desired_angle-initial_desired_angle)*(100)

    # Steady_state_error 
    steady_state_erorr = subset["Filtered Platen Rotation Angle"].iloc[-1] - final_desired_angle

    # Rise Time:
    # Extract response after second step
    response = subset[subset["Time"] >= 1].copy()

    time = response["Time"].to_numpy()
    angle = response["Filtered Platen Rotation Angle"].to_numpy()

    # Define 10% and 90% thresholds
    lower_threshold = -final_desired_angle + 2 * final_desired_angle * 0.1
    upper_threshold = -final_desired_angle + 2 * final_desired_angle * 0.9

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
    print(len(idx_last))

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
    print(f"Kp: {subset["kp"].iloc[0]} deg")
    
    print(f"Mean Abs Error: {mean_abs_error}")
    print(f"Std Abs Error: {std_abs_error}")
    print(f"RMS Error: {rms_error}")

    print(f"Mean Abs Command: {mean_abs_command}")
    print(f"Std Abs Command: {std_abs_command}")
    print(f"RMS Command: {rms_command}")

    print(f"Percent Overshoot: {percent_overshoot}")
    print(f"Steady State Error: {steady_state_erorr}")
    print(f"Rise Time: {rise_time}")
    print(f"Settling Time: {settling_time}")
    print()

    axes[i].scatter(subset["Time"], subset["Filtered Platen Rotation Angle"],
        color="blue", label="Filtered Platen Rotation Angle", s=5)
    axes[i].scatter(subset["Time"], subset["Desired Position"],
        color="red", label="Desired Position", s=5)
    axes[i].set_xlabel("Time (s)")
    axes[i].set_ylabel("Position (deg)")
    axes[i].set_title(f"Kp: {subset["kp"].iloc[0]}")

    axes[i].text(1.25, 15, f"Mean Abs Error: {mean_abs_error:.2f} deg",fontsize=7)
    axes[i].text(1.25, 10, f"Std Abs Error: {std_abs_error:.2f} deg",fontsize=7)
    axes[i].text(1.25, 5, f"RMS Error: {rms_error:.2f} deg",fontsize=7)

    axes[i].text(1.25, -5, f"Mean Abs Command: {mean_abs_command:.2f}",fontsize=7)
    axes[i].text(1.25, -10, f"Std Abs Command: {std_abs_command:.2f}",fontsize=7)
    axes[i].text(1.25, -15, f"RMS Command: {rms_command:.2f}",fontsize=7)

    axes[i].text(1.25, -25, f"Percent Overshoot: {percent_overshoot:.2f}",fontsize=7)
    axes[i].text(1.25, -30, f"Steady State Error: {steady_state_erorr:.2f}",fontsize=7)
    
    axes[i].text(1.25, -35, f"Rise Time: {rise_time:.2f}",fontsize=7)
    axes[i].text(1.25, -40, f"Settling Time: {settling_time:.2f}",fontsize=7)

plt.tight_layout()
plt.show()