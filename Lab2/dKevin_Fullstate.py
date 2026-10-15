import numpy as np
from scipy.signal import place_poles
from control import acker

tau = 0.030 # From Lab 1
dc_gain = 1.72 # From Lab 1

percent_overshoot = 1 # From Lab Assignment
settling_time = 0.2 # From Lab Assignment

damping_ratio = np.sqrt(np.log(percent_overshoot/100)**2/(np.pi**2 + np.log(percent_overshoot/100)**2))
natural_frequency = 4/(damping_ratio*settling_time)

damped_frequency = natural_frequency*np.sqrt(1-damping_ratio**2)

poles = [complex(-damping_ratio*natural_frequency,damped_frequency), complex(-damping_ratio*natural_frequency,-damped_frequency)]

#System matrices A and B
A = np.array([[0, 1], [0 , -1/tau]])
B = np.array([[0],[dc_gain/tau]])

K = acker(A, B, poles)

print("Feedback Gain Matrix K:\n", K)

print("Natural Frequency:", natural_frequency)
print("Damping Ratio:",damping_ratio)


# K = [10.22358634  0.11627907]