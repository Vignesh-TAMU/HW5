import numpy as np
import matplotlib.pyplot as plt

# Reference voltage
Vref = 1.0
G = 4  # Gain of the stage

# Input voltage range
Vin = np.linspace(-Vref, Vref, 1000)

# 2-Bit Stage (No Redundancy)
thresholds_2bit = [-0.5, 0, 0.5]  # Comparator thresholds
codes_2bit = [-3/4, -1/4, 1/4, 3/4]  # Corresponding Zero values 
Vout_2bit = np.zeros_like(Vin)
for i, v in enumerate(Vin):
    if v < thresholds_2bit[0]:
        Vout_2bit[i] = G * (v - (-3/4*Vref))
    elif v < thresholds_2bit[1]:
        Vout_2bit[i] = G * (v - (-1/4 * Vref))
    elif v < thresholds_2bit[2]:
        Vout_2bit[i] = G * ( v-(1/4 * Vref))
    else:
        Vout_2bit[i] = G * (v - (3/4 * Vref))

# 2.5-Bit Stage with Redundancy
thresholds_25bit_red = [-5/8, -3/8, -1/8, 1/8, 3/8, 5/8 ]  # 6 thresholds
codes_25bit_red = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]  # Zero voltages
Vout_25bit_red = np.zeros_like(Vin)
for i, v in enumerate(Vin):
    if v < thresholds_25bit_red[0]:
        Vout_25bit_red[i] = G * (v - (-0.75 * Vref))
    elif v < thresholds_25bit_red[1]:
        Vout_25bit_red[i] = G * (v - (-0.5 * Vref))
    elif v < thresholds_25bit_red[2]:
        Vout_25bit_red[i] = G * (v - (-0.25 * Vref))
    elif v < thresholds_25bit_red[3]:
        Vout_25bit_red[i] = G * (v - 0)
    elif v < thresholds_25bit_red[4]:
        Vout_25bit_red[i] = G * (v - (0.25 * Vref))
    elif v < thresholds_25bit_red[5]:
        Vout_25bit_red[i] = G * (v - (0.5 * Vref))
    #elif v < thresholds_25bit_red[6]:
    #    Vout_25bit_red[i] = G * (v - (0.75 * Vref))
    else:
        Vout_25bit_red[i] = G * (v - (0.75 * Vref))  # Clipped at max

# 2.5-Bit Stage without Redundancy (Simplified to 4 levels within Vref)
G=7
thresholds_25bit_no_red = [-5/7, -3/7, -1/7, 1/7, 3/7, 5/7]  # Fewer thresholds
codes_25bit_no_red = [-0.5, -0.25, 0, 0.25, 0.5]  # Zero voltages
Vout_25bit_no_red = np.zeros_like(Vin)
for i, v in enumerate(Vin):
    if v < thresholds_25bit_no_red[0]:
        Vout_25bit_no_red[i] = G * (v - (-6/7 * Vref))
    elif v < thresholds_25bit_no_red[1]:
        Vout_25bit_no_red[i] = G * (v - (-4/7 * Vref))
    elif v < thresholds_25bit_no_red[2]:
        Vout_25bit_no_red[i] = G * (v - (-2/7 * Vref))
    elif v < thresholds_25bit_no_red[3]:
        Vout_25bit_no_red[i] = G * (v - (0/7 * Vref))
    elif v < thresholds_25bit_no_red[4]:
        Vout_25bit_no_red[i] = G * (v - (2/7 * Vref))
    elif v < thresholds_25bit_no_red[5]:
        Vout_25bit_no_red[i] = G * (v - (4/7 * Vref))
    #elif v < thresholds_25bit_no_red[6]:
        #Vout_25bit_no_red[i] = G * (v - (0.25 * Vref))
    else:
        Vout_25bit_no_red[i] = G * (v - (6/7 * Vref))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(Vin, Vout_2bit , 'b-', label='2-Bit Stage')
plt.plot(Vin, Vout_25bit_red, 'r-', label='2.5-Bit with Redundancy')
plt.plot(Vin, Vout_25bit_no_red, 'g-', label='2.5-Bit without Redundancy')
plt.xlabel('Input Voltage (V)')
plt.ylabel('Output Voltage (V)')
plt.title('Transfer Functions of ADC Stages')
plt.grid(True)
plt.legend(loc='upper left')

# Add vertical lines for thresholds (optional for clarity)
for t in thresholds_2bit:
    plt.axvline(x=t, color='b', linestyle='--', alpha=0.3)
for t in thresholds_25bit_red:
    plt.axvline(x=t, color='r', linestyle='--', alpha=0.3)
for t in thresholds_25bit_no_red:
    plt.axvline(x=t, color='g', linestyle='--', alpha=0.3)

plt.show()
