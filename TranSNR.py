import numpy as np
import matplotlib.pyplot as plt

# Parameters
Vref = 1.0  # Full-scale range: -1 to 1 V
f_input = 1e9  # 1 GHz tone
f_sample = 40e9  # 20 GHz sampling
A = 1  # Input amplitude
duration = 1e-9  # sim duration
offset = 0 # Comparator offset
num_stages = 2  # stages pipeline

# Time and input signal
Ts = 1 / f_sample
time = np.arange(0, duration, Ts)
Vin = A * np.sin(2 * np.pi * f_input * time)  # Input: -0.5 to 0.5 V

# Ideal digital output (for SNR calculation)
bits_total = 8  # 4 stages, 2 bits each (effective resolution after correction)
LSB = (2 * Vref) / (2**bits_total)
Vin_ideal = np.round((Vin + Vref) / LSB) * LSB - Vref  # Quantized ideal

# 2-Bit Stage (No Redundancy)
def stage_2bit(Vin, offset=0):
    thresholds = np.array([-0.5, 0, 0.5]) + offset
    G = 4
    bits = np.zeros_like(Vin, dtype=int)
    Vout = np.zeros_like(Vin)
    dac = np.array([-0.75, -0.25, 0.25, 0.75]) * Vref  # Adjusted for symmetry
    for i, v in enumerate(Vin):
        if v < thresholds[0]:
            bits[i] = 0  # -1.5
            Vout[i] = G * (v - dac[0])
        elif v < thresholds[1]:
            bits[i] = 1  # -0.5
            Vout[i] = G * (v - dac[1])
        elif v < thresholds[2]:
            bits[i] = 2  # 0.5
            Vout[i] = G * (v - dac[2])
        else:
            bits[i] = 3  # 1.5
            Vout[i] = G * (v - dac[3])
    return Vout, bits

# 2.5-Bit Stage (With Redundancy)
def stage_25bit_red(Vin, offset=0):
    thresholds = np.array([-5/8, -3/8, -1/8, 1/8, 3/8, 5/8]) + offset
    G = 4
    bits = np.zeros_like(Vin, dtype=int)
    Vout = np.zeros_like(Vin)
    dac = np.array([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]) * Vref
    for i, v in enumerate(Vin):
        if v < thresholds[0]:
            bits[i] = 0  # -3
            Vout[i] = G * (v - dac[0])
        elif v < thresholds[1]:
            bits[i] = 1  # -2
            Vout[i] = G * (v - dac[1])
        elif v < thresholds[2]:
            bits[i] = 2  # -1
            Vout[i] = G * (v - dac[2])
        elif v < thresholds[3]:
            bits[i] = 3  # 0
            Vout[i] = G * (v - dac[3])
        elif v < thresholds[4]:
            bits[i] = 4  # 1
            Vout[i] = G * (v - dac[4])
        elif v < thresholds[5]:
            bits[i] = 5  # 2
            Vout[i] = G * (v - dac[5])
        else:
            bits[i] = 6  # 3
            Vout[i] = G * (v - dac[6])
    return Vout, bits

# 2.5-Bit Stage (No Redundancy)
def stage_25bit_no_red(Vin, offset=0):
    thresholds = np.array([-5/7, -3/7, -1/7, 1/7, 3/7, 5/7]) + offset
    G = 7  # As per your code
    bits = np.zeros_like(Vin, dtype=int)
    Vout = np.zeros_like(Vin)
    dac = np.array([-6/7, -4/7, -2/7, 0, 2/7, 4/7, 6/7]) * Vref
    for i, v in enumerate(Vin):
        if v < thresholds[0]:
            bits[i] = 0  # -3
            Vout[i] = G * (v - dac[0])
        elif v < thresholds[1]:
            bits[i] = 1  # -2
            Vout[i] = G * (v - dac[1])
        elif v < thresholds[2]:
            bits[i] = 2  # -1
            Vout[i] = G * (v - dac[2])
        elif v < thresholds[3]:
            bits[i] = 3  # 0
            Vout[i] = G * (v - dac[3])
        elif v < thresholds[4]:
            bits[i] = 4  # 1
            Vout[i] = G * (v - dac[4])
        elif v < thresholds[5]:
            bits[i] = 5  # 2
            Vout[i] = G * (v - dac[5])
        else:
            bits[i] = 6  # 3
            Vout[i] = G * (v - dac[6])
    return Vout, bits

# Pipeline Simulation
# 2-Bit per Stage
Vin_2bit = Vin.copy()
bits_2bit = []
for stage in range(num_stages):
    Vout, bits = stage_2bit(Vin_2bit, offset)
    bits_2bit.append(bits)
    Vin_2bit = Vout

# 2.5-Bit per Stage with Redundancy
Vin_25bit_red = Vin.copy()
bits_25bit_red = []
for stage in range(num_stages):
    Vout, bits = stage_25bit_red(Vin_25bit_red, offset)
    bits_25bit_red.append(bits)
    Vin_25bit_red = Vout

# 2.5-Bit per Stage without Redundancy
Vin_25bit_no_red = Vin.copy()
bits_25bit_no_red = []
for stage in range(num_stages):
    Vout, bits = stage_25bit_no_red(Vin_25bit_no_red, offset)
    bits_25bit_no_red.append(bits)
    Vin_25bit_no_red = Vout

# Digital Reconstruction
# 2-Bit: Direct binary (no redundancy)
Dout_2bit = np.zeros_like(Vin)
for stage in range(num_stages):
    Dout_2bit += bits_2bit[stage] * (4**(num_stages - 1 - stage))
Dout_2bit = (Dout_2bit / (4**num_stages)) * 2 * Vref - Vref  # Scale to -1 to 1 V

# 2.5-Bit with Redundancy: Apply digital correction
Dout_25bit_red = np.zeros_like(Vin, dtype=int)
for i in range(len(Vin)):
    raw_bits = [bits_25bit_red[stage][i] for stage in range(num_stages)]
    # Digital correction: Treat as 2-bit per stage with overlap
    corrected = 0
    for stage in range(num_stages):
        b = raw_bits[stage]
        if b == 0:
            b_val = -3
        elif b == 1:
            b_val = -2
        elif b == 2:
            b_val = -1
        elif b == 3:
            b_val = 0
        elif b == 4:
            b_val = 1
        elif b == 5:
            b_val = 2
        else:
            b_val = 3
        corrected += b_val * (4**(num_stages - 1 - stage))
    Dout_25bit_red[i] = corrected
Dout_25bit_red = (Dout_25bit_red / (4**num_stages)) * Vref #- Vref  # Scale to -1 to 1 V

# 2.5-Bit without Redundancy: No correction
Dout_25bit_no_red = np.zeros_like(Vin)
for stage in range(num_stages):
    bits = bits_25bit_no_red[stage]
    b_val = np.zeros_like(bits)
    b_val[bits == 0] = -3
    b_val[bits == 1] = -2
    b_val[bits == 2] = -1
    b_val[bits == 3] = 0
    b_val[bits == 4] = 1
    b_val[bits == 5] = 2
    b_val[bits == 6] = 3
    Dout_25bit_no_red += b_val * (4**(num_stages - 1 - stage))
Dout_25bit_no_red = (Dout_25bit_no_red / (4**num_stages)) * Vref #- Vref  # Scale to -1 to 1 V

# Calculate SNR
def calculate_snr(ideal, actual):
    signal_power = np.mean(ideal**2)
    noise_power = np.mean((ideal - actual)**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

snr_2bit = calculate_snr(Vin, Dout_2bit)
snr_25bit_red = calculate_snr(Vin, Dout_25bit_red)
snr_25bit_no_red = calculate_snr(Vin, Dout_25bit_no_red)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time * 1e9, Dout_2bit, 'b-', label=f'SNR={snr_2bit:.3f} dB (2 bit)')
plt.plot(time * 1e9, Dout_25bit_red, 'r-', label=f'SNR={snr_25bit_red:.3f} dB (2.5 bits with offset tolerance)')
plt.plot(time * 1e9, Dout_25bit_no_red, 'g-', label=f'SNR={snr_25bit_no_red:.3f} dB (2.5 bits with NO offset tolerance)')
plt.plot(time * 1e9, Vin, 'k--', label='Ideal Input', alpha=0.5)
plt.xlabel('time')
plt.ylabel('Dout')
plt.title(f"Effect of Offset\n{num_stages:.0f}-Stage Pipeline ADC with 2 and 2.5 bits per stage and Voffset= Vref*{offset/Vref:.0f}")
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

# Print SNR values
print(f"SNR 2-Bit: {snr_2bit:.3f} dB")
print(f"SNR 2.5-Bit with Redundancy: {snr_25bit_red:.3f} dB")
print(f"SNR 2.5-Bit without Redundancy: {snr_25bit_no_red:.3f} dB")
