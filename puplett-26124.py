import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
from scipy.optimize import curve_fit

# for time, it does not work for now.

f= 0.2  # Adjust frequency to control the oscillation
phi = 45
phi2 = 45

np.random.seed(42)

x = np.linspace(47,53,10000)

tau = np.linspace(5,15,10**4)#(np.max(x)-np.min(x))*len(x)/3000,10000)


# Generate Sample Data
def U(variable, A, f, phi, decay_rate):
    funct = A * np.exp(-decay_rate * (variable - np.mean(variable))**2) * np.sin(2 * np.pi * f * (variable - np.mean(variable)) + phi)
    return funct

# Find difference interferogram using generated data
def difference_interferogram(U_h, U_v):
    delta = (U_h - U_v)/( U_h + U_v)
    return delta

# Create Sample data for x space
def sample_data(variable):
    np.random.seed(42)
    num = len(variable)

    Uv = -U(variable, 1.0, f, phi , 0.2) + 2 #+ np.random.randn(num) /20
    Uh =  U(variable, 1.2, f, phi2, 0.2) + 2 #+ np.random.randn(num) / 20

    delta = difference_interferogram(Uh, Uv)
    return Uv, Uh, delta


# Plot the graph for :
# difference interferogram (delta_x)
# Uh_x ; Uvx
def plot_Uh_Uv_delta(variable):
    num = len(variable)
    Uv = sample_data(variable)[0] +2 #-U(variable,0.8, f, phi,1) + 1 #+ np.random.rand(num)/20
    np.random.seed(35)
    Uh = sample_data(variable)[1] +2 #U(variable,1.2, f, phi2,1) + 1 #+ np.random.rand(num)/20

    delta = difference_interferogram(Uh,Uv)


    plt.subplot(2,1,1)
    plt.plot(variable,Uv,"-b")
    plt.plot(variable,Uh,"-r")
    plt.legend(["U_v", "U_h"])
    plt.xlabel("roof mirror poisition (mm)")
    plt.ylabel("Signal amplitude (V)")
    plt.grid()


    plt.subplot(2,1,2)
    plt.plot(variable,delta,"-",color = "orange")
    plt.legend(["difference interferogram (x)"])
    plt.xlabel("roof mirror poisition (mm)")
    plt.ylabel("Signal amplitude (V)")
    plt.grid()

    #plt.show()

#plt.subplot(2,1,1)
plot_Uh_Uv_delta(variable= tau)

#plt.subplot(2,1,2)
#plot_Uh_Uv_delta(variable= t)
plt.show()

# propagate gaussian error for delta_x
def error_propagate_for_deltax(Uh,Uv):
    sh = np.std(Uh)
    sv = np.std(Uv)
    A = np.power(Uh*sv,2)
    B = np.power(Uv*sh,2)

    s_delta = 2*np.sqrt(A+B)/np.power(Uh + Uv, 2)
    return s_delta

error_in_delta_x = error_propagate_for_deltax(sample_data(x)[0],sample_data(x)[1])

print(error_in_delta_x)

from scipy.fftpack import idct

data= sample_data(tau)[2]
output = idct(data, type = 1)
"""
plt.plot(tau,output)
plt.title("Inverse Cosine Fourier Transform of \nDifference Interferogram")
plt.xlim([0,0.20])
plt.grid()
# plt.show()"""
"""
b_lim = 2
u_lim = 3
wk = np.linspace(b_lim,u_lim,10**4)
Iwk = []
for i in range(len(wk)):
    I = np.real(sample_data(tau)[2][i]* np.exp(1j * wk[i]*tau[i]))
    Iwk.append(I)
plt.plot(wk,Iwk,"-r")
plt.title("Eqn 6.1 is applied for I(w_k)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Intensity (Arbitrary unit) (I(w_k))")
plt.xlim([2.35, 2.65])
plt.grid()
plt.show()"""


def ST(t,sigma):
    e = 1
    St1 = np.exp(-(t ** 2) / (4 * (sigma ** 2)))
    St2 = 2 * sigma * np.exp(-np.power(t, 2) / (4 * ((sigma ** 2) + (e**2)))) / np.sqrt(
        (sigma ** 2) + (e**2))
    St3 = sigma * np.exp(-np.power(t, 2) / (4 * ((sigma ** 2) + (2 * (e**2))))) / np.sqrt(
        (sigma ** 2) + 2 * (e**2))
    St = St1 - St2 + St3
    return St

###############################
# Function to fit data using ST
def fit_data(variable, initial_guess):
    x_data = variable
    y_data = sample_data(variable)[2]  # Use the delta data for fitting

    # Perform the curve fit
    popt, pcov = curve_fit(ST, x_data, y_data, p0=initial_guess)

    # Plot the original data and the fit
    plt.plot(x_data, y_data, 'o', label='data')
    plt.plot(x_data, ST(x_data, *popt), '-', label='fit: sigma=%5.3f, e=1 ' % tuple(popt))
    plt.plot(tau, ST(tau, 10),"--r",label = "ST")
    plt.legend()
    plt.show()

# Initial guess for the parameters (adjust as needed)
initial_guess = [1]

# Call the function to fit the data
fit_data(variable=tau, initial_guess=initial_guess)

"""
def Iw(N,delta_tN,w_k,t_N):
    sum = 0
    for i in range(1,N+1):
        s = delta_tN*np.exp(complex(0,w_k*t_N))
        sum += s
    I_wk = 1/N * sum
    return I_wk


def fft_Iw(N, delta_tN, w_k, t_N):
    t_min = -4  # replace with your actual minimum time value
    t_max = t_N  # replace with your actual maximum time value

    # Generate time values
    t_values = np.linspace(t_min, t_max, N, endpoint=False)

    # Calculate the signal values at each time point
    signal_values = delta_tN * np.exp(1j * w_k * t_values)

    # Perform FFT
    fft_result = np.fft.fft(signal_values)

    # Extract non-redundant intensity values
    K = N // 2
    intensity_values = np.abs(fft_result[:K]) / N

    # Frequency values corresponding to the intensity values
    frequency_values = np.fft.fftfreq(N, d=(t_max - t_min)/N)[:K]

    return frequency_values, intensity_values

# Example usage:
N = 2000  # replace with your desired number of points
delta_tN = 0.1  # replace with your desired time step
w_k = 2 * np.pi  # replace with your desired frequency
t_N = 4  # replace with your desired time duration

frequency_values, intensity_values = fft_Iw(N, sample_data_x()[2], w_k, t_N)
print(frequency_values)
print(intensity_values)

plt.plot(frequency_values,intensity_values,"-r")
plt.grid()
# plt.show()


# Function to calculate the discrete cosine transform
def discrete_cosine_transform(signal, time_values):
    N = len(signal)
    frequencies = np.fft.fftfreq(N, d=(time_values[1] - time_values[0]))
    frequencies = frequencies[:N//2]  # Use only non-negative frequencies

    dct_values = np.fft.fft(signal)
    dct_values = np.real(dct_values)[:N//2]  # Take the real part, and consider only non-negative frequencies

    return frequencies, dct_values

# Example signal and time values
tau_min = -4
tau_max = 4
N = 2000  # Number of points
time_values = np.linspace(tau_min, tau_max, N)
signal = sample_data_x()[2]

# Compute discrete cosine transform
frequencies, dct_values = discrete_cosine_transform(signal, time_values)

# Plot the original signal and its discrete cosine transform
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time_values, signal)
plt.title('Original Signal')

plt.subplot(2, 1, 2)
plt.stem(frequencies, dct_values, markerfmt='bo', basefmt='b-', linefmt='r-')
plt.title('Discrete Cosine Transform')

plt.tight_layout()
#plt.show()

import numpy as np
from scipy.fft import fft

def calculate_spectrum(tau_min, tau_max, N):
    # Generate the time values (equidistant sampling)
    tau_values = np.linspace(tau_min, tau_max, N, endpoint=False)

    # Frequency values for the spectrum
    w_values = np.fft.fftfreq(N, d=(tau_max - tau_min) / N)

    # Compute the spectrum using the provided formula
    spectrum = np.real(fft(np.sinc(tau_values)) / N)

    return w_values, spectrum


# Calculate the spectrum
frequency_values, intensity_spectrum = calculate_spectrum(tau_min, tau_max, N)

# Plot the spectrum
import matplotlib.pyplot as plt

plt.plot(frequency_values, intensity_spectrum)
plt.xlabel('Frequency (w)')
plt.ylabel('Intensity (I(w))')
plt.title('Spectrum Calculation using Discrete Cosine Transform')
# plt.show()
"""



"""
def symmetric_double_sided_damping_sinusoidal(x):
    # Parameters
    amplitude = 1.0
    frequency = 2  # Adjust frequency to control the oscillation
    damping_factor = 0.2  # Damping factor

    # Symmetric double-sided damping sinusoidal function
    y = amplitude * np.exp(-damping_factor * (x - 50)**2) * np.sin(2 * np.pi * frequency * (x - 50))

    return y

# Generate x values
x_values = np.linspace(47, 53, 2000)

# Calculate y values
y_values = symmetric_double_sided_damping_sinusoidal(x_values)

# Plot the graph
plt.plot(x_values, y_values, label='Symmetric Double-Sided Damping Sinusoidal')
plt.title('Symmetric Double-Sided Damping Sinusoidal Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
"""