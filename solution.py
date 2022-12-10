import numpy as np
import matplotlib.pyplot as plt

"""
    WHAT WE KNOW ABOUT THE PROBLEM:
        -> The signals have to be modeled as complex data (e.i. a+jb)
        -> There is a delay between the signals, which is a multiple of the sampling period T
        -> The delay also implies a phase difference between the signals, as modeled by the complex exponential
        -> There is an added noise signal
    USEFUL FACTS
        -> Sampled signal is s(kT) with samples at integer values of k
        -> FFT figures out which frequencies exist in that set of samples (the magnitude of the FFT indicates the strength of each frequency)
        -> FFT also figures out the delay (time shift needed to apply to each of those frequencies)
        -> The delay is the phase of the FFT                                                           
"""
SYMBOLS = 100
FREQ = 200
T = 1/FREQ
TIME = range(SYMBOLS) 
NRZ = np.random.randint(0, 2, SYMBOLS)

if __name__ == "__main__":
    print(str(NRZ))
    plt.plot(TIME, NRZ,'k-')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

# FREQ = np.random.randint(50,500)# Sample rate
# T = 1/FREQ # Sample period