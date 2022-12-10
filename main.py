import numpy as np
import matplotlib.pyplot as plt

"""
    WHAT WE KNOW ABOUT THE PROBLEM:
        -> The signals have to be modeled as complex data (e.i. a+jb)
        -> There is a delay between the sampled sequences, which is a multiple of the sampling period T (no fractional samples)
        -> There is a phase difference between the sampled sequences, as modeled by the complex exponential
        -> There is added noise to both sequences
        -> Sampled signal is s(kT) with samples at integer values of k
    BONUS QUESTIONS:
        1) Assume that the delay between the two receivers is an unknown multiple of the sample period. This is generally not the case. Why? What is going on in this assumption? How would the problem be different if this assumption was not made?                          
            -> Making the assumption that the delay is an integer multiple of the sample period makes the calculation simpler because it can just be done by shifting the symbols, but there is no guarantee the delay will follow this rule in real life. 
        2) There is an unknown complex phase difference between the two receivers. Why? Where does the complex phase at each of the receivers come from? 
            ->  Because the receivers are spaced out, the signal arrives to them at different moments in time which causes the phase shift.
"""
# constants
SYMBOLS = 200 # number of samples to simulate
DELAY = np.random.randint(1,100) # random integer multiple to delay
PHASE_DIFF = np.pi/2 * np.random.uniform() # random phase difference 

# signal data
s = np.random.randint(0, 2, SYMBOLS) # 0s and 1s

# expand complex exponential with Euler's formula
complex_exp = complex(np.cos(PHASE_DIFF),np.sin(PHASE_DIFF))

# noise power
noise_power = 0.01

# AWGN with unit power
n_1 = (np.random.randn(SYMBOLS) + 1j*np.random.randn(SYMBOLS))/np.sqrt(2)
n_2 = (np.random.randn(SYMBOLS) + 1j*np.random.randn(SYMBOLS))/np.sqrt(2)

# received complex sequences
r_1 = s + (n_1 * np.sqrt(noise_power))
r_2 = np.roll(([val * complex_exp for val in s]), DELAY) + (n_2 * np.sqrt(noise_power))

# SNR
SNR_r_1= np.mean(np.abs(r_1)**2)/np.mean(np.abs(n_1)**2)
SNR_r_2= np.mean(np.abs(r_2)**2)/np.mean(np.abs(n_2)**2)

def plotIQ(signal, title):
    plt.figure()
    plt.title(title)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.plot(np.real(signal), np.imag(signal), '.')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid()

if __name__ == "__main__":
    # PRE-PROCESSING
    print("We start by generating a random data for a NRZ signal. This signal is sampled by two receivers who have an randomly generated unknown time delay and phase difference between them. Due to the nature of the question, I decided to do my signal manipulations with IQ data (complex numbers). The following are the random parameters generated for run:")
    print("     Symbols: " + str(SYMBOLS))
    print("     Unknown delay: " + str(DELAY))
    print("     Unknown phase difference (degrees): " + str(PHASE_DIFF*(180/np.pi)))
    print()

    plotIQ(r_1, "Pre-processing: r_1, IQ")
    plotIQ(r_2, "Pre-processing: r_2, IQ")

    # POST-PROCESSING
    print("SNR signal-to-noise ratio = signal power/noise power: ")
    print("     SNR_r_1: " + str(SNR_r_1))
    print("     SNR_r_2: " + str(SNR_r_2))
    print()

    # Find unknown delay
    corr = np.correlate(r_1 - np.mean(r_1), r_2 - np.mean(r_2), mode='full')
    lag = corr.argmax() - (len(r_1) - 1)
    print("For post-processing, we first do a cross correlation to find the time delay between the receivers. We correct for it by shifting the first receiver by the lag.")
    print("     lag: " + str(lag))
    print()

    # Fix delay
    r_1 = np.roll(r_1, lag)

    # Find phase shift
    r_1_mean = np.mean(r_1)
    r_2_mean = np.mean(r_2)
    phase_1 = np.arctan2(np.imag([r_1_mean]), np.real([r_1_mean]))
    phase_2 = np.arctan2(np.imag([r_2_mean]), np.real([r_2_mean]))
    difference = phase_2-phase_1
    print("Next, (and this is an estimation of a way I thought might mostly work) we take the mean of both receivers, which would give a single average complex number for each. I found the inner angle of the triangle the vector forms which I considered the phase of that receiver, and than I took their difference. This gives the phase difference between the receivers with an error margin of up to about 1 degree (which I took for the sake of the assignment:")
    print("     phase difference between receivers:" + str((difference*(180/np.pi))))
    print()
    print("We adjusted for the phase shift by shifting the phase of the first receiver by the difference. The post-processing plots show how the signals are now aligned.")
    print()
    # Fix phase shift
    complex_exp = complex(np.cos(difference), np.sin(difference))
    r_1 = [val * complex_exp for val in r_1]

    plotIQ(r_1, "Post-processing: r_1, IQ")
    plotIQ(r_2, "Post-processing: r_2, IQ")

    plt.show()