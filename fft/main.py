import numpy as np
import matplotlib.pyplot as plt
from dft import DFT

# sampling rate
sr = 2048
# sampling interval
ts = 1.0/sr
t = np.arange(0,1,ts)

freq = 1.
x = 3*np.sin(2*np.pi*freq*t)

freq = 4
x += np.sin(2*np.pi*freq*t)

freq = 7   
x += 0.5* np.sin(2*np.pi*freq*t)

fig, axs = plt.subplots(3, 1)
plt.tight_layout(pad=0.8)
axs[0].plot(t, x, 'r')
axs[0].set_ylabel('Amplitude')
axs[0].set_xlabel('Time (s)')
axs[0].set_title('Signal')

times = []
for i in range(0, 2):
    X, time = DFT(x, i)

    times.append(time)

N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T 

axs[1].stem(freq[0:int(N/2)], abs(X[0:int(N/2)]), 'b', \
         markerfmt=" ", basefmt="-b")
axs[1].set_title('DFT (slow)')
axs[1].set_xscale('log')
axs[1].set_xlabel('Freq (Hz)')
axs[1].set_ylabel('DFT Amplitude |X(freq)|')

axs[2].bar(np.arange(0, 2), times, 0.5, color='g')

plt.savefig('dft.png')
plt.show()