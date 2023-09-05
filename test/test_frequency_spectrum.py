import numpy as np
import scipy 
import matplotlib.pyplot as plt
import os,sys
sys.path.append(os.getcwd())



r1_after_presmooth = np.loadtxt("r1_after_presmooth.txt")
r1_afterprolongate = np.loadtxt("r1_after_prolongate.txt")
r1_after_postsmooth = np.loadtxt("r1_after_postsmooth.txt")


fig, ax = plt.subplots(3, 1)

frequencies, spectrum = scipy.signal.periodogram(r1_after_presmooth)

# 绘制频谱图
ax[0].plot(frequencies, spectrum, color="red")
ax[0].set_xlabel('Frequency (Hz)')
ax[0].set_ylabel('Power Spectral Density')
ax[0].set_title('Power Spectrum r1_after_pre_smooth')
ax[0].set_ylim(0, 0.1)

frequencies, spectrum = scipy.signal.periodogram(r1_afterprolongate)

# 绘制频谱图
ax[1].plot(frequencies, spectrum, color="green")
ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Power Spectral Density')
ax[1].set_title('Power Spectrum  r1_after_prolongate')
ax[1].set_ylim(0, 0.1)

frequencies, spectrum = scipy.signal.periodogram(r1_after_postsmooth)

# 绘制频谱图
ax[2].plot(frequencies, spectrum, color="blue")
ax[2].set_xlabel('Frequency (Hz)')
ax[2].set_ylabel('Power Spectral Density')
ax[2].set_title('Power Spectrum r1_after_post_smooth')
ax[2].set_ylim(0, 0.1)

plt.tight_layout()
plt.show()